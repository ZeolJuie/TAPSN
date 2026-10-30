"""Compile a ResolvedCase into an executable scenario directory.

The compiler is deterministic. It formats the resolved map, actors, weather,
vehicle density, and sampling settings into scenario.toml, validates the
resulting configuration, and writes the case metadata next to it.

Rendered TOML is parsed back before it is written, so a renderer error is
detected at build time instead of when the simulator starts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from scenariogen.models import ActorSpec, BuiltCase, ResolvedCase
from scenariogen.schema import WEATHER_PRESETS


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STEP_LENGTH = 0.05
DEFAULT_MAP_PATH = "Carla/Maps/"
DEFAULT_CARLA_TIMEOUT = 120.0
DEFAULT_SENSOR_RIG = REPO_ROOT / "config" / "sensors" / "default_multimodal.toml"
DEFAULT_SAMPLING_POLICY = REPO_ROOT / "config" / "sampling" / "default.toml"

MAX_PEDESTRIANS = 16
MAX_SPAWNERS = 16

VEHICLE_DENSITY_CONFIG = {
    "none": {"local_count": 0, "global_count": 0},
    "low": {"local_count": 8, "global_count": 8},
    "medium": {"local_count": 12, "global_count": 16},
    "high": {"local_count": 16, "global_count": 24},
}


class CompilationError(ValueError):
    """Raised when a resolved case cannot be compiled into a valid scenario."""


def compile_case(
    case: ResolvedCase,
    *,
    case_dir: Path,
    user_prompt: str = "",
) -> BuiltCase:
    validate_resolved_case(case)

    case_dir = Path(case_dir).resolve()
    config_dir = case_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(case, case_dir=case_dir)
    scenario_text = render_toml(config)
    _check_toml_round_trip(config, scenario_text)

    scenario_path = config_dir / "scenario.toml"
    scenario_path.write_text(scenario_text, encoding="utf-8")
    (config_dir / "scene_description.txt").write_text(
        case.plan.description.rstrip() + "\n",
        encoding="utf-8",
    )

    spec_path = config_dir / "scenario_spec.json"
    spec_payload = {
        "user_prompt": user_prompt,
        "plan": case.plan.to_dict(),
        "resolved_case": case.to_dict(),
    }
    spec_path.write_text(
        json.dumps(spec_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config_dir / "case.json").write_text(
        json.dumps(case.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if case.sampling_enabled:
        _write_sampling_policy(case, config_dir=config_dir)
        _copy_sensor_rig(config_dir)

    manifest_path = case_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "case_index": case.case_index,
                "map_name": case.map_name,
                "scene_id": case.scene_id,
                "scenario_name": config["scenario_name"],
                "case_dir": str(case_dir),
                "config_dir": str(config_dir),
                "scenario_path": str(scenario_path),
                "spec_path": str(spec_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return BuiltCase(
        resolved=case,
        case_dir=case_dir,
        scenario_path=scenario_path,
        spec_path=spec_path,
        config=config,
    )


def build_config(case: ResolvedCase, *, case_dir: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "scenario_name": _scenario_name(case),
        "step_length": DEFAULT_STEP_LENGTH,
        "map": {
            "map_name": case.map_name,
            "map_path": DEFAULT_MAP_PATH,
            "carla_timeout": DEFAULT_CARLA_TIMEOUT,
            "scene_id": case.scene_id,
            "scene_catalog_path": _portable_path(case.catalog_path),
            "draw_obstacles": False,
        },
        "weather": {
            "enabled": True,
            "seed": case.seed + 101,
        },
        "walker": {
            "pedestrian_seed": case.seed,
            "default_speed": 1.2,
            "despawn_on_arrival": True,
            "waypoint_threshold": 1.0,
            "draw_bounding_boxes": False,
            "ped_spawner": [render_actor(actor) for actor in case.actors],
        },
        "vehicle": {
            "vehicle_seed": case.seed + 17,
            "no_bikes": False,
        },
    }

    if case.weather_presets:
        config["weather"]["regimes"] = {"weights": dict(case.weather_presets)}

    vehicle_counts = VEHICLE_DENSITY_CONFIG[case.vehicle_density]
    config["vehicle"]["background_population"] = {
        "enabled": True,
        "seed": case.seed + 31,
        **vehicle_counts,
    }

    if case.sampling_enabled:
        config["sampling"] = {
            "enabled": True,
            "sensor_rig": "sensor_rig.toml",
            "policy": "sampling_policy.toml",
            "sequence_id": case.case_id,
            "output_root": _portable_path(case_dir.parent),
        }

    return config


def render_actor(actor: ActorSpec) -> Dict[str, Any]:
    spawner: Dict[str, Any] = {
        "behavior": actor.behavior,
        "route_id": actor.route_id,
        "quantity": actor.quantity,
        "spawn_time": actor.spawn_time,
    }
    if actor.formation is not None:
        spawner["formation"] = actor.formation
    if actor.spawn_interval is not None:
        spawner["spawn_interval"] = actor.spawn_interval
    if actor.ego:
        spawner["ego"] = True
        spawner["ego_member_index"] = int(actor.ego_member_index or 0)
    return spawner


def validate_resolved_case(case: ResolvedCase) -> None:
    errors: List[str] = []

    if not case.actors:
        errors.append("case has no actors")

    total_pedestrians = sum(int(actor.quantity) for actor in case.actors)
    if total_pedestrians > MAX_PEDESTRIANS:
        errors.append(f"total pedestrians {total_pedestrians} exceed {MAX_PEDESTRIANS}")
    if len(case.actors) > MAX_SPAWNERS:
        errors.append(f"spawner count {len(case.actors)} exceeds {MAX_SPAWNERS}")

    routes = {case.main_route.route_id: case.main_route}
    for route in case.support_routes:
        routes[route.route_id] = route

    ego_count = 0
    for index, actor in enumerate(case.actors):
        route = routes.get(actor.route_id)
        if route is None:
            errors.append(f"actor {index} uses unknown route {actor.route_id}")
        else:
            if actor.route_type != route.route_type:
                errors.append(f"actor {index} route_type does not match route {route.route_id}")
            if actor.behavior not in route.supported_behaviors:
                errors.append(
                    f"actor {index} behavior {actor.behavior} is not supported by {route.route_id}"
                )

        if int(actor.quantity) < 1:
            errors.append(f"actor {index} quantity must be positive")
        if actor.formation is not None:
            if actor.quantity not in (2, 3):
                errors.append(f"actor {index} formation requires quantity 2 or 3")
            if actor.formation not in {"abreast", "follow"}:
                errors.append(f"actor {index} has unsupported formation {actor.formation}")
            if actor.spawn_interval is None or actor.spawn_interval <= 0:
                errors.append(f"actor {index} formation requires a positive spawn_interval")

        max_spawn_time = 12.0 if actor.role == "ego" else 8.0
        if not 0.0 <= float(actor.spawn_time) <= max_spawn_time:
            errors.append(f"actor {index} spawn_time must be within [0, {max_spawn_time}]")

        if actor.ego:
            ego_count += 1
            if int(actor.quantity) != 1:
                errors.append(f"actor {index} ego must have quantity 1")

    if case.sampling_enabled and ego_count != 1:
        errors.append("sampling requires exactly one ego pedestrian")
    if not case.sampling_enabled and ego_count != 0:
        errors.append("sampling is disabled but an ego pedestrian is marked")

    if case.vehicle_density not in VEHICLE_DENSITY_CONFIG:
        errors.append(f"unsupported vehicle density {case.vehicle_density}")

    for preset_name, weight in case.weather_presets.items():
        if preset_name not in WEATHER_PRESETS:
            errors.append(f"unknown weather preset {preset_name}")
        if float(weight) <= 0.0:
            errors.append(f"weather preset {preset_name} must have a positive weight")

    if errors:
        raise CompilationError("; ".join(errors))


def render_toml(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    _render_table(lines, [], data)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


def _render_table(lines: List[str], path: List[str], data: Dict[str, Any]) -> None:
    scalar_items = []
    dict_items = []
    list_table_items = []

    for key, value in data.items():
        if isinstance(value, dict):
            dict_items.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            list_table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if path:
        lines.append(f"[{'.'.join(path)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")

    if dict_items or list_table_items:
        if lines and lines[-1] != "":
            lines.append("")

    for index, (key, value) in enumerate(dict_items):
        _render_table(lines, path + [key], value)
        if index != len(dict_items) - 1 or list_table_items:
            lines.append("")

    for key, value in list_table_items:
        for item_index, item in enumerate(value):
            lines.append(f"[[{'.'.join(path + [key])}]]")
            for item_key, item_value in item.items():
                if isinstance(item_value, dict):
                    raise CompilationError("nested tables inside array-of-table items are not supported")
                lines.append(f"{item_key} = {_format_toml_value(item_value)}")
            if item_index != len(value) - 1:
                lines.append("")


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0")
        if text.endswith("."):
            text += "0"
        return text
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise CompilationError(f"unsupported TOML value type: {type(value)!r}")


def _check_toml_round_trip(config: Dict[str, Any], text: str) -> None:
    try:
        parsed = tomllib.loads(text)
    except Exception as exc:
        raise CompilationError(f"rendered TOML is not parseable: {exc}") from exc
    if parsed != config:
        raise CompilationError("rendered TOML does not round-trip to the source config")


def _scenario_name(case: ResolvedCase) -> str:
    total_pedestrians = sum(int(actor.quantity) for actor in case.actors)
    return (
        f"case_{case.map_name.lower()}_{case.scene_id}_"
        f"{case.main_route.route_type}_p{total_pedestrians}"
    )


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write_sampling_policy(case: ResolvedCase, *, config_dir: Path) -> None:
    if not DEFAULT_SAMPLING_POLICY.exists():
        raise CompilationError(f"sampling policy template not found: {DEFAULT_SAMPLING_POLICY}")

    policy = tomllib.loads(DEFAULT_SAMPLING_POLICY.read_text(encoding="utf-8"))
    policy["output_root"] = _portable_path(config_dir.parent.parent)
    policy["sequence_id"] = case.case_id
    policy["reuse_existing_root"] = True

    policy_text = render_toml(policy)
    (config_dir / "sampling_policy.toml").write_text(policy_text, encoding="utf-8")
    (config_dir / "sampling_resolved.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _copy_sensor_rig(config_dir: Path) -> None:
    if not DEFAULT_SENSOR_RIG.exists():
        raise CompilationError(f"sensor rig template not found: {DEFAULT_SENSOR_RIG}")
    shutil.copyfile(DEFAULT_SENSOR_RIG, config_dir / "sensor_rig.toml")
