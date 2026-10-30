"""Scenario schema Sigma.

Sigma defines the admissible semantic slots, their value whitelists, the
compatibility rules between factors, and the mapping from semantic values
to simulator configuration. The planner prompt and the plan validator both
read this module, so the constraints shown to the LLM and the constraints
enforced after generation are the same.

Factor names follow the paper:
    T = topology
    E = environment
    V = vehicle traffic
    B = pedestrian social behaviour
    I = ego interaction task
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scenariogen.models import ResourceSummary, ScenePlan


SCHEMA_VERSION = "1.0"

MAX_CASES = 64


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

SCENE_KINDS = ("crossing", "sidewalk")
WALKING_ROUTE_TYPES = (
    "sidewalk_through",
    "junction_through",
    "junction_turn",
    "junction_u_turn",
)
CROSSING_ROUTE_TYPES = (
    "crosswalk_cross",
    "unmarked_junction_cross",
)
ROUTE_TYPES = WALKING_ROUTE_TYPES + CROSSING_ROUTE_TYPES

WALKING_BEHAVIORS = ("steady_walk", "slow_walk", "fast_pass")
CROSSING_BEHAVIORS = (
    "cautious_cross",
    "normal_cross",
    "assertive_cross",
    "ignore_gap_cross",
)
BEHAVIORS = WALKING_BEHAVIORS + CROSSING_BEHAVIORS

ROUTE_TYPE_TO_SCENE_KIND = {
    "sidewalk_through": "sidewalk",
    "junction_through": "crossing",
    "junction_turn": "crossing",
    "junction_u_turn": "crossing",
    "crosswalk_cross": "crossing",
    "unmarked_junction_cross": "crossing",
}

ROUTE_TYPE_TO_BEHAVIORS = {
    "sidewalk_through": WALKING_BEHAVIORS,
    "junction_through": WALKING_BEHAVIORS,
    "junction_turn": WALKING_BEHAVIORS,
    "junction_u_turn": WALKING_BEHAVIORS,
    "crosswalk_cross": CROSSING_BEHAVIORS,
    "unmarked_junction_cross": CROSSING_BEHAVIORS,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

ILLUMINATIONS = ("day", "low_sun", "night", "any")
WEATHER_CONDITIONS = (
    "clear",
    "cloudy",
    "overcast",
    "wet",
    "light_rain",
    "rain",
    "heavy_rain",
    "dust_storm",
    "any",
)
ROAD_SURFACES = ("dry", "wet", "any")
VISIBILITIES = ("normal", "reduced", "any")


# ---------------------------------------------------------------------------
# Traffic and social behaviour
# ---------------------------------------------------------------------------

VEHICLE_DENSITIES = ("none", "low", "medium", "high", "any")
PEDESTRIAN_DENSITIES = ("sparse", "medium", "crowded", "any")
GROUP_PATTERNS = ("none", "mixed", "abreast", "follow", "any")
FLOWS = ("one_way", "two_way", "any")
BEHAVIOR_MIXES = ("walking", "crossing", "mixed", "any")


# ---------------------------------------------------------------------------
# Weather preset mapping
# ---------------------------------------------------------------------------

WEATHER_PRESETS: Dict[str, Dict[str, str]] = {
    "ClearNoon": {"illumination": "day", "weather": "clear", "road_surface": "dry", "visibility": "normal"},
    "CloudyNoon": {"illumination": "day", "weather": "cloudy", "road_surface": "dry", "visibility": "normal"},
    "WetNoon": {"illumination": "day", "weather": "overcast", "road_surface": "wet", "visibility": "normal"},
    "WetCloudyNoon": {"illumination": "day", "weather": "cloudy", "road_surface": "wet", "visibility": "normal"},
    "SoftRainNoon": {"illumination": "day", "weather": "light_rain", "road_surface": "wet", "visibility": "normal"},
    "MidRainyNoon": {"illumination": "day", "weather": "rain", "road_surface": "wet", "visibility": "normal"},
    "HardRainNoon": {"illumination": "day", "weather": "heavy_rain", "road_surface": "wet", "visibility": "normal"},
    "DustStorm": {"illumination": "day", "weather": "dust_storm", "road_surface": "dry", "visibility": "reduced"},
    "ClearSunset": {"illumination": "low_sun", "weather": "clear", "road_surface": "dry", "visibility": "normal"},
    "CloudySunset": {"illumination": "low_sun", "weather": "cloudy", "road_surface": "dry", "visibility": "normal"},
    "WetSunset": {"illumination": "low_sun", "weather": "overcast", "road_surface": "wet", "visibility": "normal"},
    "WetCloudySunset": {"illumination": "low_sun", "weather": "cloudy", "road_surface": "wet", "visibility": "normal"},
    "SoftRainSunset": {"illumination": "low_sun", "weather": "light_rain", "road_surface": "wet", "visibility": "normal"},
    "MidRainSunset": {"illumination": "low_sun", "weather": "rain", "road_surface": "wet", "visibility": "normal"},
    "HardRainSunset": {"illumination": "low_sun", "weather": "heavy_rain", "road_surface": "wet", "visibility": "normal"},
    "ClearNight": {"illumination": "night", "weather": "clear", "road_surface": "dry", "visibility": "normal"},
    "CloudyNight": {"illumination": "night", "weather": "cloudy", "road_surface": "dry", "visibility": "normal"},
    "WetNight": {"illumination": "night", "weather": "overcast", "road_surface": "wet", "visibility": "normal"},
    "WetCloudyNight": {"illumination": "night", "weather": "cloudy", "road_surface": "wet", "visibility": "normal"},
    "SoftRainNight": {"illumination": "night", "weather": "light_rain", "road_surface": "wet", "visibility": "normal"},
    "MidRainyNight": {"illumination": "night", "weather": "rain", "road_surface": "wet", "visibility": "normal"},
    "HardRainNight": {"illumination": "night", "weather": "heavy_rain", "road_surface": "wet", "visibility": "reduced"},
}


# ---------------------------------------------------------------------------
# Prompt-facing schema
# ---------------------------------------------------------------------------

SCENARIO_SCHEMA: Dict[str, Any] = {
    "version": SCHEMA_VERSION,
    "factors": {
        "topology": {
            "scene_kind": list(SCENE_KINDS + ("any",)),
            "route_types": list(ROUTE_TYPES),
        },
        "environment": {
            "illumination": list(ILLUMINATIONS),
            "weather": list(WEATHER_CONDITIONS),
            "road_surface": list(ROAD_SURFACES),
            "visibility": list(VISIBILITIES),
        },
        "traffic": {
            "vehicle_density": list(VEHICLE_DENSITIES),
        },
        "social": {
            "pedestrian_density": list(PEDESTRIAN_DENSITIES),
            "group_pattern": list(GROUP_PATTERNS),
            "flow": list(FLOWS),
            "behavior_mix": list(BEHAVIOR_MIXES),
        },
        "ego": {
            "behavior": list(BEHAVIORS + ("any",)),
        },
    },
    "compatibility": {
        "route_type_to_scene_kind": dict(ROUTE_TYPE_TO_SCENE_KIND),
        "route_type_to_behaviors": {
            key: list(value) for key, value in ROUTE_TYPE_TO_BEHAVIORS.items()
        },
        "rules": [
            "A crossing ego behaviour requires at least one crossing route type.",
            "A walking ego behaviour requires at least one walking route type when route_types is specified.",
            "A sidewalk scene accepts only sidewalk_through routes.",
            "A crossing scene does not accept sidewalk_through routes.",
        ],
    },
    "output": {
        "type": "object",
        "required": ["count", "description", "topology", "environment", "traffic", "social", "ego"],
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": MAX_CASES},
            "description": {"type": "string"},
            "topology": {
                "type": "object",
                "properties": {
                    "scene_kind": {"enum": list(SCENE_KINDS + ("any",))},
                    "route_types": {"type": "array", "items": {"enum": list(ROUTE_TYPES)}},
                },
            },
            "environment": {
                "type": "object",
                "properties": {
                    "illumination": {"enum": list(ILLUMINATIONS)},
                    "weather": {"enum": list(WEATHER_CONDITIONS)},
                    "road_surface": {"enum": list(ROAD_SURFACES)},
                    "visibility": {"enum": list(VISIBILITIES)},
                },
            },
            "traffic": {
                "type": "object",
                "properties": {
                    "vehicle_density": {"enum": list(VEHICLE_DENSITIES)},
                },
            },
            "social": {
                "type": "object",
                "properties": {
                    "pedestrian_density": {"enum": list(PEDESTRIAN_DENSITIES)},
                    "group_pattern": {"enum": list(GROUP_PATTERNS)},
                    "flow": {"enum": list(FLOWS)},
                    "behavior_mix": {"enum": list(BEHAVIOR_MIXES)},
                },
            },
            "ego": {
                "type": "object",
                "properties": {
                    "behavior": {"enum": list(BEHAVIORS + ("any",))},
                },
            },
        },
    },
}


class SchemaError(ValueError):
    """Raised when a scene plan cannot be normalized or satisfied."""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_SCENE_KIND_SYNONYMS = {
    "intersection": "crossing",
    "junction": "crossing",
    "urban_junction": "crossing",
    "street": "sidewalk",
    "walkway": "sidewalk",
}

_ROUTE_TYPE_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "sidewalk": ("sidewalk_through",),
    "through": ("sidewalk_through", "junction_through"),
    "junction": ("junction_through", "junction_turn"),
    "turn": ("junction_turn",),
    "left_turn": ("junction_turn",),
    "right_turn": ("junction_turn",),
    "u_turn": ("junction_u_turn",),
    "uturn": ("junction_u_turn",),
    "crosswalk": ("crosswalk_cross",),
    "marked_crosswalk": ("crosswalk_cross",),
    "unmarked_crossing": ("unmarked_junction_cross",),
    "jaywalking": ("unmarked_junction_cross",),
}

_ILLUMINATION_SYNONYMS = {
    "daytime": "day",
    "noon": "day",
    "bright": "day",
    "sunset": "low_sun",
    "sunrise": "low_sun",
    "dusk": "low_sun",
    "dawn": "low_sun",
    "twilight": "low_sun",
    "dark": "night",
}

_WEATHER_SYNONYMS = {
    "sunny": "clear",
    "rainy": "rain",
    "drizzle": "light_rain",
    "downpour": "heavy_rain",
    "storm": "heavy_rain",
    "sandstorm": "dust_storm",
    "dust": "dust_storm",
}

_ROAD_SURFACE_SYNONYMS = {
    "rainy": "wet",
    "slippery": "wet",
}

_VISIBILITY_SYNONYMS = {
    "low": "reduced",
    "foggy": "reduced",
    "fog": "reduced",
}

_VEHICLE_DENSITY_SYNONYMS = {
    "light": "low",
    "moderate": "medium",
    "heavy": "high",
    "dense": "high",
    "no_traffic": "none",
    "empty": "none",
}

_PEDESTRIAN_DENSITY_SYNONYMS = {
    "low": "sparse",
    "few": "sparse",
    "moderate": "medium",
    "high": "crowded",
    "dense": "crowded",
    "busy": "crowded",
}

_GROUP_PATTERN_SYNONYMS = {
    "grouped": "mixed",
    "group": "mixed",
    "single": "none",
    "alone": "none",
    "file": "follow",
    "side_by_side": "abreast",
}

_FLOW_SYNONYMS = {
    "bidirectional": "two_way",
    "both_directions": "two_way",
    "single_direction": "one_way",
}

_BEHAVIOR_MIX_SYNONYMS = {
    "walk": "walking",
    "cross": "crossing",
}

_EGO_BEHAVIOR_SYNONYMS = {
    "steady": "steady_walk",
    "walk": "steady_walk",
    "walking": "steady_walk",
    "slow": "slow_walk",
    "fast": "fast_pass",
    "brisk": "fast_pass",
    "cautious": "cautious_cross",
    "assertive": "assertive_cross",
    "ignore_gap": "ignore_gap_cross",
    "jaywalk": "ignore_gap_cross",
}


def _as_text(value: Any, default: str = "any") -> str:
    if value is None:
        return default
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _dedupe(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_choice(value: Any, synonyms: Mapping[str, str], default: str = "any") -> str:
    text = _as_text(value, default)
    return synonyms.get(text, text)


def _normalize_scene_kind(value: Any) -> str:
    return _normalize_choice(value, _SCENE_KIND_SYNONYMS)


def _normalize_route_types(value: Any) -> List[str]:
    result = []
    for item in _as_list(value):
        text = _as_text(item)
        if text == "any":
            return []
        result.extend(_ROUTE_TYPE_SYNONYMS.get(text, (text,)))
    return _dedupe(result)


def _infer_scene_kind(route_types: Sequence[str]) -> str:
    if route_types and set(route_types) <= {"sidewalk_through"}:
        return "sidewalk"
    if route_types:
        return "crossing"
    return "any"


def _normalize_environment(raw: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "illumination": _normalize_choice(raw.get("illumination"), _ILLUMINATION_SYNONYMS),
        "weather": _normalize_choice(raw.get("weather"), _WEATHER_SYNONYMS),
        "road_surface": _normalize_choice(raw.get("road_surface"), _ROAD_SURFACE_SYNONYMS),
        "visibility": _normalize_choice(raw.get("visibility"), _VISIBILITY_SYNONYMS),
    }


def _normalize_social(raw: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "pedestrian_density": _normalize_choice(
            raw.get("pedestrian_density"), _PEDESTRIAN_DENSITY_SYNONYMS
        ),
        "group_pattern": _normalize_choice(raw.get("group_pattern"), _GROUP_PATTERN_SYNONYMS),
        "flow": _normalize_choice(raw.get("flow"), _FLOW_SYNONYMS),
        "behavior_mix": _normalize_choice(raw.get("behavior_mix"), _BEHAVIOR_MIX_SYNONYMS),
    }


def normalize_scene_plan(raw: Mapping[str, Any]) -> ScenePlan:
    """Convert one planner response into a canonical ScenePlan."""

    topology_raw = raw.get("topology") or {}
    route_types = _normalize_route_types(topology_raw.get("route_types"))
    scene_kind = _normalize_scene_kind(topology_raw.get("scene_kind"))
    if scene_kind == "any":
        scene_kind = _infer_scene_kind(route_types)

    try:
        count = int(raw.get("count", 1))
    except (TypeError, ValueError):
        count = 0

    return ScenePlan(
        count=count,
        description=str(raw.get("description") or "").strip(),
        topology={"scene_kind": scene_kind, "route_types": route_types},
        environment=_normalize_environment(raw.get("environment") or {}),
        traffic={
            "vehicle_density": _normalize_choice(
                (raw.get("traffic") or {}).get("vehicle_density"),
                _VEHICLE_DENSITY_SYNONYMS,
            )
        },
        social=_normalize_social(raw.get("social") or {}),
        ego={
            "behavior": _normalize_choice(
                (raw.get("ego") or {}).get("behavior"),
                _EGO_BEHAVIOR_SYNONYMS,
            )
        },
        planner_source=str(raw.get("planner_source") or "llm"),
        planner_error=raw.get("planner_error"),
    )


# ---------------------------------------------------------------------------
# Environment to weather preset mapping
# ---------------------------------------------------------------------------

_RAIN_PRESETS = {"light_rain", "rain", "heavy_rain"}


def _weather_matches(requested: str, actual: str) -> bool:
    if requested == "any" or requested == actual:
        return True
    if requested == "rain":
        return actual in _RAIN_PRESETS
    if requested == "wet":
        return actual in _RAIN_PRESETS or actual == "wet"
    if requested == "cloudy":
        return actual in {"cloudy", "overcast"}
    if requested == "overcast":
        return actual in {"overcast", "wet"}
    return False


def weather_presets_for_environment(environment: Mapping[str, Any]) -> Dict[str, float]:
    """Return CARLA weather presets that satisfy an environment request.

    An empty dictionary means the request leaves environment unspecified and
    the simulator uses its default weather distribution.
    """

    normalized = _normalize_environment(environment)
    for label, value, allowed in (
        ("illumination", normalized["illumination"], ILLUMINATIONS),
        ("weather", normalized["weather"], WEATHER_CONDITIONS),
        ("road_surface", normalized["road_surface"], ROAD_SURFACES),
        ("visibility", normalized["visibility"], VISIBILITIES),
    ):
        if value not in allowed:
            raise SchemaError(f"{label} must be one of {allowed}, got {value!r}")

    if all(value == "any" for value in normalized.values()):
        return {}

    matches = {}
    for preset_name, preset in WEATHER_PRESETS.items():
        if not _weather_matches(normalized["weather"], preset["weather"]):
            continue
        if normalized["illumination"] != "any" and normalized["illumination"] != preset["illumination"]:
            continue
        if normalized["road_surface"] != "any" and normalized["road_surface"] != preset["road_surface"]:
            continue
        if normalized["visibility"] != "any" and normalized["visibility"] != preset["visibility"]:
            continue
        matches[preset_name] = 1.0

    if not matches:
        raise SchemaError(
            "no weather preset satisfies environment request "
            f"{normalized}; available presets are {sorted(WEATHER_PRESETS)}"
        )
    return matches


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def route_types_for_behavior(behavior: str) -> Tuple[str, ...]:
    if behavior in WALKING_BEHAVIORS:
        return WALKING_ROUTE_TYPES
    if behavior in CROSSING_BEHAVIORS:
        return CROSSING_ROUTE_TYPES
    return ()


def _check_enum(errors: List[str], label: str, value: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        errors.append(f"{label} must be one of {list(allowed)}, got {value!r}")


def validate_scene_plan(
    plan: ScenePlan,
    resource_summary: Optional[ResourceSummary] = None,
) -> List[str]:
    """Check a normalized scene plan against the schema and available resources."""

    errors: List[str] = []

    if not 1 <= int(plan.count) <= MAX_CASES:
        errors.append(f"count must be between 1 and {MAX_CASES}, got {plan.count}")

    scene_kind = plan.topology.get("scene_kind", "any")
    route_types = list(plan.topology.get("route_types") or [])
    ego_behavior = plan.ego.get("behavior", "any")

    _check_enum(errors, "topology.scene_kind", scene_kind, SCENE_KINDS + ("any",))
    for route_type in route_types:
        _check_enum(errors, "topology.route_types item", route_type, ROUTE_TYPES)

    if scene_kind == "sidewalk" and any(item != "sidewalk_through" for item in route_types):
        errors.append("scene_kind=sidewalk accepts only sidewalk_through routes")
    if scene_kind == "crossing" and "sidewalk_through" in route_types:
        errors.append("scene_kind=crossing does not accept sidewalk_through routes")

    _check_enum(errors, "environment.illumination", plan.environment.get("illumination", "any"), ILLUMINATIONS)
    _check_enum(errors, "environment.weather", plan.environment.get("weather", "any"), WEATHER_CONDITIONS)
    _check_enum(errors, "environment.road_surface", plan.environment.get("road_surface", "any"), ROAD_SURFACES)
    _check_enum(errors, "environment.visibility", plan.environment.get("visibility", "any"), VISIBILITIES)
    _check_enum(errors, "traffic.vehicle_density", plan.traffic.get("vehicle_density", "any"), VEHICLE_DENSITIES)
    _check_enum(
        errors,
        "social.pedestrian_density",
        plan.social.get("pedestrian_density", "any"),
        PEDESTRIAN_DENSITIES,
    )
    _check_enum(errors, "social.group_pattern", plan.social.get("group_pattern", "any"), GROUP_PATTERNS)
    _check_enum(errors, "social.flow", plan.social.get("flow", "any"), FLOWS)
    _check_enum(errors, "social.behavior_mix", plan.social.get("behavior_mix", "any"), BEHAVIOR_MIXES)
    _check_enum(errors, "ego.behavior", ego_behavior, BEHAVIORS + ("any",))

    if ego_behavior != "any":
        compatible = set(route_types_for_behavior(ego_behavior))
        if route_types:
            if not compatible & set(route_types):
                errors.append(
                    f"ego.behavior={ego_behavior} requires a route type in {sorted(compatible)}"
                )
        elif scene_kind == "sidewalk" and ego_behavior in CROSSING_BEHAVIORS:
            errors.append(
                f"ego.behavior={ego_behavior} is a crossing behaviour and cannot be used in a sidewalk scene"
            )

    try:
        weather_presets_for_environment(plan.environment)
    except SchemaError as exc:
        errors.append(str(exc))

    if resource_summary is not None:
        errors.extend(_check_resources(plan, resource_summary))

    return errors


def _check_resources(plan: ScenePlan, summary: ResourceSummary) -> List[str]:
    if not summary.maps:
        return ["resource catalog contains no usable maps"]

    scene_kind = plan.topology.get("scene_kind", "any")
    route_types = list(plan.topology.get("route_types") or [])
    ego_behavior = plan.ego.get("behavior", "any")
    compatible_ego_routes = set(route_types_for_behavior(ego_behavior)) if ego_behavior != "any" else set()

    for capability in summary.maps:
        if scene_kind != "any" and scene_kind not in capability.scene_kinds:
            continue
        if route_types and not set(route_types) & set(capability.route_types):
            continue
        if compatible_ego_routes and not compatible_ego_routes & set(capability.route_types):
            continue
        return []

    return [
        "resource catalog has no map satisfying "
        f"scene_kind={scene_kind}, route_types={route_types}, ego_behavior={ego_behavior}"
    ]
