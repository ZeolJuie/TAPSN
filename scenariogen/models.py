"""Data contracts for the prompt-driven scenario generation pipeline.

Pipeline:
    UserRequest -> ScenePlan -> ResolvedCase -> BuiltCase -> ExecutionResult

ScenePlan is the only object produced by the LLM. It holds semantic factors:
topology, environment, traffic, social behaviour, and ego task. Simulator ids
and configuration values are introduced by the resolver and the compiler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserRequest:
    """Natural-language request supplied by the user.

    count is a command-line override. When it is None, the planner reads the
    requested number of cases from the prompt itself.
    """

    prompt: str
    count: Optional[int] = None
    seed: int = 2026


# ---------------------------------------------------------------------------
# Resource catalog summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapCapability:
    """Capabilities offered by one map, without concrete route ids."""

    map_name: str
    catalog_path: Path
    scene_kinds: List[str]
    route_types: List[str]
    behaviors: List[str]
    scene_count: int
    route_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_name": self.map_name,
            "scene_kinds": list(self.scene_kinds),
            "route_types": list(self.route_types),
            "behaviors": list(self.behaviors),
            "scene_count": self.scene_count,
            "route_count": self.route_count,
        }


@dataclass(frozen=True)
class ResourceSummary:
    """Capability summary injected into the planner prompt."""

    maps: List[MapCapability]
    weather_presets: List[str]
    vehicle_densities: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maps": [item.to_dict() for item in self.maps],
            "weather_presets": list(self.weather_presets),
            "vehicle_densities": list(self.vehicle_densities),
        }


# ---------------------------------------------------------------------------
# LLM output: the five-factor scene plan
# ---------------------------------------------------------------------------


@dataclass
class ScenePlan:
    """Structured semantic scene plan C = (T, E, V, B, I).

    Each factor is a dictionary. Admissible keys and values are defined in
    scenariogen.schema. The plan holds data only, so it can be serialized,
    validated, and stored with the generated case.
    """

    count: int
    description: str
    topology: Dict[str, Any]
    environment: Dict[str, Any]
    traffic: Dict[str, Any]
    social: Dict[str, Any]
    ego: Dict[str, Any]
    planner_source: str = "llm"
    planner_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "description": self.description,
            "topology": dict(self.topology),
            "environment": dict(self.environment),
            "traffic": dict(self.traffic),
            "social": dict(self.social),
            "ego": dict(self.ego),
            "planner_source": self.planner_source,
            "planner_error": self.planner_error,
        }


# ---------------------------------------------------------------------------
# Resolved case: concrete simulator binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteSpec:
    """One concrete route selected from a walkable-cell catalog."""

    route_id: str
    route_type: str
    route_length_m: float
    supported_behaviors: List[str]
    spawn_point_xy: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActorSpec:
    """One concrete pedestrian spawner produced by the resolver."""

    role: str
    route_id: str
    route_type: str
    behavior: str
    quantity: int = 1
    formation: Optional[str] = None
    spawn_time: float = 0.0
    spawn_interval: Optional[float] = None
    ego: bool = False
    ego_member_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedCase:
    """A scene plan bound to concrete map resources.

    All simulator-facing values are known at this point: map, scene, routes,
    actors, weather preset weights, and vehicle density. The compiler formats
    these values into scenario.toml.
    """

    case_index: int
    case_id: str
    plan: ScenePlan
    map_name: str
    catalog_path: Path
    scene_id: str
    scene_kind: str
    main_route: RouteSpec
    support_routes: List[RouteSpec]
    actors: List[ActorSpec]
    weather_presets: Dict[str, float]
    vehicle_density: str
    sampling_enabled: bool
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_index": self.case_index,
            "case_id": self.case_id,
            "plan": self.plan.to_dict(),
            "map_name": self.map_name,
            "catalog_path": str(self.catalog_path),
            "scene_id": self.scene_id,
            "scene_kind": self.scene_kind,
            "main_route": self.main_route.to_dict(),
            "support_routes": [route.to_dict() for route in self.support_routes],
            "actors": [actor.to_dict() for actor in self.actors],
            "weather_presets": dict(self.weather_presets),
            "vehicle_density": self.vehicle_density,
            "sampling_enabled": self.sampling_enabled,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------


@dataclass
class BuiltCase:
    """A case written to disk and ready for run_simulation.py."""

    resolved: ResolvedCase
    case_dir: Path
    scenario_path: Path
    spec_path: Path
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_dir": str(self.case_dir),
            "scenario_path": str(self.scenario_path),
            "spec_path": str(self.spec_path),
            "resolved": self.resolved.to_dict(),
            "config": self.config,
        }


@dataclass
class ExecutionResult:
    """Outcome of one run_simulation.py invocation."""

    command: List[str]
    returncode: int
    elapsed_s: float
    log_path: Optional[Path] = None
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "timed_out": self.timed_out,
        }
