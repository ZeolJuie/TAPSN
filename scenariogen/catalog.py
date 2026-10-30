"""Resource catalog built from walkable-cell topology.

The catalog answers two questions:

1. Which scene kinds, route types, and pedestrian behaviours does a map offer?
2. For a given topology and ego-behaviour constraint, which concrete scene
   and route should a case bind to?

Reverse routes are exposed as route_id + '__reverse'. Their geometry is
synthesized by the runtime scene catalog, so this module only records their
metadata.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scenariogen.models import MapCapability, ResourceSummary, RouteSpec
from scenariogen.schema import VEHICLE_DENSITIES, WEATHER_PRESETS


REVERSE_ROUTE_SUFFIX = "__reverse"


class CatalogError(ValueError):
    """Raised when a topology catalog cannot satisfy a binding request."""


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    scene_kind: str
    bounds: Dict[str, Any]
    routes: List[RouteSpec]


@dataclass(frozen=True)
class SceneBinding:
    map_name: str
    catalog_path: Path
    scene_id: str
    scene_kind: str
    bounds: Dict[str, Any]
    main_route: RouteSpec
    support_routes: List[RouteSpec]


class Catalog:
    """Query layer over one walkable_cells.json file."""

    def __init__(
        self,
        *,
        catalog_path: Path,
        map_name: str,
        scenes: List[SceneRecord],
    ):
        self.catalog_path = Path(catalog_path).resolve()
        self.map_name = map_name
        self.scenes = scenes

    @classmethod
    def from_path(cls, catalog_path: Path) -> "Catalog":
        path = Path(catalog_path).resolve()
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        map_name = str(payload.get("map_name") or path.parent.name)
        scenes: List[SceneRecord] = []
        for cell in payload.get("cells") or []:
            if not cell.get("attributes", {}).get("usable", False):
                continue

            routes = _routes_from_cell(cell)
            if not routes:
                continue

            scenes.append(
                SceneRecord(
                    scene_id=str(cell["id"]),
                    scene_kind=str(cell.get("kind") or "unknown"),
                    bounds=dict(cell.get("bounds") or {}),
                    routes=routes,
                )
            )

        if not scenes:
            raise CatalogError(f"catalog contains no usable scenes with routes: {path}")
        return cls(catalog_path=path, map_name=map_name, scenes=scenes)

    def capability(self) -> MapCapability:
        scene_kinds = sorted({scene.scene_kind for scene in self.scenes})
        route_types = sorted({route.route_type for scene in self.scenes for route in scene.routes})
        behaviors = sorted(
            {
                behavior
                for scene in self.scenes
                for route in scene.routes
                for behavior in route.supported_behaviors
            }
        )
        return MapCapability(
            map_name=self.map_name,
            catalog_path=self.catalog_path,
            scene_kinds=scene_kinds,
            route_types=route_types,
            behaviors=behaviors,
            scene_count=len(self.scenes),
            route_count=sum(len(scene.routes) for scene in self.scenes),
        )

    def has_eligible(
        self,
        *,
        scene_kind: str = "any",
        route_types: Sequence[str] = (),
        ego_behavior: str = "any",
        excluded_scene_ids: Iterable[str] = (),
    ) -> bool:
        return bool(
            self._candidates(
                scene_kind=scene_kind,
                route_types=route_types,
                ego_behavior=ego_behavior,
                excluded_scene_ids=excluded_scene_ids,
            )
        )

    def bind(
        self,
        *,
        scene_kind: str = "any",
        route_types: Sequence[str] = (),
        ego_behavior: str = "any",
        excluded_scene_ids: Iterable[str] = (),
        rng: random.Random,
    ) -> SceneBinding:
        candidates = self._candidates(
            scene_kind=scene_kind,
            route_types=route_types,
            ego_behavior=ego_behavior,
            excluded_scene_ids=excluded_scene_ids,
        )
        if not candidates:
            raise CatalogError(
                f"map {self.map_name} has no scene satisfying "
                f"scene_kind={scene_kind}, route_types={list(route_types)}, "
                f"ego_behavior={ego_behavior}"
            )

        scene, matching_routes = rng.choice(candidates)
        main_route = rng.choice(matching_routes)
        support_routes = [route for route in scene.routes if route.route_id != main_route.route_id]

        return SceneBinding(
            map_name=self.map_name,
            catalog_path=self.catalog_path,
            scene_id=scene.scene_id,
            scene_kind=scene.scene_kind,
            bounds=dict(scene.bounds),
            main_route=main_route,
            support_routes=support_routes,
        )

    def _candidates(
        self,
        *,
        scene_kind: str,
        route_types: Sequence[str],
        ego_behavior: str,
        excluded_scene_ids: Iterable[str],
    ) -> List[Tuple[SceneRecord, List[RouteSpec]]]:
        allowed_route_types = set(route_types)
        excluded = set(excluded_scene_ids)
        result: List[Tuple[SceneRecord, List[RouteSpec]]] = []

        for scene in self.scenes:
            if scene.scene_id in excluded:
                continue
            if scene_kind not in (None, "", "any") and scene.scene_kind != scene_kind:
                continue

            matching = [
                route
                for route in scene.routes
                if _route_matches(route, allowed_route_types, ego_behavior)
            ]
            if matching:
                result.append((scene, matching))
        return result


def load_catalogs(paths: Sequence[Path]) -> List[Catalog]:
    return [Catalog.from_path(path) for path in paths]


def summarize_catalogs(catalogs: Sequence[Catalog]) -> ResourceSummary:
    return ResourceSummary(
        maps=[catalog.capability() for catalog in catalogs],
        weather_presets=sorted(WEATHER_PRESETS),
        vehicle_densities=[value for value in VEHICLE_DENSITIES if value != "any"],
    )


def _route_matches(route: RouteSpec, allowed_route_types: set, ego_behavior: str) -> bool:
    if allowed_route_types and route.route_type not in allowed_route_types:
        return False
    if ego_behavior not in (None, "", "any") and ego_behavior not in route.supported_behaviors:
        return False
    return True


def _routes_from_cell(cell: Dict[str, Any]) -> List[RouteSpec]:
    raw_routes = list(cell.get("routes") or [])
    existing_ids = {str(route.get("route_id")) for route in raw_routes if route.get("route_id")}
    routes: List[RouteSpec] = []

    for raw_route in raw_routes:
        forward = _route_spec(raw_route)
        if forward is None:
            continue
        routes.append(forward)

        reverse_id = forward.route_id + REVERSE_ROUTE_SUFFIX
        if forward.route_id.endswith(REVERSE_ROUTE_SUFFIX) or reverse_id in existing_ids:
            continue
        reverse = _route_spec(raw_route, reverse=True)
        if reverse is not None:
            routes.append(reverse)

    return routes


def _route_spec(raw_route: Dict[str, Any], *, reverse: bool = False) -> Optional[RouteSpec]:
    route_id = raw_route.get("route_id")
    if not route_id:
        return None

    waypoints = raw_route.get("path_waypoints") or []
    spawn_point_xy = _last_xy(waypoints) if reverse else _first_xy(waypoints)
    suffix = REVERSE_ROUTE_SUFFIX if reverse else ""

    return RouteSpec(
        route_id=str(route_id) + suffix,
        route_type=str(raw_route.get("route_type") or "unknown"),
        route_length_m=float(raw_route.get("route_length_m") or 0.0),
        supported_behaviors=list(
            (raw_route.get("attributes") or {}).get("supported_behaviors") or []
        ),
        spawn_point_xy=spawn_point_xy,
    )


def _first_xy(waypoints: Sequence[Any]) -> Optional[List[float]]:
    if not waypoints:
        return None
    return _point_xy(waypoints[0])


def _last_xy(waypoints: Sequence[Any]) -> Optional[List[float]]:
    if not waypoints:
        return None
    return _point_xy(waypoints[-1])


def _point_xy(point: Any) -> Optional[List[float]]:
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return [float(point[0]), float(point[1])]
    return None
