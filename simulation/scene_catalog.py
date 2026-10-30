import json
import copy
from pathlib import Path
from typing import Dict, List, Optional

REVERSE_ROUTE_SUFFIX = '__reverse'


class SceneCatalogError(ValueError):
    """Raised when a scene catalog or scene route selection is invalid."""


class SceneCatalog:
    def __init__(self, catalog_path: Path, map_name: str):
        self.catalog_path = catalog_path
        self.map_name = map_name

        with catalog_path.open(encoding='utf-8') as catalog_file:
            catalog = json.load(catalog_file)

        catalog_map_name = catalog.get('map_name')
        if catalog_map_name is not None and catalog_map_name != map_name:
            raise SceneCatalogError(
                f'Scene catalog map_name mismatch: expected {map_name}, got {catalog_map_name}.'
            )

        self.behavior_profiles = catalog.get('behavior_profiles', {})
        self.route_type_definitions = catalog.get('route_type_definitions', {})
        self.cells_by_id = {}
        for raw_cell in catalog.get('cells', []):
            cell = copy.deepcopy(raw_cell)
            routes = cell.get('routes') or []
            routes.extend(_synthetic_reverse_routes(routes))
            self.cells_by_id[cell['id']] = cell

    @classmethod
    def from_map_config(cls, map_config: Dict[str, object]):
        map_name = map_config.get('map_name')
        if not map_name:
            raise SceneCatalogError('Scene catalog requires map.map_name to be configured.')

        default_path = Path('topology') / str(map_name) / 'walkable_cells.json'
        catalog_path = Path(__file__).resolve().parents[1] / Path(map_config.get('scene_catalog_path', default_path))
        if not catalog_path.exists():
            raise SceneCatalogError(
                f'Scene catalog file not found at {catalog_path}. Generate walkable cells first or set '
                f'map.scene_catalog_path explicitly.'
            )

        return cls(catalog_path=catalog_path, map_name=str(map_name))

    def get_behavior_profile(self, behavior_name: str) -> Dict[str, object]:
        try:
            return self.behavior_profiles[behavior_name]
        except KeyError as exc:
            raise SceneCatalogError(f'Unknown behavior "{behavior_name}" in scene catalog.') from exc

    def get_scene(self, scene_id: str) -> Dict[str, object]:
        try:
            scene = self.cells_by_id[scene_id]
        except KeyError as exc:
            raise SceneCatalogError(f'Unknown scene_id "{scene_id}" in scene catalog.') from exc

        if not scene.get('attributes', {}).get('usable', False):
            reason = scene.get('attributes', {}).get('usable_reason', 'scene_not_usable')
            raise SceneCatalogError(f'Scene "{scene_id}" is not usable: {reason}.')

        return scene

    def get_routes(
        self,
        scene_id: str,
        behavior_name: str,
        route_type: Optional[str] = None,
        route_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        scene = self.get_scene(scene_id)
        routes = scene.get('routes', [])
        if not routes:
            raise SceneCatalogError(f'Scene "{scene_id}" has no routes.')

        matching_routes = []
        for route in routes:
            if route_id is not None and route.get('route_id') != route_id:
                continue
            if route_type is not None and route.get('route_type') != route_type:
                continue
            supported_behaviors = route.get('attributes', {}).get('supported_behaviors', [])
            if behavior_name not in supported_behaviors:
                continue
            matching_routes.append(route)

        if matching_routes:
            return matching_routes

        filters = [f'behavior={behavior_name}']
        if route_type is not None:
            filters.append(f'route_type={route_type}')
        if route_id is not None:
            filters.append(f'route_id={route_id}')
        raise SceneCatalogError(
            f'No routes in scene "{scene_id}" satisfy {", ".join(filters)}.'
        )


def _synthetic_reverse_routes(routes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    reverse_routes = []
    existing_route_ids = {route.get('route_id') for route in routes}
    for route in routes:
        route_id = route.get('route_id')
        if not route_id:
            continue
        reverse_route_id = f'{route_id}{REVERSE_ROUTE_SUFFIX}'
        if route_id.endswith(REVERSE_ROUTE_SUFFIX) or reverse_route_id in existing_route_ids:
            continue

        reversed_route = copy.deepcopy(route)
        reversed_route['route_id'] = reverse_route_id
        reversed_route['start_portal'] = route.get('end_portal')
        reversed_route['end_portal'] = route.get('start_portal')

        for key in ('node_ids', 'edge_ids', 'path_waypoints'):
            _reverse_list_field(reversed_route, route, key)
        _reverse_crossing_bools(reversed_route, route, 'path_crossing_road_bools')
        _reverse_config_waypoints(reversed_route, route)

        if route.get('destination') is not None:
            reversed_route['spawn_location'] = copy.deepcopy(route.get('destination'))
        if route.get('spawn_location') is not None:
            reversed_route['destination'] = copy.deepcopy(route.get('spawn_location'))

        reversed_route['spawn_transform'] = {
            'location': copy.deepcopy(reversed_route.get('spawn_location')),
            'source': 'synthetic_reverse_runtime_projection',
            'requires_runtime_projection': True,
        }

        reverse_routes.append(reversed_route)
    return reverse_routes


def _reverse_list_field(reversed_route: Dict[str, object], route: Dict[str, object], key: str) -> None:
    value = route.get(key)
    if isinstance(value, list):
        reversed_route[key] = list(reversed(value))


def _reverse_crossing_bools(reversed_route: Dict[str, object], route: Dict[str, object], key: str) -> None:
    value = route.get(key)
    if not isinstance(value, list):
        return
    if len(value) <= 1:
        reversed_route[key] = list(value)
        return
    reversed_route[key] = [value[0]] + list(reversed(value[1:]))


def _reverse_config_waypoints(reversed_route: Dict[str, object], route: Dict[str, object]) -> None:
    path_waypoints = reversed_route.get('path_waypoints')
    if isinstance(path_waypoints, list) and len(path_waypoints) >= 2:
        reversed_route['config_waypoints'] = copy.deepcopy(path_waypoints[1:-1])

    path_crossing_bools = reversed_route.get('path_crossing_road_bools')
    if isinstance(path_crossing_bools, list) and len(path_crossing_bools) >= 1:
        reversed_route['config_crossing_road_bools'] = copy.deepcopy(path_crossing_bools[1:])
