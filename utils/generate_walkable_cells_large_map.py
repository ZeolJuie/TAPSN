import argparse
import colorsys
import hashlib
import heapq
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import carla
import networkx as nx
import numpy as np
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import unary_union

from simulation.obstacles import extract_sidewalk_borders
from simulation.path_planner import EdgeType, PedPathPlanner


ROUTE_TYPE_DEFINITIONS = {
    "sidewalk_through": {
        "description": "Traverse one connected sidewalk corridor inside a sidewalk cell.",
        "crossing_required": False,
    },
    "junction_through": {
        "description": "Move straight through a junction cell while staying on sidewalk edges only.",
        "crossing_required": False,
    },
    "crosswalk_cross": {
        "description": "Cross a road through at least one marked crosswalk edge.",
        "crossing_required": True,
    },
    "unmarked_junction_cross": {
        "description": "Cross a junction through unmarked jaywalking-at-junction edges without a marked crosswalk.",
        "crossing_required": True,
    },
    "junction_turn": {
        "description": "Move through a junction cell while staying on sidewalk edges only.",
        "crossing_required": False,
    },
    "junction_u_turn": {
        "description": "Reverse direction within a junction support graph while staying on sidewalk edges only.",
        "crossing_required": False,
    },
}

BEHAVIOR_PROFILES = {
    "steady_walk": {
        "description": "Nominal sidewalk walking profile.",
        "speed_multiplier": 1.0,
        "crossing_speed_factor": 1.5,
        "crossing_safety_margin": 1.5,
    },
    "slow_walk": {
        "description": "Slow and conservative sidewalk walking profile.",
        "speed_multiplier": 0.8,
        "crossing_speed_factor": 1.3,
        "crossing_safety_margin": 1.8,
    },
    "fast_pass": {
        "description": "Fast sidewalk traversal profile.",
        "speed_multiplier": 1.25,
        "crossing_speed_factor": 1.7,
        "crossing_safety_margin": 1.2,
    },
    "cautious_cross": {
        "description": "Cross only with a generous perceived gap.",
        "speed_multiplier": 0.95,
        "crossing_speed_factor": 1.2,
        "crossing_safety_margin": 2.0,
    },
    "normal_cross": {
        "description": "Default crossing profile aligned with current runtime defaults.",
        "speed_multiplier": 1.0,
        "crossing_speed_factor": 1.5,
        "crossing_safety_margin": 1.5,
    },
    "assertive_cross": {
        "description": "Accept smaller gaps and cross more aggressively.",
        "speed_multiplier": 1.05,
        "crossing_speed_factor": 1.7,
        "crossing_safety_margin": 0.8,
    },
    "ignore_gap_cross": {
        "description": "Treat gaps as effectively always acceptable.",
        "speed_multiplier": 1.1,
        "crossing_speed_factor": 1.9,
        "crossing_safety_margin": -0.5,
    },
}

ROUTE_TYPE_SUPPORTED_BEHAVIORS = {
    "sidewalk_through": ["steady_walk", "slow_walk", "fast_pass"],
    "junction_through": ["steady_walk", "slow_walk", "fast_pass"],
    "crosswalk_cross": ["cautious_cross", "normal_cross", "assertive_cross", "ignore_gap_cross"],
    "unmarked_junction_cross": ["cautious_cross", "normal_cross", "assertive_cross", "ignore_gap_cross"],
    "junction_turn": ["steady_walk", "slow_walk", "fast_pass"],
    "junction_u_turn": ["steady_walk", "slow_walk", "fast_pass"],
}

JUNCTION_THROUGH_MAX_DEFLECTION_DEG = 35.0
JUNCTION_U_TURN_MIN_DEFLECTION_DEG = 140.0
JUNCTION_U_TURN_MAX_LATERAL_SHIFT_M = 8.0
JUNCTION_U_TURN_MAX_LATERAL_TO_LONGITUDINAL_RATIO = 0.45

CROSSING_MIN_ROUTE_CANDIDATES = 12
CROSSING_MAX_ROUTE_CANDIDATES = 36
CROSSING_MAX_SEED_ROUTES = 8
CROSSING_TARGET_EDGE_DENSITY = 10
CROSSING_ROUTE_TYPE_LIMITS = {
    "crosswalk_cross": 12,
    "junction_turn": 12,
    "junction_through": 8,
    "unmarked_junction_cross": 12,
    "junction_u_turn": 0,
}
CROSSING_ROUTE_TYPE_MAX_DETOUR_RATIO = {
    "crosswalk_cross": 3.0,
    "junction_turn": 2.4,
    "junction_through": 1.75,
    "unmarked_junction_cross": 2.5,
    "junction_u_turn": 0.0,
}
STATIC_OBSTACLE_SIBLING_REPAIR_MAX_START_DISTANCE_M = 4.0
STATIC_OBSTACLE_SIBLING_REPAIR_MAX_END_DISTANCE_M = 12.0
STATIC_OBSTACLE_SIBLING_REPAIR_MAX_START_HEADING_DIFF_DEG = 35.0
STATIC_OBSTACLE_SIBLING_REPAIR_MAX_END_HEADING_DIFF_DEG = 55.0
STATIC_OBSTACLE_SIBLING_REPAIR_MAX_ROUTE_LENGTH_RATIO = 1.8
CROSSING_ROUTE_TYPE_PRIORITY = [
    "crosswalk_cross",
    "junction_turn",
    "junction_through",
    "unmarked_junction_cross",
    "junction_u_turn",
]

STATIC_OBSTACLE_LABEL_ALIASES = {
    "Walls": ("Walls", "Wall"),
    "Fences": ("Fences", "Fence"),
    "GuardRail": ("GuardRail",),
    "Static": ("Static",),
    "Other": ("Other",),
    "Vegetation": ("Vegetation",),
    "Poles": ("Poles",),
    "TrafficLight": ("TrafficLight",),
    "TrafficSigns": ("TrafficSigns", "TrafficSign"),
    "Buildings": ("Buildings", "Building"),
    "Bridge": ("Bridge",),
    "RailTrack": ("RailTrack",),
    "Car": ("Car",),
    "Truck": ("Truck",),
    "Bus": ("Bus",),
    "Motorcycle": ("Motorcycle",),
    "Bicycle": ("Bicycle",),
    "Train": ("Train",),
    "Vehicles": ("Vehicles", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Train"),
}
DEFAULT_STATIC_OBSTACLE_LABELS = (
    "Walls",
    "Fences",
    "GuardRail",
    "Static",
    "Other",
    "Vegetation",
    "Poles",
    "TrafficLight",
    "TrafficSigns",
    "Buildings",
    "Bridge",
    "RailTrack",
    "Car",
    "Truck",
    "Bus",
    "Motorcycle",
    "Bicycle",
    "Train",
)
DEFAULT_STATIC_OBSTACLE_MAX_BOTTOM_Z_M = 0.35
DEFAULT_STATIC_OBSTACLE_MIN_AREA_M2 = 0.08
DEFAULT_STATIC_OBSTACLE_MIN_SPAN_M = 0.12
DEFAULT_STATIC_OBSTACLE_FOOTPRINT_BUFFER_M = 0.1
DEFAULT_VEGETATION_MAX_HEIGHT_M = 2.5
DEFAULT_EDGE_CORRIDOR_SIDE_MARGIN_M = 0.15
DEFAULT_EDGE_MIN_PASSAGE_WIDTH_M = 0.9
DEFAULT_SIDEWALK_INWARD_OFFSET_MAX_M = 2.0
SIDEWALK_INWARD_OFFSET_MIN_M = 0.05
SIDEWALK_INWARD_OFFSET_BORDER_MARGIN_M = 0.2
SIDEWALK_INWARD_OFFSET_PATH_CLEARANCE_M = 0.2
SIDEWALK_INWARD_OFFSET_ATTEMPT_SCALES = (1.0, 0.5, 0.25)
STATIC_OBSTACLE_ENDPOINT_ANCHOR_RADIUS_M = 0.18
ROUTE_SPAWN_Z_OFFSET_M = 0.2
DEFAULT_TILE_CATALOG_PATH = Path("tmp") / "town12_sidewalk_tiles" / "tile_catalog.json"
DEFAULT_TOPOLOGY_ROOT = Path("topology")
DEFAULT_CARLA_MAPS_ROOT = Path(
    os.environ.get("CARLA_MAPS_ROOT", "Carla/Maps")
)

Bounds = Tuple[float, float, float, float]
Cell = Tuple[int, int]


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: int
    n1: int
    n2: int
    edge_type: EdgeType
    length: float
    road_id: int
    section_id: int
    lane_id: int
    in_junction: bool
    entry_lane_type: carla.LaneType
    exit_lane_type: carla.LaneType
    x1: float
    y1: float
    x2: float
    y2: float
    width_hint: float

    @property
    def lane_key(self) -> Tuple[int, int, int]:
        return self.road_id, self.section_id, self.lane_id

    @property
    def is_pure_sidewalk(self) -> bool:
        return (
            self.edge_type == EdgeType.SIDEWALK
            and self.entry_lane_type == carla.LaneType.Sidewalk
            and self.exit_lane_type == carla.LaneType.Sidewalk
        )

    @property
    def is_connector_sidewalk(self) -> bool:
        return self.edge_type == EdgeType.SIDEWALK and not self.is_pure_sidewalk

    @property
    def is_crossing_seed(self) -> bool:
        return self.edge_type in {EdgeType.CROSSWALK, EdgeType.JAYWALKING_JUNCTION}

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def other(self, node_id: int) -> int:
        if node_id == self.n1:
            return self.n2
        if node_id == self.n2:
            return self.n1
        raise KeyError(f"Node {node_id} is not part of edge {self.edge_id}")


@dataclass(frozen=True)
class StaticObstacleValidationConfig:
    label_names: Tuple[str, ...]
    max_bottom_z_m: float
    min_footprint_area_m2: float
    min_footprint_span_m: float
    footprint_buffer_m: float
    vegetation_max_height_m: float
    edge_corridor_side_margin_m: float
    edge_min_passage_width_m: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "label_names": list(self.label_names),
            "max_bottom_z_m": round(self.max_bottom_z_m, 3),
            "min_footprint_area_m2": round(self.min_footprint_area_m2, 3),
            "min_footprint_span_m": round(self.min_footprint_span_m, 3),
            "footprint_buffer_m": round(self.footprint_buffer_m, 3),
            "vegetation_max_height_m": round(self.vegetation_max_height_m, 3),
            "edge_corridor_side_margin_m": round(self.edge_corridor_side_margin_m, 3),
            "edge_min_passage_width_m": round(self.edge_min_passage_width_m, 3),
            "edge_block_rule": "corridor_endpoint_connectivity",
        }


@dataclass(frozen=True)
class StaticObstacleGeometrySource:
    object_id: int
    canonical_label_name: str
    source_label_name: str
    bounding_box: carla.BoundingBox
    transform: Optional[carla.Transform]


@dataclass(frozen=True)
class StaticObstacleRecord:
    object_id: int
    label_name: str
    source_label_name: str
    height_m: float
    bottom_z: float
    raw_footprint: Polygon
    buffered_footprint: Polygon

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return tuple(float(value) for value in self.buffered_footprint.bounds)


@dataclass(frozen=True)
class BlockedEdgeRecord:
    edge_id: int
    reason: str
    corridor_width_m: float
    blocking_object_ids: Tuple[int, ...]
    blocking_labels: Tuple[str, ...]


class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> int:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return root_left

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left

        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return root_left


@dataclass(frozen=True)
class LaneComponent:
    band_key: Tuple[int, int, int, Tuple[int, ...]]
    lane_key: Tuple[int, int, int]
    ordered_edge_ids: List[int]
    total_length: float
    projection_min: float
    projection_max: float


def lane_type_name(lane_type: carla.LaneType) -> str:
    mapping = {
        carla.LaneType.Sidewalk: "Sidewalk",
        carla.LaneType.Shoulder: "Shoulder",
        carla.LaneType.Driving: "Driving",
        carla.LaneType.Bidirectional: "Bidirectional",
        carla.LaneType.Parking: "Parking",
        carla.LaneType.Border: "Border",
        carla.LaneType.Median: "Median",
    }
    return mapping.get(lane_type, f"LaneType({int(lane_type)})")


def svg_color(index: int) -> str:
    hue = ((index * 137) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.58, 0.72)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def lane_sign(lane_id: int) -> int:
    return -1 if lane_id < 0 else 1


def world_to_canvas(
    point: Tuple[float, float], min_x: float, max_y: float, padding: float
) -> Tuple[float, float]:
    x, y = point
    return x - min_x + padding, max_y - y + padding


def load_world(client: carla.Client, map_name: str, *, large_map_safe: bool = False) -> carla.World:
    world = client.get_world()
    current_name = world.get_map().name
    if current_name.endswith(map_name):
        return world
    if large_map_safe:
        try:
            return client.load_world(map_name, False, carla.MapLayer.NONE)
        except TypeError:
            return client.load_world(map_name)
    return client.load_world(map_name)


def apply_large_map_safe_settings(world: carla.World) -> None:
    settings = world.get_settings()
    settings.no_rendering_mode = True
    settings.tile_stream_distance = 1000.0
    settings.actor_active_distance = 1000.0
    settings.spectator_as_ego = False
    world.apply_settings(settings)
    time.sleep(1.0)


def load_tile_catalog(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_tile_record(catalog: Dict[str, object], tile_id: str) -> Dict[str, object]:
    for tile in catalog["tiles"]:
        if tile["id"] == tile_id:
            return tile
    raise KeyError(f"Tile not found: {tile_id}")


def tile_center(bounds: Sequence[float]) -> carla.Location:
    min_x, min_y, max_x, max_y = [float(value) for value in bounds]
    return carla.Location(x=(min_x + max_x) * 0.5, y=(min_y + max_y) * 0.5, z=0.0)


def nearest_ground_height_from_waypoint(carla_map: carla.Map, location: carla.Location) -> Optional[float]:
    for lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Driving, carla.LaneType.Any):
        try:
            waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=lane_type)
        except RuntimeError:
            waypoint = None
        if waypoint is not None:
            return float(waypoint.transform.location.z)
    return None


def nearest_waypoint_height(world: carla.World, center: carla.Location) -> float:
    carla_map = world.get_map()
    ground_height = nearest_ground_height_from_waypoint(carla_map, center)
    if ground_height is not None:
        return ground_height
    for lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Driving, carla.LaneType.Any):
        waypoint = carla_map.get_waypoint(center, lane_type=lane_type)
        if waypoint is not None:
            return float(waypoint.transform.location.z)
    return 0.0


def focus_tile(world: carla.World, bounds: Sequence[float], spectator_height: float = 220.0) -> None:
    settings = world.get_settings()
    settings.spectator_as_ego = True
    world.apply_settings(settings)
    center = tile_center(bounds)
    base_z = nearest_waypoint_height(world, center)
    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(x=center.x, y=center.y, z=base_z + spectator_height),
            carla.Rotation(pitch=-88.0, yaw=0.0, roll=0.0),
        )
    )
    time.sleep(1.0)


class BoundsFilteredPedPathPlanner(PedPathPlanner):
    def __init__(
        self,
        carla_map: carla.Map,
        bounds: Sequence[float],
        waypoint_distance: float = 5.0,
        jaywalking_weight_factor: float = 2.0,
    ):
        self.bounds = tuple(float(value) for value in bounds)
        super().__init__(carla_map, waypoint_distance=waypoint_distance, jaywalking_weight_factor=jaywalking_weight_factor)

    def _in_bounds(self, location: carla.Location, extra_margin: float = 0.0) -> bool:
        min_x, min_y, max_x, max_y = self.bounds
        return (
            min_x - extra_margin <= location.x <= max_x + extra_margin
            and min_y - extra_margin <= location.y <= max_y + extra_margin
        )

    def _segment_intersects_bounds(self, segment: Tuple[carla.Waypoint, carla.Waypoint]) -> bool:
        start, end = segment
        if self._in_bounds(start.transform.location) or self._in_bounds(end.transform.location):
            return True
        midpoint = carla.Location(
            x=(start.transform.location.x + end.transform.location.x) * 0.5,
            y=(start.transform.location.y + end.transform.location.y) * 0.5,
            z=(start.transform.location.z + end.transform.location.z) * 0.5,
        )
        return self._in_bounds(midpoint)

    def _crosswalk_intersects_bounds(self, corners: Sequence[carla.Location]) -> bool:
        return any(self._in_bounds(point, extra_margin=10.0) for point in corners)

    def _generate_crosswalk_edges_local(self):
        crosswalk_corners = self.carla_map.get_crosswalks()
        filtered_crosswalk_corners = []
        crosswalk = []

        for point in crosswalk_corners:
            if point not in crosswalk:
                crosswalk.append(point)
            else:
                candidate = list(crosswalk)
                if len(candidate) == 6:
                    del candidate[4]
                    del candidate[1]
                if len(candidate) == 4 and self._crosswalk_intersects_bounds(candidate):
                    filtered_crosswalk_corners.extend(candidate)
                crosswalk = []

        crosswalk_edges = []
        for index in range(0, len(filtered_crosswalk_corners), 4):
            quad = filtered_crosswalk_corners[index : index + 4]
            if len(quad) < 4:
                continue
            crosswalk_waypoints = []
            for side_start, side_end in ((quad[0], quad[1]), (quad[2], quad[3])):
                middle = carla.Location(
                    x=(side_start.x + side_end.x) * 0.5,
                    y=(side_start.y + side_end.y) * 0.5,
                    z=(side_start.z + side_end.z) * 0.5,
                )
                middle_waypoint = self.carla_map.get_waypoint(middle, lane_type=carla.LaneType.Shoulder)
                if middle_waypoint is not None:
                    crosswalk_waypoints.append(middle_waypoint)
            crosswalk_edges.extend(self._generate_edge_dicts(crosswalk_waypoints, EdgeType.CROSSWALK))

        return crosswalk_edges

    def _build_topology(self):
        carla_topology = [segment for segment in self.carla_map.get_topology() if self._segment_intersects_bounds(segment)]
        self.ped_topology = []

        for segment in carla_topology:
            sidewalk_waypoints = self._generate_sidewalk_waypoints(segment)
            for side in sidewalk_waypoints:
                if side:
                    for lane in side.values():
                        sidewalk_edges = self._generate_edge_dicts(lane, EdgeType.SIDEWALK)
                        self.ped_topology.extend(sidewalk_edges)

        junction_edges = self._generate_junction_edges(carla_topology)
        self.ped_topology.extend(junction_edges)

        crosswalk_edges = self._generate_crosswalk_edges_local()
        connection_edges = self._generate_edges_to_crosswalks(crosswalk_edges, connection_radius=10)
        self.ped_topology.extend(connection_edges)
        self.ped_topology.extend(crosswalk_edges)


def build_node_locations(path_planner: PedPathPlanner) -> Dict[int, Tuple[float, float, float]]:
    node_locations: Dict[int, Tuple[float, float, float]] = {}
    for node_id, data in path_planner.graph.nodes(data=True):
        location = data["waypoint"].transform.location
        node_locations[node_id] = (float(location.x), float(location.y), float(location.z))
    return node_locations


def build_edge_records(path_planner: PedPathPlanner) -> Tuple[Dict[int, EdgeRecord], Dict[int, List[int]]]:
    edges: Dict[int, EdgeRecord] = {}
    adjacency: Dict[int, List[int]] = defaultdict(list)

    for edge_id, (n1, n2, data) in enumerate(path_planner.graph.edges.data()):
        entry_wp = data["entry_waypoint"]
        exit_wp = data["exit_waypoint"]
        entry_loc = entry_wp.transform.location
        exit_loc = exit_wp.transform.location
        width_hint = max(entry_wp.lane_width, exit_wp.lane_width, 1.5)

        record = EdgeRecord(
            edge_id=edge_id,
            n1=n1,
            n2=n2,
            edge_type=data["type"],
            length=float(entry_loc.distance(exit_loc)),
            road_id=int(entry_wp.road_id),
            section_id=int(entry_wp.section_id),
            lane_id=int(entry_wp.lane_id),
            in_junction=bool(entry_wp.is_junction or exit_wp.is_junction),
            entry_lane_type=entry_wp.lane_type,
            exit_lane_type=exit_wp.lane_type,
            x1=float(entry_loc.x),
            y1=float(entry_loc.y),
            x2=float(exit_loc.x),
            y2=float(exit_loc.y),
            width_hint=float(width_hint),
        )
        edges[edge_id] = record
        adjacency[n1].append(edge_id)
        adjacency[n2].append(edge_id)

    return edges, adjacency


def static_obstacle_config_from_args(args: argparse.Namespace) -> StaticObstacleValidationConfig:
    return StaticObstacleValidationConfig(
        label_names=tuple(args.obstacle_labels),
        max_bottom_z_m=float(args.obstacle_max_bottom_z),
        min_footprint_area_m2=float(args.obstacle_min_area),
        min_footprint_span_m=float(args.obstacle_min_span),
        footprint_buffer_m=float(args.obstacle_footprint_buffer),
        vegetation_max_height_m=float(args.vegetation_max_height),
        edge_corridor_side_margin_m=float(args.edge_corridor_side_margin),
        edge_min_passage_width_m=float(args.edge_min_passage_width),
    )


def static_obstacle_config_from_catalog(catalog: Dict[str, object]) -> StaticObstacleValidationConfig:
    validation_rules = catalog.get("static_obstacle_validation_rules", {})
    label_names = tuple(validation_rules.get("label_names", DEFAULT_STATIC_OBSTACLE_LABELS))
    return StaticObstacleValidationConfig(
        label_names=label_names,
        max_bottom_z_m=float(validation_rules.get("max_bottom_z_m", DEFAULT_STATIC_OBSTACLE_MAX_BOTTOM_Z_M)),
        min_footprint_area_m2=float(
            validation_rules.get("min_footprint_area_m2", DEFAULT_STATIC_OBSTACLE_MIN_AREA_M2)
        ),
        min_footprint_span_m=float(
            validation_rules.get("min_footprint_span_m", DEFAULT_STATIC_OBSTACLE_MIN_SPAN_M)
        ),
        footprint_buffer_m=float(
            validation_rules.get("footprint_buffer_m", DEFAULT_STATIC_OBSTACLE_FOOTPRINT_BUFFER_M)
        ),
        vegetation_max_height_m=float(
            validation_rules.get("vegetation_max_height_m", DEFAULT_VEGETATION_MAX_HEIGHT_M)
        ),
        edge_corridor_side_margin_m=float(
            validation_rules.get("edge_corridor_side_margin_m", DEFAULT_EDGE_CORRIDOR_SIDE_MARGIN_M)
        ),
        edge_min_passage_width_m=float(
            validation_rules.get("edge_min_passage_width_m", DEFAULT_EDGE_MIN_PASSAGE_WIDTH_M)
        ),
    )


def bbox_intersects(
    left_bounds: Tuple[float, float, float, float],
    right_bounds: Tuple[float, float, float, float],
) -> bool:
    return not (
        left_bounds[2] < right_bounds[0]
        or right_bounds[2] < left_bounds[0]
        or left_bounds[3] < right_bounds[1]
        or right_bounds[3] < left_bounds[1]
    )


def geometry_polygon_parts(geometry) -> List[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, GeometryCollection):
        parts: List[Polygon] = []
        for item in geometry.geoms:
            parts.extend(geometry_polygon_parts(item))
        return parts
    if hasattr(geometry, "geoms"):
        parts = []
        for item in geometry.geoms:
            parts.extend(geometry_polygon_parts(item))
        return parts
    return []


def rounded_xy_ring(coords: Sequence[Tuple[float, float]]) -> List[List[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in coords]


def geometry_polygon_rings(geometry) -> List[Dict[str, object]]:
    polygon_parts = geometry_polygon_parts(geometry)
    rings: List[Dict[str, object]] = []
    for polygon in polygon_parts:
        rings.append(
            {
                "outer": rounded_xy_ring(list(polygon.exterior.coords)),
                "holes": [rounded_xy_ring(list(interior.coords)) for interior in polygon.interiors],
            }
        )
    return rings


def environment_object_world_vertices(env_object) -> List[carla.Location]:
    bounding_box = env_object.bounding_box
    if hasattr(bounding_box, "get_world_vertices"):
        try:
            return list(bounding_box.get_world_vertices(env_object.transform))
        except TypeError:
            return list(bounding_box.get_world_vertices(carla.Transform(bounding_box.location, bounding_box.rotation)))
    raise RuntimeError("CARLA bounding box API does not expose get_world_vertices().")


def oriented_box_polygon(center_x: float, center_y: float, yaw_deg: float, extent_x: float, extent_y: float) -> Polygon:
    yaw_rad = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=float)
    lateral = np.array([-forward[1], forward[0]], dtype=float)
    center = np.array([center_x, center_y], dtype=float)
    corners = [
        center + forward * extent_x + lateral * extent_y,
        center + forward * extent_x - lateral * extent_y,
        center - forward * extent_x - lateral * extent_y,
        center - forward * extent_x + lateral * extent_y,
    ]
    return Polygon([(float(point[0]), float(point[1])) for point in corners])


def bounding_box_contains_location(
    bounding_box,
    location: carla.Location,
    transform: carla.Transform,
) -> bool:
    diff = bounding_box.location - location
    diff = transform.transform(diff)
    return (
        abs(float(diff.x)) < float(bounding_box.extent.x)
        and abs(float(diff.y)) < float(bounding_box.extent.y)
        and abs(float(diff.z)) < float(bounding_box.extent.z)
    )


def bounding_box_reference_transform(
    bounding_box,
    transform: Optional[carla.Transform],
) -> carla.Transform:
    if transform is None:
        return carla.Transform(
            carla.Location(
                x=float(bounding_box.location.x),
                y=float(bounding_box.location.y),
                z=float(bounding_box.location.z),
            ),
            bounding_box.rotation,
        )

    candidate_location = transform.location + (bounding_box.location - transform.location) * 0.1
    inverse_bbox_rotation = carla.Transform(
        rotation=carla.Rotation(
            pitch=-float(bounding_box.rotation.pitch),
            yaw=-float(bounding_box.rotation.yaw),
            roll=-float(bounding_box.rotation.roll),
        )
    )
    if bounding_box_contains_location(bounding_box, candidate_location, inverse_bbox_rotation):
        return carla.Transform(
            carla.Location(
                x=float(transform.location.x),
                y=float(transform.location.y),
                z=float(bounding_box.location.z),
            ),
            transform.rotation,
        )
    return carla.Transform(
        carla.Location(
            x=float(bounding_box.location.x),
            y=float(bounding_box.location.y),
            z=float(bounding_box.location.z),
        ),
        bounding_box.rotation,
    )


def bounding_box_polygon_from_transform(
    bounding_box,
    transform: carla.Transform,
) -> Tuple[Polygon, float, float]:
    polygon = oriented_box_polygon(
        center_x=float(transform.location.x),
        center_y=float(transform.location.y),
        yaw_deg=float(transform.rotation.yaw),
        extent_x=float(bounding_box.extent.x),
        extent_y=float(bounding_box.extent.y),
    )
    bottom_z = float(bounding_box.location.z - bounding_box.extent.z)
    top_z = float(bounding_box.location.z + bounding_box.extent.z)
    return polygon.buffer(0.0), bottom_z, top_z


def environment_object_reference_transform(env_object) -> carla.Transform:
    return bounding_box_reference_transform(env_object.bounding_box, env_object.transform)


def environment_object_world_vertex_footprint_polygon(env_object) -> Tuple[Polygon, float, float]:
    world_vertices = environment_object_world_vertices(env_object)
    points_xy = [(float(vertex.x), float(vertex.y)) for vertex in world_vertices]
    polygon = Polygon(points_xy).convex_hull
    bottom_z = min(float(vertex.z) for vertex in world_vertices)
    top_z = max(float(vertex.z) for vertex in world_vertices)
    return polygon.buffer(0.0), bottom_z, top_z


def environment_object_transform_bbox_polygon(env_object) -> Tuple[Polygon, float, float]:
    return bounding_box_polygon_from_transform(env_object.bounding_box, environment_object_reference_transform(env_object))


def environment_object_bbox_center_polygon(env_object) -> Tuple[Polygon, float, float]:
    return bounding_box_polygon_from_transform(
        env_object.bounding_box,
        carla.Transform(
            carla.Location(
                x=float(env_object.bounding_box.location.x),
                y=float(env_object.bounding_box.location.y),
                z=float(env_object.bounding_box.location.z),
            ),
            env_object.bounding_box.rotation,
        ),
    )


def environment_object_footprint_polygon(env_object) -> Tuple[Polygon, float, float]:
    polygon, bottom_z, top_z = environment_object_transform_bbox_polygon(env_object)
    if not polygon.is_empty and polygon.area > 1e-6:
        return polygon, bottom_z, top_z
    return environment_object_bbox_center_polygon(env_object)


def environment_object_ground_probe_location(env_object) -> carla.Location:
    return environment_object_reference_transform(env_object).location


def static_obstacle_source_reference_transform(source: StaticObstacleGeometrySource) -> carla.Transform:
    return bounding_box_reference_transform(source.bounding_box, source.transform)


def static_obstacle_source_footprint_polygon(source: StaticObstacleGeometrySource) -> Tuple[Polygon, float, float]:
    return bounding_box_polygon_from_transform(source.bounding_box, static_obstacle_source_reference_transform(source))


def static_obstacle_source_ground_probe_location(source: StaticObstacleGeometrySource) -> carla.Location:
    return static_obstacle_source_reference_transform(source).location


def static_obstacle_source_geometry_key(
    canonical_label_name: str,
    bounding_box,
) -> Tuple[object, ...]:
    return (
        canonical_label_name,
        round(float(bounding_box.location.x), 3),
        round(float(bounding_box.location.y), 3),
        round(float(bounding_box.location.z), 3),
        round(float(bounding_box.extent.x), 3),
        round(float(bounding_box.extent.y), 3),
        round(float(bounding_box.extent.z), 3),
        round(float(bounding_box.rotation.pitch), 3),
        round(float(bounding_box.rotation.yaw), 3),
        round(float(bounding_box.rotation.roll), 3),
    )


def synthetic_level_bbox_object_id(
    source_label_name: str,
    bbox_index: int,
    bounding_box,
) -> int:
    digest = hashlib.sha1(
        (
            f"{source_label_name}|{bbox_index}|"
            f"{float(bounding_box.location.x):.6f}|{float(bounding_box.location.y):.6f}|{float(bounding_box.location.z):.6f}|"
            f"{float(bounding_box.extent.x):.6f}|{float(bounding_box.extent.y):.6f}|{float(bounding_box.extent.z):.6f}|"
            f"{float(bounding_box.rotation.pitch):.6f}|{float(bounding_box.rotation.yaw):.6f}|{float(bounding_box.rotation.roll):.6f}"
        ).encode("utf-8")
    ).hexdigest()
    return -int(digest[:15], 16)


def iter_static_obstacle_sources_for_labels(
    carla_world: carla.World,
    label_names: Sequence[str],
) -> List[StaticObstacleGeometrySource]:
    sources_by_key: Dict[Tuple[object, ...], StaticObstacleGeometrySource] = {}
    for canonical_label_name in label_names:
        aliases = STATIC_OBSTACLE_LABEL_ALIASES.get(canonical_label_name, (canonical_label_name,))
        for source_label_name in aliases:
            label = getattr(carla.CityObjectLabel, source_label_name, None)
            if label is None:
                continue
            for env_object in carla_world.get_environment_objects(label):
                source = StaticObstacleGeometrySource(
                    object_id=int(env_object.id),
                    canonical_label_name=canonical_label_name,
                    source_label_name=source_label_name,
                    bounding_box=env_object.bounding_box,
                    transform=env_object.transform,
                )
                key = static_obstacle_source_geometry_key(canonical_label_name, source.bounding_box)
                sources_by_key.setdefault(key, source)
            try:
                level_bboxes = list(carla_world.get_level_bbs(label))
            except RuntimeError:
                level_bboxes = []
            for bbox_index, bounding_box in enumerate(level_bboxes):
                key = static_obstacle_source_geometry_key(canonical_label_name, bounding_box)
                if key in sources_by_key:
                    continue
                sources_by_key[key] = StaticObstacleGeometrySource(
                    object_id=synthetic_level_bbox_object_id(source_label_name, bbox_index, bounding_box),
                    canonical_label_name=canonical_label_name,
                    source_label_name=f"level_bbs:{source_label_name}",
                    bounding_box=bounding_box,
                    transform=None,
                )
    return list(sources_by_key.values())


def extract_static_obstacle_records(
    carla_world: carla.World,
    config: StaticObstacleValidationConfig,
) -> List[StaticObstacleRecord]:
    obstacle_records: List[StaticObstacleRecord] = []
    carla_map = carla_world.get_map()
    for source in iter_static_obstacle_sources_for_labels(carla_world, config.label_names):
        raw_footprint, bottom_z, top_z = static_obstacle_source_footprint_polygon(source)
        raw_footprint = raw_footprint.buffer(0.0)
        if raw_footprint.is_empty:
            continue

        ground_z = nearest_ground_height_from_waypoint(carla_map, static_obstacle_source_ground_probe_location(source))
        if ground_z is None:
            continue
        if bottom_z - ground_z > config.max_bottom_z_m + 1e-6:
            continue

        height_m = max(0.0, top_z - bottom_z)
        if source.canonical_label_name == "Vegetation" and height_m > config.vegetation_max_height_m + 1e-6:
            continue

        min_x, min_y, max_x, max_y = raw_footprint.bounds
        footprint_span = min(max_x - min_x, max_y - min_y)
        if raw_footprint.area < config.min_footprint_area_m2 - 1e-6:
            continue
        if footprint_span < config.min_footprint_span_m - 1e-6:
            continue

        buffered_footprint = raw_footprint.buffer(config.footprint_buffer_m, resolution=8)
        buffered_footprint = buffered_footprint.buffer(0.0)
        if buffered_footprint.is_empty:
            continue

        obstacle_records.append(
            StaticObstacleRecord(
                object_id=int(source.object_id),
                label_name=source.canonical_label_name,
                source_label_name=source.source_label_name,
                height_m=float(height_m),
                bottom_z=float(bottom_z),
                raw_footprint=raw_footprint,
                buffered_footprint=buffered_footprint,
            )
        )

    obstacle_records.sort(key=lambda record: (record.label_name, record.object_id))
    return obstacle_records


def relevant_obstacles_for_bounds(
    obstacle_records: Sequence[StaticObstacleRecord],
    bounds: Tuple[float, float, float, float],
) -> List[StaticObstacleRecord]:
    return [record for record in obstacle_records if bbox_intersects(record.bounds, bounds)]


def rounded_bounds_dict(bounds: Tuple[float, float, float, float]) -> Dict[str, float]:
    return {
        "min_x": round(float(bounds[0]), 3),
        "min_y": round(float(bounds[1]), 3),
        "max_x": round(float(bounds[2]), 3),
        "max_y": round(float(bounds[3]), 3),
    }


def bounds_for_edge_ids(edge_ids: Sequence[int], edges: Dict[int, EdgeRecord]) -> Tuple[float, float, float, float]:
    coordinates: List[Tuple[float, float]] = []
    for edge_id in edge_ids:
        edge = edges[edge_id]
        coordinates.extend(((edge.x1, edge.y1), (edge.x2, edge.y2)))
    if not coordinates:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(point[0] for point in coordinates),
        min(point[1] for point in coordinates),
        max(point[0] for point in coordinates),
        max(point[1] for point in coordinates),
    )


def grow_crossing_assignments(
    edges: Dict[int, EdgeRecord],
    adjacency: Dict[int, List[int]],
    approach_length: float,
) -> Dict[int, Set[int]]:
    base_edge_ids = sorted(
        edge_id
        for edge_id, edge in edges.items()
        if edge.is_crossing_seed or edge.is_connector_sidewalk or (edge.is_pure_sidewalk and edge.in_junction)
    )
    if not base_edge_ids:
        return {}

    raw_components = connected_components_for_lane(base_edge_ids, edges)
    component_edge_sets: List[Set[int]] = []
    component_node_sets: List[Set[int]] = []
    for component in raw_components:
        if not any(edges[edge_id].is_crossing_seed for edge_id in component):
            continue
        component_edge_sets.append(set(component))
        component_node_sets.append({node_id for edge_id in component for node_id in (edges[edge_id].n1, edges[edge_id].n2)})

    component_candidates: Dict[int, Dict[int, float]] = defaultdict(dict)
    for component_id, component_nodes in enumerate(component_node_sets):
        node_distances: Dict[int, float] = {}
        queue: List[Tuple[float, int]] = []
        for node_id in component_nodes:
            node_distances[node_id] = 0.0
            heapq.heappush(queue, (0.0, node_id))

        while queue:
            distance, node_id = heapq.heappop(queue)
            best_distance = node_distances.get(node_id)
            if best_distance is None or distance > best_distance + 1e-6:
                continue

            for edge_id in adjacency[node_id]:
                edge = edges[edge_id]
                if not edge.is_pure_sidewalk:
                    continue

                next_distance = distance + edge.length
                if next_distance > approach_length + 1e-6:
                    continue

                current_edge_distance = component_candidates[component_id].get(edge_id)
                if current_edge_distance is None or next_distance < current_edge_distance - 1e-6:
                    component_candidates[component_id][edge_id] = next_distance

                other_node_id = edge.other(node_id)
                current_node_distance = node_distances.get(other_node_id)
                if current_node_distance is None or next_distance < current_node_distance - 1e-6:
                    node_distances[other_node_id] = next_distance
                    heapq.heappush(queue, (next_distance, other_node_id))

    sidewalk_claims: Dict[int, Tuple[float, int]] = {}
    for component_id, candidates in component_candidates.items():
        for edge_id, distance in candidates.items():
            current = sidewalk_claims.get(edge_id)
            if current is None or distance < current[0] - 1e-6 or (
                abs(distance - current[0]) <= 1e-6 and component_id < current[1]
            ):
                sidewalk_claims[edge_id] = (distance, component_id)

    crossing_assignments: Dict[int, Set[int]] = {}
    for component_id, component_edges in enumerate(component_edge_sets):
        assigned_edges = set(component_edges)
        for edge_id, (_, claimed_component_id) in sidewalk_claims.items():
            if claimed_component_id == component_id:
                assigned_edges.add(edge_id)
        crossing_assignments[component_id] = assigned_edges

    return crossing_assignments


def connected_components_for_lane(
    edge_ids: Iterable[int], edges: Dict[int, EdgeRecord]
) -> List[Set[int]]:
    node_to_edges: Dict[int, Set[int]] = defaultdict(set)
    for edge_id in edge_ids:
        edge = edges[edge_id]
        node_to_edges[edge.n1].add(edge_id)
        node_to_edges[edge.n2].add(edge_id)

    remaining = set(edge_ids)
    components: List[Set[int]] = []
    while remaining:
        start_edge_id = min(remaining)
        queue = [start_edge_id]
        component: Set[int] = set()
        while queue:
            edge_id = queue.pop()
            if edge_id in component:
                continue
            component.add(edge_id)
            if edge_id in remaining:
                remaining.remove(edge_id)
            edge = edges[edge_id]
            for node_id in (edge.n1, edge.n2):
                for neighbor_edge_id in node_to_edges[node_id]:
                    if neighbor_edge_id not in component:
                        queue.append(neighbor_edge_id)
        components.append(component)
    return components


def contiguous_lane_bands(lane_ids: Iterable[int]) -> List[Tuple[int, ...]]:
    sorted_lane_ids = sorted(set(lane_ids), key=abs)
    if not sorted_lane_ids:
        return []

    bands: List[List[int]] = [[sorted_lane_ids[0]]]
    for lane_id in sorted_lane_ids[1:]:
        if abs(lane_id) == abs(bands[-1][-1]) + 1:
            bands[-1].append(lane_id)
        else:
            bands.append([lane_id])
    return [tuple(band) for band in bands]


def compute_axis(
    edge_ids: Iterable[int],
    edges: Dict[int, EdgeRecord],
) -> Tuple[np.ndarray, np.ndarray]:
    edge_ids = list(edge_ids)
    centers = np.array([edges[edge_id].center for edge_id in edge_ids], dtype=float)
    origin = centers.mean(axis=0)

    if len(centers) == 1:
        edge = edges[edge_ids[0]]
        axis = np.array([edge.x2 - edge.x1, edge.y2 - edge.y1], dtype=float)
    else:
        centered = centers - origin
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]

    if np.linalg.norm(axis) < 1e-6:
        edge = edges[edge_ids[0]]
        axis = np.array([edge.x2 - edge.x1, edge.y2 - edge.y1], dtype=float)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0], dtype=float)

    axis = axis / np.linalg.norm(axis)
    mean_direction = np.zeros(2, dtype=float)
    for edge_id in edge_ids:
        edge = edges[edge_id]
        direction = np.array([edge.x2 - edge.x1, edge.y2 - edge.y1], dtype=float)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            mean_direction += direction / norm
    if np.dot(mean_direction, axis) < 0:
        axis = -axis
    return origin, axis


def projection_interval(
    edge_ids: Iterable[int],
    edges: Dict[int, EdgeRecord],
    origin: np.ndarray,
    axis: np.ndarray,
) -> Tuple[float, float]:
    values = []
    for edge_id in edge_ids:
        center = np.array(edges[edge_id].center, dtype=float)
        values.append(float(np.dot(center - origin, axis)))
    return min(values), max(values)


def ordered_lane_path(component_edge_ids: Set[int], edges: Dict[int, EdgeRecord]) -> List[int]:
    node_to_edges: Dict[int, List[int]] = defaultdict(list)
    for edge_id in component_edge_ids:
        edge = edges[edge_id]
        node_to_edges[edge.n1].append(edge_id)
        node_to_edges[edge.n2].append(edge_id)

    endpoints = sorted(node_id for node_id, incident in node_to_edges.items() if len(incident) == 1)
    current_node = endpoints[0] if endpoints else min(node_to_edges)
    unused_edges = set(component_edge_ids)
    ordered_edges: List[int] = []
    previous_edge_id: Optional[int] = None

    while unused_edges:
        candidate_edge_ids = sorted(edge_id for edge_id in node_to_edges[current_node] if edge_id in unused_edges)
        if not candidate_edge_ids:
            next_edge_id = min(unused_edges)
            edge = edges[next_edge_id]
            current_node = min(edge.n1, edge.n2)
            candidate_edge_ids = [next_edge_id]

        if previous_edge_id is not None and len(candidate_edge_ids) > 1:
            candidate_edge_ids = [edge_id for edge_id in candidate_edge_ids if edge_id != previous_edge_id] or candidate_edge_ids

        next_edge_id = candidate_edge_ids[0]
        ordered_edges.append(next_edge_id)
        unused_edges.remove(next_edge_id)

        edge = edges[next_edge_id]
        current_node = edge.other(current_node)
        previous_edge_id = next_edge_id

    return ordered_edges


def split_lane_path(
    ordered_edge_ids: List[int],
    edges: Dict[int, EdgeRecord],
    target_length: float,
    min_tail_length: float,
) -> List[List[int]]:
    chunks: List[List[int]] = []
    current_chunk: List[int] = []
    current_length = 0.0

    for edge_id in ordered_edge_ids:
        current_chunk.append(edge_id)
        current_length += edges[edge_id].length
        if current_length >= target_length:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0.0

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) >= 2:
        tail_length = sum(edges[edge_id].length for edge_id in chunks[-1])
        if tail_length < min_tail_length:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

    return chunks


def chunk_end_ratios(
    ordered_edge_ids: List[int],
    edges: Dict[int, EdgeRecord],
    target_length: float,
    min_tail_length: float,
) -> List[float]:
    chunks = split_lane_path(ordered_edge_ids, edges, target_length, min_tail_length)
    total_length = sum(edges[edge_id].length for edge_id in ordered_edge_ids)
    if total_length <= 1e-6:
        return [1.0]

    ratios = []
    cumulative = 0.0
    for chunk in chunks:
        cumulative += sum(edges[edge_id].length for edge_id in chunk)
        ratios.append(cumulative / total_length)
    if ratios:
        ratios[-1] = 1.0
    return ratios


def assign_edges_to_ratio_chunks(
    ordered_edge_ids: List[int],
    edges: Dict[int, EdgeRecord],
    end_ratios: List[float],
) -> List[List[int]]:
    total_length = sum(edges[edge_id].length for edge_id in ordered_edge_ids)
    chunk_count = len(end_ratios)
    assigned = [[] for _ in range(chunk_count)]
    if chunk_count == 0:
        return assigned

    if total_length <= 1e-6:
        assigned[0].extend(ordered_edge_ids)
        return assigned

    cumulative = 0.0
    chunk_index = 0
    for edge_id in ordered_edge_ids:
        midpoint_ratio = (cumulative + edges[edge_id].length * 0.5) / total_length
        while chunk_index < chunk_count - 1 and midpoint_ratio > end_ratios[chunk_index] + 1e-6:
            chunk_index += 1
        assigned[chunk_index].append(edge_id)
        cumulative += edges[edge_id].length
    return assigned


def build_sidewalk_assignments(
    edges: Dict[int, EdgeRecord],
    reserved_edge_ids: Set[int],
    target_length: float,
    min_tail_length: float,
    min_longitudinal_length: float,
) -> List[List[int]]:
    remaining_by_lane: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for edge_id, edge in edges.items():
        if edge_id in reserved_edge_ids:
            continue
        if not edge.is_pure_sidewalk:
            continue
        remaining_by_lane[edge.lane_key].append(edge_id)

    lanes_by_sign: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for road_id, section_id, lane_id in remaining_by_lane:
        lanes_by_sign[(road_id, section_id, lane_sign(lane_id))].append(lane_id)

    lane_to_band: Dict[Tuple[int, int, int], Tuple[int, int, int, Tuple[int, ...]]] = {}
    for (road_id, section_id, sign), lane_ids in lanes_by_sign.items():
        for lane_band in contiguous_lane_bands(lane_ids):
            band_key = (road_id, section_id, sign, lane_band)
            for lane_id in lane_band:
                lane_to_band[(road_id, section_id, lane_id)] = band_key

    sidewalk_chunks: List[List[int]] = []
    components_by_band: Dict[Tuple[int, int, int, Tuple[int, ...]], List[LaneComponent]] = defaultdict(list)

    for lane_key in sorted(remaining_by_lane):
        band_key = lane_to_band[lane_key]
        band_edge_ids = []
        for candidate_lane_key, candidate_edge_ids in remaining_by_lane.items():
            if lane_to_band[candidate_lane_key] == band_key:
                band_edge_ids.extend(candidate_edge_ids)
        origin, axis = compute_axis(band_edge_ids, edges)

        components = connected_components_for_lane(remaining_by_lane[lane_key], edges)
        for component in components:
            ordered_edges = ordered_lane_path(component, edges)
            total_length = sum(edges[edge_id].length for edge_id in ordered_edges)
            projection_min, projection_max = projection_interval(ordered_edges, edges, origin, axis)
            components_by_band[band_key].append(
                LaneComponent(
                    band_key=band_key,
                    lane_key=lane_key,
                    ordered_edge_ids=ordered_edges,
                    total_length=total_length,
                    projection_min=projection_min,
                    projection_max=projection_max,
                )
            )

    for band_key in sorted(components_by_band):
        lane_components = components_by_band[band_key]
        if len(lane_components) == 1:
            component = lane_components[0]
            chunks = split_lane_path(component.ordered_edge_ids, edges, target_length, min_tail_length)
            chunks = merge_short_sidewalk_chunks(chunks, edges, min_longitudinal_length)
            sidewalk_chunks.extend(sorted(chunk) for chunk in chunks)
            continue

        union_find = UnionFind(range(len(lane_components)))
        for left_index in range(len(lane_components)):
            left = lane_components[left_index]
            for right_index in range(left_index + 1, len(lane_components)):
                right = lane_components[right_index]
                if left.lane_key == right.lane_key:
                    continue

                overlap = min(left.projection_max, right.projection_max) - max(left.projection_min, right.projection_min)
                if overlap >= -5.0:
                    union_find.union(left_index, right_index)

        grouped_components: Dict[int, List[LaneComponent]] = defaultdict(list)
        for component_index, component in enumerate(lane_components):
            grouped_components[union_find.find(component_index)].append(component)

        grouped_components_list = sorted(
            grouped_components.values(),
            key=lambda group: min(component.projection_min for component in group),
        )

        for group in grouped_components_list:
            reference = max(group, key=lambda component: component.total_length)
            end_ratios = chunk_end_ratios(reference.ordered_edge_ids, edges, target_length, min_tail_length)
            grouped_chunks = [[] for _ in range(len(end_ratios))]

            for component in group:
                component_chunks = assign_edges_to_ratio_chunks(component.ordered_edge_ids, edges, end_ratios)
                for chunk_index, chunk in enumerate(component_chunks):
                    grouped_chunks[chunk_index].extend(chunk)

            grouped_chunks = merge_short_sidewalk_chunks(grouped_chunks, edges, min_longitudinal_length)
            for grouped_chunk in grouped_chunks:
                if grouped_chunk:
                    sidewalk_chunks.append(sorted(grouped_chunk))
    return sidewalk_chunks


def edge_type_counts(edge_ids: Iterable[int], edges: Dict[int, EdgeRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for edge_id in edge_ids:
        counts[edges[edge_id].edge_type.name] += 1
    return dict(sorted(counts.items()))


def lane_keys_for_cell(edge_ids: Iterable[int], edges: Dict[int, EdgeRecord]) -> List[List[int]]:
    keys = sorted({edges[edge_id].lane_key for edge_id in edge_ids if edges[edge_id].is_pure_sidewalk})
    return [list(key) for key in keys]


def longitudinal_length_for_sidewalk_cell(
    edge_ids: Iterable[int], edges: Dict[int, EdgeRecord]
) -> Optional[float]:
    lane_lengths: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for edge_id in edge_ids:
        edge = edges[edge_id]
        if edge.is_pure_sidewalk:
            lane_lengths[edge.lane_key] += edge.length
    if not lane_lengths:
        return None
    return max(lane_lengths.values())


def longitudinal_length_for_edges(edge_ids: Iterable[int], edges: Dict[int, EdgeRecord]) -> float:
    length = longitudinal_length_for_sidewalk_cell(edge_ids, edges)
    return 0.0 if length is None else float(length)


def merge_short_sidewalk_chunks(
    chunks: List[List[int]],
    edges: Dict[int, EdgeRecord],
    min_longitudinal_length: float,
) -> List[List[int]]:
    if min_longitudinal_length <= 0 or len(chunks) <= 1:
        return chunks

    merged_chunks = [list(chunk) for chunk in chunks if chunk]
    changed = True
    while changed and len(merged_chunks) > 1:
        changed = False
        for index, chunk in enumerate(list(merged_chunks)):
            current_length = longitudinal_length_for_edges(chunk, edges)
            if current_length >= min_longitudinal_length:
                continue

            previous_index = index - 1 if index > 0 else None
            next_index = index + 1 if index + 1 < len(merged_chunks) else None
            if previous_index is None and next_index is None:
                continue

            if previous_index is None:
                target_index = next_index
            elif next_index is None:
                target_index = previous_index
            else:
                previous_length = longitudinal_length_for_edges(merged_chunks[previous_index], edges)
                next_length = longitudinal_length_for_edges(merged_chunks[next_index], edges)
                target_index = previous_index if previous_length <= next_length else next_index

            if target_index is None:
                continue

            if target_index < index:
                merged_chunks[target_index].extend(chunk)
                del merged_chunks[index]
            else:
                merged_chunks[target_index] = chunk + merged_chunks[target_index]
                del merged_chunks[index]
            changed = True
            break

    return merged_chunks


def build_cell_record(
    cell_id: str,
    cell_kind: str,
    edge_ids: Iterable[int],
    edges: Dict[int, EdgeRecord],
    color: str,
    sidewalk_min_longitudinal_length: float,
) -> Dict[str, object]:
    edge_ids = sorted(edge_ids)
    total_length = sum(edges[edge_id].length for edge_id in edge_ids)
    centers = [edges[edge_id].center for edge_id in edge_ids]
    centroid_x = sum(point[0] for point in centers) / len(centers)
    centroid_y = sum(point[1] for point in centers) / len(centers)

    road_ids = sorted({edges[edge_id].road_id for edge_id in edge_ids})
    junction_road_ids = sorted({edges[edge_id].road_id for edge_id in edge_ids if edges[edge_id].in_junction})
    coordinates = [(edges[edge_id].x1, edges[edge_id].y1) for edge_id in edge_ids]
    coordinates.extend((edges[edge_id].x2, edges[edge_id].y2) for edge_id in edge_ids)
    min_x = min(point[0] for point in coordinates)
    max_x = max(point[0] for point in coordinates)
    min_y = min(point[1] for point in coordinates)
    max_y = max(point[1] for point in coordinates)
    longitudinal_length = (
        longitudinal_length_for_sidewalk_cell(edge_ids, edges) if cell_kind == "sidewalk" else None
    )
    usable = True
    usable_reason = "usable"
    if cell_kind == "sidewalk":
        usable = bool(
            longitudinal_length is not None and longitudinal_length >= sidewalk_min_longitudinal_length - 1e-6
        )
        if not usable:
            usable_reason = "sidewalk_longitudinal_length_below_threshold"

    return {
        "id": cell_id,
        "kind": cell_kind,
        "color": color,
        "edge_count": len(edge_ids),
        "total_length_m": round(total_length, 3),
        "longitudinal_length_m": round(longitudinal_length, 3) if longitudinal_length is not None else None,
        "centroid": [round(centroid_x, 3), round(centroid_y, 3)],
        "bounds": {
            "min_x": round(min_x, 3),
            "max_x": round(max_x, 3),
            "min_y": round(min_y, 3),
            "max_y": round(max_y, 3),
        },
        "road_ids": road_ids,
        "junction_road_ids": junction_road_ids,
        "lane_keys": lane_keys_for_cell(edge_ids, edges),
        "attributes": {
            "usable": usable,
            "usable_reason": usable_reason,
            "has_junction_context": bool(junction_road_ids),
            "lane_count": len(lane_keys_for_cell(edge_ids, edges)),
            "sidewalk_min_longitudinal_length_m": sidewalk_min_longitudinal_length if cell_kind == "sidewalk" else None,
        },
        "edge_type_counts": edge_type_counts(edge_ids, edges),
        "edges": [
            {
                "edge_id": edge_id,
                "type": edges[edge_id].edge_type.name,
                "length_m": round(edges[edge_id].length, 3),
                "road_id": edges[edge_id].road_id,
                "section_id": edges[edge_id].section_id,
                "lane_id": edges[edge_id].lane_id,
                "entry_lane_type": lane_type_name(edges[edge_id].entry_lane_type),
                "exit_lane_type": lane_type_name(edges[edge_id].exit_lane_type),
                "in_junction": edges[edge_id].in_junction,
                "points": [
                    [round(edges[edge_id].x1, 3), round(edges[edge_id].y1, 3)],
                    [round(edges[edge_id].x2, 3), round(edges[edge_id].y2, 3)],
                ],
            }
            for edge_id in edge_ids
        ],
    }


def build_sidewalk_longitudinal_stats(cells: List[Dict[str, object]]) -> Dict[str, object]:
    lengths = [
        float(cell["longitudinal_length_m"])
        for cell in cells
        if cell["kind"] == "sidewalk" and cell.get("longitudinal_length_m") is not None
    ]
    if not lengths:
        return {
            "count": 0,
            "min_m": None,
            "median_m": None,
            "max_m": None,
            "histogram": [],
            "threshold_counts_ge": {},
        }

    bins = [10, 20, 30, 40, 50, 60, 80]
    histogram = []
    lower = 0.0
    for upper in bins:
        count = sum(lower <= value < upper for value in lengths)
        histogram.append({"range": f"[{int(lower)},{upper})", "count": count})
        lower = float(upper)
    histogram.append({"range": f"[{int(lower)},+inf)", "count": sum(value >= lower for value in lengths)})

    thresholds = [10, 20, 30, 40, 50, 60, 80]
    threshold_counts = {str(threshold): sum(value >= threshold for value in lengths) for threshold in thresholds}

    return {
        "count": len(lengths),
        "min_m": round(min(lengths), 3),
        "median_m": round(statistics.median(lengths), 3),
        "max_m": round(max(lengths), 3),
        "histogram": histogram,
        "threshold_counts_ge": threshold_counts,
    }


def build_candidate_scene_stats(cells: List[Dict[str, object]]) -> Dict[str, object]:
    total_count = len(cells)
    usable_count = sum(1 for cell in cells if cell["attributes"]["usable"])
    by_kind: Dict[str, int] = defaultdict(int)
    usable_by_kind: Dict[str, int] = defaultdict(int)
    unusable_by_kind: Dict[str, int] = defaultdict(int)
    unusable_reasons: Dict[str, int] = defaultdict(int)

    for cell in cells:
        kind = str(cell["kind"])
        usable = bool(cell["attributes"]["usable"])
        reason = str(cell["attributes"]["usable_reason"])
        by_kind[kind] += 1
        if usable:
            usable_by_kind[kind] += 1
        else:
            unusable_by_kind[kind] += 1
            unusable_reasons[reason] += 1

    return {
        "total_count": total_count,
        "usable_count": usable_count,
        "unusable_count": total_count - usable_count,
        "by_kind": dict(sorted(by_kind.items())),
        "usable_by_kind": dict(sorted(usable_by_kind.items())),
        "unusable_by_kind": dict(sorted(unusable_by_kind.items())),
        "unusable_reasons": dict(sorted(unusable_reasons.items())),
    }


def build_local_graph(
    edge_ids: Iterable[int],
    edges: Dict[int, EdgeRecord],
    allowed_edge_types: Optional[Set[EdgeType]] = None,
) -> nx.Graph:
    graph = nx.Graph()
    for edge_id in edge_ids:
        edge = edges[edge_id]
        if allowed_edge_types is not None and edge.edge_type not in allowed_edge_types:
            continue
        graph.add_edge(edge.n1, edge.n2, edge_id=edge_id, length=edge.length, edge_type=edge.edge_type)
    return graph


def build_edge_lookup(edges: Dict[int, EdgeRecord]) -> Dict[Tuple[int, int], int]:
    lookup: Dict[Tuple[int, int], int] = {}
    for edge_id, edge in edges.items():
        lookup[(min(edge.n1, edge.n2), max(edge.n1, edge.n2))] = edge_id
    return lookup


def node_distance_xy(
    node_locations: Dict[int, Tuple[float, float, float]],
    left_node_id: int,
    right_node_id: int,
) -> float:
    left = node_locations[left_node_id]
    right = node_locations[right_node_id]
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))


def boundary_nodes_for_edges(
    edge_ids: Iterable[int],
    edges: Dict[int, EdgeRecord],
    adjacency: Dict[int, List[int]],
) -> List[int]:
    edge_id_set = set(edge_ids)
    node_to_edges: Dict[int, List[int]] = defaultdict(list)
    for edge_id in edge_id_set:
        edge = edges[edge_id]
        node_to_edges[edge.n1].append(edge_id)
        node_to_edges[edge.n2].append(edge_id)

    nodes = []
    for node_id, local_edge_ids in node_to_edges.items():
        has_external_connection = any(neighbor_edge_id not in edge_id_set for neighbor_edge_id in adjacency[node_id])
        if len(local_edge_ids) == 1 or has_external_connection:
            nodes.append(node_id)
    return sorted(nodes)


def path_edge_ids_from_nodes(
    node_path: List[int],
    edge_lookup: Dict[Tuple[int, int], int],
) -> List[int]:
    path_edge_ids = []
    for left_node_id, right_node_id in zip(node_path, node_path[1:]):
        key = (min(left_node_id, right_node_id), max(left_node_id, right_node_id))
        path_edge_ids.append(edge_lookup[key])
    return path_edge_ids


def rounded_point(point: Tuple[float, float, float]) -> List[float]:
    return [round(point[0], 3), round(point[1], 3), round(point[2], 3)]


def yaw_from_points(start: Sequence[float], end: Sequence[float]) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    return round(math.degrees(math.atan2(dy, dx)), 3)


def snap_route_spawn_transform(
    carla_map: carla.Map,
    start_point: Sequence[float],
    next_point: Sequence[float],
) -> Dict[str, object]:
    location = carla.Location(float(start_point[0]), float(start_point[1]), float(start_point[2]))
    waypoint = None
    snapped_lane_type = None
    for lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Shoulder, carla.LaneType.Any):
        try:
            waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=lane_type)
        except RuntimeError:
            waypoint = None
        if waypoint is not None:
            snapped_lane_type = lane_type_name(waypoint.lane_type)
            break

    if waypoint is not None:
        snapped_location = waypoint.transform.location
        snapped_point = [
            round(float(snapped_location.x), 3),
            round(float(snapped_location.y), 3),
            round(float(snapped_location.z) + ROUTE_SPAWN_Z_OFFSET_M, 3),
        ]
        snap_source = "carla_map.get_waypoint"
    else:
        snapped_point = [
            round(float(start_point[0]), 3),
            round(float(start_point[1]), 3),
            round(float(start_point[2]) + ROUTE_SPAWN_Z_OFFSET_M, 3),
        ]
        snap_source = "fallback_raw_start"

    return {
        "location": snapped_point,
        "yaw": yaw_from_points(start_point, next_point),
        "lane_type": snapped_lane_type,
        "source": snap_source,
        "z_offset_m": ROUTE_SPAWN_Z_OFFSET_M,
    }


def point_distance_xy(left: Iterable[float], right: Iterable[float]) -> float:
    left_x, left_y = left[0], left[1]
    right_x, right_y = right[0], right[1]
    return float(np.hypot(left_x - right_x, left_y - right_y))


def edge_corridor_half_width(edge: EdgeRecord, config: StaticObstacleValidationConfig) -> float:
    return max(
        0.05,
        float(edge.width_hint) * 0.5 - config.edge_corridor_side_margin_m,
    )


def edge_corridor_polygon(edge: EdgeRecord, config: StaticObstacleValidationConfig) -> Polygon:
    half_width = edge_corridor_half_width(edge, config)
    line = LineString([(edge.x1, edge.y1), (edge.x2, edge.y2)])
    return line.buffer(half_width, resolution=8, cap_style=1, join_style=2)


def edge_endpoint_anchor_polygons(
    edge: EdgeRecord,
    config: StaticObstacleValidationConfig,
) -> Tuple[Polygon, Polygon]:
    anchor_radius = max(
        STATIC_OBSTACLE_ENDPOINT_ANCHOR_RADIUS_M,
        min(edge_corridor_half_width(edge, config) * 0.5, 0.3),
    )
    start_anchor = Point(edge.x1, edge.y1).buffer(anchor_radius, resolution=8)
    end_anchor = Point(edge.x2, edge.y2).buffer(anchor_radius, resolution=8)
    return start_anchor, end_anchor


def blocked_edge_record_for_edge(
    edge: EdgeRecord,
    obstacle_records: Sequence[StaticObstacleRecord],
    config: StaticObstacleValidationConfig,
) -> Optional[BlockedEdgeRecord]:
    corridor = edge_corridor_polygon(edge, config)
    corridor_bounds = tuple(float(value) for value in corridor.bounds)
    relevant_obstacles = relevant_obstacles_for_bounds(obstacle_records, corridor_bounds)
    if not relevant_obstacles:
        return None

    intersecting_obstacles = [
        record for record in relevant_obstacles if record.buffered_footprint.intersects(corridor)
    ]
    if not intersecting_obstacles:
        return None

    obstacle_union = unary_union([record.buffered_footprint for record in intersecting_obstacles]).intersection(corridor)
    if obstacle_union.is_empty:
        return None

    free_space = corridor.difference(obstacle_union)
    traversable_space = free_space.buffer(-0.5 * config.edge_min_passage_width_m, resolution=8)
    if traversable_space.is_empty:
        reason = "no_traversable_corridor_after_clearance"
    else:
        start_anchor, end_anchor = edge_endpoint_anchor_polygons(edge, config)
        has_connecting_component = any(
            component.intersects(start_anchor) and component.intersects(end_anchor)
            for component in geometry_polygon_parts(traversable_space)
        )
        if has_connecting_component:
            return None
        reason = "no_component_connects_edge_endcaps"

    blocking_object_ids = tuple(sorted(record.object_id for record in intersecting_obstacles))
    blocking_labels = tuple(sorted({record.label_name for record in intersecting_obstacles}))
    return BlockedEdgeRecord(
        edge_id=edge.edge_id,
        reason=reason,
        corridor_width_m=round(edge_corridor_half_width(edge, config) * 2.0, 3),
        blocking_object_ids=blocking_object_ids,
        blocking_labels=blocking_labels,
    )


def build_blocked_edge_records(
    edge_ids: Iterable[int],
    edges: Dict[int, EdgeRecord],
    obstacle_records: Sequence[StaticObstacleRecord],
    config: StaticObstacleValidationConfig,
) -> Dict[int, BlockedEdgeRecord]:
    blocked_edge_records: Dict[int, BlockedEdgeRecord] = {}
    for edge_id in sorted(set(edge_ids)):
        blocked_record = blocked_edge_record_for_edge(edges[edge_id], obstacle_records, config)
        if blocked_record is not None:
            blocked_edge_records[edge_id] = blocked_record
    return blocked_edge_records


def _heading_vector_from_points(
    points_xy: List[Tuple[float, float]],
    sample_distance: float,
) -> Optional[np.ndarray]:
    if len(points_xy) < 2:
        return None

    origin = np.array(points_xy[0], dtype=float)
    previous = origin
    remaining = sample_distance
    for point_xy in points_xy[1:]:
        point = np.array(point_xy, dtype=float)
        segment = point - previous
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-6:
            previous = point
            continue
        if remaining <= segment_length + 1e-6:
            target = previous + segment * max(remaining, 0.0) / segment_length
            direction = target - origin
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-6:
                return direction / direction_norm
        remaining -= segment_length
        previous = point

    fallback = np.array(points_xy[-1], dtype=float) - origin
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1e-6:
        return None
    return fallback / fallback_norm


def path_deflection_degrees(
    node_path: List[int],
    node_locations: Dict[int, Tuple[float, float, float]],
    sample_distance: float = 4.0,
) -> Optional[float]:
    if len(node_path) < 2:
        return None

    points_xy = [(node_locations[node_id][0], node_locations[node_id][1]) for node_id in node_path]
    start_heading = _heading_vector_from_points(points_xy, sample_distance)
    end_heading = _heading_vector_from_points(list(reversed(points_xy)), sample_distance)
    if start_heading is None or end_heading is None:
        return None

    end_heading = -end_heading
    cosine = float(np.clip(np.dot(start_heading, end_heading), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def path_endpoint_frame(
    node_path: List[int],
    node_locations: Dict[int, Tuple[float, float, float]],
    sample_distance: float = 4.0,
) -> Optional[Tuple[float, float]]:
    if len(node_path) < 2:
        return None

    points_xy = [(node_locations[node_id][0], node_locations[node_id][1]) for node_id in node_path]
    start_heading = _heading_vector_from_points(points_xy, sample_distance)
    if start_heading is None:
        return None

    start = np.array(points_xy[0], dtype=float)
    end = np.array(points_xy[-1], dtype=float)
    delta = end - start
    lateral_axis = np.array([-start_heading[1], start_heading[0]], dtype=float)
    longitudinal = float(np.dot(delta, start_heading))
    lateral = float(np.dot(delta, lateral_axis))
    return longitudinal, lateral


def route_type_for_path(
    cell_kind: str,
    path_edge_ids: List[int],
    edges: Dict[int, EdgeRecord],
    node_path: List[int],
    node_locations: Dict[int, Tuple[float, float, float]],
) -> str:
    edge_types = {edges[edge_id].edge_type for edge_id in path_edge_ids}
    if EdgeType.CROSSWALK in edge_types:
        return "crosswalk_cross"
    if EdgeType.JAYWALKING_JUNCTION in edge_types:
        return "unmarked_junction_cross"
    if cell_kind == "crossing":
        deflection = path_deflection_degrees(node_path, node_locations)
        if deflection is not None:
            if deflection <= JUNCTION_THROUGH_MAX_DEFLECTION_DEG + 1e-6:
                return "junction_through"
            if deflection >= JUNCTION_U_TURN_MIN_DEFLECTION_DEG - 1e-6:
                endpoint_frame = path_endpoint_frame(node_path, node_locations)
                if endpoint_frame is not None:
                    longitudinal, lateral = endpoint_frame
                    lateral_limit = max(
                        JUNCTION_U_TURN_MAX_LATERAL_SHIFT_M,
                        abs(longitudinal) * JUNCTION_U_TURN_MAX_LATERAL_TO_LONGITUDINAL_RATIO,
                    )
                    if longitudinal < 0.0 and abs(lateral) <= lateral_limit + 1e-6:
                        return "junction_u_turn"
        return "junction_turn"
    return "sidewalk_through"


def crossing_bools_for_path(path_edge_ids: List[int], edges: Dict[int, EdgeRecord]) -> List[bool]:
    crossing_edge_types = {EdgeType.CROSSWALK, EdgeType.JAYWALKING, EdgeType.JAYWALKING_JUNCTION}
    return [False] + [edges[edge_id].edge_type in crossing_edge_types for edge_id in path_edge_ids]


def polyline_tangent_vectors(points_xyz: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.array(points_xyz, dtype=float)
    tangent_vectors = np.zeros((len(points), 2), dtype=float)
    for index in range(len(points)):
        if index == 0:
            delta = points[1, :2] - points[0, :2]
        elif index == len(points) - 1:
            delta = points[-1, :2] - points[-2, :2]
        else:
            delta = points[index + 1, :2] - points[index - 1, :2]
        delta_norm = float(np.linalg.norm(delta))
        tangent_vectors[index] = np.array([1.0, 0.0], dtype=float) if delta_norm <= 1e-6 else delta / delta_norm
    return tangent_vectors


def shift_polyline_laterally(points_xyz: Sequence[Sequence[float]], signed_offset_m: float) -> np.ndarray:
    points = np.array(points_xyz, dtype=float, copy=True)
    if len(points) < 2 or abs(signed_offset_m) <= 1e-6:
        return points
    tangent_vectors = polyline_tangent_vectors(points)
    lateral_vectors = np.column_stack((-tangent_vectors[:, 1], tangent_vectors[:, 0]))
    points[:, :2] += lateral_vectors * float(signed_offset_m)
    return points


def safe_get_waypoint(
    carla_map: carla.Map,
    location: carla.Location,
    *,
    project_to_road: bool,
    lane_type: carla.LaneType,
):
    try:
        return carla_map.get_waypoint(location, project_to_road=project_to_road, lane_type=lane_type)
    except RuntimeError:
        return None


def first_driving_lane_from_sidewalk(sidewalk_waypoint, use_left: bool):
    lane = sidewalk_waypoint
    visited = set()
    while lane is not None:
        lane = lane.get_left_lane() if use_left else lane.get_right_lane()
        if lane is None:
            return None
        lane_key = (int(lane.road_id), int(lane.section_id), int(lane.lane_id))
        if lane_key in visited:
            return None
        visited.add(lane_key)
        if lane.lane_type == carla.LaneType.Driving:
            return lane
    return None


def sidewalk_inward_offset_sign_and_limit(
    path_waypoints: Sequence[Sequence[float]],
    carla_map: carla.Map,
) -> Optional[Tuple[float, float]]:
    points = np.array(path_waypoints, dtype=float)
    tangents = polyline_tangent_vectors(points)
    inward_sign_votes: List[float] = []
    max_offsets: List[float] = []

    for point, tangent in zip(points, tangents):
        location = carla.Location(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]) if len(point) > 2 else 0.0,
        )
        sidewalk_waypoint = safe_get_waypoint(
            carla_map,
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Sidewalk,
        )
        if sidewalk_waypoint is None or sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
            continue

        driving_candidates = []
        for use_left in (True, False):
            driving_waypoint = first_driving_lane_from_sidewalk(sidewalk_waypoint, use_left=use_left)
            if driving_waypoint is None:
                continue
            driving_candidates.append(
                np.array(
                    [
                        float(driving_waypoint.transform.location.x),
                        float(driving_waypoint.transform.location.y),
                    ],
                    dtype=float,
                )
            )
        if not driving_candidates:
            continue

        point_xy = np.array(point[:2], dtype=float)
        driving_xy = min(driving_candidates, key=lambda candidate: float(np.linalg.norm(candidate - point_xy)))
        left_normal = np.array([-tangent[1], tangent[0]], dtype=float)
        side = float(np.dot(driving_xy - point_xy, left_normal))
        if abs(side) <= 1e-6:
            continue

        inward_sign_votes.append(-1.0 if side > 0.0 else 1.0)
        max_offsets.append(max(0.0, float(sidewalk_waypoint.lane_width) * 1 - SIDEWALK_INWARD_OFFSET_BORDER_MARGIN_M))

    if not inward_sign_votes or not max_offsets:
        return None

    max_offset_m = min(DEFAULT_SIDEWALK_INWARD_OFFSET_MAX_M, min(max_offsets))
    if max_offset_m < SIDEWALK_INWARD_OFFSET_MIN_M:
        return None

    inward_sign = 1.0 if sum(inward_sign_votes) > 0.0 else -1.0
    return inward_sign, float(max_offset_m)


def shifted_sidewalk_route_is_valid(
    shifted_points: np.ndarray,
    obstacle_records: Sequence[StaticObstacleRecord],
    carla_map: carla.Map,
) -> bool:
    for point in shifted_points:
        location = carla.Location(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]) if len(point) > 2 else 0.0,
        )
        sidewalk_waypoint = safe_get_waypoint(
            carla_map,
            location,
            project_to_road=False,
            lane_type=carla.LaneType.Sidewalk,
        )
        if sidewalk_waypoint is None or sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
            return False

    route_buffer = LineString([(float(point[0]), float(point[1])) for point in shifted_points]).buffer(
        SIDEWALK_INWARD_OFFSET_PATH_CLEARANCE_M,
        resolution=8,
    ).buffer(0.0)
    if route_buffer.is_empty:
        return False

    for obstacle_record in relevant_obstacles_for_bounds(obstacle_records, route_buffer.bounds):
        if obstacle_record.buffered_footprint.intersects(route_buffer):
            return False
    return True


def apply_sidewalk_inward_offset_to_route(
    route: Dict[str, object],
    obstacle_records: Sequence[StaticObstacleRecord],
    carla_map: carla.Map,
) -> Dict[str, object]:
    if str(route.get('route_type')) != 'sidewalk_through':
        return route

    path_waypoints = route.get('path_waypoints')
    if not isinstance(path_waypoints, list) or len(path_waypoints) < 2:
        return route

    inward_config = sidewalk_inward_offset_sign_and_limit(path_waypoints, carla_map)
    if inward_config is None:
        return route
    inward_sign, max_offset_m = inward_config

    route_id = str(route.get('route_id', ''))
    offset_seed = int(hashlib.sha1(route_id.encode('utf-8')).hexdigest()[:16], 16)
    sampled_offset_m = random.Random(offset_seed).uniform(0.0, max_offset_m)
    if sampled_offset_m < SIDEWALK_INWARD_OFFSET_MIN_M:
        return route

    for scale in SIDEWALK_INWARD_OFFSET_ATTEMPT_SCALES:
        applied_offset_m = sampled_offset_m * float(scale)
        shifted_points = shift_polyline_laterally(path_waypoints, inward_sign * applied_offset_m)
        if not shifted_sidewalk_route_is_valid(shifted_points, obstacle_records, carla_map):
            continue

        shifted_waypoints = [rounded_point((float(point[0]), float(point[1]), float(point[2]))) for point in shifted_points]
        next_point = shifted_waypoints[1] if len(shifted_waypoints) > 1 else shifted_waypoints[0]
        spawn_transform = snap_route_spawn_transform(carla_map, shifted_waypoints[0], next_point)

        adjusted_route = dict(route)
        adjusted_route['path_waypoints'] = shifted_waypoints
        adjusted_route['spawn_location'] = spawn_transform['location']
        adjusted_route['spawn_transform'] = spawn_transform
        adjusted_route['destination'] = shifted_waypoints[-1]
        adjusted_route['config_waypoints'] = shifted_waypoints[1:-1]
        adjusted_attributes = dict(route.get('attributes', {}))
        adjusted_attributes['sidewalk_inward_offset_m'] = round(applied_offset_m, 3)
        adjusted_attributes['sidewalk_inward_offset_side'] = 'left' if inward_sign > 0.0 else 'right'
        adjusted_route['attributes'] = adjusted_attributes
        return adjusted_route

    return route


def apply_sidewalk_inward_offsets(
    routes: List[Dict[str, object]],
    obstacle_records: Sequence[StaticObstacleRecord],
    carla_map: carla.Map,
) -> List[Dict[str, object]]:
    return [apply_sidewalk_inward_offset_to_route(route, obstacle_records, carla_map) for route in routes]


def build_route_record(
    route_id: str,
    route_type: str,
    node_path: List[int],
    path_edge_ids: List[int],
    edges: Dict[int, EdgeRecord],
    node_locations: Dict[int, Tuple[float, float, float]],
    carla_map: carla.Map,
    start_portal_id: str,
    end_portal_id: str,
    component_index: Optional[int],
    path_source: str = "shortest_path",
) -> Dict[str, object]:
    path_waypoints = [rounded_point(node_locations[node_id]) for node_id in node_path]
    next_point = path_waypoints[1] if len(path_waypoints) > 1 else path_waypoints[0]
    spawn_transform = snap_route_spawn_transform(carla_map, path_waypoints[0], next_point)
    path_crossing_bools = crossing_bools_for_path(path_edge_ids, edges)
    route_length = sum(edges[edge_id].length for edge_id in path_edge_ids)
    edge_type_count = edge_type_counts(path_edge_ids, edges)
    supported_behaviors = ROUTE_TYPE_SUPPORTED_BEHAVIORS[route_type]
    return {
        "route_id": route_id,
        "route_type": route_type,
        "start_portal": start_portal_id,
        "end_portal": end_portal_id,
        "component_index": component_index,
        "path_source": path_source,
        "node_ids": node_path,
        "edge_ids": path_edge_ids,
        "route_length_m": round(route_length, 3),
        "path_waypoints": path_waypoints,
        "path_crossing_road_bools": path_crossing_bools,
        "spawn_location": spawn_transform["location"],
        "spawn_transform": spawn_transform,
        "destination": path_waypoints[-1],
        "config_waypoints": path_waypoints[1:-1],
        "config_crossing_road_bools": path_crossing_bools[1:],
        "edge_type_counts": edge_type_count,
        "attributes": {
            "uses_crosswalk": bool(edge_type_count.get("CROSSWALK", 0)),
            "uses_unmarked_junction_cross": bool(edge_type_count.get("JAYWALKING_JUNCTION", 0)),
            "crossing_required": bool(ROUTE_TYPE_DEFINITIONS[route_type]["crossing_required"]),
            "supported_behaviors": supported_behaviors,
        },
    }


def cell_edge_ids(cell: Dict[str, object]) -> List[int]:
    return [edge_info["edge_id"] for edge_info in cell["edges"]]


def build_route_support_for_cell(
    cell: Dict[str, object],
    cells_by_id: Dict[str, Dict[str, object]],
    edge_to_cell: Dict[int, str],
    adjacency: Dict[int, List[int]],
    edges: Dict[int, EdgeRecord],
) -> Tuple[List[str], List[int]]:
    core_edge_ids = cell_edge_ids(cell)
    return [str(cell["id"])], core_edge_ids


def portal_centroid(
    node_ids: Iterable[int],
    node_locations: Dict[int, Tuple[float, float, float]],
) -> List[float]:
    points = [node_locations[node_id] for node_id in node_ids]
    centroid = (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )
    return rounded_point(centroid)


def build_leaf_portals(
    cell_id: str,
    graph: nx.Graph,
    node_locations: Dict[int, Tuple[float, float, float]],
) -> List[Dict[str, object]]:
    leaf_node_ids = sorted(node_id for node_id in graph.nodes if graph.degree[node_id] == 1)
    if not leaf_node_ids:
        return []

    portals = []
    for portal_index, node_id in enumerate(leaf_node_ids, start=1):
        portals.append(
            {
                "portal_id": f"{cell_id}_portal_{portal_index:02d}",
                "node_ids": [node_id],
                "centroid": rounded_point(node_locations[node_id]),
                "kind": "leaf_node",
            }
        )
    return portals


def best_sidewalk_path_length_between_portals(
    sidewalk_graph: nx.Graph,
    left_portal: Dict[str, object],
    right_portal: Dict[str, object],
    max_length: float,
) -> Optional[float]:
    best_length: Optional[float] = None
    for start_node_id in left_portal["node_ids"]:
        for end_node_id in right_portal["node_ids"]:
            if start_node_id == end_node_id:
                return 0.0
            if start_node_id not in sidewalk_graph or end_node_id not in sidewalk_graph:
                continue
            try:
                path_length = float(
                    nx.shortest_path_length(
                        sidewalk_graph,
                        source=start_node_id,
                        target=end_node_id,
                        weight="length",
                    )
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if path_length > max_length + 1e-6:
                continue
            if best_length is None or path_length < best_length - 1e-6:
                best_length = path_length
    return best_length


def merge_crossing_portals(
    cell_id: str,
    portals: List[Dict[str, object]],
    graph: nx.Graph,
    node_locations: Dict[int, Tuple[float, float, float]],
    portal_merge_radius: float,
    portal_sidewalk_merge_length: float,
) -> List[Dict[str, object]]:
    if (
        len(portals) < 2
        or portal_merge_radius <= 1e-6
        or portal_sidewalk_merge_length <= 1e-6
    ):
        return portals

    sidewalk_graph = nx.Graph()
    for left_node_id, right_node_id, data in graph.edges(data=True):
        if data.get("edge_type") != EdgeType.SIDEWALK:
            continue
        sidewalk_graph.add_edge(left_node_id, right_node_id, length=float(data["length"]))

    union_find = UnionFind(range(len(portals)))
    for left_index, left_portal in enumerate(portals):
        for right_index, right_portal in enumerate(portals[left_index + 1 :], start=left_index + 1):
            portal_distance = point_distance_xy(left_portal["centroid"], right_portal["centroid"])
            if portal_distance > portal_merge_radius + 1e-6:
                continue
            sidewalk_distance = best_sidewalk_path_length_between_portals(
                sidewalk_graph=sidewalk_graph,
                left_portal=left_portal,
                right_portal=right_portal,
                max_length=portal_sidewalk_merge_length,
            )
            if sidewalk_distance is None:
                continue
            union_find.union(left_index, right_index)

    grouped_portals: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for portal_index, portal in enumerate(portals):
        grouped_portals[union_find.find(portal_index)].append(portal)

    merged_portals: List[Dict[str, object]] = []
    sorted_groups = sorted(
        grouped_portals.values(),
        key=lambda group: (
            min(portal["centroid"][0] for portal in group),
            min(portal["centroid"][1] for portal in group),
            min(str(portal["portal_id"]) for portal in group),
        ),
    )
    for portal_index, group in enumerate(sorted_groups, start=1):
        group_node_ids = sorted({node_id for portal in group for node_id in portal["node_ids"]})
        merged_portals.append(
            {
                "portal_id": f"{cell_id}_portal_{portal_index:02d}",
                "node_ids": group_node_ids,
                "centroid": portal_centroid(group_node_ids, node_locations),
                "kind": "merged_leaf_nodes" if len(group) > 1 else str(group[0]["kind"]),
                "member_portal_ids": [str(portal["portal_id"]) for portal in group],
            }
        )
    return merged_portals


def best_path_between_portals(
    graph: nx.Graph,
    left_portal: Dict[str, object],
    right_portal: Dict[str, object],
    edge_lookup: Dict[Tuple[int, int], int],
) -> Optional[Tuple[List[int], List[int], float]]:
    best_path: Optional[List[int]] = None
    best_edge_ids: Optional[List[int]] = None
    best_length: Optional[float] = None

    for start_node_id in left_portal["node_ids"]:
        for end_node_id in right_portal["node_ids"]:
            if start_node_id == end_node_id:
                continue
            if start_node_id not in graph or end_node_id not in graph:
                continue
            try:
                node_path = nx.shortest_path(graph, source=start_node_id, target=end_node_id, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            path_edge_ids = path_edge_ids_from_nodes(node_path, edge_lookup)
            path_length = float(sum(graph.edges[left_node_id, right_node_id]["length"] for left_node_id, right_node_id in zip(node_path, node_path[1:])))
            if best_length is None or path_length < best_length - 1e-6:
                best_path = node_path
                best_edge_ids = path_edge_ids
                best_length = path_length

    if best_path is None or best_edge_ids is None or best_length is None:
        return None
    return best_path, best_edge_ids, best_length


def serialize_blocked_edge_records(blocked_edge_records: Dict[int, BlockedEdgeRecord]) -> List[Dict[str, object]]:
    serialized = []
    for edge_id in sorted(blocked_edge_records):
        blocked_record = blocked_edge_records[edge_id]
        serialized.append(
            {
                "edge_id": blocked_record.edge_id,
                "reason": blocked_record.reason,
                "corridor_width_m": blocked_record.corridor_width_m,
                "blocking_object_ids": list(blocked_record.blocking_object_ids),
                "blocking_labels": list(blocked_record.blocking_labels),
            }
        )
    return serialized


def rejected_route_record(
    route: Dict[str, object],
    reason: str,
    blocked_edge_ids: Sequence[int],
) -> Dict[str, object]:
    return {
        "route_id": str(route["route_id"]),
        "route_type": str(route["route_type"]),
        "reason": reason,
        "start_portal": str(route["start_portal"]),
        "end_portal": str(route["end_portal"]),
        "blocked_edge_ids": list(blocked_edge_ids),
        "path_waypoints": route["path_waypoints"],
    }


def route_with_validation_metadata(
    route: Dict[str, object],
    raw_route: Dict[str, object],
    status: str,
    blocked_edge_ids: Sequence[int],
) -> Dict[str, object]:
    validated_route = dict(route)
    validated_route["validation"] = {
        "status": status,
        "raw_route_type": str(raw_route["route_type"]),
        "raw_route_length_m": float(raw_route["route_length_m"]),
        "blocked_edge_ids": list(blocked_edge_ids),
    }
    return validated_route


def validate_routes_against_static_obstacles(
    cell_kind: str,
    routes: List[Dict[str, object]],
    portals: List[Dict[str, object]],
    support_edge_ids: Sequence[int],
    edges: Dict[int, EdgeRecord],
    edge_lookup: Dict[Tuple[int, int], int],
    node_locations: Dict[int, Tuple[float, float, float]],
    blocked_edge_records: Dict[int, BlockedEdgeRecord],
    obstacle_records: Sequence[StaticObstacleRecord],
    carla_map: carla.Map,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    blocked_edge_id_set = set(blocked_edge_records)
    portal_by_id = {str(portal["portal_id"]): portal for portal in portals}
    obstacle_aware_edge_ids = [edge_id for edge_id in support_edge_ids if edge_id not in blocked_edge_id_set]
    obstacle_aware_graph = build_local_graph(obstacle_aware_edge_ids, edges)

    retained_routes: List[Dict[str, object]] = []
    rejected_routes: List[Dict[str, object]] = []
    rerouted_route_count = 0
    repaired_route_count = 0
    route_evaluations: List[Dict[str, object]] = []

    for route in routes:
        start_portal = portal_by_id.get(str(route["start_portal"]))
        end_portal = portal_by_id.get(str(route["end_portal"]))
        blocked_edge_ids = sorted(set(route["edge_ids"]) & blocked_edge_id_set)

        if start_portal is None or end_portal is None:
            route_evaluations.append(
                {
                    "raw_route": route,
                    "blocked_edge_ids": blocked_edge_ids,
                    "rejection_reason": "portal_definition_missing",
                    "validated_route": None,
                }
            )
            continue

        inward_shifted_route = apply_sidewalk_inward_offset_to_route(route, obstacle_records, carla_map)
        if inward_shifted_route is not route:
            route_evaluations.append(
                {
                    "raw_route": route,
                    "blocked_edge_ids": blocked_edge_ids,
                    "rejection_reason": None,
                    "validated_route": route_with_validation_metadata(
                        inward_shifted_route,
                        route,
                        "retained_as_is",
                        blocked_edge_ids,
                    ),
                }
            )
            continue

        best = best_path_between_portals(obstacle_aware_graph, start_portal, end_portal, edge_lookup)
        if best is None:
            route_evaluations.append(
                {
                    "raw_route": route,
                    "blocked_edge_ids": blocked_edge_ids,
                    "rejection_reason": "semantic_regions_disconnected_after_static_obstacles",
                    "validated_route": None,
                }
            )
            continue

        node_path, path_edge_ids, _ = best
        path_signature = canonical_path_signature(path_edge_ids)
        raw_signature = canonical_path_signature(route["edge_ids"])
        validation_status = "retained_as_is" if path_signature == raw_signature else "rerouted_around_static_obstacles"
        if validation_status == "rerouted_around_static_obstacles":
            rerouted_route_count += 1

        validated_route = build_route_record(
            route_id=str(route["route_id"]),
            route_type=route_type_for_path(cell_kind, path_edge_ids, edges, node_path, node_locations),
            node_path=node_path,
            path_edge_ids=path_edge_ids,
            edges=edges,
            node_locations=node_locations,
            carla_map=carla_map,
            start_portal_id=str(route["start_portal"]),
            end_portal_id=str(route["end_portal"]),
            component_index=route.get("component_index"),
            path_source="static_obstacle_aware_shortest_path"
            if validation_status != "retained_as_is"
            else str(route.get("path_source", "shortest_path")),
        )
        validated_route = apply_sidewalk_inward_offset_to_route(validated_route, obstacle_records, carla_map)
        route_evaluations.append(
            {
                "raw_route": route,
                "blocked_edge_ids": blocked_edge_ids,
                "rejection_reason": None,
                "validated_route": route_with_validation_metadata(
                    validated_route,
                    route,
                    validation_status,
                    blocked_edge_ids,
                ),
            }
        )

    feasible_validated_routes = [
        evaluation["validated_route"]
        for evaluation in route_evaluations
        if evaluation["validated_route"] is not None
    ]

    for evaluation in route_evaluations:
        raw_route = evaluation["raw_route"]
        blocked_edge_ids = evaluation["blocked_edge_ids"]
        validated_route = evaluation["validated_route"]
        rejection_reason = evaluation["rejection_reason"]

        if validated_route is not None:
            retained_routes.append(validated_route)
            continue

        if rejection_reason == "semantic_regions_disconnected_after_static_obstacles":
            repair_candidates: List[Tuple[Tuple[float, ...], Dict[str, object]]] = []
            for candidate_route in feasible_validated_routes:
                if str(candidate_route["route_id"]) == str(raw_route["route_id"]):
                    continue
                sort_key = route_sibling_repair_sort_key(raw_route, candidate_route, node_locations)
                if sort_key is None:
                    continue
                repair_candidates.append((sort_key, candidate_route))
            if repair_candidates:
                repair_candidates.sort(key=lambda item: item[0])
                retained_routes.append(
                    build_repaired_route_from_sibling(
                        raw_route=raw_route,
                        sibling_route=repair_candidates[0][1],
                        blocked_edge_ids=blocked_edge_ids,
                    )
                )
                repaired_route_count += 1
                continue

        rejected_routes.append(rejected_route_record(raw_route, str(rejection_reason), blocked_edge_ids))

    return retained_routes, {
        "raw_route_count": len(routes),
        "retained_route_count": len(retained_routes),
        "rejected_route_count": len(rejected_routes),
        "rerouted_route_count": rerouted_route_count,
        "repaired_route_count": repaired_route_count,
        "blocked_edge_count": len(blocked_edge_id_set),
        "blocked_edge_ids": sorted(blocked_edge_id_set),
        "blocked_edges": serialize_blocked_edge_records(blocked_edge_records),
        "rejected_routes": rejected_routes,
    }


def best_path_between_portal_and_node(
    graph: nx.Graph,
    portal: Dict[str, object],
    target_node_id: int,
    edge_lookup: Dict[Tuple[int, int], int],
) -> Optional[Tuple[List[int], List[int], float]]:
    best_path: Optional[List[int]] = None
    best_edge_ids: Optional[List[int]] = None
    best_length: Optional[float] = None
    for start_node_id in portal["node_ids"]:
        if start_node_id not in graph or target_node_id not in graph:
            continue
        if start_node_id == target_node_id:
            node_path = [target_node_id]
            path_edge_ids: List[int] = []
            path_length = 0.0
        else:
            try:
                node_path = nx.shortest_path(graph, source=start_node_id, target=target_node_id, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            path_edge_ids = path_edge_ids_from_nodes(node_path, edge_lookup)
            path_length = float(
                sum(graph.edges[left_node_id, right_node_id]["length"] for left_node_id, right_node_id in zip(node_path, node_path[1:]))
            )
        if best_length is None or path_length < best_length - 1e-6:
            best_path = node_path
            best_edge_ids = path_edge_ids
            best_length = path_length

    if best_path is None or best_edge_ids is None or best_length is None:
        return None
    return best_path, best_edge_ids, best_length


def portal_path_candidates_to_node(
    graph: nx.Graph,
    portals: List[Dict[str, object]],
    target_node_id: int,
    edge_lookup: Dict[Tuple[int, int], int],
    limit: int = 6,
) -> List[Tuple[Dict[str, object], List[int], float]]:
    candidates: List[Tuple[Dict[str, object], List[int], float]] = []
    for portal in portals:
        best = best_path_between_portal_and_node(graph, portal, target_node_id, edge_lookup)
        if best is None:
            continue
        node_path, _, path_length = best
        candidates.append((portal, node_path, path_length))
    candidates.sort(key=lambda item: (item[2], str(item[0]["portal_id"])))
    return candidates[:limit]


def concatenate_node_paths(*paths: List[int]) -> List[int]:
    combined: List[int] = []
    for path in paths:
        if not path:
            continue
        if not combined:
            combined.extend(path)
            continue
        if combined[-1] == path[0]:
            combined.extend(path[1:])
        else:
            combined.extend(path)
    return combined


def component_diameter_path(
    graph: nx.Graph,
    candidate_node_ids: List[int],
    edge_lookup: Dict[Tuple[int, int], int],
) -> Optional[Tuple[List[int], List[int], float, int, int]]:
    best_path: Optional[List[int]] = None
    best_edge_ids: Optional[List[int]] = None
    best_length = -1.0
    best_pair: Optional[Tuple[int, int]] = None

    for left_index, left_node_id in enumerate(candidate_node_ids):
        for right_node_id in candidate_node_ids[left_index + 1:]:
            try:
                node_path = nx.shortest_path(graph, source=left_node_id, target=right_node_id, weight="length")
                path_length = float(
                    sum(graph.edges[u, v]["length"] for u, v in zip(node_path, node_path[1:]))
                )
            except nx.NetworkXNoPath:
                continue
            if path_length > best_length + 1e-6:
                best_path = node_path
                best_edge_ids = path_edge_ids_from_nodes(node_path, edge_lookup)
                best_length = path_length
                best_pair = (left_node_id, right_node_id)

    if best_path is None or best_edge_ids is None or best_pair is None:
        return None
    return best_path, best_edge_ids, best_length, best_pair[0], best_pair[1]


def canonical_path_signature(path_edge_ids: List[int]) -> Tuple[int, ...]:
    forward = tuple(path_edge_ids)
    backward = tuple(reversed(path_edge_ids))
    return forward if forward <= backward else backward


def heading_difference_degrees(
    left_heading: Optional[np.ndarray],
    right_heading: Optional[np.ndarray],
) -> Optional[float]:
    if left_heading is None or right_heading is None:
        return None
    cosine = float(np.clip(np.dot(left_heading, right_heading), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def route_endpoint_heading_vectors(
    route: Dict[str, object],
    node_locations: Dict[int, Tuple[float, float, float]],
    sample_distance: float = 4.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    node_path = list(route["node_ids"])
    if len(node_path) < 2:
        return None, None
    points_xy = [(node_locations[node_id][0], node_locations[node_id][1]) for node_id in node_path]
    start_heading = _heading_vector_from_points(points_xy, sample_distance)
    end_heading = _heading_vector_from_points(list(reversed(points_xy)), sample_distance)
    if end_heading is not None:
        end_heading = -end_heading
    return start_heading, end_heading


def route_sibling_repair_sort_key(
    raw_route: Dict[str, object],
    candidate_route: Dict[str, object],
    node_locations: Dict[int, Tuple[float, float, float]],
) -> Optional[Tuple[float, ...]]:
    if str(raw_route["route_type"]) != str(candidate_route["route_type"]):
        return None

    raw_start = raw_route["path_waypoints"][0]
    raw_end = raw_route["path_waypoints"][-1]
    candidate_start = candidate_route["path_waypoints"][0]
    candidate_end = candidate_route["path_waypoints"][-1]
    start_distance = point_distance_xy(raw_start, candidate_start)
    end_distance = point_distance_xy(raw_end, candidate_end)
    if start_distance > STATIC_OBSTACLE_SIBLING_REPAIR_MAX_START_DISTANCE_M + 1e-6:
        return None
    if end_distance > STATIC_OBSTACLE_SIBLING_REPAIR_MAX_END_DISTANCE_M + 1e-6:
        return None

    raw_start_heading, raw_end_heading = route_endpoint_heading_vectors(raw_route, node_locations)
    candidate_start_heading, candidate_end_heading = route_endpoint_heading_vectors(candidate_route, node_locations)
    start_heading_diff = heading_difference_degrees(raw_start_heading, candidate_start_heading)
    end_heading_diff = heading_difference_degrees(raw_end_heading, candidate_end_heading)
    if start_heading_diff is None or end_heading_diff is None:
        return None
    if start_heading_diff > STATIC_OBSTACLE_SIBLING_REPAIR_MAX_START_HEADING_DIFF_DEG + 1e-6:
        return None
    if end_heading_diff > STATIC_OBSTACLE_SIBLING_REPAIR_MAX_END_HEADING_DIFF_DEG + 1e-6:
        return None

    raw_route_length = float(raw_route["route_length_m"])
    candidate_route_length = float(candidate_route["route_length_m"])
    if candidate_route_length > raw_route_length * STATIC_OBSTACLE_SIBLING_REPAIR_MAX_ROUTE_LENGTH_RATIO + 1e-6:
        return None

    return (
        start_distance,
        end_distance,
        start_heading_diff,
        end_heading_diff,
        abs(candidate_route_length - raw_route_length),
        candidate_route_length,
        str(candidate_route["route_id"]),
    )


def build_repaired_route_from_sibling(
    raw_route: Dict[str, object],
    sibling_route: Dict[str, object],
    blocked_edge_ids: Sequence[int],
) -> Dict[str, object]:
    repaired_route = dict(sibling_route)
    repaired_route["route_id"] = str(raw_route["route_id"])
    repaired_route["path_source"] = "static_obstacle_repaired_from_sibling_route"
    repaired_route["validation"] = {
        "status": "repaired_from_sibling_route",
        "raw_route_type": str(raw_route["route_type"]),
        "raw_route_length_m": float(raw_route["route_length_m"]),
        "blocked_edge_ids": list(blocked_edge_ids),
        "repaired_from_route_id": str(sibling_route["route_id"]),
        "repaired_from_start_portal": str(sibling_route["start_portal"]),
        "repaired_from_end_portal": str(sibling_route["end_portal"]),
        "raw_start_portal": str(raw_route["start_portal"]),
        "raw_end_portal": str(raw_route["end_portal"]),
    }
    return repaired_route


def build_component_seed_route(
    cell_id: str,
    component_edge_ids: Set[int],
    graph: nx.Graph,
    portals: List[Dict[str, object]],
    adjacency: Dict[int, List[int]],
    edge_lookup: Dict[Tuple[int, int], int],
    edges: Dict[int, EdgeRecord],
    node_locations: Dict[int, Tuple[float, float, float]],
    carla_map: carla.Map,
    seen_signatures: Set[Tuple[int, ...]],
    route_index: int,
) -> Optional[Dict[str, object]]:
    component_graph = build_local_graph(component_edge_ids, edges)
    candidate_node_ids = boundary_nodes_for_edges(component_edge_ids, edges, adjacency)
    if len(candidate_node_ids) < 2:
        candidate_node_ids = sorted(node_id for node_id in component_graph.nodes if component_graph.degree[node_id] == 1)
    if len(candidate_node_ids) < 2:
        candidate_node_ids = sorted(component_graph.nodes)
    if len(candidate_node_ids) < 2:
        return None

    diameter = component_diameter_path(component_graph, candidate_node_ids, edge_lookup)
    if diameter is None:
        return None
    component_node_path, _, _, start_node_id, end_node_id = diameter
    start_candidates = portal_path_candidates_to_node(graph, portals, start_node_id, edge_lookup)
    end_candidates = portal_path_candidates_to_node(graph, portals, end_node_id, edge_lookup)
    if not start_candidates or not end_candidates:
        return None

    best_key: Optional[Tuple[float, ...]] = None
    best_route: Optional[Dict[str, object]] = None
    component_edge_id_set = set(component_edge_ids)
    for start_portal, start_path, start_length in start_candidates:
        for end_portal, end_path, end_length in end_candidates:
            if str(start_portal["portal_id"]) == str(end_portal["portal_id"]):
                continue
            node_path = concatenate_node_paths(start_path, component_node_path, list(reversed(end_path)))
            if len(node_path) < 2:
                continue
            path_edge_ids = path_edge_ids_from_nodes(node_path, edge_lookup)
            signature = canonical_path_signature(path_edge_ids)
            if signature in seen_signatures:
                continue
            covered_component_weight = sum(
                route_coverage_weight(edges[edge_id])
                for edge_id in set(path_edge_ids)
                if edge_id in component_edge_id_set
            )
            total_length = start_length + end_length + sum(
                edges[edge_id].length for edge_id in path_edge_ids if edge_id in component_edge_id_set
            )
            route_type = route_type_for_path("crossing", path_edge_ids, edges, node_path, node_locations)
            route_record = build_route_record(
                route_id=f"{cell_id}/coverage_seed_{route_index:02d}",
                route_type=route_type,
                node_path=node_path,
                path_edge_ids=path_edge_ids,
                edges=edges,
                node_locations=node_locations,
                carla_map=carla_map,
                start_portal_id=str(start_portal["portal_id"]),
                end_portal_id=str(end_portal["portal_id"]),
                component_index=None,
                path_source="coverage_component_seed",
            )
            route_record["selection_metrics"] = route_selection_metrics(route_record, edges)
            quality_key = route_candidate_sort_key(route_record)
            candidate_key = (
                covered_component_weight,
                -total_length,
                *tuple(-float(value) for value in quality_key),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_route = route_record

    if best_route is None:
        return None
    seen_signatures.add(canonical_path_signature(best_route["edge_ids"]))
    return best_route


def build_coverage_seed_routes(
    cell_id: str,
    required_edge_ids: Iterable[int],
    portals: List[Dict[str, object]],
    graph: nx.Graph,
    adjacency: Dict[int, List[int]],
    edge_lookup: Dict[Tuple[int, int], int],
    edges: Dict[int, EdgeRecord],
    node_locations: Dict[int, Tuple[float, float, float]],
    carla_map: carla.Map,
    existing_routes: List[Dict[str, object]],
    max_seed_routes: int = 12,
) -> List[Dict[str, object]]:
    required_edge_id_set = set(required_edge_ids)
    covered_edge_ids = {edge_id for route in existing_routes for edge_id in route["edge_ids"] if edge_id in required_edge_id_set}
    seen_signatures = {canonical_path_signature(route["edge_ids"]) for route in existing_routes}
    seed_routes: List[Dict[str, object]] = []

    while len(seed_routes) < max_seed_routes:
        uncovered_edge_ids = sorted(required_edge_id_set - covered_edge_ids)
        if not uncovered_edge_ids:
            break
        uncovered_components = connected_components_for_lane(uncovered_edge_ids, edges)
        uncovered_components = sorted(
            uncovered_components,
            key=lambda component: -sum(route_coverage_weight(edges[edge_id]) for edge_id in component),
        )
        progress = False
        for component_edge_ids in uncovered_components:
            if len(seed_routes) >= max_seed_routes:
                break
            route = build_component_seed_route(
                cell_id=cell_id,
                component_edge_ids=component_edge_ids,
                graph=graph,
                portals=portals,
                adjacency=adjacency,
                edge_lookup=edge_lookup,
                edges=edges,
                node_locations=node_locations,
                carla_map=carla_map,
                seen_signatures=seen_signatures,
                route_index=len(seed_routes) + 1,
            )
            if route is None:
                continue
            newly_covered = {
                edge_id for edge_id in route["edge_ids"] if edge_id in required_edge_id_set and edge_id not in covered_edge_ids
            }
            if not newly_covered:
                continue
            seed_routes.append(route)
            covered_edge_ids.update(newly_covered)
            progress = True
        if not progress:
            break
    return seed_routes


def route_selection_metrics(
    route: Dict[str, object],
    edges: Dict[int, EdgeRecord],
) -> Dict[str, float]:
    route_length_m = float(route["route_length_m"])
    path_waypoints = route["path_waypoints"]
    endpoint_distance_m = point_distance_xy(path_waypoints[0], path_waypoints[-1])
    crossing_length_m = sum(
        edges[edge_id].length
        for edge_id in route["edge_ids"]
        if edges[edge_id].edge_type in {EdgeType.CROSSWALK, EdgeType.JAYWALKING_JUNCTION}
    )
    detour_ratio = route_length_m / max(endpoint_distance_m, 1.0)
    crossing_share = crossing_length_m / max(route_length_m, 1e-6)
    return {
        "route_length_m": round(route_length_m, 3),
        "endpoint_distance_m": round(endpoint_distance_m, 3),
        "detour_ratio": round(detour_ratio, 3),
        "crossing_length_m": round(crossing_length_m, 3),
        "crossing_share": round(crossing_share, 3),
    }


def route_coverage_weight(edge: EdgeRecord) -> float:
    if edge.edge_type == EdgeType.CROSSWALK:
        return edge.length * 5.0
    if edge.edge_type == EdgeType.JAYWALKING_JUNCTION:
        return edge.length * 4.0
    if edge.is_connector_sidewalk:
        return edge.length * 3.0
    if edge.in_junction:
        return edge.length * 1.75
    return edge.length


def desired_crossing_route_candidate_count(
    raw_route_count: int,
    portal_count: int,
    required_edge_count: int,
) -> int:
    target = max(
        CROSSING_MIN_ROUTE_CANDIDATES,
        portal_count + 2,
        int(math.ceil(required_edge_count / max(CROSSING_TARGET_EDGE_DENSITY, 1))),
    )
    return min(raw_route_count, CROSSING_MAX_ROUTE_CANDIDATES, target)


def route_candidate_sort_key(route: Dict[str, object]) -> Tuple[float, ...]:
    metrics = route["selection_metrics"]
    route_type = str(route["route_type"])
    if route_type in {"crosswalk_cross", "unmarked_junction_cross"}:
        return (
            float(metrics["detour_ratio"]),
            -float(metrics["crossing_share"]),
            float(metrics["route_length_m"]),
            float(metrics["endpoint_distance_m"]),
        )
    return (
        float(metrics["detour_ratio"]),
        float(metrics["route_length_m"]),
        float(metrics["endpoint_distance_m"]),
    )


def route_coverage_gain(
    route: Dict[str, object],
    required_edge_ids: Set[int],
    edges: Dict[int, EdgeRecord],
    edge_selection_count: Dict[int, int],
) -> Tuple[float, float, int]:
    uncovered_weight = 0.0
    diminishing_weight = 0.0
    uncovered_edge_count = 0
    for edge_id in route["edge_ids"]:
        if edge_id not in required_edge_ids:
            continue
        edge_weight = route_coverage_weight(edges[edge_id])
        prior_count = edge_selection_count.get(edge_id, 0)
        diminishing_weight += edge_weight / (1.0 + float(prior_count))
        if prior_count == 0:
            uncovered_weight += edge_weight
            uncovered_edge_count += 1
    return uncovered_weight, diminishing_weight, uncovered_edge_count


def route_coverage_priority(
    route: Dict[str, object],
    required_edge_ids: Set[int],
    edges: Dict[int, EdgeRecord],
    edge_selection_count: Dict[int, int],
) -> Tuple[float, ...]:
    uncovered_weight, diminishing_weight, uncovered_edge_count = route_coverage_gain(
        route=route,
        required_edge_ids=required_edge_ids,
        edges=edges,
        edge_selection_count=edge_selection_count,
    )
    metrics = route["selection_metrics"]
    return (
        uncovered_weight,
        diminishing_weight,
        float(uncovered_edge_count),
        float(metrics["crossing_share"]),
        -float(metrics["detour_ratio"]),
        -float(metrics["route_length_m"]),
    )


def greedy_select_crossing_routes(
    candidates: List[Dict[str, object]],
    selected: List[Dict[str, object]],
    required_edge_ids: Set[int],
    edges: Dict[int, EdgeRecord],
    edge_selection_count: Dict[int, int],
    retained_route_count_by_type: Dict[str, int],
    target_count: int,
    ignore_type_limits: bool = False,
    count_towards_type_limits: bool = True,
) -> List[Dict[str, object]]:
    remaining = list(candidates)
    while remaining and len(selected) < target_count:
        best_index: Optional[int] = None
        best_priority: Optional[Tuple[float, ...]] = None
        for candidate_index, route in enumerate(remaining):
            route_type = str(route["route_type"])
            type_limit = CROSSING_ROUTE_TYPE_LIMITS.get(route_type, CROSSING_MAX_ROUTE_CANDIDATES)
            if not ignore_type_limits and retained_route_count_by_type[route_type] >= type_limit:
                continue
            priority = route_coverage_priority(
                route=route,
                required_edge_ids=required_edge_ids,
                edges=edges,
                edge_selection_count=edge_selection_count,
            )
            if best_priority is None or priority > best_priority:
                best_priority = priority
                best_index = candidate_index

        if best_index is None:
            break

        route = remaining.pop(best_index)
        selected.append(route)
        if count_towards_type_limits:
            retained_route_count_by_type[str(route["route_type"])] += 1
        for edge_id in route["edge_ids"]:
            if edge_id in required_edge_ids:
                edge_selection_count[edge_id] += 1
    return remaining


def prune_crossing_route_candidates(
    routes: List[Dict[str, object]],
    edges: Dict[int, EdgeRecord],
    required_edge_ids: Iterable[int],
    portal_count: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not routes:
        return [], {
            "raw_route_count": 0,
            "retained_route_count": 0,
            "target_route_count": 0,
            "raw_route_count_by_type": {},
            "retained_route_count_by_type": {},
        }

    required_edge_id_set = set(required_edge_ids)
    raw_route_count_by_type: Dict[str, int] = defaultdict(int)
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for route in routes:
        route["selection_metrics"] = route_selection_metrics(route, edges)
        route_type = str(route["route_type"])
        raw_route_count_by_type[route_type] += 1
        buckets[route_type].append(route)

    for route_type, bucket in buckets.items():
        bucket.sort(
            key=lambda route: (
                route_candidate_sort_key(route),
                str(route["start_portal"]),
                str(route["end_portal"]),
            )
        )
        for rank, route in enumerate(bucket, start=1):
            route["selection_rank_within_type"] = rank

    reliable_candidates: List[Dict[str, object]] = []
    rejected_route_count_by_type: Dict[str, int] = defaultdict(int)
    for route in routes:
        route_type = str(route["route_type"])
        max_detour_ratio = CROSSING_ROUTE_TYPE_MAX_DETOUR_RATIO.get(route_type)
        detour_ratio = float(route["selection_metrics"]["detour_ratio"])
        if max_detour_ratio is not None and detour_ratio > max_detour_ratio + 1e-6:
            rejected_route_count_by_type[route_type] += 1
            continue
        reliable_candidates.append(route)

    route_type_order = [
        route_type for route_type in CROSSING_ROUTE_TYPE_PRIORITY if route_type in buckets
    ]
    route_type_order.extend(
        sorted(route_type for route_type in buckets if route_type not in CROSSING_ROUTE_TYPE_PRIORITY)
    )

    target_count = desired_crossing_route_candidate_count(
        raw_route_count=len(routes),
        portal_count=portal_count,
        required_edge_count=len(required_edge_id_set),
    )
    retained_route_count_by_type: Dict[str, int] = defaultdict(int)
    seed_candidates = sorted(
        [route for route in reliable_candidates if str(route.get("path_source", "")) == "coverage_component_seed"],
        key=lambda route: (
            route_candidate_sort_key(route),
            route_type_order.index(str(route["route_type"])),
            str(route["start_portal"]),
            str(route["end_portal"]),
        ),
    )
    preferred_candidates = sorted(
        [route for route in reliable_candidates if route not in seed_candidates],
        key=lambda route: (
            route_candidate_sort_key(route),
            route_type_order.index(str(route["route_type"])),
            str(route["start_portal"]),
            str(route["end_portal"]),
        ),
    )
    fallback_candidates = sorted(
        [route for route in routes if route not in reliable_candidates],
        key=lambda route: (
            route_candidate_sort_key(route),
            route_type_order.index(str(route["route_type"])),
            str(route["start_portal"]),
            str(route["end_portal"]),
        ),
    )

    edge_selection_count: Dict[int, int] = defaultdict(int)
    selected: List[Dict[str, object]] = []
    greedy_select_crossing_routes(
        candidates=seed_candidates,
        selected=selected,
        required_edge_ids=required_edge_id_set,
        edges=edges,
        edge_selection_count=edge_selection_count,
        retained_route_count_by_type=retained_route_count_by_type,
        target_count=min(CROSSING_MAX_SEED_ROUTES, len(seed_candidates)),
        ignore_type_limits=True,
        count_towards_type_limits=False,
    )
    selected_seed_count = len(selected)
    remaining_preferred = greedy_select_crossing_routes(
        candidates=preferred_candidates,
        selected=selected,
        required_edge_ids=required_edge_id_set,
        edges=edges,
        edge_selection_count=edge_selection_count,
        retained_route_count_by_type=retained_route_count_by_type,
        target_count=target_count,
    )
    fallback_selected_count = 0
    if len(selected) < target_count and fallback_candidates:
        before_fallback = len(selected)
        remaining_fallback = greedy_select_crossing_routes(
            candidates=fallback_candidates,
            selected=selected,
            required_edge_ids=required_edge_id_set,
            edges=edges,
            edge_selection_count=edge_selection_count,
            retained_route_count_by_type=retained_route_count_by_type,
            target_count=target_count,
        )
        fallback_selected_count = len(selected) - before_fallback
    else:
        remaining_fallback = list(fallback_candidates)

    covered_edge_ids = {edge_id for edge_id, count in edge_selection_count.items() if count > 0}
    if (
        len(covered_edge_ids & required_edge_id_set) < len(required_edge_id_set)
        and len(selected) < CROSSING_MAX_ROUTE_CANDIDATES
    ):
        remaining_preferred = greedy_select_crossing_routes(
            candidates=remaining_preferred,
            selected=selected,
            required_edge_ids=required_edge_id_set,
            edges=edges,
            edge_selection_count=edge_selection_count,
            retained_route_count_by_type=retained_route_count_by_type,
            target_count=CROSSING_MAX_ROUTE_CANDIDATES,
        )
        covered_edge_ids = {edge_id for edge_id, count in edge_selection_count.items() if count > 0}
        if (
            len(covered_edge_ids & required_edge_id_set) < len(required_edge_id_set)
            and len(selected) < CROSSING_MAX_ROUTE_CANDIDATES
            and remaining_fallback
        ):
            before_fallback = len(selected)
            remaining_fallback = greedy_select_crossing_routes(
                candidates=remaining_fallback,
                selected=selected,
                required_edge_ids=required_edge_id_set,
                edges=edges,
                edge_selection_count=edge_selection_count,
                retained_route_count_by_type=retained_route_count_by_type,
                target_count=CROSSING_MAX_ROUTE_CANDIDATES,
            )
            fallback_selected_count += len(selected) - before_fallback

    if not selected:
        shortest_route = min(routes, key=lambda route: (float(route["route_length_m"]), str(route["route_id"])))
        selected = [shortest_route]
        retained_route_count_by_type[str(shortest_route["route_type"])] = 1
        for edge_id in shortest_route["edge_ids"]:
            if edge_id in required_edge_id_set:
                edge_selection_count[edge_id] += 1

    selected.sort(
        key=lambda route: (
            CROSSING_ROUTE_TYPE_PRIORITY.index(route["route_type"])
            if route["route_type"] in CROSSING_ROUTE_TYPE_PRIORITY
            else len(CROSSING_ROUTE_TYPE_PRIORITY),
            route_candidate_sort_key(route),
            str(route["start_portal"]),
            str(route["end_portal"]),
        )
    )
    covered_edge_ids = {edge_id for edge_id, count in edge_selection_count.items() if count > 0}
    total_required_weight = sum(route_coverage_weight(edges[edge_id]) for edge_id in required_edge_id_set)
    covered_required_weight = sum(
        route_coverage_weight(edges[edge_id]) for edge_id in covered_edge_ids if edge_id in required_edge_id_set
    )
    return selected, {
        "raw_route_count": len(routes),
        "retained_route_count": len(selected),
        "target_route_count": target_count,
        "raw_route_count_by_type": dict(sorted(raw_route_count_by_type.items())),
        "rejected_route_count_by_type": dict(sorted(rejected_route_count_by_type.items())),
        "retained_route_count_by_type": dict(sorted(retained_route_count_by_type.items())),
        "selected_seed_count": selected_seed_count,
        "fallback_selected_count": fallback_selected_count,
        "required_edge_count": len(required_edge_id_set),
        "covered_edge_count": len(covered_edge_ids & required_edge_id_set),
        "weighted_edge_coverage_ratio": round(
            covered_required_weight / max(total_required_weight, 1e-6),
            3,
        ),
    }


def build_cell_routes(
    cell: Dict[str, object],
    edges: Dict[int, EdgeRecord],
    adjacency: Dict[int, List[int]],
    node_locations: Dict[int, Tuple[float, float, float]],
    edge_lookup: Dict[Tuple[int, int], int],
    carla_map: carla.Map,
    portal_merge_radius: float = 0.0,
    portal_sidewalk_merge_length: float = 0.0,
    route_edge_ids: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not cell["attributes"]["usable"]:
        return [], []

    core_edge_ids = cell_edge_ids(cell)
    edge_ids = list(route_edge_ids) if route_edge_ids is not None else list(core_edge_ids)
    cell_kind = str(cell["kind"])

    if cell_kind == "sidewalk":
        routes: List[Dict[str, object]] = []
        portals: List[Dict[str, object]] = []
        portal_count = 0
        components = connected_components_for_lane(core_edge_ids, edges)
        sorted_components = sorted(
            components,
            key=lambda component: min(edges[edge_id].center[0] for edge_id in component),
        )
        for component_index, component_edge_ids in enumerate(sorted_components):
            graph = build_local_graph(component_edge_ids, edges)
            candidate_node_ids = boundary_nodes_for_edges(component_edge_ids, edges, adjacency)
            if len(candidate_node_ids) < 2:
                candidate_node_ids = sorted(node_id for node_id in graph.nodes if graph.degree[node_id] == 1)
            if len(candidate_node_ids) < 2:
                candidate_node_ids = sorted(graph.nodes)
            if len(candidate_node_ids) < 2:
                continue

            diameter = component_diameter_path(graph, candidate_node_ids, edge_lookup)
            if diameter is None:
                continue
            node_path, path_edge_ids, _, start_node_id, end_node_id = diameter

            start_portal_id = f"{cell['id']}_portal_{portal_count + 1:02d}"
            end_portal_id = f"{cell['id']}_portal_{portal_count + 2:02d}"
            portals.extend(
                [
                    {
                        "portal_id": start_portal_id,
                        "node_ids": [start_node_id],
                        "centroid": rounded_point(node_locations[start_node_id]),
                        "kind": "component_endpoint",
                        "component_index": component_index,
                    },
                    {
                        "portal_id": end_portal_id,
                        "node_ids": [end_node_id],
                        "centroid": rounded_point(node_locations[end_node_id]),
                        "kind": "component_endpoint",
                        "component_index": component_index,
                    },
                ]
            )
            portal_count += 2

            route_id = f"{cell['id']}/route_{len(routes) + 1:02d}"
            routes.append(
                build_route_record(
                    route_id=route_id,
                    route_type="sidewalk_through",
                    node_path=node_path,
                    path_edge_ids=path_edge_ids,
                    edges=edges,
                    node_locations=node_locations,
                    carla_map=carla_map,
                    start_portal_id=start_portal_id,
                    end_portal_id=end_portal_id,
                    component_index=component_index,
                    path_source="sidewalk_component_diameter",
                )
            )
        route_count_by_type: Dict[str, int] = defaultdict(int)
        for route in routes:
            route_count_by_type[str(route["route_type"])] += 1
        return portals, routes

    graph = build_local_graph(edge_ids, edges)
    portals = build_leaf_portals(
        cell_id=str(cell["id"]),
        graph=graph,
        node_locations=node_locations,
    )
    if len(portals) < 2:
        return portals, []
    seen_signatures: Set[Tuple[int, ...]] = set()
    routes = []
    for left_index, left_portal in enumerate(portals):
        for right_portal in portals[left_index + 1:]:
            best = best_path_between_portals(graph, left_portal, right_portal, edge_lookup)
            if best is None:
                continue
            node_path, path_edge_ids, _ = best
            signature = canonical_path_signature(path_edge_ids)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            route_type = route_type_for_path(cell_kind, path_edge_ids, edges, node_path, node_locations)
            route_id = f"{cell['id']}/route_{len(routes) + 1:02d}"
            routes.append(
                build_route_record(
                    route_id=route_id,
                    route_type=route_type,
                    node_path=node_path,
                    path_edge_ids=path_edge_ids,
                    edges=edges,
                    node_locations=node_locations,
                    carla_map=carla_map,
                    start_portal_id=str(left_portal["portal_id"]),
                    end_portal_id=str(right_portal["portal_id"]),
                    component_index=None,
                    path_source="shortest_path",
                )
            )

    routes.sort(key=lambda route: (route["route_type"], route["route_length_m"], route["route_id"]))
    for route_index, route in enumerate(routes, start=1):
        route["route_id"] = f"{cell['id']}/route_{route_index:02d}"
    return portals, routes


def attach_routes_to_cells(
    cells: List[Dict[str, object]],
    edges: Dict[int, EdgeRecord],
    edge_to_cell: Dict[int, str],
    adjacency: Dict[int, List[int]],
    node_locations: Dict[int, Tuple[float, float, float]],
    carla_map: carla.Map,
    portal_merge_radius: float,
    portal_sidewalk_merge_length: float,
    obstacle_records: Sequence[StaticObstacleRecord],
    obstacle_validation_config: StaticObstacleValidationConfig,
) -> None:
    edge_lookup = build_edge_lookup(edges)
    cells_by_id = {str(cell["id"]): cell for cell in cells}
    for cell in cells:
        support_cell_ids, support_edge_ids = build_route_support_for_cell(
            cell=cell,
            cells_by_id=cells_by_id,
            edge_to_cell=edge_to_cell,
            adjacency=adjacency,
            edges=edges,
        )
        portals, routes = build_cell_routes(
            cell=cell,
            edges=edges,
            adjacency=adjacency,
            node_locations=node_locations,
            edge_lookup=edge_lookup,
            carla_map=carla_map,
            portal_merge_radius=portal_merge_radius,
            portal_sidewalk_merge_length=portal_sidewalk_merge_length,
            route_edge_ids=support_edge_ids,
        )
        cell["route_support"] = {
            "cell_ids": support_cell_ids,
            "edge_ids": support_edge_ids,
            "includes_adjacent_sidewalk_cells": bool(cell["kind"] == "crossing" and len(support_cell_ids) > 1),
        }
        cell["portals"] = portals
        support_bounds = bounds_for_edge_ids(support_edge_ids, edges)
        cell_obstacle_records = relevant_obstacles_for_bounds(obstacle_records, support_bounds)
        blocked_edge_records = build_blocked_edge_records(
            edge_ids=support_edge_ids,
            edges=edges,
            obstacle_records=cell_obstacle_records,
            config=obstacle_validation_config,
        )
        validated_routes, validation_summary = validate_routes_against_static_obstacles(
            cell_kind=str(cell["kind"]),
            routes=routes,
            portals=portals,
            support_edge_ids=support_edge_ids,
            edges=edges,
            edge_lookup=edge_lookup,
            node_locations=node_locations,
            blocked_edge_records=blocked_edge_records,
            obstacle_records=cell_obstacle_records,
            carla_map=carla_map,
        )
        obstacle_counts_by_label: Dict[str, int] = defaultdict(int)
        for obstacle_record in cell_obstacle_records:
            obstacle_counts_by_label[obstacle_record.label_name] += 1

        validation_summary["support_bounds"] = rounded_bounds_dict(support_bounds)
        validation_summary["candidate_obstacle_count"] = len(cell_obstacle_records)
        validation_summary["candidate_obstacle_ids"] = [record.object_id for record in cell_obstacle_records]
        validation_summary["candidate_obstacle_counts_by_label"] = dict(sorted(obstacle_counts_by_label.items()))
        cell["static_obstacle_validation"] = validation_summary
        cell["routes"] = validated_routes
        route_type_counts: Dict[str, int] = defaultdict(int)
        for route in validated_routes:
            route_type_counts[str(route["route_type"])] += 1
        cell["route_stats"] = {
            "route_count": len(validated_routes),
            "by_type": dict(sorted(route_type_counts.items())),
        }


def build_route_stats(cells: List[Dict[str, object]]) -> Dict[str, object]:
    total_count = 0
    by_type: Dict[str, int] = defaultdict(int)
    by_cell_kind: Dict[str, int] = defaultdict(int)
    cells_with_routes = 0

    for cell in cells:
        routes = cell.get("routes", [])
        if routes:
            cells_with_routes += 1
        for route in routes:
            total_count += 1
            by_type[str(route["route_type"])] += 1
            by_cell_kind[str(cell["kind"])] += 1

    return {
        "total_count": total_count,
        "cells_with_routes": cells_with_routes,
        "cells_without_routes": len(cells) - cells_with_routes,
        "by_type": dict(sorted(by_type.items())),
        "by_cell_kind": dict(sorted(by_cell_kind.items())),
    }


def partition_walkable_cells(
    edges: Dict[int, EdgeRecord],
    adjacency: Dict[int, List[int]],
    approach_length: float,
    sidewalk_chunk_length: float,
    sidewalk_tail_merge_length: float,
    sidewalk_min_longitudinal_length: float,
) -> Tuple[List[Dict[str, object]], Dict[int, str], Dict[str, int]]:
    crossing_assignments = grow_crossing_assignments(edges, adjacency, approach_length)
    reserved_edge_ids = set().union(*crossing_assignments.values()) if crossing_assignments else set()
    sidewalk_chunks = build_sidewalk_assignments(
        edges,
        reserved_edge_ids=reserved_edge_ids,
        target_length=sidewalk_chunk_length,
        min_tail_length=sidewalk_tail_merge_length,
        min_longitudinal_length=sidewalk_min_longitudinal_length,
    )

    cells: List[Dict[str, object]] = []
    edge_to_cell: Dict[int, str] = {}
    color_index = 0

    for crossing_index, root in enumerate(sorted(crossing_assignments), start=1):
        cell_id = f"crossing_{crossing_index:03d}"
        color = svg_color(color_index)
        color_index += 1
        cell = build_cell_record(
            cell_id,
            "crossing",
            crossing_assignments[root],
            edges,
            color,
            sidewalk_min_longitudinal_length,
        )
        cells.append(cell)
        for edge_id in crossing_assignments[root]:
            edge_to_cell[edge_id] = cell_id

    for sidewalk_index, chunk in enumerate(sidewalk_chunks, start=1):
        cell_id = f"sidewalk_{sidewalk_index:03d}"
        color = svg_color(color_index)
        color_index += 1
        cell = build_cell_record(
            cell_id,
            "sidewalk",
            chunk,
            edges,
            color,
            sidewalk_min_longitudinal_length,
        )
        cells.append(cell)
        for edge_id in chunk:
            edge_to_cell[edge_id] = cell_id

    included_edge_ids = set(edge_to_cell)
    stats = {
        "cell_count": len(cells),
        "crossing_cell_count": sum(1 for cell in cells if cell["kind"] == "crossing"),
        "sidewalk_cell_count": sum(1 for cell in cells if cell["kind"] == "sidewalk"),
        "included_edge_count": len(included_edge_ids),
        "excluded_edge_count": len(edges) - len(included_edge_ids),
        "excluded_midblock_jaywalking_count": sum(1 for edge in edges.values() if edge.edge_type == EdgeType.JAYWALKING),
        "excluded_sidewalk_to_road_count": sum(1 for edge in edges.values() if edge.edge_type == EdgeType.SIDEWALK_TO_ROAD),
    }
    return cells, edge_to_cell, stats


def write_svg(
    svg_path: Path,
    cells: List[Dict[str, object]],
    edges: Dict[int, EdgeRecord],
    edge_to_cell: Dict[int, str],
    title: str,
) -> None:
    included_edge_ids = sorted(edge_to_cell)
    if not included_edge_ids:
        raise RuntimeError("No edges were assigned to cells; SVG export would be empty.")

    coordinates = []
    for edge_id in included_edge_ids:
        edge = edges[edge_id]
        coordinates.extend([(edge.x1, edge.y1), (edge.x2, edge.y2)])

    min_x = min(point[0] for point in coordinates)
    max_x = max(point[0] for point in coordinates)
    min_y = min(point[1] for point in coordinates)
    max_y = max(point[1] for point in coordinates)
    padding = 12.0
    canvas_width = (max_x - min_x) + 2 * padding
    canvas_height = (max_y - min_y) + 2 * padding

    pixel_width = 1800
    pixel_height = max(900, int(pixel_width * canvas_height / max(canvas_width, 1.0)))

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_width}" height="{pixel_height}" '
            f'viewBox="0 0 {canvas_width:.3f} {canvas_height:.3f}">'
        ),
        f'<title>{title}</title>',
        '<rect width="100%" height="100%" fill="#f7f6f1"/>',
        (
            '<text x="6" y="5" font-family="monospace" font-size="4.5" fill="#333">'
            f'{title}'
            "</text>"
        ),
        '<g opacity="0.16" stroke="#111" stroke-linecap="round" stroke-linejoin="round">',
    ]

    for edge_id in included_edge_ids:
        edge = edges[edge_id]
        x1, y1 = world_to_canvas((edge.x1, edge.y1), min_x, max_y, padding)
        x2, y2 = world_to_canvas((edge.x2, edge.y2), min_x, max_y, padding)
        base_width = max(0.9, edge.width_hint)
        lines.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke-width="{base_width + 0.8:.3f}"/>'
        )
    lines.append("</g>")

    for cell in cells:
        lines.append(
            f'<g id="{cell["id"]}" stroke="{cell["color"]}" fill="none" '
            'stroke-linecap="round" stroke-linejoin="round">'
        )
        for edge_info in cell["edges"]:
            edge = edges[edge_info["edge_id"]]
            x1, y1 = world_to_canvas((edge.x1, edge.y1), min_x, max_y, padding)
            x2, y2 = world_to_canvas((edge.x2, edge.y2), min_x, max_y, padding)

            if edge.edge_type == EdgeType.CROSSWALK:
                stroke_width = max(3.0, edge.width_hint)
            elif edge.edge_type == EdgeType.JAYWALKING_JUNCTION:
                stroke_width = max(2.4, edge.width_hint * 0.9)
            elif edge.is_connector_sidewalk:
                stroke_width = max(1.8, edge.width_hint * 0.9)
            else:
                stroke_width = max(1.4, edge.width_hint)

            lines.append(
                f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                f'stroke-width="{stroke_width:.3f}"/>'
            )
        lines.append("</g>")

    for cell in cells:
        centroid_x, centroid_y = cell["centroid"]
        x, y = world_to_canvas((centroid_x, centroid_y), min_x, max_y, padding)
        label = cell["id"].split("_")[-1]
        lines.append(
            f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="monospace" font-size="4.4" fill="#1e1e1e">{label}</text>'
        )

    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")


def write_catalog(
    json_path: Path,
    map_name: str,
    approach_length: float,
    sidewalk_chunk_length: float,
    sidewalk_tail_merge_length: float,
    sidewalk_min_longitudinal_length: float,
    route_portal_merge_radius: float,
    route_portal_sidewalk_merge_length: float,
    static_obstacle_validation_config: StaticObstacleValidationConfig,
    cells: List[Dict[str, object]],
    stats: Dict[str, int],
) -> None:
    payload = {
        "map_name": map_name,
        "catalog_type": "candidate_scene_catalog",
        "cells_are_candidate_scenes": True,
        "rules": {
            "crossing_seed_types": ["CROSSWALK", "JAYWALKING_JUNCTION"],
            "crossing_sidewalk_approach_length_m": approach_length,
            "remaining_sidewalk_chunk_length_m": sidewalk_chunk_length,
            "remaining_sidewalk_tail_merge_length_m": sidewalk_tail_merge_length,
            "sidewalk_min_longitudinal_length_m": sidewalk_min_longitudinal_length,
            "route_portal_merge_radius_m": route_portal_merge_radius,
            "route_portal_sidewalk_merge_length_m": route_portal_sidewalk_merge_length,
            "excluded_types": ["JAYWALKING", "SIDEWALK_TO_ROAD"],
        },
        "static_obstacle_validation_rules": static_obstacle_validation_config.to_dict(),
        "route_type_definitions": ROUTE_TYPE_DEFINITIONS,
        "behavior_profiles": BEHAVIOR_PROFILES,
        "stats": stats,
        "sidewalk_longitudinal_stats": build_sidewalk_longitudinal_stats(cells),
        "candidate_scene_stats": build_candidate_scene_stats(cells),
        "route_stats": build_route_stats(cells),
        "cells": cells,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def canonical_map_name(map_name: str) -> str:
    cleaned = map_name.rstrip("/")
    if not cleaned:
        raise ValueError("map_name must not be empty")
    return cleaned.split("/")[-1]


def resolve_xodr_path(map_name: str, explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        resolved = explicit_path.expanduser().resolve()
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(f"OpenDRIVE file not found: {resolved}")

    canonical_name = canonical_map_name(map_name)
    candidates = [
        DEFAULT_CARLA_MAPS_ROOT / canonical_name / "OpenDrive" / f"{canonical_name}.xodr",
        DEFAULT_CARLA_MAPS_ROOT / "OpenDrive" / f"{canonical_name}.xodr",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if DEFAULT_CARLA_MAPS_ROOT.exists():
        matches = sorted(DEFAULT_CARLA_MAPS_ROOT.rglob(f"{canonical_name}.xodr"))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        f"Unable to locate {canonical_name}.xodr under {DEFAULT_CARLA_MAPS_ROOT}. "
        "Pass --xodr-path explicitly if the file lives elsewhere."
    )


def load_offline_map(map_name: str, xodr_path: Path) -> carla.Map:
    content = xodr_path.read_text(encoding="utf-8")
    return carla.Map(canonical_map_name(map_name), content)


def bounds_for_points(points: Iterable[Tuple[float, float]]) -> Bounds:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if not xs or not ys:
        raise ValueError("Cannot compute bounds for an empty point set")
    return (min(xs), min(ys), max(xs), max(ys))


def expand_bounds(bounds: Bounds, margin: float) -> Bounds:
    return (bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin)


def bounds_intersect(left: Bounds, right: Bounds) -> bool:
    return not (
        left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1]
    )


def coarse_cell(x: float, y: float, cell_size: float) -> Cell:
    return (math.floor(x / cell_size), math.floor(y / cell_size))


def cell_bounds(cell: Cell, cell_size: float) -> Bounds:
    grid_x, grid_y = cell
    min_x = grid_x * cell_size
    min_y = grid_y * cell_size
    return (min_x, min_y, min_x + cell_size, min_y + cell_size)


def format_tile_id(tile_x: int, tile_y: int) -> str:
    return f"tile_{tile_x:+05d}_{tile_y:+05d}"


def tile_core_bounds(tile_x: int, tile_y: int, tile_size: float) -> Bounds:
    min_x = tile_x * tile_size
    min_y = tile_y * tile_size
    return (min_x, min_y, min_x + tile_size, min_y + tile_size)


def sample_grid_cells_for_border(border: Sequence[carla.Vector3D], coarse_grid_size: float) -> Set[Cell]:
    cells: Set[Cell] = set()
    for point in border:
        cells.add(coarse_cell(point.x, point.y, coarse_grid_size))
    return cells


def connected_components(cells: Set[Cell]) -> List[Set[Cell]]:
    components: List[Set[Cell]] = []
    remaining = set(cells)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while remaining:
        start = next(iter(remaining))
        queue = deque([start])
        component: Set[Cell] = set()
        remaining.remove(start)
        while queue:
            cell = queue.popleft()
            component.add(cell)
            for dx, dy in neighbors:
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
        components.append(component)
    return components


def tile_indices_for_bounds(bounds: Bounds, tile_size: float) -> Iterable[Tuple[int, int]]:
    min_tx = math.floor(bounds[0] / tile_size)
    max_tx = math.floor((bounds[2] - 1e-6) / tile_size)
    min_ty = math.floor(bounds[1] / tile_size)
    max_ty = math.floor((bounds[3] - 1e-6) / tile_size)
    for tile_x in range(min_tx, max_tx + 1):
        for tile_y in range(min_ty, max_ty + 1):
            yield tile_x, tile_y


def rect_points(bounds: Bounds) -> List[Tuple[float, float]]:
    return [
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
        (bounds[0], bounds[3]),
        (bounds[0], bounds[1]),
    ]


def build_tile_catalog(
    map_name: str,
    requested_map_name: str,
    xodr_path: Path,
    border_resolution: float,
    coarse_grid_size: float,
    tile_size: float,
    tile_halo: float,
) -> Dict[str, object]:
    offline_map = load_offline_map(map_name, xodr_path)
    carla_borders, _ = extract_sidewalk_borders(offline_map, border_resolution)
    borders_xy: List[List[Tuple[float, float]]] = [
        [(float(point.x), float(point.y)) for point in border]
        for border in carla_borders
        if border
    ]
    if not borders_xy:
        raise RuntimeError("No sidewalk borders extracted from the OpenDRIVE map.")

    occupied_cells: Set[Cell] = set()
    border_bounds: List[Bounds] = []
    for border, carla_border in zip(borders_xy, carla_borders):
        border_bounds.append(bounds_for_points(border))
        occupied_cells.update(sample_grid_cells_for_border(carla_border, coarse_grid_size))

    components = connected_components(occupied_cells)
    raw_tiles: Dict[str, Dict[str, object]] = {}
    for component_index, component in enumerate(components):
        component_points = [point for cell in component for point in rect_points(cell_bounds(cell, coarse_grid_size))]
        component_bounds = bounds_for_points(component_points)
        for tile_x, tile_y in tile_indices_for_bounds(component_bounds, tile_size):
            core_bounds = tile_core_bounds(tile_x, tile_y, tile_size)
            halo_bounds = expand_bounds(core_bounds, tile_halo)
            matched_border_indices = {
                index for index, bounds in enumerate(border_bounds) if bounds_intersect(bounds, halo_bounds)
            }
            if not matched_border_indices:
                continue
            tile_cells = {
                cell for cell in component if bounds_intersect(cell_bounds(cell, coarse_grid_size), core_bounds)
            }
            if not tile_cells:
                continue

            name = format_tile_id(tile_x, tile_y)
            tile_entry = raw_tiles.setdefault(
                name,
                {
                    "id": name,
                    "tile_index": [tile_x, tile_y],
                    "core_bounds": [round(value, 3) for value in core_bounds],
                    "halo_bounds": [round(value, 3) for value in halo_bounds],
                    "component_indices": set(),
                    "coarse_cells": set(),
                    "border_indices": set(),
                },
            )
            tile_entry["component_indices"].add(component_index)
            tile_entry["coarse_cells"].update(tile_cells)
            tile_entry["border_indices"].update(matched_border_indices)

    tiles: List[Dict[str, object]] = []
    for name in sorted(raw_tiles):
        entry = raw_tiles[name]
        component_indices = sorted(entry["component_indices"])
        coarse_cells = sorted(entry["coarse_cells"])
        border_indices = sorted(entry["border_indices"])
        tiles.append(
            {
                "id": entry["id"],
                "tile_index": entry["tile_index"],
                "core_bounds": entry["core_bounds"],
                "halo_bounds": entry["halo_bounds"],
                "component_indices": component_indices,
                "component_count": len(component_indices),
                "coarse_cell_count": len(coarse_cells),
                "border_count": len(border_indices),
                "coarse_cells": [[cell[0], cell[1]] for cell in coarse_cells],
                "border_indices": border_indices,
            }
        )

    return {
        "map_name": map_name,
        "requested_map_name": requested_map_name,
        "xodr_path": str(xodr_path),
        "border_resolution": border_resolution,
        "coarse_grid_size": coarse_grid_size,
        "tile_size": tile_size,
        "tile_halo": tile_halo,
        "sidewalk_border_count": len(borders_xy),
        "occupied_coarse_cell_count": len(occupied_cells),
        "component_count": len(components),
        "tile_count": len(tiles),
        "tiles": tiles,
    }


def write_tile_catalog_json(catalog: Dict[str, object], catalog_path: Path) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def generate_tile_outputs(
    *,
    world: carla.World,
    carla_map: carla.Map,
    args: argparse.Namespace,
    map_name: str,
    map_output_root: Path,
    tile_record: Dict[str, object],
    tile_catalog_path: Path,
    static_obstacle_validation_config: StaticObstacleValidationConfig,
    skip_static_obstacles: bool,
) -> Dict[str, object]:
    tile_id = str(tile_record["id"])
    output_dir = map_output_root / tile_id
    output_dir.mkdir(parents=True, exist_ok=True)

    focus_tile(world, tile_record["core_bounds"], spectator_height=args.tile_focus_height)

    path_planner = BoundsFilteredPedPathPlanner(
        carla_map,
        bounds=tile_record["halo_bounds"],
        waypoint_distance=args.waypoint_distance,
    )

    obstacle_records: List[StaticObstacleRecord]
    if skip_static_obstacles:
        obstacle_records = []
    else:
        obstacle_records = extract_static_obstacle_records(world, static_obstacle_validation_config)

    node_locations = build_node_locations(path_planner)
    edges, adjacency = build_edge_records(path_planner)
    cells, edge_to_cell, stats = partition_walkable_cells(
        edges,
        adjacency=adjacency,
        approach_length=args.approach_length,
        sidewalk_chunk_length=args.sidewalk_chunk_length,
        sidewalk_tail_merge_length=args.sidewalk_tail_merge_length,
        sidewalk_min_longitudinal_length=args.sidewalk_min_longitudinal_length,
    )
    attach_routes_to_cells(
        cells=cells,
        edges=edges,
        edge_to_cell=edge_to_cell,
        adjacency=adjacency,
        node_locations=node_locations,
        carla_map=carla_map,
        portal_merge_radius=args.route_portal_merge_radius,
        portal_sidewalk_merge_length=args.route_portal_sidewalk_merge_length,
        obstacle_records=obstacle_records,
        obstacle_validation_config=static_obstacle_validation_config,
    )

    json_path = output_dir / "walkable_cells.json"
    svg_path = output_dir / "walkable_cells.svg"
    title = (
        f"{map_name} {tile_id} walkable cells "
        f"(crossing approach={args.approach_length:.0f}m, sidewalk chunk={args.sidewalk_chunk_length:.0f}m)"
    )

    write_catalog(
        json_path=json_path,
        map_name=map_name,
        approach_length=args.approach_length,
        sidewalk_chunk_length=args.sidewalk_chunk_length,
        sidewalk_tail_merge_length=args.sidewalk_tail_merge_length,
        sidewalk_min_longitudinal_length=args.sidewalk_min_longitudinal_length,
        route_portal_merge_radius=args.route_portal_merge_radius,
        route_portal_sidewalk_merge_length=args.route_portal_sidewalk_merge_length,
        static_obstacle_validation_config=static_obstacle_validation_config,
        cells=cells,
        stats=stats,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["tile"] = {
        "id": tile_record["id"],
        "tile_index": tile_record["tile_index"],
        "core_bounds": tile_record["core_bounds"],
        "halo_bounds": tile_record["halo_bounds"],
        "catalog": str(tile_catalog_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    svg_written = bool(edge_to_cell)
    if svg_written:
        write_svg(svg_path=svg_path, cells=cells, edges=edges, edge_to_cell=edge_to_cell, title=title)

    return {
        "tile_id": tile_id,
        "cell_count": stats["cell_count"],
        "crossing_cell_count": stats["crossing_cell_count"],
        "sidewalk_cell_count": stats["sidewalk_cell_count"],
        "included_edge_count": stats["included_edge_count"],
        "excluded_edge_count": stats["excluded_edge_count"],
        "static_obstacle_count": len(obstacle_records),
        "skip_static_obstacles": bool(skip_static_obstacles),
        "json": str(json_path),
        "svg": str(svg_path) if svg_written else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partition large CARLA maps into sidewalk-backed tiles and export walkable cells per tile.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA port")
    parser.add_argument("--timeout", default=180.0, type=float, help="CARLA RPC timeout in seconds")
    parser.add_argument("--map-name", default="Town12", help="CARLA map name to analyze")
    parser.add_argument(
        "--xodr-path",
        type=Path,
        default=None,
        help="Optional explicit OpenDRIVE path. Defaults to auto-discovery from the CARLA maps directory.",
    )
    parser.add_argument(
        "--waypoint-distance",
        default=5.0,
        type=float,
        help="Waypoint spacing used to build the pedestrian topology graph [m]",
    )
    parser.add_argument(
        "--approach-length",
        default=20.0,
        type=float,
        help="How far crossing cells extend into connected sidewalk edges [m]",
    )
    parser.add_argument(
        "--sidewalk-chunk-length",
        default=50.0,
        type=float,
        help="Target length for remaining continuous one-side sidewalk cells [m]",
    )
    parser.add_argument(
        "--sidewalk-tail-merge-length",
        default=15.0,
        type=float,
        help="Merge final short sidewalk fragments shorter than this length into the previous chunk [m]",
    )
    parser.add_argument(
        "--sidewalk-min-longitudinal-length",
        default=30.0,
        type=float,
        help="Merge sidewalk chunks shorter than this longitudinal length into adjacent chunks when possible [m]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TOPOLOGY_ROOT,
        help="Root output directory. Results are written to <output-dir>/<map>/<tile_id>/.",
    )
    parser.add_argument(
        "--route-portal-merge-radius",
        default=6.0,
        type=float,
        help="Merge nearby boundary nodes into one crossing portal when they stay connected by sidewalk within this radius [m]",
    )
    parser.add_argument(
        "--route-portal-sidewalk-merge-length",
        default=20.0,
        type=float,
        help="Maximum local sidewalk path length used to merge crossing boundary nodes into one portal [m]",
    )
    parser.add_argument(
        "--obstacle-labels",
        nargs="+",
        default=list(DEFAULT_STATIC_OBSTACLE_LABELS),
        help="CARLA city object label groups treated as static obstacle candidates",
    )
    parser.add_argument(
        "--obstacle-max-bottom-z",
        default=DEFAULT_STATIC_OBSTACLE_MAX_BOTTOM_Z_M,
        type=float,
        help="Keep only obstacle footprints whose bottom clearance above nearby ground is at or below this threshold [m]",
    )
    parser.add_argument(
        "--obstacle-min-area",
        default=DEFAULT_STATIC_OBSTACLE_MIN_AREA_M2,
        type=float,
        help="Discard obstacle footprints smaller than this area [m^2]",
    )
    parser.add_argument(
        "--obstacle-min-span",
        default=DEFAULT_STATIC_OBSTACLE_MIN_SPAN_M,
        type=float,
        help="Discard obstacle footprints whose minimum x/y span is below this threshold [m]",
    )
    parser.add_argument(
        "--obstacle-footprint-buffer",
        default=DEFAULT_STATIC_OBSTACLE_FOOTPRINT_BUFFER_M,
        type=float,
        help="Small geometry robustness buffer applied to each obstacle footprint [m]",
    )
    parser.add_argument(
        "--vegetation-max-height",
        default=DEFAULT_VEGETATION_MAX_HEIGHT_M,
        type=float,
        help="Vegetation taller than this is ignored to avoid tree-canopy false positives [m]",
    )
    parser.add_argument(
        "--edge-corridor-side-margin",
        default=DEFAULT_EDGE_CORRIDOR_SIDE_MARGIN_M,
        type=float,
        help="Margin trimmed from each side of an edge corridor before obstacle validation [m]",
    )
    parser.add_argument(
        "--edge-min-passage-width",
        default=DEFAULT_EDGE_MIN_PASSAGE_WIDTH_M,
        type=float,
        help="Required minimum free-space width inside an edge corridor [m]",
    )
    parser.add_argument(
        "--large-map-safe",
        action="store_true",
        help=(
            "Use a large-map safe loading path: request minimal map layers, disable rendering, "
            "and shrink large-map streaming distances before topology extraction."
        ),
    )
    parser.add_argument(
        "--skip-static-obstacles",
        action="store_true",
        help="Skip static environment object extraction and obstacle pruning during tile export.",
    )
    parser.add_argument(
        "--tile-catalog",
        type=Path,
        default=None,
        help="Optional output path for the generated tile catalog JSON. Defaults to <output-dir>/<map>/tile_catalog.json.",
    )
    parser.add_argument(
        "--tile-id",
        default=None,
        help="Optional tile id to restrict processing after the tile catalog has been generated.",
    )
    parser.add_argument(
        "--tile-border-resolution",
        default=2.0,
        type=float,
        help="Sampling resolution used when extracting offline sidewalk borders [m].",
    )
    parser.add_argument(
        "--coarse-grid-size",
        default=100.0,
        type=float,
        help="Coarse occupancy grid size used to eliminate non-sidewalk regions before tiling [m].",
    )
    parser.add_argument(
        "--tile-size",
        default=400.0,
        type=float,
        help="Tile core size [m].",
    )
    parser.add_argument(
        "--tile-halo",
        default=80.0,
        type=float,
        help="Halo margin added around each tile core during local topology extraction [m].",
    )
    parser.add_argument(
        "--tile-focus-height",
        default=220.0,
        type=float,
        help="Spectator height used to focus each large-map tile before extraction [m].",
    )
    return parser.parse_args()


def load_or_build_tile_catalog(
    *,
    map_name: str,
    requested_map_name: str,
    xodr_path: Path,
    tile_catalog_path: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if tile_catalog_path.exists():
        catalog = load_tile_catalog(tile_catalog_path)
        if canonical_map_name(str(catalog.get("map_name", ""))) == map_name:
            return catalog

    catalog = build_tile_catalog(
        map_name=map_name,
        requested_map_name=requested_map_name,
        xodr_path=xodr_path,
        border_resolution=args.tile_border_resolution,
        coarse_grid_size=args.coarse_grid_size,
        tile_size=args.tile_size,
        tile_halo=args.tile_halo,
    )
    write_tile_catalog_json(catalog, tile_catalog_path)
    return catalog


def process_single_tile(
    *,
    requested_map_name: str,
    map_name: str,
    tile_catalog: Dict[str, object],
    tile_catalog_path: Path,
    map_output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if args.tile_id is None:
        raise ValueError("process_single_tile requires --tile-id")

    tile_record = find_tile_record(tile_catalog, args.tile_id)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = load_world(client, requested_map_name, large_map_safe=args.large_map_safe)
    if args.large_map_safe:
        apply_large_map_safe_settings(world)
    carla_map = world.get_map()

    static_obstacle_validation_config = static_obstacle_config_from_args(args)
    summary = generate_tile_outputs(
        world=world,
        carla_map=carla_map,
        args=args,
        map_name=map_name,
        map_output_root=map_output_root,
        tile_record=tile_record,
        tile_catalog_path=tile_catalog_path,
        static_obstacle_validation_config=static_obstacle_validation_config,
        skip_static_obstacles=args.skip_static_obstacles,
    )

    print(f"map_name={map_name}")
    print(f"tile_catalog={tile_catalog_path}")
    print(f"tile_id={summary['tile_id']}")
    print(f"cell_count={summary['cell_count']}")
    print(f"crossing_cell_count={summary['crossing_cell_count']}")
    print(f"sidewalk_cell_count={summary['sidewalk_cell_count']}")
    print(f"included_edge_count={summary['included_edge_count']}")
    print(f"excluded_edge_count={summary['excluded_edge_count']}")
    print(f"static_obstacle_count={summary['static_obstacle_count']}")
    print(f"skip_static_obstacles={summary['skip_static_obstacles']}")
    print(f"json={summary['json']}")
    if summary["svg"]:
        print(f"svg={summary['svg']}")
    else:
        print("svg_skipped=true")

    return summary


def build_subprocess_command(
    *,
    script_path: Path,
    args: argparse.Namespace,
    requested_map_name: str,
    tile_catalog_path: Path,
    tile_id: str,
) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
        "--map-name",
        requested_map_name,
        "--tile-catalog",
        str(tile_catalog_path),
        "--tile-id",
        tile_id,
        "--output-dir",
        str(args.output_dir),
        "--waypoint-distance",
        str(args.waypoint_distance),
        "--approach-length",
        str(args.approach_length),
        "--sidewalk-chunk-length",
        str(args.sidewalk_chunk_length),
        "--sidewalk-tail-merge-length",
        str(args.sidewalk_tail_merge_length),
        "--sidewalk-min-longitudinal-length",
        str(args.sidewalk_min_longitudinal_length),
        "--route-portal-merge-radius",
        str(args.route_portal_merge_radius),
        "--route-portal-sidewalk-merge-length",
        str(args.route_portal_sidewalk_merge_length),
        "--tile-focus-height",
        str(args.tile_focus_height),
        "--xodr-path",
        str(args.xodr_path) if args.xodr_path is not None else "",
    ]
    if args.large_map_safe:
        command.append("--large-map-safe")
    if args.skip_static_obstacles:
        command.append("--skip-static-obstacles")
    if args.obstacle_labels:
        command.extend(["--obstacle-labels", *args.obstacle_labels])
    command.extend(
        [
            "--obstacle-max-bottom-z",
            str(args.obstacle_max_bottom_z),
            "--obstacle-min-area",
            str(args.obstacle_min_area),
            "--obstacle-min-span",
            str(args.obstacle_min_span),
            "--obstacle-footprint-buffer",
            str(args.obstacle_footprint_buffer),
            "--vegetation-max-height",
            str(args.vegetation_max_height),
            "--edge-corridor-side-margin",
            str(args.edge_corridor_side_margin),
            "--edge-min-passage-width",
            str(args.edge_min_passage_width),
        ]
    )
    if args.xodr_path is None:
        index = command.index("--xodr-path")
        del command[index : index + 2]
    return command


def main() -> None:
    args = parse_args()

    requested_map_name = args.map_name
    map_name = canonical_map_name(requested_map_name)
    map_output_root = args.output_dir / map_name
    map_output_root.mkdir(parents=True, exist_ok=True)

    xodr_path = resolve_xodr_path(requested_map_name, args.xodr_path)
    tile_catalog_path = args.tile_catalog if args.tile_catalog is not None else map_output_root / "tile_catalog.json"
    tile_catalog = load_or_build_tile_catalog(
        map_name=map_name,
        requested_map_name=requested_map_name,
        xodr_path=xodr_path,
        tile_catalog_path=tile_catalog_path,
        args=args,
    )

    if args.tile_id is not None:
        process_single_tile(
            requested_map_name=requested_map_name,
            map_name=map_name,
            tile_catalog=tile_catalog,
            tile_catalog_path=tile_catalog_path,
            map_output_root=map_output_root,
            args=args,
        )
        return

    target_tiles = list(tile_catalog["tiles"])
    # NOTICE: tile太多了, 挑几个生成
    target_tiles = target_tiles[::10]
    print(f"map_name={map_name}")
    print(f"requested_map_name={requested_map_name}")
    print(f"xodr_path={xodr_path}")
    print(f"tile_catalog={tile_catalog_path}")
    print(f"tile_count={len(tile_catalog['tiles'])}")
    print(f"target_tile_count={len(target_tiles)}")
    print("execution_mode=subprocess_per_tile")

    generation_summary: List[Dict[str, object]] = []
    failed_tiles: List[Dict[str, object]] = []
    script_path = Path(__file__).resolve()

    for index, tile_record in enumerate(target_tiles, start=1):
        tile_id = str(tile_record["id"])
        print(f"processing_tile={tile_id} ({index}/{len(target_tiles)})")
        command = build_subprocess_command(
            script_path=script_path,
            args=args,
            requested_map_name=requested_map_name,
            tile_catalog_path=tile_catalog_path,
            tile_id=tile_id,
        )
        started_at = time.time()
        completed = subprocess.run(command, check=False)
        duration_s = round(time.time() - started_at, 3)
        if completed.returncode == 0:
            tile_output_dir = map_output_root / tile_id
            summary = {
                "tile_id": tile_id,
                "returncode": completed.returncode,
                "duration_s": duration_s,
                "json": str(tile_output_dir / "walkable_cells.json"),
                "svg": str(tile_output_dir / "walkable_cells.svg"),
            }
            generation_summary.append(summary)
            print(f"tile_dispatch_status=ok tile_id={tile_id} duration_s={duration_s}")
        else:
            failed_tiles.append(
                {
                    "tile_id": tile_id,
                    "returncode": completed.returncode,
                    "duration_s": duration_s,
                }
            )
            print(f"tile_dispatch_status=failed tile_id={tile_id} returncode={completed.returncode} duration_s={duration_s}")

    summary_path = map_output_root / "generation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "map_name": map_name,
                "requested_map_name": requested_map_name,
                "xodr_path": str(xodr_path),
                "tile_catalog": str(tile_catalog_path),
                "tile_count": len(tile_catalog["tiles"]),
                "target_tile_count": len(target_tiles),
                "successful_tile_count": len(generation_summary),
                "failed_tile_count": len(failed_tiles),
                "failed_tiles": failed_tiles,
                "tiles": generation_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"successful_tile_count={len(generation_summary)}")
    print(f"failed_tile_count={len(failed_tiles)}")
    print(f"summary={summary_path}")

    if failed_tiles:
        raise RuntimeError("One or more tiles failed. See generation_summary.json for details.")


if __name__ == "__main__":
    main()
