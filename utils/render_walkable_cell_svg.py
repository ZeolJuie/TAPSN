import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import carla
import numpy as np

from generate_walkable_cells import (
    EdgeRecord,
    StaticObstacleRecord,
    bounds_for_edge_ids,
    build_edge_records,
    edge_corridor_polygon,
    extract_static_obstacle_records,
    geometry_polygon_rings,
    load_world,
    relevant_obstacles_for_bounds,
    static_obstacle_config_from_catalog,
)
from simulation.path_planner import EdgeType, PedPathPlanner


EDGE_STYLE = {
    EdgeType.SIDEWALK: {"fill": "#eadfbe", "stroke": "#8d7e56", "fill_opacity": 0.24, "stroke_opacity": 0.75},
    EdgeType.CROSSWALK: {"fill": "#d5e7f7", "stroke": "#3a6f9d", "fill_opacity": 0.4, "stroke_opacity": 0.95},
    EdgeType.JAYWALKING_JUNCTION: {
        "fill": "#f0dbe7",
        "stroke": "#8d5877",
        "fill_opacity": 0.18,
        "stroke_opacity": 0.7,
    },
}

ROUTE_STYLE = {
    "sidewalk_through": {"stroke": "#19647e", "width": 0.56, "dash": ""},
    "junction_through": {"stroke": "#4f772d", "width": 0.6, "dash": ""},
    "crosswalk_cross": {"stroke": "#d1495b", "width": 0.68, "dash": ""},
    "unmarked_junction_cross": {"stroke": "#e28f00", "width": 0.62, "dash": "1.7 1.2"},
    "junction_turn": {"stroke": "#2a9d4b", "width": 0.6, "dash": "1.0 1.0"},
    "junction_u_turn": {"stroke": "#6a4c93", "width": 0.62, "dash": "0.9 0.9"},
}

OBSTACLE_STYLE = {
    "buffered": {"fill": "#cc5c33", "fill_opacity": 0.16, "stroke": "#9f2a00", "stroke_width": 0.18},
    "blocking": {"fill": "#d62828", "fill_opacity": 0.22, "stroke": "#8f0000", "stroke_width": 0.22},
    "raw": {"stroke": "#52311f", "stroke_width": 0.18, "stroke_opacity": 0.72},
    "blocked_edge": {"fill": "#d62828", "fill_opacity": 0.14, "stroke": "#9d0208", "stroke_width": 0.22},
    "rejected_route": {"stroke": "#666666", "width": 0.52, "dash": "1.0 0.8"},
}


def world_to_canvas(
    point: Tuple[float, float], min_x: float, max_y: float, padding: float
) -> Tuple[float, float]:
    x, y = point
    return x - min_x + padding, max_y - y + padding


def expand_bounds(bounds: Dict[str, float], margin: float) -> Dict[str, float]:
    return {
        "min_x": float(bounds["min_x"]) - margin,
        "max_x": float(bounds["max_x"]) + margin,
        "min_y": float(bounds["min_y"]) - margin,
        "max_y": float(bounds["max_y"]) + margin,
    }


def edge_polygon(edge: EdgeRecord) -> List[Tuple[float, float]]:
    dx = edge.x2 - edge.x1
    dy = edge.y2 - edge.y1
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-6:
        return [(edge.x1, edge.y1)] * 4

    nx = -dy / norm
    ny = dx / norm
    half_width = max(edge.width_hint * 0.5, 0.9)
    return [
        (edge.x1 + nx * half_width, edge.y1 + ny * half_width),
        (edge.x2 + nx * half_width, edge.y2 + ny * half_width),
        (edge.x2 - nx * half_width, edge.y2 - ny * half_width),
        (edge.x1 - nx * half_width, edge.y1 - ny * half_width),
    ]


def polyline_midpoint(points: List[List[float]]) -> Tuple[float, float]:
    if len(points) == 1:
        return float(points[0][0]), float(points[0][1])

    segment_lengths = []
    total = 0.0
    for left, right in zip(points, points[1:]):
        length = float(np.hypot(right[0] - left[0], right[1] - left[1]))
        segment_lengths.append(length)
        total += length

    if total <= 1e-6:
        mid = points[len(points) // 2]
        return float(mid[0]), float(mid[1])

    target = total * 0.5
    traversed = 0.0
    for index, length in enumerate(segment_lengths):
        if traversed + length >= target:
            ratio = 0.0 if length <= 1e-6 else (target - traversed) / length
            left = points[index]
            right = points[index + 1]
            x = left[0] + (right[0] - left[0]) * ratio
            y = left[1] + (right[1] - left[1]) * ratio
            return float(x), float(y)
        traversed += length

    last = points[-1]
    return float(last[0]), float(last[1])


def svg_poly_points(
    points: Iterable[Tuple[float, float]], min_x: float, max_y: float, padding: float
) -> str:
    canvas_points = [world_to_canvas(point, min_x, max_y, padding) for point in points]
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in canvas_points)


def svg_path_points(
    points: Iterable[List[float]], min_x: float, max_y: float, padding: float
) -> str:
    canvas_points = [world_to_canvas((float(point[0]), float(point[1])), min_x, max_y, padding) for point in points]
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in canvas_points)


def svg_path_commands(
    points: Iterable[Tuple[float, float]], min_x: float, max_y: float, padding: float
) -> str:
    canvas_points = [world_to_canvas(point, min_x, max_y, padding) for point in points]
    if not canvas_points:
        return ""
    head_x, head_y = canvas_points[0]
    commands = [f"M {head_x:.3f},{head_y:.3f}"]
    commands.extend(f"L {x:.3f},{y:.3f}" for x, y in canvas_points[1:])
    commands.append("Z")
    return " ".join(commands)


def svg_geometry_path(rings: List[Dict[str, object]], min_x: float, max_y: float, padding: float) -> str:
    commands: List[str] = []
    for ring in rings:
        commands.append(svg_path_commands(ring["outer"], min_x, max_y, padding))
        for hole in ring["holes"]:
            commands.append(svg_path_commands(hole, min_x, max_y, padding))
    return " ".join(command for command in commands if command)


def route_short_id(route_id: str) -> str:
    return "R" + route_id.split("_")[-1]


def portal_short_id(portal_id: str) -> str:
    return "P" + portal_id.split("_")[-1]


def render_cell_svg(
    output_path: Path,
    map_name: str,
    cell: Dict[str, object],
    edges: Dict[int, EdgeRecord],
    obstacle_records: List[StaticObstacleRecord],
    obstacle_validation_config,
) -> None:
    core_edge_ids = [edge_info["edge_id"] for edge_info in cell["edges"]]
    support_edge_ids = cell.get("route_support", {}).get("edge_ids", core_edge_ids)
    validation = cell.get("static_obstacle_validation", {})
    blocked_edge_ids = set(validation.get("blocked_edge_ids", []))
    blocked_object_ids = {
        int(object_id)
        for blocked_edge in validation.get("blocked_edges", [])
        for object_id in blocked_edge.get("blocking_object_ids", [])
    }
    rejected_routes = validation.get("rejected_routes", [])
    support_coordinates = []
    for edge_id in support_edge_ids:
        edge = edges[edge_id]
        support_coordinates.extend([(edge.x1, edge.y1), (edge.x2, edge.y2)])
    support_bounds = bounds_for_edge_ids(support_edge_ids, edges)
    candidate_obstacles = relevant_obstacles_for_bounds(obstacle_records, support_bounds)
    bounds = expand_bounds(
        {
            "min_x": min(point[0] for point in support_coordinates),
            "max_x": max(point[0] for point in support_coordinates),
            "min_y": min(point[1] for point in support_coordinates),
            "max_y": max(point[1] for point in support_coordinates),
        },
        margin=5.0,
    )
    min_x = bounds["min_x"]
    max_x = bounds["max_x"]
    min_y = bounds["min_y"]
    max_y = bounds["max_y"]
    padding = 10.0
    canvas_width = (max_x - min_x) + 2 * padding
    canvas_height = (max_y - min_y) + 2 * padding

    pixel_width = 2200
    pixel_height = max(1200, int(pixel_width * canvas_height / max(canvas_width, 1.0)))
    route_type_counts = cell.get("route_stats", {}).get("by_type", {})
    present_route_types = [route_type for route_type in ROUTE_STYLE if route_type_counts.get(route_type, 0)]
    legend_items = 3 + len(present_route_types)
    if candidate_obstacles:
        legend_items += 2
    if blocked_edge_ids:
        legend_items += 1
    if rejected_routes:
        legend_items += 1

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_width}" height="{pixel_height}" '
            f'viewBox="0 0 {canvas_width:.3f} {canvas_height:.3f}">'
        ),
        f'<title>{cell["id"]} routes</title>',
        '<rect width="100%" height="100%" fill="#f7f5ef"/>',
    ]

    lines.append('<g id="walkable_areas">')
    core_edge_id_set = set(core_edge_ids)
    for edge_id in support_edge_ids:
        if edge_id in core_edge_id_set:
            continue
        edge = edges[edge_id]
        style = EDGE_STYLE.get(edge.edge_type, EDGE_STYLE[EdgeType.SIDEWALK])
        polygon = edge_polygon(edge)
        lines.append(
            f'<polygon points="{svg_poly_points(polygon, min_x, max_y, padding)}" '
            f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"] * 0.55:.2f}" '
            f'stroke="{style["stroke"]}" stroke-opacity="{style["stroke_opacity"] * 0.45:.2f}" stroke-width="0.32"/>'
        )
    for edge_id in core_edge_ids:
        edge = edges[edge_id]
        style = EDGE_STYLE.get(edge.edge_type, EDGE_STYLE[EdgeType.SIDEWALK])
        polygon = edge_polygon(edge)
        lines.append(
            f'<polygon points="{svg_poly_points(polygon, min_x, max_y, padding)}" '
            f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
            f'stroke="{style["stroke"]}" stroke-opacity="{style["stroke_opacity"]:.2f}" stroke-width="0.45"/>'
        )
    lines.append("</g>")

    if candidate_obstacles:
        lines.append('<g id="obstacle_buffers" fill-rule="evenodd">')
        for obstacle_record in candidate_obstacles:
            rings = geometry_polygon_rings(obstacle_record.buffered_footprint)
            if not rings:
                continue
            style = OBSTACLE_STYLE["blocking"] if obstacle_record.object_id in blocked_object_ids else OBSTACLE_STYLE["buffered"]
            lines.append(
                f'<path d="{svg_geometry_path(rings, min_x, max_y, padding)}" '
                f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
                f'stroke="{style["stroke"]}" stroke-width="{style["stroke_width"]:.2f}"/>'
            )
        lines.append("</g>")

        lines.append('<g id="obstacle_raw_footprints" fill="none">')
        for obstacle_record in candidate_obstacles:
            rings = geometry_polygon_rings(obstacle_record.raw_footprint)
            if not rings:
                continue
            lines.append(
                f'<path d="{svg_geometry_path(rings, min_x, max_y, padding)}" '
                f'stroke="{OBSTACLE_STYLE["raw"]["stroke"]}" stroke-width="{OBSTACLE_STYLE["raw"]["stroke_width"]:.2f}" '
                f'stroke-opacity="{OBSTACLE_STYLE["raw"]["stroke_opacity"]:.2f}"/>'
            )
        lines.append("</g>")

    lines.append('<g id="edge_centerlines" opacity="0.28" stroke="#222" stroke-width="0.32" stroke-linecap="round">')
    for edge_id in support_edge_ids:
        edge = edges[edge_id]
        x1, y1 = world_to_canvas((edge.x1, edge.y1), min_x, max_y, padding)
        x2, y2 = world_to_canvas((edge.x2, edge.y2), min_x, max_y, padding)
        lines.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}"/>')
    lines.append("</g>")

    if blocked_edge_ids:
        lines.append('<g id="blocked_edge_corridors" fill-rule="evenodd">')
        for edge_id in sorted(blocked_edge_ids):
            rings = geometry_polygon_rings(edge_corridor_polygon(edges[edge_id], obstacle_validation_config))
            if not rings:
                continue
            lines.append(
                f'<path d="{svg_geometry_path(rings, min_x, max_y, padding)}" '
                f'fill="{OBSTACLE_STYLE["blocked_edge"]["fill"]}" fill-opacity="{OBSTACLE_STYLE["blocked_edge"]["fill_opacity"]:.2f}" '
                f'stroke="{OBSTACLE_STYLE["blocked_edge"]["stroke"]}" stroke-width="{OBSTACLE_STYLE["blocked_edge"]["stroke_width"]:.2f}"/>'
            )
        lines.append("</g>")
        lines.append('<g id="blocked_edge_labels">')
        for edge_id in sorted(blocked_edge_ids):
            edge = edges[edge_id]
            x, y = world_to_canvas(((edge.x1 + edge.x2) * 0.5, (edge.y1 + edge.y2) * 0.5), min_x, max_y, padding)
            lines.append(
                f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="middle" dominant-baseline="middle" '
                'font-family="monospace" font-size="1.45" fill="#8f0000">'
                f'E{edge_id}</text>'
            )
        lines.append("</g>")

    if rejected_routes:
        lines.append('<g id="rejected_routes" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.92">')
        for route in rejected_routes:
            lines.append(
                f'<polyline points="{svg_path_points(route["path_waypoints"], min_x, max_y, padding)}" '
                f'stroke="{OBSTACLE_STYLE["rejected_route"]["stroke"]}" '
                f'stroke-width="{OBSTACLE_STYLE["rejected_route"]["width"]:.2f}" '
                f'stroke-dasharray="{OBSTACLE_STYLE["rejected_route"]["dash"]}"/>'
            )
        lines.append("</g>")

    lines.append('<g id="routes_halo" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.88">')
    for route in cell.get("routes", []):
        style = ROUTE_STYLE[route["route_type"]]
        lines.append(
            f'<polyline points="{svg_path_points(route["path_waypoints"], min_x, max_y, padding)}" '
            f'stroke="#ffffff" stroke-width="{style["width"] + 0.34:.2f}"/>'
        )
    lines.append("</g>")
    lines.append('<g id="routes" fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for route in cell.get("routes", []):
        style = ROUTE_STYLE[route["route_type"]]
        dash_attr = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        lines.append(
            f'<polyline points="{svg_path_points(route["path_waypoints"], min_x, max_y, padding)}" '
            f'stroke="{style["stroke"]}" stroke-width="{style["width"]:.2f}"{dash_attr}/>'
        )
    lines.append("</g>")

    lines.append('<g id="portals">')
    for portal in cell.get("portals", []):
        x, y = world_to_canvas((portal["centroid"][0], portal["centroid"][1]), min_x, max_y, padding)
        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="0.7" fill="#1d3557" stroke="#ffffff" stroke-width="0.22"/>')
    lines.append("</g>")

    meta_box_width = min(26.0, canvas_width * 0.28)
    meta_box_height = 16.2
    lines.append(
        f'<rect x="7" y="7" width="{meta_box_width:.3f}" height="{meta_box_height:.3f}" rx="2.2" fill="#ffffff" fill-opacity="0.82" stroke="#ddd7ca" stroke-width="0.45"/>'
    )
    lines.append(
        f'<text x="9" y="11.7" font-family="monospace" font-size="2.45" fill="#222">{cell["id"]}</text>'
    )
    lines.append(
        f'<text x="9" y="15.1" font-family="monospace" font-size="1.7" fill="#444">p={len(cell.get("portals", []))} r={len(cell.get("routes", []))}</text>'
    )
    lines.append(
        f'<text x="9" y="18.2" font-family="monospace" font-size="1.52" fill="#444">'
        f'obs={validation.get("candidate_obstacle_count", 0)} blocked={validation.get("blocked_edge_count", 0)}</text>'
    )
    lines.append(
        f'<text x="9" y="21.0" font-family="monospace" font-size="1.52" fill="#444">'
        f'reject={validation.get("rejected_route_count", 0)} reroute={validation.get("rerouted_route_count", 0)}</text>'
    )

    legend_x = max(8.0, canvas_width - 24.0)
    legend_y = 7.0
    legend_height = 10.5 + 4.9 * legend_items
    lines.append(
        f'<rect x="{legend_x:.3f}" y="{legend_y:.3f}" width="17" height="{legend_height:.3f}" rx="2.2" fill="#ffffff" fill-opacity="0.82" stroke="#ddd7ca" stroke-width="0.45"/>'
    )
    lines.append('<g id="legend">')
    lines.append(
        f'<text x="{legend_x + 2:.3f}" y="{legend_y + 4.2:.3f}" font-family="monospace" font-size="2.1" fill="#222">G</text>'
    )
    legend_y += 7.0
    for label, edge_type in [("SW", EdgeType.SIDEWALK), ("CW", EdgeType.CROSSWALK), ("JWJ", EdgeType.JAYWALKING_JUNCTION)]:
        style = EDGE_STYLE[edge_type]
        lines.append(
            f'<rect x="{legend_x + 2:.3f}" y="{legend_y - 2.8:.3f}" width="3.2" height="2.4" '
            f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
            f'stroke="{style["stroke"]}" stroke-width="0.3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 6.6:.3f}" y="{legend_y - 0.9:.3f}" font-family="monospace" font-size="1.7" fill="#333">'
            f'{label}</text>'
        )
        legend_y += 4.5

    legend_y += 2.0
    lines.append(
        f'<text x="{legend_x + 2:.3f}" y="{legend_y:.3f}" font-family="monospace" font-size="2.1" fill="#222">R</text>'
    )
    legend_y += 4.7
    for route_type in present_route_types:
        style = ROUTE_STYLE[route_type]
        dash_attr = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        lines.append(
            f'<line x1="{legend_x + 2:.3f}" y1="{legend_y - 1.2:.3f}" x2="{legend_x + 6.6:.3f}" y2="{legend_y - 1.2:.3f}" '
            f'stroke="#ffffff" stroke-width="{style["width"] + 0.34:.2f}"/>'
        )
        lines.append(
            f'<line x1="{legend_x + 2:.3f}" y1="{legend_y - 1.2:.3f}" x2="{legend_x + 6.6:.3f}" y2="{legend_y - 1.2:.3f}" '
            f'stroke="{style["stroke"]}" stroke-width="{style["width"]:.2f}"{dash_attr}/>'
        )
        lines.append(
            f'<text x="{legend_x + 7.9:.3f}" y="{legend_y - 0.45:.3f}" font-family="monospace" font-size="1.55" fill="#333">'
            f'{route_type_counts[route_type]}</text>'
        )
        legend_y += 4.4

    if candidate_obstacles:
        legend_y += 1.7
        lines.append(
            f'<text x="{legend_x + 2:.3f}" y="{legend_y:.3f}" font-family="monospace" font-size="2.1" fill="#222">O</text>'
        )
        legend_y += 4.5
        lines.append(
            f'<rect x="{legend_x + 2:.3f}" y="{legend_y - 2.8:.3f}" width="3.2" height="2.4" '
            f'fill="{OBSTACLE_STYLE["buffered"]["fill"]}" fill-opacity="{OBSTACLE_STYLE["buffered"]["fill_opacity"]:.2f}" '
            f'stroke="{OBSTACLE_STYLE["buffered"]["stroke"]}" stroke-width="0.3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 6.6:.3f}" y="{legend_y - 0.9:.3f}" font-family="monospace" font-size="1.7" fill="#333">Obs</text>'
        )
        legend_y += 4.4
        lines.append(
            f'<rect x="{legend_x + 2:.3f}" y="{legend_y - 2.8:.3f}" width="3.2" height="2.4" '
            f'fill="{OBSTACLE_STYLE["blocking"]["fill"]}" fill-opacity="{OBSTACLE_STYLE["blocking"]["fill_opacity"]:.2f}" '
            f'stroke="{OBSTACLE_STYLE["blocking"]["stroke"]}" stroke-width="0.3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 6.6:.3f}" y="{legend_y - 0.9:.3f}" font-family="monospace" font-size="1.7" fill="#333">BlkObs</text>'
        )
        legend_y += 4.4

    if blocked_edge_ids:
        lines.append(
            f'<rect x="{legend_x + 2:.3f}" y="{legend_y - 2.8:.3f}" width="3.2" height="2.4" '
            f'fill="{OBSTACLE_STYLE["blocked_edge"]["fill"]}" fill-opacity="{OBSTACLE_STYLE["blocked_edge"]["fill_opacity"]:.2f}" '
            f'stroke="{OBSTACLE_STYLE["blocked_edge"]["stroke"]}" stroke-width="0.3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 6.6:.3f}" y="{legend_y - 0.9:.3f}" font-family="monospace" font-size="1.62" fill="#333">BlkEdge</text>'
        )
        legend_y += 4.4

    if rejected_routes:
        lines.append(
            f'<line x1="{legend_x + 2:.3f}" y1="{legend_y - 1.2:.3f}" x2="{legend_x + 6.6:.3f}" y2="{legend_y - 1.2:.3f}" '
            f'stroke="{OBSTACLE_STYLE["rejected_route"]["stroke"]}" stroke-width="{OBSTACLE_STYLE["rejected_route"]["width"]:.2f}" '
            f'stroke-dasharray="{OBSTACLE_STYLE["rejected_route"]["dash"]}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 7.2:.3f}" y="{legend_y - 0.45:.3f}" font-family="monospace" font-size="1.55" fill="#333">Rej</text>'
        )
    lines.append("</g>")

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one walkable cell with its supported routes as an SVG.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA port")
    parser.add_argument("--timeout", default=180.0, type=float, help="CARLA RPC timeout in seconds")
    parser.add_argument("--map-name", default="Town10HD_Opt", help="CARLA map name")
    parser.add_argument("--cell-id", required=True, help="Cell id from walkable_cells.json, e.g. crossing_002")
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="Path to walkable_cells.json (defaults to topology/<map_name>/walkable_cells.json)",
    )
    parser.add_argument(
        "--waypoint-distance",
        default=5.0,
        type=float,
        help="Waypoint spacing used when reconstructing the pedestrian topology [m]",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path (defaults to artifacts/<map_name>/topology/<cell_id>_routes.svg)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    catalog_path = args.catalog_path or Path("topology") / args.map_name / "walkable_cells.json"
    output_path = args.output or Path("artifacts") / args.map_name / "topology" / f"{args.cell_id}_routes.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    try:
        cell = next(cell for cell in catalog["cells"] if cell["id"] == args.cell_id)
    except StopIteration as exc:
        raise SystemExit(f"Cell {args.cell_id!r} not found in {catalog_path}") from exc

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = load_world(client, args.map_name)
    carla_map = world.get_map()
    path_planner = PedPathPlanner(carla_map, waypoint_distance=args.waypoint_distance)
    edges, _ = build_edge_records(path_planner)
    obstacle_validation_config = static_obstacle_config_from_catalog(catalog)
    obstacle_records = extract_static_obstacle_records(world, obstacle_validation_config)

    render_cell_svg(
        output_path=output_path,
        map_name=args.map_name,
        cell=cell,
        edges=edges,
        obstacle_records=obstacle_records,
        obstacle_validation_config=obstacle_validation_config,
    )
    print(f"cell_id={args.cell_id}")
    print(f"routes={len(cell.get('routes', []))}")
    print(f"blocked_edges={len(cell.get('static_obstacle_validation', {}).get('blocked_edge_ids', []))}")
    print(f"svg={output_path}")
