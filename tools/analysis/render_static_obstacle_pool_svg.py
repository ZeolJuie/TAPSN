import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import carla
import numpy as np

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


from generate_walkable_cells import (
    bbox_intersects,
    build_edge_records,
    extract_static_obstacle_records,
    geometry_polygon_rings,
    load_world,
    static_obstacle_config_from_catalog,
    static_obstacle_config_from_args,
)
from simulation.path_planner import EdgeType, PedPathPlanner


EDGE_STYLE = {
    EdgeType.SIDEWALK: {"fill": "#eadfbe", "stroke": "#8d7e56", "fill_opacity": 0.18, "stroke_opacity": 0.45},
    EdgeType.CROSSWALK: {"fill": "#d5e7f7", "stroke": "#3a6f9d", "fill_opacity": 0.28, "stroke_opacity": 0.65},
    EdgeType.JAYWALKING_JUNCTION: {
        "fill": "#f0dbe7",
        "stroke": "#8d5877",
        "fill_opacity": 0.16,
        "stroke_opacity": 0.42,
    },
}

OBSTACLE_STYLE = {
    "Walls": {"fill": "#3d405b", "stroke": "#22223b", "fill_opacity": 0.28},
    "Fences": {"fill": "#277da1", "stroke": "#1d3557", "fill_opacity": 0.25},
    "Static": {"fill": "#f77f00", "stroke": "#bc6c25", "fill_opacity": 0.24},
    "Other": {"fill": "#d62828", "stroke": "#9d0208", "fill_opacity": 0.24},
    "Vegetation": {"fill": "#588157", "stroke": "#344e41", "fill_opacity": 0.22},
}


def world_to_canvas(
    point: Tuple[float, float], min_x: float, max_y: float, padding: float
) -> Tuple[float, float]:
    x, y = point
    return x - min_x + padding, max_y - y + padding


def edge_polygon_points(edge) -> List[Tuple[float, float]]:
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


def svg_poly_points(
    points: Iterable[Tuple[float, float]], min_x: float, max_y: float, padding: float
) -> str:
    canvas_points = [world_to_canvas(point, min_x, max_y, padding) for point in points]
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


def svg_text(text: object) -> str:
    return html.escape(str(text), quote=False)


def expanded_bounds(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    margin: float,
) -> Tuple[float, float, float, float]:
    return (
        min_x - margin,
        min_y - margin,
        max_x + margin,
        max_y + margin,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the global static obstacle candidate pool on top of the CARLA pedestrian topology."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA port")
    parser.add_argument("--timeout", default=180.0, type=float, help="CARLA RPC timeout in seconds")
    parser.add_argument("--map-name", default="Town10HD_Opt", help="CARLA map name")
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="Optional walkable_cells.json used to load the exact static obstacle validation rules",
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
        help="Output SVG path (defaults to tmp/static_obstacle_debug/<map_name>/global_candidate_obstacles.svg)",
    )
    parser.add_argument(
        "--walkable-bounds-margin",
        default=20.0,
        type=float,
        help="Only render candidate obstacles whose buffered bounds intersect the global walkable topology bounds expanded by this margin [m]",
    )
    parser.add_argument(
        "--obstacle-labels",
        nargs="+",
        default=None,
        help="Override obstacle labels when no catalog rules are provided",
    )
    parser.add_argument(
        "--obstacle-max-bottom-z",
        default=None,
        type=float,
        help="Override max obstacle bottom z when no catalog rules are provided [m]",
    )
    parser.add_argument(
        "--obstacle-min-area",
        default=None,
        type=float,
        help="Override minimum footprint area when no catalog rules are provided [m^2]",
    )
    parser.add_argument(
        "--obstacle-min-span",
        default=None,
        type=float,
        help="Override minimum footprint span when no catalog rules are provided [m]",
    )
    parser.add_argument(
        "--obstacle-footprint-buffer",
        default=None,
        type=float,
        help="Override footprint robustness buffer when no catalog rules are provided [m]",
    )
    parser.add_argument(
        "--vegetation-max-height",
        default=None,
        type=float,
        help="Override vegetation height filter when no catalog rules are provided [m]",
    )
    parser.add_argument(
        "--edge-corridor-side-margin",
        default=None,
        type=float,
        help="Unused here, but accepted so the same ruleset can be reproduced from CLI",
    )
    parser.add_argument(
        "--edge-min-passage-width",
        default=None,
        type=float,
        help="Unused here, but accepted so the same ruleset can be reproduced from CLI",
    )
    return parser.parse_args()


def resolved_config(args: argparse.Namespace):
    if args.catalog_path is not None and args.catalog_path.exists():
        catalog = json.loads(args.catalog_path.read_text(encoding="utf-8"))
        return static_obstacle_config_from_catalog(catalog)

    if args.obstacle_labels is None:
        args.obstacle_labels = ["Walls", "Fences", "Static", "Other", "Vegetation", "Poles", "Vehicles"]
    if args.obstacle_max_bottom_z is None:
        args.obstacle_max_bottom_z = 0.35
    if args.obstacle_min_area is None:
        args.obstacle_min_area = 0.08
    if args.obstacle_min_span is None:
        args.obstacle_min_span = 0.12
    if args.obstacle_footprint_buffer is None:
        args.obstacle_footprint_buffer = 0.1
    if args.vegetation_max_height is None:
        args.vegetation_max_height = 2.5
    if args.edge_corridor_side_margin is None:
        args.edge_corridor_side_margin = 0.15
    if args.edge_min_passage_width is None:
        args.edge_min_passage_width = 0.9
    return static_obstacle_config_from_args(args)


def main() -> None:
    args = parse_args()
    output_path = args.output or Path("tmp") / "static_obstacle_debug" / args.map_name / "global_candidate_obstacles.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = load_world(client, args.map_name)
    carla_map = world.get_map()
    path_planner = PedPathPlanner(carla_map, waypoint_distance=args.waypoint_distance)
    edges, _ = build_edge_records(path_planner)

    obstacle_config = resolved_config(args)
    all_obstacle_records = extract_static_obstacle_records(world, obstacle_config)

    included_edges = [
        edge
        for edge in edges.values()
        if edge.edge_type in {EdgeType.SIDEWALK, EdgeType.CROSSWALK, EdgeType.JAYWALKING_JUNCTION}
    ]
    if not included_edges:
        raise RuntimeError("No walkable edges found for rendering.")

    walkable_min_x = min(min(edge.x1, edge.x2) for edge in included_edges)
    walkable_max_x = max(max(edge.x1, edge.x2) for edge in included_edges)
    walkable_min_y = min(min(edge.y1, edge.y2) for edge in included_edges)
    walkable_max_y = max(max(edge.y1, edge.y2) for edge in included_edges)
    walkable_bounds = expanded_bounds(
        walkable_min_x,
        walkable_min_y,
        walkable_max_x,
        walkable_max_y,
        max(0.0, float(args.walkable_bounds_margin)),
    )

    obstacle_records = [
        obstacle_record
        for obstacle_record in all_obstacle_records
        if bbox_intersects(obstacle_record.bounds, walkable_bounds)
    ]

    coordinates: List[Tuple[float, float]] = []
    for edge in included_edges:
        coordinates.extend([(edge.x1, edge.y1), (edge.x2, edge.y2)])
    for obstacle_record in obstacle_records:
        min_x, min_y, max_x, max_y = obstacle_record.buffered_footprint.bounds
        coordinates.extend([(min_x, min_y), (max_x, max_y)])

    min_x = min(point[0] for point in coordinates)
    max_x = max(point[0] for point in coordinates)
    min_y = min(point[1] for point in coordinates)
    max_y = max(point[1] for point in coordinates)
    padding = 18.0
    canvas_width = (max_x - min_x) + 2 * padding
    canvas_height = (max_y - min_y) + 2 * padding

    pixel_width = 2800
    pixel_height = max(1500, int(pixel_width * canvas_height / max(canvas_width, 1.0)))

    obstacle_counts: Dict[str, int] = defaultdict(int)
    for obstacle_record in obstacle_records:
        obstacle_counts[obstacle_record.label_name] += 1

    meta_line_1 = (
        f"candidate_obstacles={len(obstacle_records)} "
        f"raw_pool={len(all_obstacle_records)} labels={','.join(obstacle_config.label_names)}"
    )
    meta_line_2 = (
        f"max_bottom_z={obstacle_config.max_bottom_z_m:.2f} "
        f"area>={obstacle_config.min_footprint_area_m2:.2f} "
        f"span>={obstacle_config.min_footprint_span_m:.2f}"
    )
    meta_line_3 = (
        f"buffer={obstacle_config.footprint_buffer_m:.2f} "
        f"vegetation_h<={obstacle_config.vegetation_max_height_m:.2f} "
        f"walkable_margin={max(0.0, float(args.walkable_bounds_margin)):.1f}"
    )

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_width}" height="{pixel_height}" '
            f'viewBox="0 0 {canvas_width:.3f} {canvas_height:.3f}">'
        ),
        f'<title>{args.map_name} global static obstacle candidate pool</title>',
        '<rect width="100%" height="100%" fill="#f7f5ef"/>',
        '<g id="walkable_areas">',
    ]

    for edge in included_edges:
        style = EDGE_STYLE.get(edge.edge_type, EDGE_STYLE[EdgeType.SIDEWALK])
        lines.append(
            f'<polygon points="{svg_poly_points(edge_polygon_points(edge), min_x, max_y, padding)}" '
            f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
            f'stroke="{style["stroke"]}" stroke-opacity="{style["stroke_opacity"]:.2f}" stroke-width="0.32"/>'
        )
    lines.append("</g>")

    lines.append('<g id="edge_centerlines" opacity="0.16" stroke="#222" stroke-width="0.18" stroke-linecap="round">')
    for edge in included_edges:
        x1, y1 = world_to_canvas((edge.x1, edge.y1), min_x, max_y, padding)
        x2, y2 = world_to_canvas((edge.x2, edge.y2), min_x, max_y, padding)
        lines.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}"/>')
    lines.append("</g>")

    lines.append('<g id="candidate_obstacle_buffers" fill-rule="evenodd">')
    for obstacle_record in obstacle_records:
        rings = geometry_polygon_rings(obstacle_record.buffered_footprint)
        if not rings:
            continue
        style = OBSTACLE_STYLE.get(obstacle_record.label_name, OBSTACLE_STYLE["Other"])
        lines.append(
            f'<path d="{svg_geometry_path(rings, min_x, max_y, padding)}" '
            f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
            f'stroke="{style["stroke"]}" stroke-width="0.18"/>'
        )
    lines.append("</g>")

    lines.append('<g id="candidate_obstacle_raw" fill="none">')
    for obstacle_record in obstacle_records:
        rings = geometry_polygon_rings(obstacle_record.raw_footprint)
        if not rings:
            continue
        style = OBSTACLE_STYLE.get(obstacle_record.label_name, OBSTACLE_STYLE["Other"])
        lines.append(
            f'<path d="{svg_geometry_path(rings, min_x, max_y, padding)}" '
            f'stroke="{style["stroke"]}" stroke-width="0.14" stroke-opacity="0.8"/>'
        )
    lines.append("</g>")

    meta_box_width = min(52.0, canvas_width * 0.32)
    meta_box_height = 20.0
    lines.append(
        f'<rect x="8" y="8" width="{meta_box_width:.3f}" height="{meta_box_height:.3f}" rx="2.4" '
        'fill="#ffffff" fill-opacity="0.88" stroke="#ddd7ca" stroke-width="0.45"/>'
    )
    lines.append(
        f'<text x="11" y="13.0" font-family="monospace" font-size="2.4" fill="#222">{svg_text(args.map_name)}</text>'
    )
    lines.append(
        f'<text x="11" y="16.2" font-family="monospace" font-size="1.65" fill="#444">'
        f'{svg_text(meta_line_1)}</text>'
    )
    lines.append(
        f'<text x="11" y="19.0" font-family="monospace" font-size="1.5" fill="#444">'
        f'{svg_text(meta_line_2)}</text>'
    )
    lines.append(
        f'<text x="11" y="21.8" font-family="monospace" font-size="1.5" fill="#444">'
        f'{svg_text(meta_line_3)}</text>'
    )

    legend_x = max(8.0, canvas_width - 34.0)
    legend_y = 8.0
    legend_rows = 3 + len(obstacle_counts)
    legend_height = 8.5 + 4.6 * legend_rows
    lines.append(
        f'<rect x="{legend_x:.3f}" y="{legend_y:.3f}" width="26" height="{legend_height:.3f}" rx="2.4" '
        'fill="#ffffff" fill-opacity="0.88" stroke="#ddd7ca" stroke-width="0.45"/>'
    )
    lines.append('<g id="legend">')
    lines.append(
        f'<text x="{legend_x + 2:.3f}" y="{legend_y + 3.8:.3f}" font-family="monospace" font-size="2.0" fill="#222">{svg_text("Walkable")}</text>'
    )
    legend_y += 7.2
    for label, edge_type in [("SW", EdgeType.SIDEWALK), ("CW", EdgeType.CROSSWALK), ("JWJ", EdgeType.JAYWALKING_JUNCTION)]:
        style = EDGE_STYLE[edge_type]
        lines.append(
            f'<rect x="{legend_x + 2:.3f}" y="{legend_y - 2.5:.3f}" width="3.0" height="2.2" '
            f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
            f'stroke="{style["stroke"]}" stroke-width="0.28"/>'
        )
        lines.append(
            f'<text x="{legend_x + 6.0:.3f}" y="{legend_y - 0.8:.3f}" font-family="monospace" font-size="1.65" fill="#333">{svg_text(label)}</text>'
        )
        legend_y += 4.2

    legend_y += 1.2
    lines.append(
        f'<text x="{legend_x + 2:.3f}" y="{legend_y:.3f}" font-family="monospace" font-size="2.0" fill="#222">{svg_text("Obstacle Pool")}</text>'
    )
    legend_y += 4.8
    for label_name in sorted(obstacle_counts):
        style = OBSTACLE_STYLE.get(label_name, OBSTACLE_STYLE["Other"])
        count = obstacle_counts[label_name]
        lines.append(
            f'<rect x="{legend_x + 2:.3f}" y="{legend_y - 2.5:.3f}" width="3.0" height="2.2" '
            f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
            f'stroke="{style["stroke"]}" stroke-width="0.28"/>'
        )
        lines.append(
            f'<text x="{legend_x + 6.0:.3f}" y="{legend_y - 0.8:.3f}" font-family="monospace" font-size="1.58" fill="#333">'
            f'{svg_text(f"{label_name}={count}")}</text>'
        )
        legend_y += 4.2
    lines.append("</g>")

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"map_name={args.map_name}")
    print(f"candidate_obstacle_count={len(obstacle_records)}")
    print(f"raw_candidate_obstacle_count={len(all_obstacle_records)}")
    print("candidate_obstacle_counts_by_label=" + json.dumps(dict(sorted(obstacle_counts.items()))))
    print(f"svg={output_path}")


if __name__ == "__main__":
    main()
