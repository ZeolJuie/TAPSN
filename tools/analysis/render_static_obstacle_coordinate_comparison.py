import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import carla
import numpy as np
from shapely.geometry import Polygon

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


from generate_walkable_cells import (
    STATIC_OBSTACLE_LABEL_ALIASES,
    bbox_intersects,
    build_edge_records,
    environment_object_bbox_center_polygon,
    environment_object_footprint_polygon,
    environment_object_transform_bbox_polygon,
    environment_object_world_vertex_footprint_polygon,
    extract_static_obstacle_records,
    geometry_polygon_rings,
    load_world,
    oriented_box_polygon,
    static_obstacle_config_from_catalog,
)
from simulation.path_planner import EdgeType, PedPathPlanner


EDGE_STYLE = {
    EdgeType.SIDEWALK: {"fill": "#eadfbe", "stroke": "#8d7e56", "fill_opacity": 0.16, "stroke_opacity": 0.42},
    EdgeType.CROSSWALK: {"fill": "#d5e7f7", "stroke": "#3a6f9d", "fill_opacity": 0.26, "stroke_opacity": 0.62},
    EdgeType.JAYWALKING_JUNCTION: {
        "fill": "#f0dbe7",
        "stroke": "#8d5877",
        "fill_opacity": 0.14,
        "stroke_opacity": 0.36,
    },
}

OBSTACLE_STYLE = {
    "Walls": {"fill": "#3d405b", "stroke": "#22223b", "fill_opacity": 0.26},
    "Fences": {"fill": "#277da1", "stroke": "#1d3557", "fill_opacity": 0.22},
    "Static": {"fill": "#f77f00", "stroke": "#bc6c25", "fill_opacity": 0.22},
    "Other": {"fill": "#d62828", "stroke": "#9d0208", "fill_opacity": 0.22},
    "Vegetation": {"fill": "#588157", "stroke": "#344e41", "fill_opacity": 0.20},
}

PANEL_DEFINITIONS = [
    {
        "key": "world_vertices",
        "title": "A: get_world_vertices(env_object.transform)",
        "subtitle": "Current implementation",
    },
    {
        "key": "transform_center",
        "title": "B: env_object.transform + bbox.extent",
        "subtitle": "Actor transform center",
    },
    {
        "key": "bbox_center",
        "title": "C: bounding_box.location/rotation + extent",
        "subtitle": "BBox local center as world",
    },
]


def svg_text(value: object) -> str:
    return html.escape(str(value), quote=False)


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


def world_to_canvas(
    point: Tuple[float, float],
    min_x: float,
    max_y: float,
    padding: float,
    x_offset: float,
    y_offset: float,
) -> Tuple[float, float]:
    x, y = point
    return x_offset + (x - min_x + padding), y_offset + (max_y - y + padding)


def svg_poly_points(
    points: Iterable[Tuple[float, float]],
    min_x: float,
    max_y: float,
    padding: float,
    x_offset: float,
    y_offset: float,
) -> str:
    canvas_points = [world_to_canvas(point, min_x, max_y, padding, x_offset, y_offset) for point in points]
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in canvas_points)


def svg_path_commands(
    points: Iterable[Tuple[float, float]],
    min_x: float,
    max_y: float,
    padding: float,
    x_offset: float,
    y_offset: float,
) -> str:
    canvas_points = [world_to_canvas(point, min_x, max_y, padding, x_offset, y_offset) for point in points]
    if not canvas_points:
        return ""
    head_x, head_y = canvas_points[0]
    commands = [f"M {head_x:.3f},{head_y:.3f}"]
    commands.extend(f"L {x:.3f},{y:.3f}" for x, y in canvas_points[1:])
    commands.append("Z")
    return " ".join(commands)


def svg_geometry_path(
    rings: List[Dict[str, object]],
    min_x: float,
    max_y: float,
    padding: float,
    x_offset: float,
    y_offset: float,
) -> str:
    commands: List[str] = []
    for ring in rings:
        commands.append(svg_path_commands(ring["outer"], min_x, max_y, padding, x_offset, y_offset))
        for hole in ring["holes"]:
            commands.append(svg_path_commands(hole, min_x, max_y, padding, x_offset, y_offset))
    return " ".join(command for command in commands if command)


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


def panel_footprint_map(
    objects_by_id: Dict[int, object],
    object_ids: Sequence[int],
    footprint_buffer_m: float,
) -> Dict[str, Dict[int, Polygon]]:
    panel_polygons: Dict[str, Dict[int, Polygon]] = defaultdict(dict)
    for object_id in object_ids:
        env_object = objects_by_id[object_id]
        world_poly, _, _ = environment_object_world_vertex_footprint_polygon(env_object)
        transform_poly, _, _ = environment_object_transform_bbox_polygon(env_object)
        bbox_poly, _, _ = environment_object_bbox_center_polygon(env_object)

        panel_polygons["world_vertices"][object_id] = world_poly.buffer(footprint_buffer_m, resolution=8).buffer(0.0)
        panel_polygons["transform_center"][object_id] = (
            transform_poly.buffer(footprint_buffer_m, resolution=8).buffer(0.0)
        )
        panel_polygons["bbox_center"][object_id] = bbox_poly.buffer(footprint_buffer_m, resolution=8).buffer(0.0)
    return panel_polygons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a three-panel comparison of static obstacle coordinate interpretations."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA port")
    parser.add_argument("--timeout", default=180.0, type=float, help="CARLA RPC timeout in seconds")
    parser.add_argument("--map-name", default="Town10HD_Opt", help="CARLA map name")
    parser.add_argument(
        "--catalog-path",
        type=Path,
        required=True,
        help="walkable_cells.json used to load the current static obstacle validation rules",
    )
    parser.add_argument(
        "--waypoint-distance",
        default=5.0,
        type=float,
        help="Waypoint spacing used when reconstructing the pedestrian topology [m]",
    )
    parser.add_argument(
        "--walkable-bounds-margin",
        default=20.0,
        type=float,
        help="Expand the global walkable topology bounds by this margin before selecting objects for comparison [m]",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path (defaults to tmp/static_obstacle_debug/<map_name>/coordinate_comparison.svg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or Path("tmp") / "static_obstacle_debug" / args.map_name / "coordinate_comparison.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(args.catalog_path.read_text(encoding="utf-8"))
    obstacle_config = static_obstacle_config_from_catalog(catalog)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = load_world(client, args.map_name)
    carla_map = world.get_map()
    path_planner = PedPathPlanner(carla_map, waypoint_distance=args.waypoint_distance)
    edges, _ = build_edge_records(path_planner)

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

    candidate_records = extract_static_obstacle_records(world, obstacle_config)
    candidate_ids = sorted(record.object_id for record in candidate_records if bbox_intersects(record.bounds, walkable_bounds))
    label_by_id = {record.object_id: record.label_name for record in candidate_records if record.object_id in candidate_ids}

    objects_by_id: Dict[int, object] = {}
    for canonical_label_name in obstacle_config.label_names:
        aliases = STATIC_OBSTACLE_LABEL_ALIASES.get(canonical_label_name, (canonical_label_name,))
        for source_label_name in aliases:
            label = getattr(carla.CityObjectLabel, source_label_name, None)
            if label is None:
                continue
            for env_object in world.get_environment_objects(label):
                object_id = int(env_object.id)
                if object_id in label_by_id:
                    objects_by_id[object_id] = env_object

    missing_object_ids = [object_id for object_id in candidate_ids if object_id not in objects_by_id]
    if missing_object_ids:
        raise RuntimeError(f"Failed to resolve {len(missing_object_ids)} candidate objects from CARLA.")

    panel_polygons = panel_footprint_map(objects_by_id, candidate_ids, obstacle_config.footprint_buffer_m)

    coordinates: List[Tuple[float, float]] = []
    for edge in included_edges:
        coordinates.extend(((edge.x1, edge.y1), (edge.x2, edge.y2)))
    for polygons in panel_polygons.values():
        for polygon in polygons.values():
            min_x, min_y, max_x, max_y = polygon.bounds
            coordinates.extend(((min_x, min_y), (max_x, max_y)))

    min_x = min(point[0] for point in coordinates)
    max_x = max(point[0] for point in coordinates)
    min_y = min(point[1] for point in coordinates)
    max_y = max(point[1] for point in coordinates)

    padding = 16.0
    panel_gap = 14.0
    panel_width = (max_x - min_x) + 2 * padding
    panel_height = (max_y - min_y) + 2 * padding

    canvas_width = panel_width * len(PANEL_DEFINITIONS) + panel_gap * (len(PANEL_DEFINITIONS) - 1)
    header_height = 34.0
    footer_height = 18.0
    canvas_height = header_height + panel_height + footer_height

    pixel_width = 4800
    pixel_height = max(1800, int(pixel_width * canvas_height / max(canvas_width, 1.0)))

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_width}" height="{pixel_height}" '
            f'viewBox="0 0 {canvas_width:.3f} {canvas_height:.3f}">'
        ),
        f'<title>{svg_text(args.map_name)} static obstacle coordinate comparison</title>',
        '<rect width="100%" height="100%" fill="#f7f5ef"/>',
    ]

    top_title = (
        f"{args.map_name} static obstacle coordinate comparison | "
        f"objects={len(candidate_ids)} | raw_pool={len(candidate_records)} | "
        f"walkable_margin={max(0.0, float(args.walkable_bounds_margin)):.1f}m"
    )
    lines.append(
        f'<text x="10" y="10.5" font-family="monospace" font-size="3.1" fill="#222">{svg_text(top_title)}</text>'
    )
    lines.append(
        f'<text x="10" y="15.0" font-family="monospace" font-size="1.75" fill="#444">'
        f'{svg_text("Each panel uses the same object ids, same walkable topology, and same world-coordinate view extent.")}</text>'
    )

    walkable_bounds_points = [
        (walkable_bounds[0], walkable_bounds[1]),
        (walkable_bounds[2], walkable_bounds[1]),
        (walkable_bounds[2], walkable_bounds[3]),
        (walkable_bounds[0], walkable_bounds[3]),
    ]

    panel_label_counts: Dict[str, Dict[str, int]] = {}
    panel_hits_walkable: Dict[str, int] = {}
    for panel in PANEL_DEFINITIONS:
        counts: Dict[str, int] = defaultdict(int)
        hits = 0
        for object_id, polygon in panel_polygons[panel["key"]].items():
            counts[label_by_id[object_id]] += 1
            if bbox_intersects(tuple(float(value) for value in polygon.bounds), walkable_bounds):
                hits += 1
        panel_label_counts[panel["key"]] = dict(sorted(counts.items()))
        panel_hits_walkable[panel["key"]] = hits

    for panel_index, panel in enumerate(PANEL_DEFINITIONS):
        x_offset = panel_index * (panel_width + panel_gap)
        y_offset = header_height
        lines.append(
            f'<rect x="{x_offset:.3f}" y="{y_offset:.3f}" width="{panel_width:.3f}" height="{panel_height:.3f}" '
            'rx="2.6" fill="#ffffff" fill-opacity="0.55" stroke="#ddd7ca" stroke-width="0.45"/>'
        )
        lines.append(
            f'<text x="{x_offset + 6:.3f}" y="{y_offset - 16:.3f}" font-family="monospace" font-size="2.45" fill="#222">'
            f'{svg_text(panel["title"])}</text>'
        )
        lines.append(
            f'<text x="{x_offset + 6:.3f}" y="{y_offset - 12.0:.3f}" font-family="monospace" font-size="1.55" fill="#555">'
            f'{svg_text(panel["subtitle"])}</text>'
        )
        meta_text = (
            f"intersects_walkable_bounds={panel_hits_walkable[panel['key']]}/{len(candidate_ids)} "
            f"labels={panel_label_counts[panel['key']]}"
        )
        lines.append(
            f'<text x="{x_offset + 6:.3f}" y="{y_offset - 8.0:.3f}" font-family="monospace" font-size="1.35" fill="#555">'
            f'{svg_text(meta_text)}</text>'
        )

        lines.append(f'<g id="{panel["key"]}_walkable">')
        for edge in included_edges:
            style = EDGE_STYLE.get(edge.edge_type, EDGE_STYLE[EdgeType.SIDEWALK])
            lines.append(
                f'<polygon points="{svg_poly_points(edge_polygon_points(edge), min_x, max_y, padding, x_offset, y_offset)}" '
                f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
                f'stroke="{style["stroke"]}" stroke-opacity="{style["stroke_opacity"]:.2f}" stroke-width="0.28"/>'
            )
        lines.append("</g>")

        lines.append(
            f'<g id="{panel["key"]}_walkable_bounds" fill="none" stroke="#b5179e" stroke-width="0.55" stroke-dasharray="2.2 1.4">'
        )
        lines.append(
            f'<polygon points="{svg_poly_points(walkable_bounds_points, min_x, max_y, padding, x_offset, y_offset)}"/>'
        )
        lines.append("</g>")

        lines.append(
            f'<g id="{panel["key"]}_obstacles" fill-rule="evenodd" stroke-linejoin="round" stroke-linecap="round">'
        )
        for object_id in candidate_ids:
            polygon = panel_polygons[panel["key"]][object_id]
            rings = geometry_polygon_rings(polygon)
            if not rings:
                continue
            style = OBSTACLE_STYLE.get(label_by_id[object_id], OBSTACLE_STYLE["Other"])
            lines.append(
                f'<path d="{svg_geometry_path(rings, min_x, max_y, padding, x_offset, y_offset)}" '
                f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" '
                f'stroke="{style["stroke"]}" stroke-width="0.12"/>'
            )
        lines.append("</g>")

    legend_x = 10.0
    legend_y = canvas_height - 12.0
    legend_entries = ["SW", "CW", "JWJ", "Walls", "Fences", "Static", "Other", "Vegetation", "walkable_bounds"]
    legend_width = 5.5
    cursor_x = legend_x
    for entry in legend_entries:
        if entry == "SW":
            style = EDGE_STYLE[EdgeType.SIDEWALK]
            lines.append(
                f'<rect x="{cursor_x:.3f}" y="{legend_y - 3.0:.3f}" width="2.6" height="1.8" '
                f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" stroke="{style["stroke"]}" stroke-width="0.2"/>'
            )
        elif entry == "CW":
            style = EDGE_STYLE[EdgeType.CROSSWALK]
            lines.append(
                f'<rect x="{cursor_x:.3f}" y="{legend_y - 3.0:.3f}" width="2.6" height="1.8" '
                f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" stroke="{style["stroke"]}" stroke-width="0.2"/>'
            )
        elif entry == "JWJ":
            style = EDGE_STYLE[EdgeType.JAYWALKING_JUNCTION]
            lines.append(
                f'<rect x="{cursor_x:.3f}" y="{legend_y - 3.0:.3f}" width="2.6" height="1.8" '
                f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" stroke="{style["stroke"]}" stroke-width="0.2"/>'
            )
        elif entry == "walkable_bounds":
            lines.append(
                f'<rect x="{cursor_x:.3f}" y="{legend_y - 3.0:.3f}" width="2.6" height="1.8" '
                'fill="none" stroke="#b5179e" stroke-width="0.3" stroke-dasharray="1.2 0.8"/>'
            )
        else:
            style = OBSTACLE_STYLE[entry]
            lines.append(
                f'<rect x="{cursor_x:.3f}" y="{legend_y - 3.0:.3f}" width="2.6" height="1.8" '
                f'fill="{style["fill"]}" fill-opacity="{style["fill_opacity"]:.2f}" stroke="{style["stroke"]}" stroke-width="0.2"/>'
            )
        lines.append(
            f'<text x="{cursor_x + 3.2:.3f}" y="{legend_y - 1.4:.3f}" font-family="monospace" font-size="1.35" fill="#444">{svg_text(entry)}</text>'
        )
        cursor_x += legend_width + len(entry) * 0.78

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"map_name={args.map_name}")
    print(f"object_count={len(candidate_ids)}")
    print("panel_hits_walkable_bounds=" + json.dumps(panel_hits_walkable, ensure_ascii=False))
    print(f"svg={output_path}")


if __name__ == "__main__":
    main()
