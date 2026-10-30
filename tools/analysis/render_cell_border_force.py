import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import carla
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


from generate_walkable_cells import EdgeRecord, build_edge_records, load_world
from simulation.obstacles import extract_sidewalk
from simulation.path_planner import EdgeType, PedPathPlanner
from render_cell_snapshot_comparison import compute_cell_bounds, panel_geometry
from render_walkable_cell_svg import EDGE_STYLE, edge_polygon


BACKGROUND = "#f7f5ef"
SUPPORT_LINE = "#2a2a2a"
PORTAL_FILL = "#12355b"
BORDER_POINT = "#5b6472"
TEXT_PRIMARY = "#202020"
TEXT_SECONDARY = "#5d5d5d"
TITLE_HEIGHT_PX = 72
PANEL_PADDING_M = 10.0
SNAPSHOT_MARGIN_M = 5.0


def world_to_canvas(
    point: Tuple[float, float], min_x: float, max_y: float, padding: float, scale: float
) -> Tuple[float, float]:
    x, y = point
    return (x - min_x + padding) * scale, (max_y - y + padding) * scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render topology subgraph and BorderForce vectors for one cell.")
    parser.add_argument("--map-name", default="Town10HD_Opt", help="CARLA map name")
    parser.add_argument("--cell-id", required=True, help="Cell id from walkable_cells.json")
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="Path to walkable_cells.json (defaults to topology/<map>/walkable_cells.json)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA port")
    parser.add_argument("--timeout", default=120.0, type=float, help="CARLA RPC timeout in seconds")
    parser.add_argument(
        "--grid-step",
        default=1.6,
        type=float,
        help="World-space grid spacing in meters for sampling the border force field.",
    )
    parser.add_argument(
        "--min-force",
        default=0.18,
        type=float,
        help="Minimum BorderForce magnitude to draw an arrow.",
    )
    parser.add_argument(
        "--border-resolution",
        default=0.1,
        type=float,
        help="Resolution in meters used when extracting sidewalk border points.",
    )
    parser.add_argument(
        "--border-step",
        default=4,
        type=int,
        help="Draw every Nth border point to keep the image readable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to artifacts/<map>/topology/<cell_id>_border_force.png",
    )
    return parser.parse_args()


def load_cell(catalog_path: Path, cell_id: str) -> Dict[str, object]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    try:
        return next(cell for cell in catalog["cells"] if cell["id"] == cell_id)
    except StopIteration as exc:
        raise SystemExit(f'Cell "{cell_id}" not found in {catalog_path}.') from exc


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) == (y2 > y):
            continue
        intersects = x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        if intersects:
            inside = not inside
    return inside


def build_support_polygons(cell: Dict[str, object], edges: Dict[int, EdgeRecord]) -> List[List[Tuple[float, float]]]:
    core_edge_ids = [edge_info["edge_id"] for edge_info in cell["edges"]]
    support_edge_ids = cell.get("route_support", {}).get("edge_ids", core_edge_ids)
    polygons: List[List[Tuple[float, float]]] = []
    for edge_id in support_edge_ids:
        poly = edge_polygon(edges[edge_id])
        polygons.append([(float(x), float(y)) for x, y in poly])
    return polygons


def point_in_support(point: Tuple[float, float], support_polygons: Sequence[Sequence[Tuple[float, float]]]) -> bool:
    return any(point_in_polygon(point, polygon) for polygon in support_polygons)


def border_near_bounds(border: np.ndarray, bounds: Dict[str, float], margin: float) -> bool:
    min_x = bounds["min_x"] - margin
    max_x = bounds["max_x"] + margin
    min_y = bounds["min_y"] - margin
    max_y = bounds["max_y"] + margin
    xs = border[:, 0]
    ys = border[:, 1]
    return bool(np.any((xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)))


def prepare_borders(
    borders: Sequence[np.ndarray],
    section_info: Sequence[Sequence[object]],
    bounds: Dict[str, float],
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    filtered_borders: List[np.ndarray] = []
    centers: List[np.ndarray] = []
    lengths: List[float] = []
    for border, section in zip(borders, section_info):
        array = np.asarray(border, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            continue
        if not border_near_bounds(array, bounds, margin=6.0):
            continue
        filtered_borders.append(array)
        centers.append(np.asarray(section[0], dtype=float))
        lengths.append(float(section[1]))
    if centers:
        return filtered_borders, np.vstack(centers), np.asarray(lengths, dtype=float)
    return [], np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)


def compute_border_force(
    point: np.ndarray,
    borders: Sequence[np.ndarray],
    section_centers: np.ndarray,
    section_lengths: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    if not borders:
        return np.zeros(2, dtype=float)

    distances = np.linalg.norm(point - section_centers, axis=-1)
    close_mask = distances < section_lengths
    if not np.any(close_mask):
        return np.zeros(2, dtype=float)

    force = np.zeros(2, dtype=float)
    for border in [candidate_border for candidate_border, keep in zip(borders, close_mask) if keep]:
        deltas = point - border
        norms = np.linalg.norm(deltas, axis=-1)
        closest_index = int(np.argmin(norms))
        distance = float(norms[closest_index])
        if distance <= 1e-9:
            continue
        direction = deltas[closest_index] / distance
        force += direction * a * math.exp(-distance / b)
    return force


def collect_force_samples(
    bounds: Dict[str, float],
    support_polygons: Sequence[Sequence[Tuple[float, float]]],
    borders: Sequence[np.ndarray],
    section_centers: np.ndarray,
    section_lengths: np.ndarray,
    a: float,
    b: float,
    grid_step: float,
    min_force: float,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    x_values = np.arange(bounds["min_x"], bounds["max_x"] + 0.5 * grid_step, grid_step)
    y_values = np.arange(bounds["min_y"], bounds["max_y"] + 0.5 * grid_step, grid_step)
    for x in x_values:
        for y in y_values:
            point = (float(x), float(y))
            if not point_in_support(point, support_polygons):
                continue
            force = compute_border_force(np.asarray(point, dtype=float), borders, section_centers, section_lengths, a, b)
            magnitude = float(np.linalg.norm(force))
            if magnitude < min_force:
                continue
            samples.append({"point": point, "force": force, "magnitude": magnitude})
    return samples


def mix_color(color_a: str, color_b: str, alpha: float) -> Tuple[int, int, int]:
    alpha = min(max(alpha, 0.0), 1.0)
    a_rgb = np.asarray(ImageColor.getrgb(color_a), dtype=float)
    b_rgb = np.asarray(ImageColor.getrgb(color_b), dtype=float)
    mixed = (1.0 - alpha) * a_rgb + alpha * b_rgb
    return tuple(int(round(channel)) for channel in mixed)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    direction: np.ndarray,
    magnitude: float,
    max_magnitude: float,
) -> None:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return
    unit = direction / norm
    ratio = 0.0 if max_magnitude <= 1e-6 else min(magnitude / max_magnitude, 1.0)
    length_px = 12.0 + 28.0 * math.sqrt(ratio)
    color = mix_color("#4d7ea8", "#d1495b", ratio)
    end = (start[0] + unit[0] * length_px, start[1] - unit[1] * length_px)
    draw.line([start, end], fill=color, width=3)

    angle = math.atan2(-(end[1] - start[1]), end[0] - start[0])
    head_len = 8.0
    head_angle = math.radians(28.0)
    left = (
        end[0] - head_len * math.cos(angle - head_angle),
        end[1] + head_len * math.sin(angle - head_angle),
    )
    right = (
        end[0] - head_len * math.cos(angle + head_angle),
        end[1] + head_len * math.sin(angle + head_angle),
    )
    draw.polygon([end, left, right], fill=color)


def render_border_force_panel(
    cell: Dict[str, object],
    edges: Dict[int, EdgeRecord],
    bounds: Dict[str, float],
    support_polygons: Sequence[Sequence[Tuple[float, float]]],
    cropped_borders: Sequence[np.ndarray],
    force_samples: Sequence[Dict[str, object]],
    panel_width: int,
    panel_height: int,
    scale: float,
    border_step: int,
) -> Image.Image:
    image = Image.new("RGB", (panel_width, panel_height), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")
    min_x = bounds["min_x"]
    max_y = bounds["max_y"]
    core_edge_ids = [edge_info["edge_id"] for edge_info in cell["edges"]]
    support_edge_ids = cell.get("route_support", {}).get("edge_ids", core_edge_ids)
    core_edge_id_set = set(core_edge_ids)

    for edge_id in support_edge_ids:
        if edge_id in core_edge_id_set:
            continue
        edge = edges[edge_id]
        style = EDGE_STYLE.get(edge.edge_type, EDGE_STYLE[EdgeType.SIDEWALK])
        polygon = [world_to_canvas(point, min_x, max_y, PANEL_PADDING_M, scale) for point in edge_polygon(edge)]
        draw.polygon(
            polygon,
            fill=(*ImageColor.getrgb(style["fill"]), int(style["fill_opacity"] * 0.45 * 255)),
            outline=(*ImageColor.getrgb(style["stroke"]), int(style["stroke_opacity"] * 0.35 * 255)),
        )

    for edge_id in core_edge_ids:
        edge = edges[edge_id]
        style = EDGE_STYLE.get(edge.edge_type, EDGE_STYLE[EdgeType.SIDEWALK])
        polygon = [world_to_canvas(point, min_x, max_y, PANEL_PADDING_M, scale) for point in edge_polygon(edge)]
        draw.polygon(
            polygon,
            fill=(*ImageColor.getrgb(style["fill"]), int(style["fill_opacity"] * 0.82 * 255)),
            outline=(*ImageColor.getrgb(style["stroke"]), int(style["stroke_opacity"] * 0.65 * 255)),
        )

    for edge_id in support_edge_ids:
        edge = edges[edge_id]
        p1 = world_to_canvas((edge.x1, edge.y1), min_x, max_y, PANEL_PADDING_M, scale)
        p2 = world_to_canvas((edge.x2, edge.y2), min_x, max_y, PANEL_PADDING_M, scale)
        draw.line([p1, p2], fill=(*ImageColor.getrgb(SUPPORT_LINE), 78), width=1)

    for index, border in enumerate(cropped_borders):
        sampled = border[:: max(1, border_step)]
        for x, y in sampled:
            px, py = world_to_canvas((float(x), float(y)), min_x, max_y, PANEL_PADDING_M, scale)
            radius = 2 if index % 3 else 3
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=BORDER_POINT)

    max_magnitude = max((sample["magnitude"] for sample in force_samples), default=1.0)
    for sample in force_samples:
        px, py = world_to_canvas(sample["point"], min_x, max_y, PANEL_PADDING_M, scale)
        draw_arrow(draw, (px, py), sample["force"], sample["magnitude"], max_magnitude)

    for portal in cell.get("portals", []):
        x, y = world_to_canvas(
            (float(portal["centroid"][0]), float(portal["centroid"][1])),
            min_x,
            max_y,
            PANEL_PADDING_M,
            scale,
        )
        radius = max(4, int(round(0.6 * scale)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=PORTAL_FILL, outline="#ffffff", width=2)

    return image


def add_title_bar(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + TITLE_HEIGHT_PX), "#f5f3ee")
    canvas.paste(image, (0, TITLE_HEIGHT_PX))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 16), title, fill=TEXT_PRIMARY, font=font)
    draw.text((20, 40), subtitle, fill=TEXT_SECONDARY, font=font)
    return canvas


if __name__ == "__main__":
    args = parse_args()
    catalog_path = args.catalog_path or Path("topology") / args.map_name / "walkable_cells.json"
    output_path = args.output or Path("artifacts") / args.map_name / "topology" / f"{args.cell_id}_border_force.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell = load_cell(catalog_path, args.cell_id)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = load_world(client, args.map_name)
    carla_map = world.get_map()

    path_planner = PedPathPlanner(carla_map, waypoint_distance=5.0)
    edges, _ = build_edge_records(path_planner)
    bounds = compute_cell_bounds(cell, edges)
    panel_width, panel_height, scale = panel_geometry(bounds)

    support_polygons = build_support_polygons(cell, edges)
    sidewalk_borders, section_info, _ = extract_sidewalk(
        carla_map,
        {"obstacles": {"resolution": args.border_resolution}},
    )
    cropped_borders, section_centers, section_lengths = prepare_borders(sidewalk_borders, section_info, bounds)

    border_force_config = {"a": 6.0, "b": 0.3}
    with Path("config/sfm_config.toml").open("rb") as handle:
        border_force_config = tomllib.load(handle).get("border_force", border_force_config)

    force_samples = collect_force_samples(
        bounds=bounds,
        support_polygons=support_polygons,
        borders=cropped_borders,
        section_centers=section_centers,
        section_lengths=section_lengths,
        a=float(border_force_config.get("a", 6.0)),
        b=float(border_force_config.get("b", 0.3)),
        grid_step=args.grid_step,
        min_force=args.min_force,
    )

    panel = render_border_force_panel(
        cell=cell,
        edges=edges,
        bounds=bounds,
        support_polygons=support_polygons,
        cropped_borders=cropped_borders,
        force_samples=force_samples,
        panel_width=panel_width,
        panel_height=panel_height,
        scale=scale,
        border_step=args.border_step,
    )
    titled = add_title_bar(
        panel,
        f"Border Force | {args.map_name} / {args.cell_id}",
        f'borders={len(cropped_borders)} vectors={len(force_samples)} a={border_force_config.get("a", 6.0)} b={border_force_config.get("b", 0.3)}',
    )
    titled.save(output_path)

    print(f"cell_id={args.cell_id}")
    print(f"borders={len(cropped_borders)}")
    print(f"vectors={len(force_samples)}")
    print(f"output={output_path}")
