import argparse
import json
import math
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Sequence, Tuple

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
from simulation.path_planner import EdgeType, PedPathPlanner
from render_walkable_cell_svg import EDGE_STYLE, ROUTE_STYLE, edge_polygon


PANEL_WIDTH_PX = 1600
MIN_PANEL_HEIGHT_PX = 900
PANEL_PADDING_M = 10.0
SNAPSHOT_MARGIN_M = 5.0
PANEL_GAP_PX = 36
TITLE_HEIGHT_PX = 72
SNAPSHOT_FOV_DEG = 32.0


def world_to_canvas(
    point: Tuple[float, float], min_x: float, max_y: float, padding: float, scale: float
) -> Tuple[float, float]:
    x, y = point
    return (x - min_x + padding) * scale, (max_y - y + padding) * scale


def expand_bounds(bounds: Dict[str, float], margin: float) -> Dict[str, float]:
    return {
        "min_x": float(bounds["min_x"]) - margin,
        "max_x": float(bounds["max_x"]) + margin,
        "min_y": float(bounds["min_y"]) - margin,
        "max_y": float(bounds["max_y"]) + margin,
    }


def compute_cell_bounds(cell: Dict[str, object], edges: Dict[int, EdgeRecord]) -> Dict[str, float]:
    core_edge_ids = [edge_info["edge_id"] for edge_info in cell["edges"]]
    support_edge_ids = cell.get("route_support", {}).get("edge_ids", core_edge_ids)
    support_coordinates: List[Tuple[float, float]] = []
    for edge_id in support_edge_ids:
        edge = edges[edge_id]
        support_coordinates.extend([(edge.x1, edge.y1), (edge.x2, edge.y2)])
    return expand_bounds(
        {
            "min_x": min(point[0] for point in support_coordinates),
            "max_x": max(point[0] for point in support_coordinates),
            "min_y": min(point[1] for point in support_coordinates),
            "max_y": max(point[1] for point in support_coordinates),
        },
        margin=SNAPSHOT_MARGIN_M,
    )


def panel_geometry(bounds: Dict[str, float]) -> Tuple[int, int, float]:
    world_width = (bounds["max_x"] - bounds["min_x"]) + 2.0 * PANEL_PADDING_M
    world_height = (bounds["max_y"] - bounds["min_y"]) + 2.0 * PANEL_PADDING_M
    scale = PANEL_WIDTH_PX / max(world_width, 1e-6)
    panel_height = max(MIN_PANEL_HEIGHT_PX, int(round(world_height * scale)))
    return PANEL_WIDTH_PX, panel_height, scale


def _dash_segments(points: List[Tuple[float, float]], dash_length_px: float, gap_length_px: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if len(points) < 2:
        return []
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    pattern_length = dash_length_px + gap_length_px
    carry = 0.0
    draw_phase = True
    for left, right in zip(points, points[1:]):
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        ux = dx / length
        uy = dy / length
        traversed = 0.0
        while traversed < length - 1e-6:
            remaining_in_phase = (dash_length_px if draw_phase else gap_length_px) - carry
            step = min(remaining_in_phase, length - traversed)
            start = (left[0] + ux * traversed, left[1] + uy * traversed)
            end = (left[0] + ux * (traversed + step), left[1] + uy * (traversed + step))
            if draw_phase:
                segments.append((start, end))
            traversed += step
            carry += step
            if carry >= (dash_length_px if draw_phase else gap_length_px) - 1e-6:
                carry = 0.0
                draw_phase = not draw_phase
        if traversed >= length - 1e-6:
            continue
    return segments


def _draw_polyline(draw: ImageDraw.ImageDraw, points: List[Tuple[float, float]], fill: str, width: int, dash: str = "") -> None:
    if len(points) < 2:
        return
    if not dash:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    dash_parts = [float(part) for part in dash.split() if part]
    if len(dash_parts) != 2:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    dash_segments = _dash_segments(points, dash_parts[0] * 7.0, dash_parts[1] * 7.0)
    for start, end in dash_segments:
        draw.line([start, end], fill=fill, width=width)


def render_topology_panel(
    cell: Dict[str, object],
    edges: Dict[int, EdgeRecord],
    bounds: Dict[str, float],
    panel_width: int,
    panel_height: int,
    scale: float,
) -> Image.Image:
    image = Image.new("RGB", (panel_width, panel_height), "#f7f5ef")
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
        polygon = [
            world_to_canvas(point, min_x, max_y, PANEL_PADDING_M, scale)
            for point in edge_polygon(edge)
        ]
        draw.polygon(
            polygon,
            fill=(*ImageColor.getrgb(style["fill"]), int(style["fill_opacity"] * 0.55 * 255)),
            outline=(*ImageColor.getrgb(style["stroke"]), int(style["stroke_opacity"] * 0.45 * 255)),
        )

    for edge_id in core_edge_ids:
        edge = edges[edge_id]
        style = EDGE_STYLE.get(edge.edge_type, EDGE_STYLE[EdgeType.SIDEWALK])
        polygon = [
            world_to_canvas(point, min_x, max_y, PANEL_PADDING_M, scale)
            for point in edge_polygon(edge)
        ]
        draw.polygon(
            polygon,
            fill=(*ImageColor.getrgb(style["fill"]), int(style["fill_opacity"] * 255)),
            outline=(*ImageColor.getrgb(style["stroke"]), int(style["stroke_opacity"] * 255)),
        )

    for edge_id in support_edge_ids:
        edge = edges[edge_id]
        p1 = world_to_canvas((edge.x1, edge.y1), min_x, max_y, PANEL_PADDING_M, scale)
        p2 = world_to_canvas((edge.x2, edge.y2), min_x, max_y, PANEL_PADDING_M, scale)
        draw.line([p1, p2], fill=(34, 34, 34, 72), width=max(1, int(round(0.8 * scale * 0.32))))

    for route in cell.get("routes", []):
        style = ROUTE_STYLE[route["route_type"]]
        points = [
            world_to_canvas((float(point[0]), float(point[1])), min_x, max_y, PANEL_PADDING_M, scale)
            for point in route["path_waypoints"]
        ]
        _draw_polyline(
            draw,
            points,
            fill="#ffffff",
            width=max(2, int(round((style["width"] + 0.34) * scale))),
        )
        _draw_polyline(
            draw,
            points,
            fill=style["stroke"],
            width=max(2, int(round(style["width"] * scale))),
            dash=style["dash"],
        )

    for portal in cell.get("portals", []):
        x, y = world_to_canvas(
            (float(portal["centroid"][0]), float(portal["centroid"][1])),
            min_x,
            max_y,
            PANEL_PADDING_M,
            scale,
        )
        radius = max(4, int(round(0.7 * scale)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#1d3557", outline="#ffffff", width=2)

    return image


def _image_from_carla(raw_image: carla.Image) -> Image.Image:
    array = np.frombuffer(raw_image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (raw_image.height, raw_image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return Image.fromarray(rgb)


def _camera_height(bounds: Dict[str, float], image_width: int, image_height: int, fov_deg: float) -> float:
    width_m = bounds["max_x"] - bounds["min_x"]
    height_m = bounds["max_y"] - bounds["min_y"]
    aspect = image_width / max(image_height, 1)
    hfov = math.radians(fov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / max(aspect, 1e-6))
    z_width = (width_m * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    z_height = (height_m * 0.5) / max(math.tan(vfov * 0.5), 1e-6)
    return max(z_width, z_height) + 6.0


def capture_snapshot_panel(
    world: carla.World,
    bounds: Dict[str, float],
    panel_width: int,
    panel_height: int,
) -> Image.Image:
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(panel_width))
    camera_bp.set_attribute("image_size_y", str(panel_height))
    camera_bp.set_attribute("fov", f"{SNAPSHOT_FOV_DEG:.2f}")
    camera_bp.set_attribute("sensor_tick", "0.0")

    center_x = 0.5 * (bounds["min_x"] + bounds["max_x"])
    center_y = 0.5 * (bounds["min_y"] + bounds["max_y"])
    center_z = _camera_height(bounds, panel_width, panel_height, SNAPSHOT_FOV_DEG)
    transform = carla.Transform(
        carla.Location(x=center_x, y=center_y, z=center_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )

    image_queue: Queue[carla.Image] = Queue()
    camera = world.spawn_actor(camera_bp, transform)
    try:
        camera.listen(image_queue.put)
        settings = world.get_settings()
        if settings.synchronous_mode:
            for _ in range(3):
                world.tick()
                try:
                    image = image_queue.get(timeout=5.0)
                    return _image_from_carla(image)
                except Empty:
                    continue
        else:
            for _ in range(8):
                world.wait_for_tick(2.0)
                try:
                    image = image_queue.get(timeout=1.0)
                    return _image_from_carla(image)
                except Empty:
                    continue
        raise RuntimeError("Timed out while waiting for CARLA snapshot image.")
    finally:
        camera.stop()
        camera.destroy()


def add_title_bar(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + TITLE_HEIGHT_PX), "#f5f3ee")
    canvas.paste(image, (0, TITLE_HEIGHT_PX))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 16), title, fill="#1f1f1f", font=font)
    draw.text((20, 40), subtitle, fill="#555555", font=font)
    return canvas


def compose_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    if left.size != right.size:
        raise ValueError("Left and right panels must have the same size.")
    canvas = Image.new("RGB", (left.width + PANEL_GAP_PX + right.width, left.height), "#f2efe8")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + PANEL_GAP_PX, 0))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a CARLA BEV snapshot beside a topology cell route view.")
    parser.add_argument("--map-name", default="Town10HD_Opt", help="CARLA map name")
    parser.add_argument("--cell-id", required=True, help="Cell id from walkable_cells.json")
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="Path to walkable_cells.json (defaults to topology/<map_name>/walkable_cells.json)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA port")
    parser.add_argument("--timeout", default=120.0, type=float, help="CARLA RPC timeout in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the side-by-side comparison image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    catalog_path = args.catalog_path or Path("topology") / args.map_name / "walkable_cells.json"
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    try:
        cell = next(cell for cell in catalog["cells"] if cell["id"] == args.cell_id)
    except StopIteration as exc:
        raise SystemExit(f'Cell "{args.cell_id}" not found in {catalog_path}.') from exc

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = load_world(client, args.map_name)
    path_planner = PedPathPlanner(world.get_map(), waypoint_distance=5.0)
    edges, _ = build_edge_records(path_planner)

    bounds = compute_cell_bounds(cell, edges)
    panel_width, panel_height, scale = panel_geometry(bounds)

    snapshot_panel = capture_snapshot_panel(world, bounds, panel_width, panel_height)
    topology_panel = render_topology_panel(cell, edges, bounds, panel_width, panel_height, scale)

    snapshot_panel = add_title_bar(snapshot_panel, "CARLA Snapshot", f'{args.map_name} / {args.cell_id}')
    topology_panel = add_title_bar(
        topology_panel,
        "Topology Routes",
        f'portals={len(cell.get("portals", []))} routes={len(cell.get("routes", []))}',
    )

    comparison = compose_side_by_side(snapshot_panel, topology_panel)
    comparison.save(output_path)

    print(f"cell_id={args.cell_id}")
    print(f"output={output_path}")
