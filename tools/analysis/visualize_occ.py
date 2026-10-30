#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "occ" / "default.toml"
DEFAULT_EMPTY_LABEL = 0
DEFAULT_YAW_DEG = 90.0
DEFAULT_PITCH_DEG = 36.0
DEFAULT_HEIGHT_SCALE = 1.15
DEFAULT_DPI = 220


LABEL_COLORS = np.array([
    (255, 255, 255),  # 0  None / empty
    (70, 70, 70),     # 1  Buildings
    (100, 40, 40),    # 2  Fences
    (55, 90, 80),     # 3  Other
    (220, 20, 60),    # 4  Pedestrians
    (153, 153, 153),  # 5  Poles
    (157, 234, 50),   # 6  RoadLines
    (128, 64, 128),   # 7  Roads
    (244, 35, 232),   # 8  Sidewalks
    (107, 142, 35),   # 9  Vegetation
    (0, 0, 142),      # 10 Vehicles
    (102, 102, 156),  # 11 Walls
    (220, 220, 0),    # 12 TrafficSigns
    (70, 130, 180),   # 13 Sky
    (81, 0, 81),      # 14 Ground
    (150, 100, 100),  # 15 Bridge
    (230, 150, 140),  # 16 RailTrack
    (180, 165, 180),  # 17 GuardRail
    (250, 170, 30),   # 18 TrafficLight
    (110, 190, 160),  # 19 Static
    (170, 120, 50),   # 20 Dynamic
    (45, 60, 150),    # 21 Water
    (145, 170, 100),  # 22 Terrain
    (0, 0, 0),        # 23 unused / unknown
    (0, 0, 0),        # 24 unused / unknown
    (255, 0, 255),    # 25 unknown extended tag
], dtype=np.uint8)


LABEL_NAMES = {
    0: "None",
    1: "Buildings",
    2: "Fences",
    3: "Other",
    4: "Pedestrians",
    5: "Poles",
    6: "RoadLines",
    7: "Roads",
    8: "Sidewalks",
    9: "Vegetation",
    10: "Vehicles",
    11: "Walls",
    12: "TrafficSigns",
    13: "Sky",
    14: "Ground",
    15: "Bridge",
    16: "RailTrack",
    17: "GuardRail",
    18: "TrafficLight",
    19: "Static",
    20: "Dynamic",
    21: "Water",
    22: "Terrain",
    25: "Unknown25",
}


BACKGROUND_COLOR = "#f5f1ea"
TITLE_COLOR = "#1f1f1f"
SUBTITLE_COLOR = "#5b5b5b"
GRID_COLOR = (0.12, 0.12, 0.12, 0.12)
EGO_COLOR = "#111111"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render one OCC sample directory or labels.npz as a single angled bird's-eye-view image."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to an OCC sample directory like .../occ/sample_002760 or directly to labels.npz.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="OCC TOML config used to recover voxel size and point-cloud range.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <sample_dir>/occ_bev_angle.png.",
    )
    parser.add_argument(
        "--empty-label",
        type=int,
        default=DEFAULT_EMPTY_LABEL,
        help="Semantic label treated as empty and hidden from rendering.",
    )
    parser.add_argument(
        "--hide-label",
        type=int,
        action="append",
        default=[],
        help="Additional semantic label to hide. Can be passed multiple times.",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=DEFAULT_YAW_DEG,
        help="BEV rotation around the vertical axis in degrees. Default keeps forward roughly upward.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=DEFAULT_PITCH_DEG,
        help="Tilt angle in degrees. Larger values make the view flatter.",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=DEFAULT_HEIGHT_SCALE,
        help="Vertical exaggeration applied before projection.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output PNG DPI.",
    )
    return parser.parse_args()


def resolve_labels_path(input_path):
    path = Path(input_path).resolve()
    if path.is_file():
        return path
    candidate = path / "labels.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not find labels.npz under {}".format(path))


def load_occ_config(path):
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    occ_cfg = payload.get("occ", payload)
    return {
        "voxel_size": float(occ_cfg["voxel_size"]),
        "pc_range": np.asarray(occ_cfg["pc_range"], dtype=np.float64),
        "occ_size": tuple(int(value) for value in occ_cfg["occ_size"]),
    }


def load_sample(labels_path):
    payload = np.load(str(labels_path))
    return payload["semantics"]


def default_output_path(labels_path):
    return labels_path.parent / "occ_bev_angle.png"


def resolve_output_path(requested_output, labels_path):
    if requested_output:
        return Path(requested_output).resolve()

    preferred = default_output_path(labels_path)
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        probe = preferred.parent / ".occ_bev_angle_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path("/tmp") / "occ_bev_angle_{}.png".format(labels_path.parent.name)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def label_rgb(label):
    if 0 <= int(label) < len(LABEL_COLORS):
        return LABEL_COLORS[int(label)].astype(np.float64) / 255.0
    return np.array([1.0, 0.0, 1.0], dtype=np.float64)


def shade_rgb(rgb, factor):
    return np.clip(np.asarray(rgb, dtype=np.float64) * float(factor), 0.0, 1.0)


def build_top_surface(semantics, empty_label, hidden_labels):
    hidden = set(int(value) for value in hidden_labels)
    hidden.add(int(empty_label))

    top_label = np.full(semantics.shape[:2], -1, dtype=np.int16)
    top_z = np.full(semantics.shape[:2], -1, dtype=np.int16)
    bottom_z = np.full(semantics.shape[:2], -1, dtype=np.int16)

    for z_index in range(semantics.shape[2]):
        slice_labels = semantics[:, :, z_index]
        visible = np.ones(slice_labels.shape, dtype=bool)
        for label in hidden:
            visible &= slice_labels != label
        write_mask = visible & (bottom_z < 0)
        bottom_z[write_mask] = z_index

    for z_index in range(semantics.shape[2] - 1, -1, -1):
        slice_labels = semantics[:, :, z_index]
        visible = np.ones(slice_labels.shape, dtype=bool)
        for label in hidden:
            visible &= slice_labels != label
        write_mask = visible & (top_z < 0)
        top_z[write_mask] = z_index
        top_label[write_mask] = slice_labels[write_mask].astype(np.int16)

    return top_label, top_z, bottom_z


def cell_corners_world(x_index, y_index, z_world, voxel_size, pc_range, occ_size):
    x0 = pc_range[0] + (float(x_index) * voxel_size)
    x1 = x0 + voxel_size

    y_base_index = (occ_size[1] - 1) - int(y_index)
    y0 = pc_range[1] + (float(y_base_index) * voxel_size)
    y1 = y0 + voxel_size

    return np.array(
        [
            [x0, y0, z_world],
            [x1, y0, z_world],
            [x1, y1, z_world],
            [x0, y1, z_world],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(yaw_deg):
    yaw_rad = np.deg2rad(float(yaw_deg))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_x(pitch_deg):
    pitch_rad = np.deg2rad(float(pitch_deg))
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float64,
    )


def project_points(points_xyz, yaw_deg, pitch_deg, height_scale):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    transformed = points_xyz.copy()
    transformed[:, 2] *= float(height_scale)

    rotation = rotation_matrix_x(pitch_deg) @ rotation_matrix_z(yaw_deg)
    camera_xyz = (rotation @ transformed.T).T
    return camera_xyz[:, :2], camera_xyz[:, 2]


def build_projected_scene(top_label, top_z, bottom_z, voxel_size, pc_range, occ_size, yaw_deg, pitch_deg, height_scale):
    occupied_xy = np.argwhere(top_z >= 0)
    if occupied_xy.size == 0:
        return {
            "base_polys": [],
            "top_polys": [],
            "connector_segments": [],
            "base_colors": [],
            "top_colors": [],
            "connector_colors": [],
            "all_points_2d": np.empty((0, 2), dtype=np.float64),
            "labels_present": np.empty((0,), dtype=np.int16),
        }

    items = []
    for x_index, y_index in occupied_xy:
        label = int(top_label[x_index, y_index])
        bottom_index = int(bottom_z[x_index, y_index])
        top_index = int(top_z[x_index, y_index])

        z_bottom = pc_range[2] + (float(bottom_index) * voxel_size)
        z_top = pc_range[2] + (float(top_index + 1) * voxel_size)

        base_corners_world = cell_corners_world(x_index, y_index, z_bottom, voxel_size, pc_range, occ_size)
        top_corners_world = cell_corners_world(x_index, y_index, z_top, voxel_size, pc_range, occ_size)

        base_poly_2d, base_depth = project_points(base_corners_world, yaw_deg, pitch_deg, height_scale)
        top_poly_2d, top_depth = project_points(top_corners_world, yaw_deg, pitch_deg, height_scale)

        centroid_world = np.array([
            pc_range[0] + ((float(x_index) + 0.5) * voxel_size),
            pc_range[1] + (((occ_size[1] - 1) - float(y_index) + 0.5) * voxel_size),
            z_top,
        ])
        _, centroid_depth = project_points(centroid_world[np.newaxis, :], yaw_deg, pitch_deg, height_scale)

        rgb = label_rgb(label)
        height_ratio = float(top_index + 1) / float(max(1, occ_size[2]))
        top_rgb = shade_rgb(rgb, 0.82 + (0.28 * height_ratio))
        base_rgb = shade_rgb(rgb, 0.22 + (0.18 * height_ratio))
        line_rgb = shade_rgb(rgb, 0.50)

        connector_segments = []
        for vertex_index in range(4):
            connector_segments.append(np.stack([base_poly_2d[vertex_index], top_poly_2d[vertex_index]], axis=0))

        items.append({
            "depth": float(centroid_depth[0]),
            "label": label,
            "base_poly": base_poly_2d,
            "top_poly": top_poly_2d,
            "connector_segments": connector_segments,
            "base_color": (base_rgb[0], base_rgb[1], base_rgb[2], 0.18),
            "top_color": (top_rgb[0], top_rgb[1], top_rgb[2], 0.98),
            "connector_color": (line_rgb[0], line_rgb[1], line_rgb[2], 0.22),
        })

    items.sort(key=lambda item: item["depth"])

    base_polys = [item["base_poly"] for item in items]
    top_polys = [item["top_poly"] for item in items]
    base_colors = [item["base_color"] for item in items]
    top_colors = [item["top_color"] for item in items]
    connector_segments = []
    connector_colors = []
    labels_present = []
    point_cloud_2d = []
    for item in items:
        labels_present.append(item["label"])
        point_cloud_2d.append(item["base_poly"])
        point_cloud_2d.append(item["top_poly"])
        for segment in item["connector_segments"]:
            connector_segments.append(segment)
            connector_colors.append(item["connector_color"])

    return {
        "base_polys": base_polys,
        "top_polys": top_polys,
        "connector_segments": connector_segments,
        "base_colors": base_colors,
        "top_colors": top_colors,
        "connector_colors": connector_colors,
        "all_points_2d": np.concatenate(point_cloud_2d, axis=0),
        "labels_present": np.asarray(labels_present, dtype=np.int16),
    }


def add_ground_grid(ax, projected_scene):
    points = projected_scene["all_points_2d"]
    if points.size == 0:
        return

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    x_values = np.linspace(min_xy[0], max_xy[0], num=7)
    y_values = np.linspace(min_xy[1], max_xy[1], num=7)

    for x_value in x_values:
        ax.plot([x_value, x_value], [min_xy[1], max_xy[1]], color=GRID_COLOR, linewidth=0.6, zorder=0)
    for y_value in y_values:
        ax.plot([min_xy[0], max_xy[0]], [y_value, y_value], color=GRID_COLOR, linewidth=0.6, zorder=0)


def add_ego_marker(ax, pc_range, yaw_deg, pitch_deg, height_scale):
    origin = np.array([[0.0, 0.0, pc_range[2]]], dtype=np.float64)
    forward = np.array([[2.2, 0.0, pc_range[2]]], dtype=np.float64)
    left = np.array([[0.0, 1.2, pc_range[2]]], dtype=np.float64)

    origin_2d, _ = project_points(origin, yaw_deg, pitch_deg, height_scale)
    forward_2d, _ = project_points(forward, yaw_deg, pitch_deg, height_scale)
    left_2d, _ = project_points(left, yaw_deg, pitch_deg, height_scale)

    origin_pt = origin_2d[0]
    forward_pt = forward_2d[0]
    left_pt = left_2d[0]

    ax.scatter([origin_pt[0]], [origin_pt[1]], s=38.0, c=[EGO_COLOR], zorder=5)
    ax.annotate(
        "",
        xy=(forward_pt[0], forward_pt[1]),
        xytext=(origin_pt[0], origin_pt[1]),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": EGO_COLOR},
        zorder=5,
    )
    ax.plot(
        [origin_pt[0], left_pt[0]],
        [origin_pt[1], left_pt[1]],
        color=EGO_COLOR,
        linewidth=1.2,
        alpha=0.55,
        zorder=5,
    )
    ax.text(
        origin_pt[0],
        origin_pt[1],
        " ego",
        color=EGO_COLOR,
        fontsize=9,
        va="bottom",
        ha="left",
        zorder=5,
    )


def add_title(ax, labels_path, labels_present):
    unique_labels = sorted(set(int(value) for value in labels_present.tolist()))
    subtitle = "occupied labels: {}".format(
        ", ".join("{}:{}".format(label, LABEL_NAMES.get(label, "Unknown")) for label in unique_labels[:8])
    )
    if len(unique_labels) > 8:
        subtitle += ", ..."

    ax.text(
        0.015,
        0.985,
        labels_path.parent.name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        color=TITLE_COLOR,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": (1.0, 1.0, 1.0, 0.78), "edgecolor": "none"},
    )
    ax.text(
        0.015,
        0.945,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=SUBTITLE_COLOR,
    )


def render_angled_bev(labels_path, output_path, config, semantics, empty_label, hidden_labels, yaw_deg, pitch_deg, height_scale, dpi):
    top_label, top_z, bottom_z = build_top_surface(semantics, empty_label, hidden_labels)
    projected_scene = build_projected_scene(
        top_label=top_label,
        top_z=top_z,
        bottom_z=bottom_z,
        voxel_size=config["voxel_size"],
        pc_range=config["pc_range"],
        occ_size=config["occ_size"],
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        height_scale=height_scale,
    )

    figure, ax = plt.subplots(figsize=(11.5, 10.0), facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    if projected_scene["all_points_2d"].size == 0:
        ax.text(0.5, 0.5, "No occupied voxels", ha="center", va="center", transform=ax.transAxes, fontsize=16)
        ax.axis("off")
    else:
        add_ground_grid(ax, projected_scene)

        base_collection = PolyCollection(
            projected_scene["base_polys"],
            facecolors=projected_scene["base_colors"],
            edgecolors="none",
            zorder=1,
        )
        connector_collection = LineCollection(
            projected_scene["connector_segments"],
            colors=projected_scene["connector_colors"],
            linewidths=0.45,
            zorder=2,
        )
        top_collection = PolyCollection(
            projected_scene["top_polys"],
            facecolors=projected_scene["top_colors"],
            edgecolors=(0.08, 0.08, 0.08, 0.20),
            linewidths=0.18,
            zorder=3,
        )

        ax.add_collection(base_collection)
        ax.add_collection(connector_collection)
        ax.add_collection(top_collection)

        add_ego_marker(ax, config["pc_range"], yaw_deg, pitch_deg, height_scale)
        add_title(ax, labels_path, projected_scene["labels_present"])

        points_2d = projected_scene["all_points_2d"]
        min_xy = points_2d.min(axis=0)
        max_xy = points_2d.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        margin = span * 0.06
        ax.set_xlim(min_xy[0] - margin[0], max_xy[0] + margin[0])
        ax.set_ylim(min_xy[1] - margin[1], max_xy[1] + margin[1])
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main():
    args = parse_args()
    labels_path = resolve_labels_path(args.input_path)
    output_path = resolve_output_path(args.output, labels_path)

    config = load_occ_config(args.config)
    semantics = load_sample(labels_path)
    if semantics.shape != tuple(config["occ_size"]):
        raise ValueError(
            "labels.npz semantics shape {} does not match config occ_size {}".format(
                semantics.shape,
                config["occ_size"],
            )
        )

    render_angled_bev(
        labels_path=labels_path,
        output_path=output_path,
        config=config,
        semantics=semantics,
        empty_label=args.empty_label,
        hidden_labels=args.hide_label,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        height_scale=args.height_scale,
        dpi=args.dpi,
    )
    print(output_path)


if __name__ == "__main__":
    main()
