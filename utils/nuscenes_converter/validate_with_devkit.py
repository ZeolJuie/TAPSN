#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPL_CONFIG_DIR = REPO_ROOT / "artifacts" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from pyquaternion import Quaternion


try:
    from loguru import logger
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    _fallback_logger = logging.getLogger("nuscenes_validate")

    class _LoggerAdapter:
        def info(self, message, *args):
            if args:
                message = message.format(*args)
            _fallback_logger.info(message)

        def warning(self, message, *args):
            if args:
                message = message.format(*args)
            _fallback_logger.warning(message)

        def error(self, message, *args):
            if args:
                message = message.format(*args)
            _fallback_logger.error(message)

    logger = _LoggerAdapter()


CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]
BOX_NEAR_PLANE_M = 0.1
MOSAIC_SCALE = 0.4
BOX_EDGE_INDICES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
BOX_FRONT_FACE_INDICES = [(0, 1), (1, 2), (2, 3), (3, 0)]
BOX_REAR_FACE_INDICES = [(4, 5), (5, 6), (6, 7), (7, 4)]
MOSAIC_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]
BOX_COLOR_PALETTE = [
    (255, 99, 71),
    (64, 224, 208),
    (255, 215, 0),
    (50, 205, 50),
    (30, 144, 255),
    (255, 105, 180),
    (255, 140, 0),
    (186, 85, 211),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a converted nuScenes-style dataset with nuscenes-devkit and generate visualizations."
    )
    parser.add_argument("dataroot", help="Converted nuScenes dataset root")
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="Metadata version directory",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in nusc.sample used for visualization",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/nuscenes_validation",
        help="Directory for validation outputs",
    )
    return parser.parse_args()


def load_table(dataroot, version, table_name):
    path = Path(dataroot) / version / "{}.json".format(table_name)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_table_counts(nusc):
    logger.info("Loaded dataset with {} samples, {} sample_data, {} annotations, {} instances.",
                len(nusc.sample), len(nusc.sample_data), len(nusc.sample_annotation), len(nusc.instance))

    if not nusc.sample:
        raise RuntimeError("No samples found in dataset")

    if not nusc.sample_data:
        raise RuntimeError("No sample_data found in dataset")

    if not nusc.calibrated_sensor:
        raise RuntimeError("No calibrated_sensor records found in dataset")

    logger.info("Sensors: {}", sorted({row["channel"] for row in nusc.sensor}))
    logger.info("Categories: {}", sorted({row["name"] for row in nusc.category}))


def category_color(name):
    palette_index = sum(ord(character) for character in str(name)) % len(BOX_COLOR_PALETTE)
    return BOX_COLOR_PALETTE[palette_index]


def clip_segment_to_near_plane(point_a, point_b, near_plane=BOX_NEAR_PLANE_M):
    z1 = float(point_a[2])
    z2 = float(point_b[2])

    if z1 < near_plane and z2 < near_plane:
        return None

    p1 = np.array(point_a, dtype=np.float64)
    p2 = np.array(point_b, dtype=np.float64)

    if z1 >= near_plane and z2 >= near_plane:
        return p1, p2

    if abs(z2 - z1) < 1e-12:
        return None

    t = (near_plane - z1) / (z2 - z1)
    t = min(max(t, 0.0), 1.0)
    intersection = p1 + t * (p2 - p1)
    intersection[2] = near_plane

    if z1 < near_plane:
        p1 = intersection
    else:
        p2 = intersection
    return p1, p2


def project_optical_point(point_optical, intrinsic):
    projected = np.asarray(intrinsic, dtype=np.float64) @ np.asarray(point_optical, dtype=np.float64)
    return projected[:2] / projected[2]


def segment_overlaps_image(p1, p2, width, height):
    min_x = min(float(p1[0]), float(p2[0]))
    max_x = max(float(p1[0]), float(p2[0]))
    min_y = min(float(p1[1]), float(p2[1]))
    max_y = max(float(p1[1]), float(p2[1]))
    return not (max_x < 0.0 or max_y < 0.0 or min_x >= width or min_y >= height)


def clip_line_to_image(p1, p2, width, height):
    x_min = 0.0
    y_min = 0.0
    x_max = float(width - 1)
    y_max = float(height - 1)

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1

    u1 = 0.0
    u2 = 1.0
    p = [-dx, dx, -dy, dy]
    q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]

    for p_i, q_i in zip(p, q):
        if abs(p_i) < 1e-12:
            if q_i < 0.0:
                return None
            continue

        t = q_i / p_i
        if p_i < 0.0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t

    clipped_start = (x1 + u1 * dx, y1 + u1 * dy)
    clipped_end = (x1 + u2 * dx, y1 + u2 * dy)
    return clipped_start, clipped_end


def draw_text_with_background(draw, position_xy, text, fill_color):
    font = ImageFont.load_default()
    text_width, text_height = font.getsize(text)
    padding_x = 4
    padding_y = 3
    x0 = float(position_xy[0])
    y0 = float(position_xy[1])
    x1 = x0 + text_width + 2 * padding_x
    y1 = y0 + text_height + 2 * padding_y
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    draw.text((x0 + padding_x, y0 + padding_y), text, fill=fill_color, font=font)


def draw_box_on_image(draw, box, intrinsic, image_width, image_height):
    corners_optical = box.corners().T
    if not np.any(corners_optical[:, 2] > BOX_NEAR_PLANE_M):
        return False

    projected_points = []
    base_color = category_color(box.name or "Unknown")

    for start_index, end_index in BOX_EDGE_INDICES:
        clipped_segment = clip_segment_to_near_plane(
            corners_optical[start_index],
            corners_optical[end_index],
        )
        if clipped_segment is None:
            continue

        p1 = project_optical_point(clipped_segment[0], intrinsic)
        p2 = project_optical_point(clipped_segment[1], intrinsic)
        if not segment_overlaps_image(p1, p2, image_width, image_height):
            continue

        clipped_pixels = clip_line_to_image(p1, p2, image_width, image_height)
        if clipped_pixels is None:
            continue

        c1, c2 = clipped_pixels
        projected_points.extend([c1, c2])

        edge = (start_index, end_index)
        reverse_edge = (end_index, start_index)
        if edge in BOX_FRONT_FACE_INDICES or reverse_edge in BOX_FRONT_FACE_INDICES:
            color = (80, 255, 120)
            width = 3
        elif edge in BOX_REAR_FACE_INDICES or reverse_edge in BOX_REAR_FACE_INDICES:
            color = (255, 180, 60)
            width = 2
        else:
            color = base_color
            width = 2
        draw.line((float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1])), fill=color, width=width)

    if projected_points:
        points_array = np.asarray(projected_points, dtype=np.float64)
        label_position = (float(np.min(points_array[:, 0])) + 4.0, float(np.min(points_array[:, 1])) + 4.0)
        draw_text_with_background(draw, label_position, str(box.name or "Unknown"), base_color)
        return True
    return False


def camera_box_image(nusc, sample_token, camera_channel):
    sample = nusc.get("sample", sample_token)
    camera_token = sample["data"][camera_channel]
    image_path, boxes, camera_intrinsic = nusc.get_sample_data(
        camera_token,
        box_vis_level=BoxVisibility.ANY,
    )

    logger.info(
        "Rendering 3D boxes for sample={} camera={} boxes={}",
        sample_token,
        camera_channel,
        len(boxes),
    )

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    for box in boxes:
        draw_box_on_image(draw, box, camera_intrinsic, image_width, image_height)

    draw_text_with_background(draw, (12, 12), camera_channel, (255, 255, 255))
    return image


def resize_image(image, scale):
    new_width = max(1, int(round(image.width * scale)))
    new_height = max(1, int(round(image.height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((new_width, new_height), resample=resampling.BILINEAR)


def render_camera_box_mosaic(nusc, sample_token, output_path, scale=MOSAIC_SCALE):
    logger.info("Rendering 6-view camera box mosaic for sample={}", sample_token)

    images = {camera: resize_image(camera_box_image(nusc, sample_token, camera), scale) for camera in CAMERAS}
    tile_width = next(iter(images.values())).width
    tile_height = next(iter(images.values())).height
    margin = 12

    mosaic_width = tile_width * 3 + margin * 4
    mosaic_height = tile_height * 2 + margin * 3
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), color=(18, 18, 18))

    for row_index, row in enumerate(MOSAIC_LAYOUT):
        for col_index, camera in enumerate(row):
            tile = images[camera]
            x = margin + col_index * (tile_width + margin)
            y = margin + row_index * (tile_height + margin)
            mosaic.paste(tile, (x, y))

    mosaic.save(output_path)
    logger.info("Saved camera box mosaic to {}", output_path)


def render_lidar_on_camera(nusc, sample_token, camera_channel, output_path):
    sample = nusc.get("sample", sample_token)
    camera_token = sample["data"][camera_channel]
    lidar_token = sample["data"]["LIDAR_TOP"]

    image_path, boxes, camera_intrinsic = nusc.get_sample_data(
        camera_token,
        box_vis_level=BoxVisibility.ANY,
    )
    del boxes

    camera_sd = nusc.get("sample_data", camera_token)
    lidar_sd = nusc.get("sample_data", lidar_token)
    camera_cs = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    logger.info("Rendering lidar projection for sample={} camera={}", sample_token, camera_channel)

    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / lidar_sd["filename"]))

    # Lidar -> ego at lidar timestamp.
    pc.rotate(Quaternion(lidar_cs["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_cs["translation"]))

    # Ego -> global.
    pc.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_pose["translation"]))

    # Global -> ego at camera timestamp.
    pc.translate(-np.array(camera_pose["translation"]))
    pc.rotate(Quaternion(camera_pose["rotation"]).rotation_matrix.T)

    # Ego -> camera.
    pc.translate(-np.array(camera_cs["translation"]))
    pc.rotate(Quaternion(camera_cs["rotation"]).rotation_matrix.T)

    image = Image.open(image_path).convert("RGB")
    points = pc.points[:3, :]
    depths = points[2, :]
    visible = depths > 0.1
    projected = view_points(points[:, visible], np.array(camera_intrinsic), normalize=True)

    width, height = image.size
    in_image = (
        (projected[0, :] >= 0)
        & (projected[0, :] < width)
        & (projected[1, :] >= 0)
        & (projected[1, :] < height)
    )

    draw = ImageDraw.Draw(image)
    pixels = projected[:, in_image]
    point_depths = depths[visible][in_image]
    order = np.argsort(point_depths)[::-1]

    for index in order:
        x = float(pixels[0, index])
        y = float(pixels[1, index])
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 80))

    image.save(output_path)
    logger.info("Saved lidar projection image to {}", output_path)


def box_bottom_corners(box):
    return box.bottom_corners()[:2, :]


def render_boxes_on_lidar_bev(nusc, sample_token, output_path):
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE)

    logger.info("Rendering lidar BEV with {} boxes for sample={}", len(boxes), sample_token)

    pc = LidarPointCloud.from_file(str(lidar_path))
    points = pc.points[:3, :]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.scatter(points[0, :], points[1, :], s=0.15, c="0.55", alpha=0.55, linewidths=0)

    for box in boxes:
        corners = box_bottom_corners(box)
        corners = np.concatenate([corners, corners[:, :1]], axis=1)
        color = np.array(category_color(box.name or "Unknown"), dtype=np.float64) / 255.0
        ax.plot(corners[0, :], corners[1, :], color=color, linewidth=1.4)
        center_bottom_forward = np.mean(box.bottom_corners()[:2, :2], axis=1)
        center_bottom = np.mean(box.bottom_corners()[:2, :], axis=1)
        ax.plot(
            [center_bottom[0], center_bottom_forward[0]],
            [center_bottom[1], center_bottom_forward[1]],
            color=color,
            linewidth=1.8,
        )
        center = box.center
        ax.text(center[0], center[1], box.name, color=color, fontsize=5, ha="center", va="center")

    ax.set_title("LIDAR_TOP BEV with 3D boxes")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved lidar BEV image to {}", output_path)


def render_all_camera_projections(nusc, sample_token, output_dir):
    for camera in CAMERAS:
        render_lidar_on_camera(
            nusc=nusc,
            sample_token=sample_token,
            camera_channel=camera,
            output_path=output_dir / "lidar_on_{}.png".format(camera.lower()),
        )


def main():
    args = parse_args()
    dataroot = Path(args.dataroot).resolve()
    output_dir = Path(args.output_dir).resolve() / dataroot.name / args.version
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting nuscenes-devkit validation.")
    logger.info("dataroot={}", dataroot)
    logger.info("version={}", args.version)
    logger.info("output_dir={}", output_dir)

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=True)
    validate_table_counts(nusc)

    if args.sample_index < 0 or args.sample_index >= len(nusc.sample):
        raise IndexError("sample-index {} out of range [0, {})".format(args.sample_index, len(nusc.sample)))

    sample = nusc.sample[args.sample_index]
    sample_token = sample["token"]
    logger.info("Selected sample_index={} sample_token={}", args.sample_index, sample_token)

    render_all_camera_projections(nusc, sample_token, output_dir)
    render_camera_box_mosaic(nusc, sample_token, output_dir / "camera_box_mosaic.png")
    render_boxes_on_lidar_bev(nusc, sample_token, output_dir / "lidar_bev_boxes.png")

    summary = {
        "dataroot": str(dataroot),
        "version": args.version,
        "sample_index": args.sample_index,
        "sample_token": sample_token,
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote validation summary to {}", summary_path)


if __name__ == "__main__":
    main()
