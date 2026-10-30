#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WAIST_CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

MOSAIC_ROWS = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]

BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CLASS_COLORS = {
    "Pedestrian": (255, 166, 0),
    "Car": (0, 214, 143),
    "TrafficLight": (255, 73, 92),
    "TrafficSigns": (61, 145, 255),
    "Poles": (140, 140, 140),
    "Static": (203, 118, 255),
    "Fences": (173, 123, 84),
    "GuardRail": (72, 216, 255),
    "Other": (255, 255, 255),
    "Dynamic": (255, 216, 107),
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DISPLAY_SCALE = 0.6
LIDAR_MAX_DRAW_POINTS = 60000
LIDAR_MAX_DEPTH_M = 50.0
BBOX_NEAR_PLANE_M = 0.25
LIDAR_POINT_RADIUS = 1
MOSAIC_GAP_PX = 18
TITLE_BAR_PX = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render one sequence frame into two six-view mosaics: "
            "top lidar projected to waist cameras, and labels.json 3D boxes projected to waist cameras."
        )
    )
    parser.add_argument("case_dir", help="Sequence case directory, e.g. artifacts/sequences/<sequence_id>.")
    parser.add_argument("frame_id", help="Frame id to visualize, e.g. 1991230.")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Draw class at the top-right corner and object ID at the bottom-right corner of each visible bbox.",
    )
    return parser.parse_args()


def normalize_frame_id(raw_value):
    return str(int(str(raw_value).strip()))


def resolve_frame_key(raw_value, *indexes):
    raw_text = str(raw_value).strip()
    if not raw_text:
        raise KeyError("Empty frame id")

    available_keys = set()
    for index in indexes:
        available_keys.update(str(key) for key in index.keys())

    if raw_text in available_keys:
        return raw_text

    try:
        normalized = normalize_frame_id(raw_text)
    except (TypeError, ValueError):
        normalized = raw_text
    if normalized in available_keys:
        return normalized

    try:
        target = int(raw_text)
    except ValueError as exc:
        raise KeyError("Frame {} not found".format(raw_value)) from exc

    numeric_matches = [key for key in available_keys if key.isdigit() and int(key) == target]
    if numeric_matches:
        return max(numeric_matches, key=lambda key: (len(key), key))

    raise KeyError("Frame {} not found".format(raw_value))


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_index(path, frame_key="frame"):
    index = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index[str(row[frame_key])] = row
    return index


def load_sensor_pose_index(path):
    index = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = str(row["frame"])
            sensor_name = str(row["sensor_name"])
            index.setdefault(frame_id, {})[sensor_name] = row
    return index


def load_label_index(path):
    labels = load_json(path)
    return {str(frame_record["frame_id"]): frame_record for frame_record in labels}


def resolve_sensor_pose_frame(frame_id, sensor_pose_by_frame):
    pose_frames = sorted(sensor_pose_by_frame.keys(), key=lambda value: int(str(value).strip()))
    if not pose_frames:
        return None

    target = int(str(frame_id).strip())
    candidate_index = None
    for index, pose_frame in enumerate(pose_frames):
        pose_value = int(str(pose_frame).strip())
        if pose_value == target:
            candidate_index = index
            break
        if pose_value > target:
            candidate_index = max(0, index - 1)
            break

    if candidate_index is None:
        candidate_index = len(pose_frames) - 1

    if int(str(pose_frames[candidate_index]).strip()) == target and candidate_index > 0:
        return pose_frames[candidate_index - 1]
    return pose_frames[candidate_index]


def carla_transform_matrix(location_xyz, roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    c_y = math.cos(yaw)
    s_y = math.sin(yaw)
    c_r = math.cos(roll)
    s_r = math.sin(roll)
    c_p = math.cos(pitch)
    s_p = math.sin(pitch)

    matrix = np.array(
        [
            [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r, float(location_xyz[0])],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r, float(location_xyz[1])],
            [s_p, -c_p * s_r, c_p * c_r, float(location_xyz[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def invert_rigid_transform(matrix):
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def transform_points(matrix, points_xyz):
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    homogeneous = np.concatenate(
        [points_xyz.astype(np.float64), np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def sensor_record_index(calibration_payload):
    return {str(sensor["name"]): sensor for sensor in calibration_payload["sensors"]}


def resize_image_and_intrinsic(image, intrinsic, scale=DISPLAY_SCALE):
    if scale >= 0.999:
        return image, intrinsic

    new_width = max(1, int(round(image.width * scale)))
    new_height = max(1, int(round(image.height * scale)))
    resized = image.resize((new_width, new_height), resample=Image.LANCZOS)

    scaled_intrinsic = np.array(intrinsic, dtype=np.float64, copy=True)
    scaled_intrinsic[0, :] *= new_width / float(image.width)
    scaled_intrinsic[1, :] *= new_height / float(image.height)
    return resized, scaled_intrinsic


def resolve_sensor_relative_pose(sensor_name, calibration_record, frame_sensor_pose):
    if frame_sensor_pose is not None and sensor_name in frame_sensor_pose:
        row = frame_sensor_pose[sensor_name]
        return {
            "location": [
                float(row["relative_location_x"]),
                float(row["relative_location_y"]),
                float(row["relative_location_z"]),
            ],
            "roll_deg": float(row["relative_rotation_roll"]),
            "pitch_deg": float(row["relative_rotation_pitch"]),
            "yaw_deg": float(row["relative_rotation_yaw"]),
        }

    pitch_deg, yaw_deg, roll_deg = calibration_record.get("rotation", [0.0, 0.0, 0.0])
    return {
        "location": [float(value) for value in calibration_record.get("location", [0.0, 0.0, 0.0])],
        "roll_deg": float(roll_deg),
        "pitch_deg": float(pitch_deg),
        "yaw_deg": float(yaw_deg),
    }


def build_sensor_world_matrix(ego_row, sensor_name, calibration_record, frame_sensor_pose):
    ego_matrix = carla_transform_matrix(
        [
            float(ego_row["location_x"]),
            float(ego_row["location_y"]),
            float(ego_row["location_z"]),
        ],
        roll_deg=float(ego_row["rotation_roll"]),
        pitch_deg=float(ego_row["rotation_pitch"]),
        yaw_deg=float(ego_row["rotation_yaw"]),
    )
    relative_pose = resolve_sensor_relative_pose(sensor_name, calibration_record, frame_sensor_pose)
    relative_matrix = carla_transform_matrix(
        relative_pose["location"],
        roll_deg=relative_pose["roll_deg"],
        pitch_deg=relative_pose["pitch_deg"],
        yaw_deg=relative_pose["yaw_deg"],
    )
    return ego_matrix @ relative_matrix


def project_camera_points(camera_local_points, intrinsic, image_width, image_height):
    if camera_local_points.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)

    points_cv = np.stack(
        [
            camera_local_points[:, 1],
            -camera_local_points[:, 2],
            camera_local_points[:, 0],
        ],
        axis=1,
    )
    depth = points_cv[:, 2]
    visible = depth > 0.1
    pixels = np.full((camera_local_points.shape[0], 2), np.nan, dtype=np.float64)
    if not np.any(visible):
        return pixels, depth, visible

    projected = (intrinsic @ points_cv[visible].T).T
    pixels_visible = projected[:, :2] / projected[:, 2:3]
    in_image = (
        (pixels_visible[:, 0] >= 0.0)
        & (pixels_visible[:, 0] < image_width)
        & (pixels_visible[:, 1] >= 0.0)
        & (pixels_visible[:, 1] < image_height)
    )
    visible_indices = np.flatnonzero(visible)
    pixels[visible_indices[in_image]] = pixels_visible[in_image]
    visible[:] = False
    visible[visible_indices[in_image]] = True
    return pixels, depth, visible


def project_camera_points_unclipped(camera_local_points, intrinsic):
    if camera_local_points.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)

    points_cv = np.stack(
        [
            camera_local_points[:, 1],
            -camera_local_points[:, 2],
            camera_local_points[:, 0],
        ],
        axis=1,
    )
    depth = points_cv[:, 2]
    in_front = depth > 0.1
    pixels = np.full((camera_local_points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(in_front):
        projected = (intrinsic @ points_cv[in_front].T).T
        pixels[in_front] = projected[:, :2] / projected[:, 2:3]
    return pixels, depth, in_front


def camera_local_to_cv(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return np.stack(
        [
            points[:, 1],
            -points[:, 2],
            points[:, 0],
        ],
        axis=1,
    )


def clip_segment_to_near_plane(point_a_cv, point_b_cv, near_plane=BBOX_NEAR_PLANE_M):
    z1 = float(point_a_cv[2])
    z2 = float(point_b_cv[2])

    if z1 < near_plane and z2 < near_plane:
        return None

    p1 = np.array(point_a_cv, dtype=np.float64)
    p2 = np.array(point_b_cv, dtype=np.float64)

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


def project_cv_point(point_cv, intrinsic):
    projected = intrinsic @ np.asarray(point_cv, dtype=np.float64)
    return projected[:2] / projected[2]


def read_lidar_points(path):
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("Unexpected lidar frame layout: {}".format(path))
    points = raw.reshape((-1, 4))
    points_xyz = points[:, :3].copy()
    points_xyz[:, 1] *= -1.0
    return points_xyz


def depth_to_color(depth_value, max_depth=LIDAR_MAX_DEPTH_M):
    clipped = min(max(float(depth_value) / max_depth, 0.0), 1.0)
    hue = (1.0 - clipped) * 0.66
    r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def overlay_lidar_on_image(image, world_points, camera_world_matrix, intrinsic):
    image_width, image_height = image.size
    world_to_camera = invert_rigid_transform(camera_world_matrix)
    camera_points = transform_points(world_to_camera, world_points)
    pixels, depth, visible = project_camera_points(camera_points, intrinsic, image_width, image_height)
    visible_indices = np.flatnonzero(visible)
    if visible_indices.size == 0:
        return image

    if visible_indices.size > LIDAR_MAX_DRAW_POINTS:
        sample_positions = np.linspace(0, visible_indices.size - 1, LIDAR_MAX_DRAW_POINTS, dtype=np.int64)
        visible_indices = visible_indices[sample_positions]

    ordered = visible_indices[np.argsort(depth[visible_indices])[::-1]]
    draw = ImageDraw.Draw(image)
    for index in ordered:
        x, y = pixels[index]
        color = depth_to_color(depth[index])
        draw.ellipse(
            (
                x - LIDAR_POINT_RADIUS,
                y - LIDAR_POINT_RADIUS,
                x + LIDAR_POINT_RADIUS,
                y + LIDAR_POINT_RADIUS,
            ),
            fill=color,
        )
    return image


def rotation_matrix_from_radians(roll_rad, pitch_rad, yaw_rad):
    c_y = math.cos(yaw_rad)
    s_y = math.sin(yaw_rad)
    c_r = math.cos(roll_rad)
    s_r = math.sin(roll_rad)
    c_p = math.cos(pitch_rad)
    s_p = math.sin(pitch_rad)
    return np.array(
        [
            [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r],
            [s_p, -c_p * s_r, c_p * c_r],
        ],
        dtype=np.float64,
    )


def build_bbox_corners_world(object_record):
    dimensions = np.array(object_record["dimensions"], dtype=np.float64)
    half = dimensions * 0.5
    local_corners = np.array(
        [
            [half[0], half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], -half[2]],
            [half[0], half[1], half[2]],
            [half[0], -half[1], half[2]],
            [-half[0], half[1], half[2]],
            [-half[0], -half[1], half[2]],
        ],
        dtype=np.float64,
    )
    roll_rad, pitch_rad, yaw_rad = [float(value) for value in object_record["rotation"]]
    rotation = rotation_matrix_from_radians(roll_rad, pitch_rad, yaw_rad)
    location = np.array(object_record["location"], dtype=np.float64)
    return (rotation @ local_corners.T).T + location


def segment_overlaps_image(p1, p2, width, height):
    min_x = min(p1[0], p2[0])
    max_x = max(p1[0], p2[0])
    min_y = min(p1[1], p2[1])
    max_y = max(p1[1], p2[1])
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


def draw_text_with_background(draw, position_xy, text, fill_color, anchor):
    font = ImageFont.load_default()
    text_width, text_height = font.getsize(text)
    padding_x = 3
    padding_y = 2

    if anchor == "rt":
        x0 = float(position_xy[0]) - text_width - 2 * padding_x
        y0 = float(position_xy[1])
    elif anchor == "rb":
        x0 = float(position_xy[0]) - text_width - 2 * padding_x
        y0 = float(position_xy[1]) - text_height - 2 * padding_y
    else:
        x0 = float(position_xy[0])
        y0 = float(position_xy[1])

    x1 = x0 + text_width + 2 * padding_x
    y1 = y0 + text_height + 2 * padding_y
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    draw.text((x0 + padding_x, y0 + padding_y), text, fill=fill_color, font=font)


def draw_bboxes_on_image(image, objects, camera_world_matrix, intrinsic, detail=False):
    image_width, image_height = image.size
    world_to_camera = invert_rigid_transform(camera_world_matrix)
    draw = ImageDraw.Draw(image)

    for object_record in objects:
        if int(object_record.get("occlusion", 0)) == 4:
            continue

        # if str(object_record.get("class", "Unknown")) != "Bicycle":
        #     continue

        world_corners = build_bbox_corners_world(object_record)
        camera_points = transform_points(world_to_camera, world_corners)
        points_cv = camera_local_to_cv(camera_points)
        if not np.any(points_cv[:, 2] > BBOX_NEAR_PLANE_M):
            continue

        color = CLASS_COLORS.get(object_record.get("class"), (255, 255, 255))
        line_width = 3 if object_record.get("object_id", -1) >= 0 else 2
        projected_points = []
        for start_index, end_index in BOX_EDGES:
            clipped_3d = clip_segment_to_near_plane(points_cv[start_index], points_cv[end_index])
            if clipped_3d is None:
                continue
            p1 = project_cv_point(clipped_3d[0], intrinsic)
            p2 = project_cv_point(clipped_3d[1], intrinsic)
            if not segment_overlaps_image(p1, p2, image_width, image_height):
                continue
            clipped = clip_line_to_image(p1, p2, image_width, image_height)
            if clipped is None:
                continue
            c1, c2 = clipped
            projected_points.extend([c1, c2])
            draw.line((float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1])), fill=color, width=line_width)

        if detail and projected_points:
            points_array = np.asarray(projected_points, dtype=np.float64)
            right_x = float(np.max(points_array[:, 0]))
            top_y = float(np.min(points_array[:, 1]))
            bottom_y = float(np.max(points_array[:, 1]))
            class_name = str(object_record.get("class", "Unknown"))
            object_id = object_record.get("object_id", -1)
            draw_text_with_background(draw, (right_x, top_y), class_name, color, anchor="rt")
            draw_text_with_background(draw, (right_x, bottom_y), str(object_id), color, anchor="rb")
    return image


def open_camera_image(case_dir, camera_name, frame_id):
    camera_dir = case_dir / "sampling" / "image" / camera_name
    for extension in IMAGE_EXTENSIONS:
        candidate = camera_dir / "{}{}".format(frame_id, extension)
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    raise FileNotFoundError("Missing image for {} frame {} under {}".format(camera_name, frame_id, camera_dir))


def add_tile_title(image, title):
    titled = Image.new("RGB", (image.width, image.height + TITLE_BAR_PX), color=(14, 18, 24))
    titled.paste(image, (0, TITLE_BAR_PX))
    draw = ImageDraw.Draw(titled)
    draw.text((12, 8), title, fill=(240, 242, 245), font=ImageFont.load_default())
    return titled


def build_mosaic(images_by_camera, heading):
    sample_image = next(iter(images_by_camera.values()))
    tile_width = sample_image.width
    tile_height = sample_image.height + TITLE_BAR_PX
    rows = len(MOSAIC_ROWS)
    cols = len(MOSAIC_ROWS[0])
    canvas_width = cols * tile_width + (cols + 1) * MOSAIC_GAP_PX
    canvas_height = rows * tile_height + (rows + 1) * MOSAIC_GAP_PX + 42

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(8, 10, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((MOSAIC_GAP_PX, 10), heading, fill=(244, 246, 248), font=ImageFont.load_default())

    for row_index, row in enumerate(MOSAIC_ROWS):
        for col_index, camera_name in enumerate(row):
            tile = add_tile_title(images_by_camera[camera_name], camera_name)
            offset_x = MOSAIC_GAP_PX + col_index * (tile_width + MOSAIC_GAP_PX)
            offset_y = 32 + MOSAIC_GAP_PX + row_index * (tile_height + MOSAIC_GAP_PX)
            canvas.paste(tile, (offset_x, offset_y))
    return canvas


def ensure_sequence_layout(case_dir):
    sampling_dir = case_dir / "sampling"
    if not sampling_dir.is_dir():
        raise FileNotFoundError("Expected sampling directory under {}".format(case_dir))
    return sampling_dir


def main():
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()

    sampling_dir = ensure_sequence_layout(case_dir)
    visualization_dir = case_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    calibration_payload = load_json(sampling_dir / "calibration.json")
    calibration_by_sensor = sensor_record_index(calibration_payload)
    ego_by_frame = load_csv_index(sampling_dir / "ego.csv")
    sensor_pose_by_frame = load_sensor_pose_index(sampling_dir / "sensor_pose.csv")
    labels_by_frame = load_label_index(sampling_dir / "labels.json")
    frame_id = resolve_frame_key(args.frame_id, ego_by_frame, labels_by_frame)

    if frame_id not in ego_by_frame:
        raise KeyError("Frame {} not found in ego.csv".format(frame_id))
    if frame_id not in labels_by_frame:
        raise KeyError("Frame {} not found in labels.json".format(frame_id))

    lidar_path = sampling_dir / "velodyne" / "LIDAR_TOP" / "{}.bin".format(frame_id)
    if not lidar_path.exists():
        raise FileNotFoundError("Missing top lidar frame {}".format(lidar_path))

    ego_row = ego_by_frame[frame_id]
    pose_frame_id = resolve_sensor_pose_frame(frame_id, sensor_pose_by_frame)
    frame_sensor_pose = sensor_pose_by_frame.get(pose_frame_id, {})
    frame_objects = labels_by_frame[frame_id]["objects"]

    lidar_world_matrix = build_sensor_world_matrix(
        ego_row=ego_row,
        sensor_name="LIDAR_TOP",
        calibration_record=calibration_by_sensor["LIDAR_TOP"],
        frame_sensor_pose=frame_sensor_pose,
    )
    lidar_points_world = transform_points(lidar_world_matrix, read_lidar_points(lidar_path))

    lidar_images = {}
    bbox_images = {}
    for camera_name in WAIST_CAMERAS:
        calibration_record = calibration_by_sensor[camera_name]
        intrinsic = np.asarray(calibration_record["camera_intrinsic"], dtype=np.float64)
        camera_world_matrix = build_sensor_world_matrix(
            ego_row=ego_row,
            sensor_name=camera_name,
            calibration_record=calibration_record,
            frame_sensor_pose=frame_sensor_pose,
        )

        base_image = open_camera_image(case_dir, camera_name, frame_id)
        base_image, intrinsic = resize_image_and_intrinsic(base_image, intrinsic)
        lidar_image = overlay_lidar_on_image(base_image.copy(), lidar_points_world, camera_world_matrix, intrinsic)
        bbox_image = draw_bboxes_on_image(
            base_image.copy(),
            frame_objects,
            camera_world_matrix,
            intrinsic,
            detail=args.detail,
        )

        lidar_images[camera_name] = lidar_image
        bbox_images[camera_name] = bbox_image

    lidar_mosaic = build_mosaic(lidar_images, "Frame {}  Top Lidar Projection".format(frame_id))
    bbox_mosaic = build_mosaic(bbox_images, "Frame {}  labels.json 3D BBox Projection".format(frame_id))

    lidar_output = visualization_dir / "frame_{}_lidar_overlay_mosaic.png".format(frame_id)
    bbox_output = visualization_dir / "frame_{}_bbox_overlay_mosaic.png".format(frame_id)
    lidar_mosaic.save(lidar_output)
    bbox_mosaic.save(bbox_output)

    print(
        json.dumps(
            {
                "frame_id": frame_id,
                "pose_frame_id": pose_frame_id,
                "case_dir": str(case_dir),
                "lidar_mosaic": str(lidar_output),
                "bbox_mosaic": str(bbox_output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
