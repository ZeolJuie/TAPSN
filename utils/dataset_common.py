import csv
import json
import math
from pathlib import Path

import numpy as np


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_csv_index(path, key="frame"):
    index = {}
    for row in load_csv_rows(path):
        index[str(row[key])] = row
    return index


def load_sensor_pose_index(path):
    index = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = str(row["frame"])
            sensor_name = str(row["sensor_name"])
            index.setdefault(frame_id, {})[sensor_name] = row
    return index


def normalize_frame_id(value):
    return str(int(str(value).strip()))


def sensor_record_index(calibration_payload):
    return {str(sensor["name"]): sensor for sensor in calibration_payload["sensors"]}


def sequence_dirs_from_root(data_root):
    root = Path(data_root)
    return [
        path for path in sorted(root.iterdir())
        if path.is_dir() and (path / "sampling" / "labels.json").exists()
    ]


def iter_sequence_dirs(input_path):
    path = Path(input_path)
    if (path / "sampling" / "labels.json").exists():
        return [path]
    return sequence_dirs_from_root(path)


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

    return np.array(
        [
            [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r, float(location_xyz[0])],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r, float(location_xyz[1])],
            [s_p, -c_p * s_r, c_p * c_r, float(location_xyz[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_points(matrix, points_xyz):
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    homogeneous = np.concatenate(
        [points_xyz.astype(np.float64), np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def resolve_sensor_pose_frame(frame_id, sensor_pose_by_frame):
    pose_frames = sorted(int(value) for value in sensor_pose_by_frame.keys())
    if not pose_frames:
        return None

    target = int(frame_id)
    candidate_index = None
    for index, pose_frame in enumerate(pose_frames):
        if pose_frame == target:
            candidate_index = index
            break
        if pose_frame > target:
            candidate_index = max(0, index - 1)
            break

    if candidate_index is None:
        candidate_index = len(pose_frames) - 1

    # Sensor pose rows are applied after the frame is captured, so export uses the previous row.
    if pose_frames[candidate_index] == target and candidate_index > 0:
        return str(pose_frames[candidate_index - 1])
    return str(pose_frames[candidate_index])


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


def build_ego_world_matrix(ego_row):
    return carla_transform_matrix(
        [
            float(ego_row["location_x"]),
            float(ego_row["location_y"]),
            float(ego_row["location_z"]),
        ],
        roll_deg=float(ego_row["rotation_roll"]),
        pitch_deg=float(ego_row["rotation_pitch"]),
        yaw_deg=float(ego_row["rotation_yaw"]),
    )


def build_sensor_world_matrix(ego_row, sensor_name, calibration_record, frame_sensor_pose):
    ego_matrix = build_ego_world_matrix(ego_row)
    relative_pose = resolve_sensor_relative_pose(sensor_name, calibration_record, frame_sensor_pose)
    relative_matrix = carla_transform_matrix(
        relative_pose["location"],
        roll_deg=relative_pose["roll_deg"],
        pitch_deg=relative_pose["pitch_deg"],
        yaw_deg=relative_pose["yaw_deg"],
    )
    return ego_matrix @ relative_matrix
