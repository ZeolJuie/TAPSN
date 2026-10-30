#!/usr/bin/env python3
import hashlib
import math

import numpy as np

from utils.dataset_common import normalize_frame_id


SUPPORTED_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]
SUPPORTED_LIDAR = "LIDAR_TOP"
SUPPORTED_CHANNELS = SUPPORTED_CAMERAS + [SUPPORTED_LIDAR]

CATEGORY_ORDER = [
    "Pedestrian",
    "Car",
    "Bicycle",
    "Motorcycle",
    "Truck",
    "Bus",
    "Van",
    "TrafficLight",
    "TrafficSigns",
    "Static",
    "Poles",
    "Fences",
    "Dynamic",
    "Other",
    "GuardRail",
]


def standard_scene_index_from_ids(map_id, sequence_id):
    return int(str(map_id).zfill(2)) * 100 + int(str(sequence_id).zfill(2))


def standard_scene_name_from_ids(map_id, sequence_id):
    return "scene-{:04d}".format(standard_scene_index_from_ids(map_id, sequence_id))


def standard_scene_prefix_from_ids(map_id, sequence_id):
    return "{:04d}".format(standard_scene_index_from_ids(map_id, sequence_id))


def deterministic_token(*parts):
    digest = hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:32]


def deterministic_timestamp_us(sequence_id, frame_id, sample_time_seconds):
    frame_component = int(normalize_frame_id(frame_id))
    sequence_component = int(hashlib.sha1(str(sequence_id).encode("utf-8")).hexdigest()[:6], 16)
    return sequence_component * 10**9 + int(round(float(sample_time_seconds) * 1_000_000.0)) + frame_component


def matrix_to_nuscenes_translation_rotation(matrix):
    translation_carla = matrix[:3, 3]
    rotation_carla = matrix[:3, :3]

    # CARLA world: x forward, y right, z up
    # nuScenes convention used here: x forward, y left, z up
    axis_flip = np.diag([1.0, -1.0, 1.0])
    rotation_nusc = axis_flip @ rotation_carla @ axis_flip
    translation_nusc = axis_flip @ translation_carla

    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation_nusc)
    quaternion_wxyz = [
        quaternion_xyzw[3],
        quaternion_xyzw[0],
        quaternion_xyzw[1],
        quaternion_xyzw[2],
    ]
    return translation_nusc.tolist(), quaternion_wxyz


def sensor_matrix_to_nuscenes_translation_rotation(matrix, sensor_modality):
    translation_carla = matrix[:3, 3]
    rotation_carla = matrix[:3, :3]

    # CARLA world: x forward, y right, z up
    # nuScenes-like world used here: x forward, y left, z up
    axis_flip = np.diag([1.0, -1.0, 1.0])
    rotation_nusc = axis_flip @ rotation_carla @ axis_flip
    translation_nusc = axis_flip @ translation_carla

    if sensor_modality == "camera":
        # `rotation_nusc` above has already converted the local frame from
        # CARLA (x forward, y right, z up) into a right-handed nuScenes-like
        # local basis (x forward, y left, z up) via the axis flip on both sides.
        # The docs define the projection frame on raw CARLA camera coords as:
        #   [x_carla, y_carla, z_carla] -> [y_carla, -z_carla, x_carla]
        # Re-expressed after the local y-axis flip, this becomes:
        #   [x_forward, y_left, z_up] -> [x_right, y_down, z_forward]
        #                                = [-y_left, -z_up, x_forward]
        camera_to_optical = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        rotation_nusc = rotation_nusc @ camera_to_optical.T

    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation_nusc)
    quaternion_wxyz = [
        quaternion_xyzw[3],
        quaternion_xyzw[0],
        quaternion_xyzw[1],
        quaternion_xyzw[2],
    ]
    return translation_nusc.tolist(), quaternion_wxyz


def rotation_matrix_to_quaternion_xyzw(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
        return [x, y, z, w]

    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
        return [x, y, z, w]

    if rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
        return [x, y, z, w]

    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    w = (rotation[1, 0] - rotation[0, 1]) / s
    x = (rotation[0, 2] + rotation[2, 0]) / s
    y = (rotation[1, 2] + rotation[2, 1]) / s
    z = 0.25 * s
    return [x, y, z, w]


def object_rotation_to_nuscenes_quaternion(rotation_rpy_rad):
    roll = float(rotation_rpy_rad[0])
    pitch = float(rotation_rpy_rad[1])
    yaw = float(rotation_rpy_rad[2])

    c_y = math.cos(yaw * 0.5)
    s_y = math.sin(yaw * 0.5)
    c_p = math.cos(pitch * 0.5)
    s_p = math.sin(pitch * 0.5)
    c_r = math.cos(roll * 0.5)
    s_r = math.sin(roll * 0.5)

    x = s_r * c_p * c_y - c_r * s_p * s_y
    y = c_r * s_p * c_y + s_r * c_p * s_y
    z = c_r * c_p * s_y - s_r * s_p * c_y
    w = c_r * c_p * c_y + s_r * s_p * s_y

    # CARLA world to nuScenes-like world axis flip.
    q_flip = [0.0, 1.0, 0.0, 0.0]
    q_out = quaternion_multiply(quaternion_multiply(q_flip, [x, y, z, w]), q_flip)
    return [q_out[3], q_out[0], q_out[1], q_out[2]]


def quaternion_multiply(q1_xyzw, q2_xyzw):
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def category_names_from_labels(label_frames):
    seen = {obj["class"] for frame in label_frames for obj in frame["objects"]}
    ordered = [name for name in CATEGORY_ORDER if name in seen]
    for extra in sorted(seen - set(CATEGORY_ORDER)):
        ordered.append(extra)
    return ordered
