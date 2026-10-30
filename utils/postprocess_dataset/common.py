#!/usr/bin/env python3
import math

import numpy as np


SEMANTIC_LIDAR_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("cos_angle", np.float32),
        ("obj_idx", np.uint32),
        ("obj_tag", np.uint32),
    ]
)


def read_semantic_lidar_points_xyz(path):
    raw = np.fromfile(str(path), dtype=SEMANTIC_LIDAR_DTYPE)
    points_xyz = np.column_stack([raw["x"], -raw["y"], raw["z"]]).astype(np.float64, copy=False)
    return points_xyz, raw


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


def make_oriented_box_test_data(object_record):
    roll_rad, pitch_rad, yaw_rad = [float(value) for value in object_record["rotation"]]
    rotation = rotation_matrix_from_radians(roll_rad, pitch_rad, yaw_rad)
    center = np.array(object_record["location"], dtype=np.float64)
    half_extents = np.array(object_record["dimensions"], dtype=np.float64) * 0.5
    return center, rotation, half_extents


def count_points_in_oriented_box(points_world, object_record, eps=1e-6):
    if points_world.size == 0:
        return 0
    center, rotation, half_extents = make_oriented_box_test_data(object_record)
    local_points = (rotation.T @ (points_world - center).T).T
    inside = np.all(np.abs(local_points) <= (half_extents + eps), axis=1)
    return int(np.count_nonzero(inside))
