#!/usr/bin/env python3
import os
import csv
import yaml
import json
import re
import torch
import chamfer
import glob
import time
import open3d as o3d
import numpy as np
import sys
from pathlib import Path

import carla
from mmcv.ops.points_in_boxes import points_in_boxes_cpu 
from scipy.spatial.transform import Rotation
from copy import deepcopy

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_common import (  # noqa: E402
    build_sensor_world_matrix,
    normalize_frame_id,
    resolve_sensor_pose_frame,
    sensor_record_index,
)
from utils.nuscenes_converter.common import (  # noqa: E402
    standard_scene_name_from_ids,
)


camera_configs = [
    {
        'name': 'CAM_FRONT',
        'transforms': {
            'location': {'x': 0.25, 'y': 0.0, 'z': 0.20}, 
            'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        },
        'fov': 120,
        'image_size': (1600, 900)
    },
    {
        'name': 'CAM_FRONT_RIGHT',
        'transforms': {
            'location': {'x': 0.13, 'y': 0.22, 'z': 0.20}, 
            'rotation': {'pitch': 0.0, 'yaw': 60.0, 'roll': 0.0}
        },
        'fov': 70,
        'image_size': (1600, 900)
    },
    {
        'name': 'CAM_FRONT_LEFT',
        'transforms': {
            'location': {'x': 0.13, 'y': -0.22, 'z': 0.20}, 
            'rotation': {'pitch': 0.0, 'yaw': -60.0, 'roll': 0.0}
        },
        'fov': 70,
        'image_size': (1600, 900)
    },
    {
        'name': 'CAM_BACK',
        'transforms': {
            'location': {'x': -0.25, 'y': 0.0, 'z': 0.20}, 
            'rotation': {'pitch': 0.0, 'yaw': 180.0, 'roll': 0.0}
        },
        'fov': 110,
        'image_size': (1600, 900)
    },
    {
        'name': 'CAM_BACK_RIGHT',
        'transforms': {
            'location': {'x': -0.13, 'y': 0.22, 'z': 0.20}, 
            'rotation': {'pitch': 0.0, 'yaw': 120.0, 'roll': 0.0}
        },
        'fov': 70,
        'image_size': (1600, 900)
    },
    {
        'name': 'CAM_BACK_LEFT',
        'transforms': {
            'location': {'x': -0.13, 'y': -0.22, 'z': 0.20}, 
            'rotation': {'pitch': 0.0, 'yaw': -120.0, 'roll': 0.0}
        },
        'fov': 70,
        'image_size': (1600, 900)
    },
    {
        'name': 'CAM_TOP',
        'transforms': {
            'location': {'x': 0.0, 'y': 0.0, 'z': 1.00}, 
            'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        },
        'fov': 120,
        'image_size': (1600, 900)
    }
]


def infer_standard_scene_name_from_path(sequence_path):
    sequence_dir = Path(sequence_path).resolve()
    if sequence_dir.name == "sampling":
        sequence_dir = sequence_dir.parent
    town_name = sequence_dir.parent.name
    sequence_id = sequence_dir.name
    match = re.match(r"^Town(\d+)", town_name)
    if match is None:
        raise RuntimeError(
            "Sequence {} parent directory must start with Town<number>, got {}".format(
                sequence_dir,
                town_name,
            )
        )
    map_id = match.group(1).zfill(2)
    return standard_scene_name_from_ids(map_id, sequence_id)


def world_to_camera(point, cam_pose):
    """将点从世界坐标系转换到相机坐标系"""
    # 提取旋转和平移
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]
    
    # 转换到相机坐标系
    point_cam = R.T @ (point - t)
    return point_cam


def is_in_frustum(point_cam, fov_x, fov_y, near=0.1, far=100.0):
    """检查点是否在相机视锥体内"""
    x, y, z = point_cam
    
    # 检查深度范围
    if not (near <= z <= far):
        return False
    
    # 检查水平视角
    half_fov_x = np.radians(fov_x) / 2
    if abs(x) > z * np.tan(half_fov_x):
        return False
    
    # 检查垂直视角
    half_fov_y = np.radians(fov_y) / 2
    if abs(y) > z * np.tan(half_fov_y):
        return False
    
    return True


def bresenham_3d(start, end, pc_range=None, voxel_size=None):
    """
    严格修正的3D Bresenham算法：
    1. 动态选择主步进轴（X/Y/Z中跨度最大的方向）
    2. 处理所有边界条件（包括起点=终点的情况）
    3. 返回np.ndarray类型的体素坐标数组
    
    参数:
        start: [x,y,z] 起点坐标（世界坐标系）
        end: [x,y,z] 终点坐标（世界坐标系）
        pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: 体素大小（单位：米）
    
    返回:
        np.ndarray: 射线穿过的体素坐标 [[x_idx, y_idx, z_idx], ...]
    """
    # === 1. 参数校验 ===
    assert pc_range is not None and voxel_size is not None, "必须提供pc_range和voxel_size"
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    
    # === 2. 转换为体素网格坐标 ===
    grid_min = np.array(pc_range[:3])
    start_voxel = ((start - grid_min) / voxel_size).astype(int)
    end_voxel = ((end - grid_min) / voxel_size).astype(int)
    
    # === 3. 初始化Bresenham算法参数 ===
    x0, y0, z0 = start_voxel
    x1, y1, z1 = end_voxel
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    sz = 1 if z0 < z1 else -1

    voxels = []
    
    # === 4. 动态选择主步进轴 ===
    if dx >= dy and dx >= dz:
        # X轴为主步进轴
        err1 = 2 * dy - dx
        err2 = 2 * dz - dx
        for _ in range(dx + 1):
            voxels.append([x0, y0, z0])
            if x0 == x1:
                break
            if err1 > 0:
                y0 += sy
                err1 -= 2 * dx
            if err2 > 0:
                z0 += sz
                err2 -= 2 * dx
            err1 += 2 * dy
            err2 += 2 * dz
            x0 += sx
    elif dy >= dx and dy >= dz:
        # Y轴为主步进轴
        err1 = 2 * dx - dy
        err2 = 2 * dz - dy
        for _ in range(dy + 1):
            voxels.append([x0, y0, z0])
            if y0 == y1:
                break
            if err1 > 0:
                x0 += sx
                err1 -= 2 * dy
            if err2 > 0:
                z0 += sz
                err2 -= 2 * dy
            err1 += 2 * dx
            err2 += 2 * dz
            y0 += sy
    else:
        # Z轴为主步进轴
        err1 = 2 * dy - dz
        err2 = 2 * dx - dz
        for _ in range(dz + 1):
            voxels.append([x0, y0, z0])
            if z0 == z1:
                break
            if err1 > 0:
                y0 += sy
                err1 -= 2 * dz
            if err2 > 0:
                x0 += sx
                err2 -= 2 * dz
            err1 += 2 * dy
            err2 += 2 * dx
            z0 += sz

    # === 5. 返回结果 ===
    return np.array(voxels, dtype=int)


def compute_camera_visibility_mask(
    occupied_voxels, camera_poses, camera_configs, 
    pc_range, voxel_size, grid_shape
):
    """改进的相机可见性计算"""
    mask = np.zeros(grid_shape, dtype=np.int8)
    occupied_voxels_set = {tuple(v) for v in occupied_voxels}
    
    for cam_pose, cam_config in zip(camera_poses, camera_configs):
        cam_center = cam_pose[:3, 3]
        
        for voxel in occupied_voxels:
            voxel_center = (voxel + 0.5) * voxel_size + pc_range[:3]

            # 进行射线追踪
            ray_voxels = bresenham_3d(cam_center, voxel_center, pc_range, voxel_size)
            
            for idx, v in enumerate(ray_voxels):
                v_tuple = tuple(v)
                # 检查体素是否在网格范围内
                if not (0 <= v[0] < grid_shape[0] and 
                        0 <= v[1] < grid_shape[1] and 
                        0 <= v[2] < grid_shape[2]):
                    continue
                
                # 标记为可见
                mask[v_tuple] = 1
                
                # 如果是被占用体素，终止射线
                if v_tuple in occupied_voxels_set:
                    break
    
    return mask


def point_transform_3d(loc, M):
    """ 
        Transform a 3D point using a 4x4 matrix
    """
 
    point = np.array([loc.x, loc.y, loc.z, 1]) if isinstance(loc, carla.libcarla.Location) else np.array([loc[0], loc[1], loc[2], 1])
    point_transformed = np.dot(M, point)
    # normalize, 其实最后一位就是1.0
    point_transformed[0] /= point_transformed[3]
    point_transformed[1] /= point_transformed[3]
    point_transformed[2] /= point_transformed[3]
    return point_transformed[:3]


def point_transform_3d_batch(loc, M):
    """ 
        Transform a 3D point using a 4x4 matrix
        loc: Nx3 array point location 
        out: Nx3 array transformed point location 
    """
    point = np.concatenate((loc, np.ones((loc.shape[0], 1))), axis=1)
    point = point[:, :, np.newaxis]
    M = M[np.newaxis, :, :]
    point_transformed = np.matmul(M, point)[:, :, 0]
    # normalize, 其实最后一位就是1.0
    point_transformed[:, 0] /= point_transformed[:, 3]
    point_transformed[:, 1] /= point_transformed[:, 3]
    point_transformed[:, 2] /= point_transformed[:, 3]
    return point_transformed[:, :3]


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=n_threads
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities


def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original=None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )


def world_to_lidar_obj(obj, lidar_to_world):
    """
    将世界坐标系下的物体转换到LiDAR坐标系
    
    参数:
        obj: dict - 世界坐标系下的物体信息（包含location和rotation）
        lidar_to_world: np.ndarray - 4x4的LiDAR到世界坐标系的变换矩阵
    
    返回:
        dict - LiDAR坐标系下的物体信息（保持原JSON格式）
    """
    # 提取世界坐标系下的位置和旋转
    world_loc = np.array([obj["location"][0], obj["location"][1], obj["location"][2]])
    world_rot = np.array(obj["rotation"])  # 欧拉角（通常为ZYX顺序）

    # 计算world_to_lidar的变换矩阵（求逆）
    world_to_lidar = np.linalg.inv(lidar_to_world)
    
    # 转换位置（齐次坐标）
    world_loc_hom = np.append(world_loc, 1.0)  # [x, y, z, 1]
    lidar_loc_hom = np.dot(world_to_lidar, world_loc_hom)
    lidar_loc = lidar_loc_hom[:3].tolist()  # 取前3维并转为list

    # 转换旋转（假设旋转是绕Z轴的偏航角yaw）
    # 方法：从变换矩阵中提取旋转部分，并调整欧拉角
    # 注意：这里简化处理，仅适用于3D旋转中的偏航角（Yaw）
    lidar_yaw = world_rot[2] - np.arctan2(lidar_to_world[1, 0], lidar_to_world[0, 0])
    # 保持roll和pitch为0
    lidar_rot = [0.0, 0.0, lidar_yaw]

    # 构建输出对象（其他字段不变）
    lidar_obj = {
        "object_id": obj["object_id"],
        "class": obj["class"],
        "truncation": obj["truncation"],
        "occlusion": obj["occlusion"],
        "location": lidar_loc,
        "dimensions": obj["dimensions"],
        "rotation": lidar_rot
    }
    
    return lidar_obj


def convert_bbox_format(original_bbox):
    """
    将原始边界框格式转换为标准格式 [x, y, z, dx, dy, dz, rz]
    
    参数:
        original_bbox (dict): 原始边界框数据，包含location(中心点), dimensions和rotation
    
    返回:
        np.ndarray: 转换后的边界框 [x, y, z, dx, dy, dz, rz]
    """
    # 提取原始数据
    location = original_bbox["location"]  # [x, y, z] (原始中心点坐标)
    dimensions = [
        original_bbox["dimensions"][0],
        original_bbox["dimensions"][1],
        original_bbox["dimensions"][2]
    ]# [dx, dy, dz] (长宽高)
    rotation_z = original_bbox["rotation"][2]  # 只需要z轴旋转角度
    
    # 计算底部中心坐标 (y轴向下调整半个高度)
    bottom_center = [
        location[0],  # x不变
        location[1],  # y不变
        # location[2] - dimensions[2]/2  # z向下移动半个高度
        location[2]
    ]
    
    # 组装为标准格式
    converted = bottom_center + dimensions + [rotation_z]
    
    return converted


def get_category_index(category):
    category_index_map = {
        'Bus': 15,
        'Car': 14,
        'Truck': 16,
        'Motorcycle': 18,
        'Bicycle': 19,
        'Pedestrian': 12,
        'Van': 15,
        }
    return category_index_map[category]


def get_extrinsic_matrix(x, y, z, roll, pitch, yaw):
    """
    根据平移和欧拉角生成4x4外参矩阵
    
    参数:
        x, y, z: 平移分量 (单位: 米)
        roll, pitch, yaw: 绕X/Y/Z轴的旋转角度 (单位: 弧度)
        (roll: X轴, pitch: Y轴, yaw: Z轴)
    
    返回:
        4x4 numpy数组表示的齐次变换矩阵
        格式:
        [[R11, R12, R13, x],
         [R21, R22, R23, y],
         [R31, R32, R33, z],
         [0,   0,   0,   1]]
    """

    # 计算旋转矩阵的各元素 (使用简写符号)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # 构建旋转矩阵 (Z-Y-X顺序，即yaw->pitch->roll)
    rotation_matrix = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr]
    ])
    
    # 构建4x4齐次变换矩阵
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = [x, y, z]
    
    return extrinsic


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为旋转矩阵。
    :param roll: 绕 X 轴的旋转角度（弧度）。
    :param pitch: 绕 Y 轴的旋转角度（弧度）。
    :param yaw: 绕 Z 轴的旋转角度（弧度）。
    :return: 3x3 旋转矩阵。
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx  # 旋转顺序：Z-Y-X
    return R


def calculate_cube_vertices(transform, rotation, dimension):
    """
    计算立方体的8个顶点坐标
    
    参数:
        transform (list/tuple/np.array): 立方体中心的平移向量 [x, y, z]
        rotation (list/tuple/np.array): 立方体的旋转角度 [rx, ry, rz] (弧度)
        dimension (list/tuple/np.array): 立方体的尺寸 [width, height, depth]
    
    返回:
        np.array: 8个顶点坐标的数组，形状为(8, 3)
    """
    # 转换为numpy数组
    transform = np.array(transform)
    rotation = np.array(rotation)
    dimension = np.array(dimension)
    
    # 立方体的局部坐标 (未旋转和平移前)
    half_dim = dimension / 2.0
    vertices_local = np.array([
        [-1, -1, -1],  # 0: 左前下
        [-1,  1, -1],  # 1: 右前下
        [ 1, -1, -1],  # 2: 右后下
        [ 1,  1, -1],  # 3: 左后下
        [-1, -1,  1],  # 4: 左前上
        [-1,  1,  1],  # 5: 右前上
        [ 1, -1,  1],  # 6: 右后上
        [ 1,  1,  1]   # 7: 左后上
    ]) * half_dim
    
    # 创建旋转矩阵 (绕x, y, z轴旋转)
    rx, ry, rz = rotation
    rotation_matrix = euler_to_rotation_matrix(rx, ry, rz)
    
    # 应用旋转和平移
    vertices_global = np.dot(vertices_local, rotation_matrix.T) + transform
    
    return vertices_global


def load_config(config_path):
    if str(config_path).lower().endswith('.toml'):
        with open(config_path, 'rb') as stream:
            payload = tomllib.load(stream)
        config = payload.get('occ', payload)
    else:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

    config = dict(config)
    config.setdefault('max_nn', 20)
    config.setdefault('depth', 11)
    config.setdefault('n_threads', 8)
    config.setdefault('min_density', None)
    config.setdefault('self_range', [1.0, 1.0, 1.5])
    config.setdefault('len_sequence', 40)
    config.setdefault('len_side', 4)
    return config


def load_sensor_pose_index(path):
    index = {}
    with open(path, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = normalize_frame_id(row['frame'])
            sensor_name = str(row['sensor_name'])
            index.setdefault(frame_id, {})[sensor_name] = row
    return index


def parse_actor_attribute_value(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value)
    marker = 'value='
    start = text.find(marker)
    if start < 0:
        return text
    value = text[start + len(marker):]
    if value.endswith(')'):
        value = value[:-1]
    value = value.strip()
    try:
        if '.' in value or 'e' in value or 'E' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def is_rgb_camera_sensor(sensor_record):
    return str(sensor_record.get('type', '')).lower() == 'camera.rgb'


def build_camera_configs_from_calibration(camera_configs_in, calibration_by_sensor):
    base_camera_configs = {cam_cfg['name']: deepcopy(cam_cfg) for cam_cfg in camera_configs_in}
    camera_configs_out = []
    for sensor_name, sensor_record in calibration_by_sensor.items():
        if not is_rgb_camera_sensor(sensor_record):
            continue

        cam_cfg = base_camera_configs.get(sensor_name, {'name': sensor_name})
        cam_cfg = deepcopy(cam_cfg)

        intrinsic = np.asarray(sensor_record.get('camera_intrinsic', []), dtype=float)
        if intrinsic.shape != (3, 3):
            continue

        location = sensor_record.get('location', [0.0, 0.0, 0.0])
        rotation = sensor_record.get('rotation', [0.0, 0.0, 0.0])
        cam_cfg['transforms'] = {
            'location': {
                'x': float(location[0]),
                'y': float(location[1]),
                'z': float(location[2]),
            },
            'rotation': {
                'pitch': float(rotation[0]),
                'yaw': float(rotation[1]),
                'roll': float(rotation[2]),
            }
        }

        attrs = sensor_record.get('attributes', {}) or {}
        image_size_x = parse_actor_attribute_value(attrs.get('image_size_x'))
        image_size_y = parse_actor_attribute_value(attrs.get('image_size_y'))
        if image_size_x is None:
            image_size_x = int(round(float(intrinsic[0, 2]) * 2.0))
        if image_size_y is None:
            image_size_y = int(round(float(intrinsic[1, 2]) * 2.0))

        cam_cfg['image_size'] = (int(image_size_x), int(image_size_y))
        cam_cfg['fov'] = float(2 * np.degrees(np.arctan(float(image_size_x) / (2.0 * float(intrinsic[0, 0])))))
        camera_configs_out.append(cam_cfg)
    return camera_configs_out


def convert_carla_camera_pose_to_reference(camera_pose_carla):
    carla_camera_to_projection = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = camera_pose_carla[:3, :3] @ carla_camera_to_projection.T
    camera_pose[:3, 3] = camera_pose_carla[:3, 3]
    return camera_pose


def build_camera_poses(camera_configs_runtime, calibration_by_sensor, ego_row, frame_sensor_pose, lidar_2_world_ref):
    world_2_lidar_ref = np.linalg.inv(lidar_2_world_ref)
    camera_poses = []
    for cam_cfg in camera_configs_runtime:
        camera_2_world = build_sensor_world_matrix(
            ego_row=ego_row,
            sensor_name=cam_cfg['name'],
            calibration_record=calibration_by_sensor[cam_cfg['name']],
            frame_sensor_pose=frame_sensor_pose,
        )
        camera_pose = world_2_lidar_ref @ camera_2_world
        camera_pose = convert_carla_camera_pose_to_reference(camera_pose)
        camera_poses.append(camera_pose)
    return camera_poses


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--data_path', type=str, required=True, help='TAPSN sequence directory, e.g. TAPSN/Town10HD_Opt/001')
    parse.add_argument('--out_path', type=str, default=None)
    parse.add_argument('--config_path', type=str, default=str(REPO_ROOT / 'config' / 'occ' / 'default.toml'))
    parse.add_argument('--sensor_name', type=str, default='LIDAR_TOP_SEMANTIC')
    parse.add_argument('--len_sequence', type=int, default=None)
    parse.add_argument('--len_side', type=int, default=None)
    parse.add_argument('--to_mesh', action='store_true', default=False)
    parse.add_argument('--with_semantic', action='store_true', default=True)
    parse.add_argument('--whole_scene_to_mesh', action='store_true', default=False)


    args = parse.parse_args()

    # load config
    config = load_config(args.config_path)

    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']
    
     
    path = args.data_path
    out_path = args.out_path or os.path.join(path, "occ")
    os.makedirs(out_path, exist_ok=True)
    scene_name = infer_standard_scene_name_from_path(path)
    scene_out_path = os.path.join(out_path, scene_name)
    os.makedirs(scene_out_path, exist_ok=True)

    sampling_path = os.path.join(path, 'sampling')
    if not os.path.exists(sampling_path):
        sampling_path = path
    
    # semantic point cloud path, N*3
    pc_dir = os.path.join(sampling_path, 'velodyne_semantic', args.sensor_name)
    transform_infor_dir = os.path.join(sampling_path, 'sensor_pose.csv')

    # lidar to world = ego_to_world @ lidar_to_ego
    # 当前TAPSN使用ego.csv + calibration.json + sensor_pose.csv重建每帧LiDAR到世界坐标系的外参
    ego_transform_path = os.path.join(sampling_path, 'ego.csv')
    calibration_path = os.path.join(sampling_path, 'calibration.json')
    sensor_pose_path = os.path.join(sampling_path, 'sensor_pose.csv')
 
    pc_seman_pathes = sorted(glob.glob(os.path.join(pc_dir, "*.bin")))
    
    # bounding boxes in world coordinate
    label_path = os.path.join(sampling_path, 'labels.json')

    transform_infor_pathes = []

    # 'CATEGOTIES_INDEX': {'bus': 15, 'car': 14, 'truck': 16, 'motorcycle': 18,'bicycle': 19, 'Pedestrian': 13}
    # object_category_pathes = sorted(glob.glob(os.path.join(bbox_dir, "object_category_*.npy")))

    # object_id
    # boxes_token_pathes = sorted(glob.glob(os.path.join(bbox_dir, "boxes_token_*.npy")))
    # transform_infor_pathes = sorted(glob.glob(os.path.join(transform_infor_dir, "lidar_2_world_*.npy")))
     
    len_sequence = args.len_sequence if args.len_sequence is not None else config['len_sequence']
    len_side = args.len_side if args.len_side is not None else config['len_side']
     
    i = 0

    # 加载数据
    point_dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('cos_angle', np.float32), ('obj_idx', np.uint32), ('obj_tag', np.uint32)
    ])
    
    # load boxes
    with open(label_path, "r") as f:
        label_frames = json.load(f)
    valid_frame_ids = {
        frame["frame_id"]
        for frame in label_frames
        if frame.get("frame_valid") is True
    }
    pc_seman_pathes = [
        path for path in pc_seman_pathes
        if os.path.basename(path).split('.')[0] in valid_frame_ids
    ]
    all_sample_number = len(pc_seman_pathes)
    labels = {frame["frame_id"]: frame['objects'] for frame in label_frames}

    with open(calibration_path, 'r') as f:
        calibration_payload = json.load(f)
    calibration_by_sensor = sensor_record_index(calibration_payload)
    lidar_calibration = calibration_by_sensor[args.sensor_name]
    sensor_pose_by_frame = load_sensor_pose_index(sensor_pose_path) if os.path.exists(sensor_pose_path) else {}
    camera_configs_runtime = build_camera_configs_from_calibration(camera_configs, calibration_by_sensor)

    lidar_2_world_map = dict()
    ego_rows = []
    with open(ego_transform_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for ego in reader:
            ego_rows.append(dict(ego))
            pose_frame_id = resolve_sensor_pose_frame(normalize_frame_id(ego['frame']), sensor_pose_by_frame)
            frame_sensor_pose = sensor_pose_by_frame.get(pose_frame_id, {}) if pose_frame_id is not None else {}
            lidar_2_world_map[ego['frame']] = build_sensor_world_matrix(
                ego_row=ego,
                sensor_name=args.sensor_name,
                calibration_record=lidar_calibration,
                frame_sensor_pose=frame_sensor_pose,
            )

    ego_by_frame = {ego['frame']: ego for ego in ego_rows}


    while i < all_sample_number:
        name_i = os.path.basename(pc_seman_pathes[i]).split('.')[0]
         
        iter_start = time.time()
        
        dict_list = []
  
        left_side = i if i == 0 else i - len_side
        right_side = left_side + len_sequence
     
        if right_side + len_sequence - len_side >= all_sample_number:
            right_side = all_sample_number
        print(f"process: {left_side}/{all_sample_number}")
        print(f"left_side: {left_side}, right_side: {right_side}")
        for j in range(left_side, right_side):
            # load data / frame
            name = os.path.basename(pc_seman_pathes[j]).split('.')[0]

            # load point cloud
            points = np.fromfile(pc_seman_pathes[j], dtype=point_dtype)

            # process to N*3
            pc0 = np.column_stack([points['x'], -points['y'], points['z'], points['obj_tag']])
            instance_idx = points['obj_idx']

            if not args.with_semantic:
                pc0 = pc0[:, :3]
            
            lidar_2_world = lidar_2_world_map[name]
            
            lidar_2_world_ref = lidar_2_world_map[name_i]
            
            # # 第i帧的lidar_to_world
            # lidar_2_world_ref = list(lidar_2_world_map.values())[i]
            
            objects = labels[name]

            # transform to lidar coodinate 
            # dynamic_objects = [world_to_lidar_obj(obj, lidar_2_world) for obj in objects if 0 < obj['object_id'] < 10000]

            dynamic_objects = [obj for obj in objects if 0 < obj['object_id'] < 10000]

            boxes = []
            # box: [x, y, z, x_size, y_size, z_size, rz] in Lidar coordinate, (x, y, z) is the bottom center)
            # 读到的是世界坐标系下的Box -> 构建carla.Box
            for obj in dynamic_objects:
                verts = calculate_cube_vertices(
                    transform=obj["location"],
                    rotation=obj["rotation"],
                    dimension=obj["dimensions"]
                )

                world_2_lidar = np.linalg.inv(lidar_2_world)
                bbox_vert_lidar = [point_transform_3d(vert, world_2_lidar) for vert in verts]
                bbox_vert_lidar = np.array(bbox_vert_lidar)

                bottom_center = (bbox_vert_lidar[0, :] + bbox_vert_lidar[1, :] + bbox_vert_lidar[2, :] + bbox_vert_lidar[3, :]) / 4

                sz = np.array([obj["dimensions"][0], obj["dimensions"][1], obj["dimensions"][2]])
                rz = bbox_vert_lidar[1, :] - bbox_vert_lidar[0, :]
                rz = - np.array([np.arctan(-rz[0] / rz[1])])
                bbox_infor = np.concatenate((bottom_center, sz, rz))
                boxes.append(bbox_infor)

            boxes = np.array(boxes, dtype=float).reshape((-1, 7))
            
            # 'CATEGOTIES_INDEX': {'bus': 15, 'car': 14, 'truck': 16, 'motorcycle': 18,'bicycle': 19, 'Pedestrian': 13}
            object_category = [get_category_index(obj['class']) for obj in dynamic_objects]
            # OBJECT_ID
            boxes_token = [obj['object_id'] for obj in dynamic_objects]

            # def points_in_boxes_cpu(points: Tensor, boxes: Tensor) -> Tensor
            # Args:
            # points (torch.Tensor): [B, M, 3], [x, y, z] in
            #     LiDAR/DEPTH coordinate
            # boxes (torch.Tensor): [B, T, 7],
            #     num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            #     (x, y, z) is the bottom center.

            # Returns:
            # torch.Tensor: Return the box indices of points with the shape of
            # (B, M, T). Default background = 0.

            B = torch.from_numpy(pc0[:, :3][np.newaxis, : , :])
            M = torch.from_numpy(boxes[np.newaxis, :])
            if M.shape[1] > 0:
                M[0, :, 6] *= -1
            # 1 * Points_Num * Object_Num
            # points_in_boxes = points_in_boxes_cpu(B, M)
            points_in_boxes = torch.zeros((1, pc0.shape[0], boxes.shape[0]), dtype=torch.int32)

            # 使用semantic lidar的obj_idx关联boxes_token, 进而得到points_in_boxes
            for k, box_token in enumerate(boxes_token):
                points_in_box = torch.from_numpy(instance_idx == box_token)
                points_in_boxes[0][:, k] = points_in_box

            object_points_list = []
 
            for k in range(0, points_in_boxes.shape[-1]):
                object_points_mask = points_in_boxes[0][:, k].bool()
                object_points = pc0[object_points_mask]
                object_points_list.append(object_points)
         
            moving_mask = torch.ones_like(points_in_boxes)
            points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool() # points_in_boxes * moving_mask 和 points_in_boxes有啥区别
            points_mask = ~(points_in_boxes[0])

            ############################# get point mask out of the vehicle itself ##########################
            ego_range = config['self_range']
            oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > ego_range[0]) |
                                            (np.abs(pc0[:, 1]) > ego_range[1]) |
                                            (np.abs(pc0[:, 2]) > ego_range[2]))
            ############################# get static scene segment ##########################
            points_mask = points_mask & oneself_mask
            pc = pc0[points_mask]
            
            ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
            world_2_lidar = np.linalg.inv(lidar_2_world)
            world_2_lidar_ref = np.linalg.inv(lidar_2_world_ref)
            lidar_2_lidar_ref = world_2_lidar_ref @ lidar_2_world
            lidar_ref_2_lidar = world_2_lidar @ lidar_2_world_ref
            
            # static point cloud to the first LiDAR coordinate
            lidar_pc = point_transform_3d_batch(pc[:, :3], lidar_2_lidar_ref)
            lidar_pc = np.concatenate((lidar_pc, pc[:, 3:]), axis=1)
            dict = {"object_tokens": boxes_token,
                    "object_points_list": object_points_list,
                    "lidar_pc": lidar_pc,
                    "gt_bbox_3d": boxes,
                    "converted_object_category": object_category,
                    "lidar_2_lidar_ref": lidar_2_lidar_ref,
                    "lidar_ref_2_lidar": lidar_ref_2_lidar,
                    "name": name}
            dict_list.append(dict)


        ################## concatenate all static scene segments  ########################
        lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
        lidar_pc = np.concatenate(lidar_pc_list, axis=0)
    
        ################## concatenate all object segments (including non-key frames)  ########################
        object_token_zoo = []
        object_semantic = []
        for dict in dict_list:
            for k, object_token in enumerate(dict['object_tokens']):
                if object_token not in object_token_zoo:
                    if (dict['object_points_list'][k].shape[0] > 0):
                        object_token_zoo.append(object_token)
                        object_semantic.append(dict['converted_object_category'][k])
                    else:
                        continue

        object_points_dict = {}
        for query_object_token in object_token_zoo:
            # 将各帧的同一动态障碍物合并到一起
            object_points_dict[query_object_token] = []
            for dict in dict_list:
                for k, object_token in enumerate(dict['object_tokens']):
                    if query_object_token == object_token:
                        object_points = dict['object_points_list'][k]
                        if object_points.shape[0] > 0:
                            # objects points相对于 box bottom center的便宜
                            object_points[:, :3] = object_points[:, :3] - dict['gt_bbox_3d'][k][:3]
                            rots = dict['gt_bbox_3d'][k][6]
                            Rot = Rotation.from_euler('z', -rots, degrees=False)
                            object_points[:, :3] = Rot.apply(object_points[:, :3])
                            object_points_dict[query_object_token].append(object_points)
                            
                    else:
                        continue
            object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                    axis=0)

        object_points_vertice = []
        for key in object_points_dict.keys():
            point_cloud = object_points_dict[key]
            object_points_vertice.append(point_cloud)
        
        if args.whole_scene_to_mesh:
            point_cloud_original = o3d.geometry.PointCloud()
            with_normal2 = o3d.geometry.PointCloud()
            point_cloud_original.points = o3d.utility.Vector3dVector(lidar_pc[:, :3])
            with_normal = preprocess(point_cloud_original, config)
            with_normal2.points = with_normal.points
            with_normal2.normals = with_normal.normals
            mesh, _ = create_mesh_from_map(None, 11, config['n_threads'],
                                        config['min_density'], with_normal2)
            lidar_pc = np.asarray(mesh.vertices, dtype=float)
            lidar_pc = np.concatenate((lidar_pc, np.ones_like(lidar_pc[:, 0:1])), axis=1)

        print("cloud merging done!")
        left_side_gen = 0
        right_side_gen = len(dict_list) if right_side == all_sample_number else len(dict_list) - len_side
        print(f"left_side_gen: {left_side_gen}, right_side_gen: {right_side_gen}")
        for j in range(left_side_gen, right_side_gen):
            dict = dict_list[j]
            name = dict['name']
            ################## convert the static scene to the target coordinate system ##############
            lidar_ref_2_lidar = dict['lidar_ref_2_lidar']

            lidar_pc_i = point_transform_3d_batch(lidar_pc[:, :3], lidar_ref_2_lidar)
            lidar_pc_i = np.concatenate((lidar_pc_i, lidar_pc[:, 3:]), axis=1)
            point_cloud = lidar_pc_i[:, :3]
            if args.with_semantic:
                point_cloud_with_semantic = lidar_pc_i[:, :4]
           
            gt_bbox_3d = dict['gt_bbox_3d']
            locs = gt_bbox_3d[:, 0:3]
            dims = gt_bbox_3d[:, 3:6]
            rots = gt_bbox_3d[:, 6:7]
            # gt_bbox_3d[:, 2] += dims[:, 2] / 2.


            ################## bbox placement ##############
            object_points_list = []
            object_semantic_list = []
            for k, object_token in enumerate(dict['object_tokens']):
                for l, object_token_in_zoo in enumerate(object_token_zoo):
                    if object_token == object_token_in_zoo:
                        
                        points = deepcopy(object_points_vertice[l])
                   
                        Rot = Rotation.from_euler('z', rots[k], degrees=False)
                        rot_point = points[:, :3].copy()
                        rotated_object_points = Rot.apply(rot_point)
                        points[:, :3] = rotated_object_points + locs[k]
                    
                        if points.shape[0] >= 5:
                            points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]), torch.from_numpy(gt_bbox_3d[k:k+1][np.newaxis, :]))
                            points = points[points_in_boxes[0, :, 0].bool()]
                 
                        object_points_list.append(points[:, :3])
                        # semantics = np.ones_like(points[:, 0:1]) * object_semantic[l]
                        # object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))
                      
                        object_semantic_list.append(points)
  
            try: # avoid concatenate an empty array
                temp = np.concatenate(object_points_list)
                scene_points = np.concatenate([point_cloud, temp])
                # scene_points = point_cloud
            except Exception:
                scene_points = point_cloud

            if args.with_semantic:
                try:
                    temp = np.concatenate(object_semantic_list)
                    scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
                    # scene_semantic_points = point_cloud_with_semantic
                except Exception:
                    scene_semantic_points = point_cloud_with_semantic
                
            ################## remain points with a spatial range ##############
            # mask = (np.abs(scene_points[:, 0]) < 50.0) & (np.abs(scene_points[:, 1]) < 50.0) \
            #        & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 3.0) # 50, 5, 3
            # scene_points = scene_points[mask]

            if args.to_mesh and not args.whole_scene_to_mesh:
                ################## get mesh via Possion Surface Reconstruction ##############
                point_cloud_original = o3d.geometry.PointCloud()
                with_normal2 = o3d.geometry.PointCloud()
                point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
                with_normal = preprocess(point_cloud_original, config)
                with_normal2.points = with_normal.points
                with_normal2.normals = with_normal.normals
                mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                            config['min_density'], with_normal2)
                scene_points = np.asarray(mesh.vertices, dtype=float)


            ################## remain points with a spatial range ##############
            mask = (np.abs(scene_points[:, 0]) < pc_range[3]) & (np.abs(scene_points[:, 1]) < pc_range[4]) \
                & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
            scene_points = scene_points[mask]

            ################## convert points to voxels ##############
            pcd_np = scene_points
            pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
            pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
            pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
            pcd_np = np.floor(pcd_np).astype(np.int32)

            # fov_voxels 是空间中存在占用的坐标
            fov_voxels = np.unique(pcd_np, axis=0).astype(np.float64)
 
            ################## convert voxel coordinates to LiDAR system  ##############
            fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size # + 0.5 是 + voxel_size 吗, fov_voxels[:, :3] 不就是 fov_voxels吗
            fov_voxels[:, 0] += pc_range[0]
            fov_voxels[:, 1] += pc_range[1]
            fov_voxels[:, 2] += pc_range[2]
            # 这里没看懂有啥区别
            # np.save(os.path.join(out_path, f'no_semantic_{name}.npy'), fov_voxels)

            if args.with_semantic:
                ################## remain points with a spatial range  ##############
                mask = (np.abs(scene_semantic_points[:, 0]) < pc_range[3]) & (np.abs(scene_semantic_points[:, 1]) < pc_range[4]) \
                    & (scene_semantic_points[:, 2] > pc_range[2]) & (scene_semantic_points[:, 2] < pc_range[5])
                scene_semantic_points = scene_semantic_points[mask]

                ################## Nearest Neighbor to assign semantics ##############
                dense_voxels = fov_voxels
                sparse_voxels_semantic = scene_semantic_points

                x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
                y = torch.from_numpy(sparse_voxels_semantic[:, :3]).cuda().unsqueeze(0).float()
    
                d1, d2, idx1, idx2 = chamfer.forward(x, y)
                indices = idx1[0].cpu().numpy()
 
                dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
                dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

                # to voxel coordinate
                pcd_np = dense_voxels_with_semantic
                pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
                pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
                pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
                dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int32)
 
                dense_voxels_with_semantic[:, 1] *= -1 # convert to the nuscenes IMU coordinate
                semantics = np.zeros(occ_size)
                
                pcd_np = np.floor(pcd_np).astype(np.int32)
                semantics[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = pcd_np[:, 3]
                semantics = np.flip(semantics, axis=1).astype(np.int8) # coordinate: carla to nuscenes 	沿Y轴（axis=1）镜像翻转3D体素网格
                pose_frame_id = resolve_sensor_pose_frame(normalize_frame_id(name), sensor_pose_by_frame)
                frame_sensor_pose = sensor_pose_by_frame.get(pose_frame_id, {}) if pose_frame_id is not None else {}
                camera_poses = build_camera_poses(
                    camera_configs_runtime,
                    calibration_by_sensor,
                    ego_by_frame[name],
                    frame_sensor_pose,
                    lidar_2_world_map[name],
                )
                mask_camera = np.ones(occ_size).astype(np.int8)

                occupied_voxels = np.argwhere(semantics != 0)
                mask_camera = compute_camera_visibility_mask(
                    occupied_voxels, camera_poses, camera_configs_runtime, pc_range, voxel_size, occ_size
                )

                mask_lidar = np.ones(occ_size).astype(np.int8)
                occ_out_dir = os.path.join(scene_out_path, f'sample_{name}')
                dict_npz = {"semantics": semantics,  "mask_camera": mask_camera, "mask_lidar": mask_lidar}
                os.makedirs(occ_out_dir, exist_ok=True)
                np.savez(os.path.join(occ_out_dir, 'labels.npz'), **dict_npz)
                print(f"{os.path.join(occ_out_dir, 'labels.npz')} saved")
                # np.save(os.path.join(occ_out_dir, 'labels.npy'), dense_voxels_with_semantic)
                 
        cost_time = time.time() - iter_start
        print(f'iter cost: {cost_time}s')
        i = right_side
