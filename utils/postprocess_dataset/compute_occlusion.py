#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_common import (
    build_sensor_world_matrix,
    iter_sequence_dirs,
    load_csv_index,
    load_json,
    load_sensor_pose_index,
    normalize_frame_id,
    resolve_sensor_pose_frame,
    sensor_record_index,
    transform_points,
    write_json,
)
from utils.postprocess_dataset.common import (
    count_points_in_oriented_box,
    read_semantic_lidar_points_xyz,
)


DEFAULT_SENSOR_NAME = "LIDAR_TOP_SEMANTIC"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute LiDAR-based occlusion and num_lidar_pts for sampled CARLA labels.json files "
            "using per-frame sensor poses."
        )
    )
    parser.add_argument(
        "input_path",
        help="Sequence directory or dataset root, e.g. data/Town01_Opt/001 or data/Town01_Opt.",
    )
    parser.add_argument(
        "--sensor-name",
        default=DEFAULT_SENSOR_NAME,
        help="Semantic LiDAR sensor name under sampling/velodyne_semantic.",
    )
    parser.add_argument(
        "--zero-points-occlusion",
        type=int,
        default=4,
        help="Occlusion value assigned when an object has zero LiDAR points inside its box.",
    )
    parser.add_argument(
        "--skip-missing-lidar",
        action="store_true",
        help="Skip frames with missing semantic LiDAR instead of failing.",
    )
    return parser.parse_args()


def occlusion_level_from_points(points_in_box, zero_points_occlusion):
    if points_in_box == 0:
        return int(zero_points_occlusion)
    if points_in_box > 20:
        return 0
    if points_in_box > 10:
        return 1
    if points_in_box > 5:
        return 2
    return 3


def add_normalized_frame_aliases(frame_index):
    for frame_key, frame_record in list(frame_index.items()):
        frame_index.setdefault(normalize_frame_id(frame_key), frame_record)


def compute_sequence_occlusion(
    sequence_dir,
    sensor_name,
    zero_points_occlusion,
    skip_missing_lidar,
    valid_only=False,
):
    sequence_dir = Path(sequence_dir)
    sampling_dir = sequence_dir / "sampling"
    labels_path = sampling_dir / "labels.json"
    calibration_path = sampling_dir / "calibration.json"
    ego_csv_path = sampling_dir / "ego.csv"
    sensor_pose_path = sampling_dir / "sensor_pose.csv"
    lidar_dir = sampling_dir / "velodyne_semantic" / sensor_name

    labels = load_json(labels_path)
    calibration_payload = load_json(calibration_path)
    ego_by_frame = load_csv_index(ego_csv_path)
    sensor_pose_by_frame = load_sensor_pose_index(sensor_pose_path) if sensor_pose_path.exists() else {}
    add_normalized_frame_aliases(ego_by_frame)
    add_normalized_frame_aliases(sensor_pose_by_frame)
    calibration_by_sensor = sensor_record_index(calibration_payload)

    if sensor_name not in calibration_by_sensor:
        raise KeyError("Sensor {} not found in {}".format(sensor_name, calibration_path))

    calibration_record = calibration_by_sensor[sensor_name]
    processed_frames = 0
    skipped_frames = 0
    updated_objects = 0

    for frame_record in labels:
        if valid_only and not frame_record.get("frame_valid", True):
            continue

        frame_name = str(frame_record["frame_id"]).strip()
        frame_id = normalize_frame_id(frame_name)
        ego_row = ego_by_frame.get(frame_name) or ego_by_frame.get(frame_id)
        if ego_row is None:
            raise KeyError("Frame {} missing from {}".format(frame_name, ego_csv_path))

        lidar_path = lidar_dir / "{}.bin".format(frame_name)
        if not lidar_path.exists() and frame_name != frame_id:
            lidar_path = lidar_dir / "{}.bin".format(frame_id)
        if not lidar_path.exists():
            if skip_missing_lidar:
                skipped_frames += 1
                continue
            raise FileNotFoundError("Missing semantic LiDAR frame {}".format(lidar_path))

        pose_frame_id = resolve_sensor_pose_frame(frame_id, sensor_pose_by_frame)
        frame_sensor_pose = sensor_pose_by_frame.get(pose_frame_id, {}) if pose_frame_id is not None else {}
        sensor_world = build_sensor_world_matrix(
            ego_row=ego_row,
            sensor_name=sensor_name,
            calibration_record=calibration_record,
            frame_sensor_pose=frame_sensor_pose,
        )

        points_lidar, _ = read_semantic_lidar_points_xyz(lidar_path)
        points_world = transform_points(sensor_world, points_lidar)

        for obj in frame_record["objects"]:
            points_in_box = count_points_in_oriented_box(points_world, obj)
            obj["num_lidar_pts"] = points_in_box
            obj["occlusion"] = occlusion_level_from_points(points_in_box, zero_points_occlusion)
            updated_objects += 1

        processed_frames += 1

    write_json(labels_path, labels)
    return {
        "sequence": sequence_dir.name,
        "frames_processed": processed_frames,
        "frames_skipped": skipped_frames,
        "objects_updated": updated_objects,
        "sensor_name": sensor_name,
    }


def main():
    args = parse_args()
    sequence_dirs = iter_sequence_dirs(args.input_path)
    if not sequence_dirs:
        raise RuntimeError("No sequence directories found under {}".format(args.input_path))

    for sequence_dir in sequence_dirs:
        summary = compute_sequence_occlusion(
            sequence_dir=sequence_dir,
            sensor_name=args.sensor_name,
            zero_points_occlusion=args.zero_points_occlusion,
            skip_missing_lidar=args.skip_missing_lidar,
        )
        print(
            "{sequence}: frames_processed={frames_processed} frames_skipped={frames_skipped} "
            "objects_updated={objects_updated} sensor={sensor_name}".format(**summary)
        )


if __name__ == "__main__":
    main()
