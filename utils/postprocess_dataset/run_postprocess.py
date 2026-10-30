#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_common import iter_sequence_dirs
from utils.postprocess_dataset.compute_occlusion import compute_sequence_occlusion
from utils.postprocess_dataset.repair_static_ids import repair_sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run sampled CARLA dataset post-processing: static object instance ID repair and "
            "LiDAR-based occlusion computation."
        )
    )
    parser.add_argument(
        "input_path",
        help="Sequence directory or dataset root, e.g. data/Town01_Opt/001 or data/Town01_Opt.",
    )
    parser.add_argument(
        "--sensor-name",
        default="LIDAR_TOP_SEMANTIC",
        help="Semantic LiDAR sensor used for occlusion computation.",
    )
    parser.add_argument(
        "--keep-small-poles",
        action="store_true",
        help="Keep tiny Poles entries instead of filtering them out during ID repair.",
    )
    parser.add_argument(
        "--skip-missing-lidar",
        action="store_true",
        help="Skip frames with missing semantic LiDAR during occlusion computation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_dirs = iter_sequence_dirs(args.input_path)
    if not sequence_dirs:
        raise RuntimeError("No sequence directories found under {}".format(args.input_path))

    for sequence_dir in sequence_dirs:
        id_summary = repair_sequence(
            sequence_dir=sequence_dir,
            start_id=10000,
            keep_small_poles=args.keep_small_poles,
            valid_only=True,
        )
        occ_summary = compute_sequence_occlusion(
            sequence_dir=sequence_dir,
            sensor_name=args.sensor_name,
            zero_points_occlusion=4,
            skip_missing_lidar=args.skip_missing_lidar,
            valid_only=True,
        )
        print(
            "{sequence}: changed_objects={changed_objects} new_static_instances={new_static_instances} "
            "remaining_neg1={remaining_neg1} frames_processed={frames_processed} objects_updated={objects_updated}".format(
                sequence=id_summary["sequence"],
                changed_objects=id_summary["changed_objects"],
                new_static_instances=id_summary["new_static_instances"],
                remaining_neg1=id_summary["remaining_neg1"],
                frames_processed=occ_summary["frames_processed"],
                objects_updated=occ_summary["objects_updated"],
            )
        )


if __name__ == "__main__":
    main()
