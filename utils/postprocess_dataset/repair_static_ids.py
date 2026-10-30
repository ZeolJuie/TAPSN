#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_common import iter_sequence_dirs, load_json, write_json


DEFAULT_START_ID = 10000
POLE_HEIGHT_KEEP_THRESHOLD_M = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair static object instance IDs in sampled CARLA labels.json files."
    )
    parser.add_argument(
        "input_path",
        help="Sequence directory or dataset root, e.g. data/Town01_Opt/001 or data/Town01_Opt.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=DEFAULT_START_ID,
        help="First synthetic object_id assigned to static city objects.",
    )
    parser.add_argument(
        "--keep-small-poles",
        action="store_true",
        help="Keep tiny Poles entries instead of filtering them out.",
    )
    return parser.parse_args()


def repair_sequence(sequence_dir, start_id, keep_small_poles, valid_only=False):
    label_path = Path(sequence_dir) / "sampling" / "labels.json"
    labels = load_json(label_path)

    object_map = {}
    assigned_ids = set()
    next_id = int(start_id)
    removed_counter = Counter()
    changed_count = 0

    for frame in labels:
        existing_ids = [int(obj["object_id"]) for obj in frame["objects"] if int(obj["object_id"]) >= 0]
        assigned_ids.update(existing_ids)

        if valid_only and not frame.get("frame_valid", True):
            continue

        filtered_objects = []
        for obj in frame["objects"]:
            if (
                not keep_small_poles
                and obj.get("class") == "Poles"
                and float(obj["dimensions"][2]) < POLE_HEIGHT_KEEP_THRESHOLD_M
            ):
                removed_counter[obj.get("class", "unknown")] += 1
                continue
            filtered_objects.append(obj)
        frame["objects"] = filtered_objects

        for obj in frame["objects"]:
            if int(obj["object_id"]) != -1:
                continue

            location_key = tuple(round(float(value), 6) for value in obj["location"])
            if location_key in object_map:
                obj["object_id"] = object_map[location_key]
            else:
                while next_id in assigned_ids:
                    next_id += 1
                object_map[location_key] = next_id
                assigned_ids.add(next_id)
                obj["object_id"] = next_id
                next_id += 1
            changed_count += 1

        seen_frame_ids = set()
        deduplicated_objects = []
        for obj in frame["objects"]:
            object_id = int(obj["object_id"])
            if object_id in seen_frame_ids:
                removed_counter["duplicate_object_id"] += 1
                continue
            seen_frame_ids.add(object_id)
            deduplicated_objects.append(obj)
        frame["objects"] = deduplicated_objects

    write_json(label_path, labels)

    remaining_neg1 = sum(
        1
        for frame in labels
        for obj in frame["objects"]
        if int(obj["object_id"]) == -1
    )
    return {
        "sequence": Path(sequence_dir).name,
        "frames": len(labels),
        "changed_objects": changed_count,
        "new_static_instances": len(object_map),
        "remaining_neg1": remaining_neg1,
        "removed": dict(removed_counter),
    }


def main():
    args = parse_args()
    sequence_dirs = iter_sequence_dirs(args.input_path)
    if not sequence_dirs:
        raise RuntimeError("No sequence directories found under {}".format(args.input_path))

    for sequence_dir in sequence_dirs:
        summary = repair_sequence(
            sequence_dir=sequence_dir,
            start_id=args.start_id,
            keep_small_poles=args.keep_small_poles,
        )
        print(
            "{sequence}: frames={frames} changed_objects={changed_objects} "
            "new_static_instances={new_static_instances} remaining_neg1={remaining_neg1} removed={removed}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
