#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_common import (
    build_ego_world_matrix,
    build_sensor_world_matrix,
    load_csv_index,
    load_json,
    load_sensor_pose_index,
    normalize_frame_id,
    resolve_sensor_pose_frame,
    sensor_record_index,
    write_json,
)
from utils.nuscenes_converter.common import (
    SUPPORTED_CHANNELS,
    SUPPORTED_LIDAR,
    category_names_from_labels,
    deterministic_timestamp_us,
    deterministic_token,
    matrix_to_nuscenes_translation_rotation,
    object_rotation_to_nuscenes_quaternion,
    sensor_matrix_to_nuscenes_translation_rotation,
    standard_scene_name_from_ids,
    standard_scene_prefix_from_ids,
)


ATTRIBUTE_EMPTY = ""
PED_SPEED_THRESHOLD = 0.2
VEHICLE_SPEED_THRESHOLD = 0.2
TABLE_NAMES = [
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export sampled CARLA data into nuScenes-style format.")
    parser.add_argument("sequence_dir", help="Sequence directory, e.g. data/Town01_Opt/001")
    parser.add_argument("output_dir", help="nuScenes output directory")
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="Target nuScenes metadata version directory",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append this sequence into an existing nuScenes output directory instead of overwriting metadata tables.",
    )
    parser.add_argument(
        "--replace-sequence",
        action="store_true",
        help="When used with --append, replace an existing scene for this sequence before appending it again.",
    )
    return parser.parse_args()


class SequenceReader:
    def __init__(self, sequence_dir):
        self.sequence_dir = Path(sequence_dir).resolve()
        self.sampling_dir = self.sequence_dir / "sampling"
        self.sequence_id = self.sequence_dir.name
        self.town_name = self.sequence_dir.parent.name
        self.map_id = self._parse_map_id()
        self.scene_key = "{}-{}".format(self.map_id, self.sequence_id)
        self.scene_name = standard_scene_name_from_ids(self.map_id, self.sequence_id)
        self.scene_prefix = standard_scene_prefix_from_ids(self.map_id, self.sequence_id)

        self.log = load_json(self.sampling_dir / "log.json")
        self.all_labels = load_json(self.sampling_dir / "labels.json")
        self.labels = self._select_valid_labels()
        self.label_by_frame = {normalize_frame_id(frame["frame_id"]): frame for frame in self.labels}
        self.source_frame_id_by_frame = {
            normalize_frame_id(frame["frame_id"]): str(frame["frame_id"])
            for frame in self.labels
        }
        self.ego_by_frame = self._normalize_frame_index(
            load_csv_index(self.sampling_dir / "ego.csv"),
            "ego.csv",
        )
        self.sensor_pose_by_frame = self._normalize_sensor_pose_index(
            load_sensor_pose_index(self.sampling_dir / "sensor_pose.csv")
        )
        self.calibration_payload = load_json(self.sampling_dir / "calibration.json")
        self.calibration_by_sensor = sensor_record_index(self.calibration_payload)

        self.frames = [normalize_frame_id(frame["frame_id"]) for frame in self.labels]

    def _parse_map_id(self):
        match = re.match(r"^Town(\d+)", self.town_name)
        if match is None:
            raise RuntimeError(
                "Sequence {} parent directory must start with Town<number>, got {}".format(
                    self.sequence_dir,
                    self.town_name,
                )
            )
        return match.group(1).zfill(2)

    def _select_valid_labels(self):
        missing = [
            normalize_frame_id(frame["frame_id"])
            for frame in self.all_labels
            if "frame_valid" not in frame
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                "Sequence {} labels.json is missing frame_valid on {} frames, examples {}".format(
                    self.sequence_dir,
                    len(missing),
                    preview,
                )
            )

        non_bool = [
            normalize_frame_id(frame["frame_id"])
            for frame in self.all_labels
            if not isinstance(frame["frame_valid"], bool)
        ]
        if non_bool:
            preview = ", ".join(non_bool[:5])
            raise RuntimeError(
                "Sequence {} labels.json has non-boolean frame_valid on {} frames, examples {}".format(
                    self.sequence_dir,
                    len(non_bool),
                    preview,
                )
            )

        valid_indices = [
            index for index, frame in enumerate(self.all_labels) if frame["frame_valid"]
        ]
        if not valid_indices:
            raise RuntimeError(
                "Sequence {} has no frame_valid=true frames in labels.json".format(self.sequence_dir)
            )

        contiguous_indices = list(range(valid_indices[0], valid_indices[-1] + 1))
        if valid_indices != contiguous_indices:
            raise RuntimeError(
                "Sequence {} frame_valid=true frames are not contiguous".format(self.sequence_dir)
            )

        return [self.all_labels[index] for index in valid_indices]

    def _normalize_frame_index(self, index, source_name):
        normalized = {}
        for frame_id, row in index.items():
            normalized_frame_id = normalize_frame_id(frame_id)
            if normalized_frame_id in normalized:
                raise RuntimeError(
                    "Sequence {} {} has duplicate normalized frame id {}".format(
                        self.sequence_dir,
                        source_name,
                        normalized_frame_id,
                    )
                )
            normalized[normalized_frame_id] = row
        return normalized

    def _normalize_sensor_pose_index(self, index):
        normalized = {}
        for frame_id, rows_by_sensor in index.items():
            normalized_frame_id = normalize_frame_id(frame_id)
            if normalized_frame_id in normalized:
                raise RuntimeError(
                    "Sequence {} sensor_pose.csv has duplicate normalized frame id {}".format(
                        self.sequence_dir,
                        normalized_frame_id,
                    )
                )
            normalized[normalized_frame_id] = rows_by_sensor
        return normalized

    def pose_frame_id(self, frame_id):
        return resolve_sensor_pose_frame(frame_id, self.sensor_pose_by_frame)

    def frame_sensor_pose(self, frame_id):
        pose_frame = self.pose_frame_id(frame_id)
        if pose_frame is None:
            return {}
        return self.sensor_pose_by_frame.get(pose_frame, {})

    def sensor_world_matrix(self, frame_id, sensor_name):
        ego_row = self.ego_by_frame[frame_id]
        calibration_record = self.calibration_by_sensor[sensor_name]
        return build_sensor_world_matrix(
            ego_row=ego_row,
            sensor_name=sensor_name,
            calibration_record=calibration_record,
            frame_sensor_pose=self.frame_sensor_pose(frame_id),
        )

    def frame_timestamp_us(self, frame_id):
        ego_row = self.ego_by_frame[frame_id]
        return deterministic_timestamp_us(
            self.scene_key,
            frame_id,
            ego_row["timestamp"],
        )

    def image_path(self, frame_id, camera_name):
        return self.sampling_dir / "image" / camera_name / "{}.jpg".format(
            self.source_frame_id_by_frame[frame_id]
        )

    def lidar_path(self, frame_id):
        return self.sampling_dir / "velodyne" / SUPPORTED_LIDAR / "{}.bin".format(
            self.source_frame_id_by_frame[frame_id]
        )


class NuScenesExporter:
    def __init__(self, reader, output_dir, version, append=False, replace_sequence=False):
        self.reader = reader
        self.output_dir = Path(output_dir).resolve()
        self.version = version
        self.version_dir = self.output_dir / version
        self.append = bool(append)
        self.replace_sequence = bool(replace_sequence)

        self.data = self._empty_tables()
        self.sample_tokens_by_frame = {}
        self.ego_pose_tokens_by_frame = {}
        self.instance_state = {}
        self.sensor_tokens = {}
        self.attribute_tokens = {}
        self.category_tokens = {}
        self.existing_data = None

    def export(self):
        self._prepare_dirs()
        self._validate_sequence_labels()
        if self.append:
            self.existing_data = self._load_existing_tables()
            self._validate_append_target()
        self._create_visibility()
        self._create_sensors()
        self._create_categories()
        self._create_attributes()
        self._create_log()
        self._create_map()
        self._create_scene_and_samples()
        self._create_ego_poses()
        self._create_sample_data_and_calibrated_sensors()
        self._create_sample_annotations()
        self._create_instances()
        if self.append:
            self._merge_with_existing()
        self._write_tables()

    def _empty_tables(self):
        return {table_name: [] for table_name in TABLE_NAMES}

    def _prepare_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        for folder in ["samples", "sweeps", "maps"]:
            (self.output_dir / folder).mkdir(exist_ok=True)
        for channel in SUPPORTED_CHANNELS:
            (self.output_dir / "samples" / channel).mkdir(parents=True, exist_ok=True)

    def _load_existing_tables(self):
        existing = self._empty_tables()
        for table_name in TABLE_NAMES:
            table_path = self.version_dir / "{}.json".format(table_name)
            if table_path.exists():
                existing[table_name] = load_json(table_path)
        return existing

    def _scene_token(self):
        return deterministic_token("scene", self.reader.scene_key)

    def _log_token(self):
        return deterministic_token("log", self.reader.scene_key)

    def _validate_append_target(self):
        scene_token = self._scene_token()
        existing_scene_tokens = {record["token"] for record in self.existing_data["scene"]}
        if scene_token in existing_scene_tokens and not self.replace_sequence:
            raise RuntimeError(
                "Sequence {} already exists in {}. Use --replace-sequence with --append to re-export it.".format(
                    self.reader.scene_key,
                    self.output_dir,
                )
            )

    def _validate_sequence_labels(self):
        unresolved = []
        for frame in self.reader.labels:
            count = sum(1 for obj in frame.get("objects", []) if int(obj.get("object_id", -1)) == -1)
            if count:
                unresolved.append((normalize_frame_id(frame["frame_id"]), count))
        if not unresolved:
            return

        preview = ", ".join("{}:{}".format(frame_id, count) for frame_id, count in unresolved[:5])
        raise RuntimeError(
            "Sequence {} still contains object_id=-1 labels ({} frames, examples {}). "
            "Run utils/postprocess_dataset/run_postprocess.py on the sequence before exporting.".format(
                self.reader.scene_key,
                len(unresolved),
                preview,
            )
        )

    def _create_visibility(self):
        self.data["visibility"] = [
            {"description": "visibility of whole object is 0", "token": "4", "level": "v0"},
            {"description": "visibility of whole object is between 0 and 40%", "token": "3", "level": "v0-40"},
            {"description": "visibility of whole object is between 40 and 60%", "token": "2", "level": "v40-60"},
            {"description": "visibility of whole object is between 60 and 80%", "token": "1", "level": "v60-80"},
            {"description": "visibility of whole object is between 80 and 100%", "token": "0", "level": "v80-100"},
        ]

    def _create_sensors(self):
        sensors = []
        for channel in SUPPORTED_CHANNELS:
            modality = "lidar" if channel == SUPPORTED_LIDAR else "camera"
            token = deterministic_token("sensor", channel)
            self.sensor_tokens[channel] = token
            sensors.append({
                "token": token,
                "channel": channel,
                "modality": modality,
            })
        self.data["sensor"] = sensors

    def _create_categories(self):
        categories = []
        for name in category_names_from_labels(self.reader.labels):
            token = deterministic_token("category", name)
            self.category_tokens[name] = token
            categories.append({
                "token": token,
                "name": name,
                "description": "",
            })
        self.data["category"] = categories

    def _create_attributes(self):
        names = {ATTRIBUTE_EMPTY}
        for frame in self.reader.labels:
            for obj in frame["objects"]:
                attribute_name = self._attribute_name(obj)
                if attribute_name is not None:
                    names.add(attribute_name)

        attributes = []
        for name in sorted(names):
            token = deterministic_token("attribute", name)
            self.attribute_tokens[name] = token
            attributes.append({
                "token": token,
                "name": name,
                "description": "",
            })
        self.data["attribute"] = attributes

    def _create_log(self):
        token = self._log_token()
        self.log_token = token
        walker = self.reader.log.get("walker", {})
        walker_name = walker.get("ped_name") or walker.get("blueprint_id") or "unknown"
        self.data["log"] = [
            {
                "token": token,
                "logfile": "{}.log".format(self.reader.scene_key),
                "vehicle": walker_name,
                "date_captured": self.reader.log.get("date_captured", ""),
                "location": self.reader.log.get("location", ""),
            }
        ]

    def _create_map(self):
        location = self.reader.log.get("location", "")
        filename = "maps/{}.png".format(location)
        token = deterministic_token("map", location)
        map_path = self.output_dir / filename
        if not map_path.exists():
            from PIL import Image

            map_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), color=(120, 120, 120)).save(map_path)
        self.data["map"] = [
            {
                "category": "semantic_prior",
                "token": token,
                "filename": filename,
                "log_tokens": [self.log_token],
            }
        ]

    def _create_scene_and_samples(self):
        scene_token = self._scene_token()
        sample_records = []
        prev_token = ""
        for frame_id in self.reader.frames:
            timestamp = self.reader.frame_timestamp_us(frame_id)
            sample_token = deterministic_token("sample", self.reader.scene_key, frame_id)
            self.sample_tokens_by_frame[frame_id] = sample_token
            sample_record = {
                "token": sample_token,
                "timestamp": timestamp,
                "scene_token": scene_token,
                "prev": prev_token,
                "next": "",
            }
            if sample_records:
                sample_records[-1]["next"] = sample_token
            sample_records.append(sample_record)
            prev_token = sample_token

        self.data["sample"] = sample_records
        self.data["scene"] = [
            {
                "token": scene_token,
                "name": self.reader.scene_name,
                "description": self.reader.log.get("scenario_name", ""),
                "log_token": self.log_token,
                "nbr_samples": len(sample_records),
                "first_sample_token": sample_records[0]["token"] if sample_records else "",
                "last_sample_token": sample_records[-1]["token"] if sample_records else "",
            }
        ]

    def _create_ego_poses(self):
        poses = []
        for frame_id in self.reader.frames:
            ego_row = self.reader.ego_by_frame[frame_id]
            ego_matrix = build_ego_world_matrix(ego_row)
            translation, rotation = matrix_to_nuscenes_translation_rotation(ego_matrix)
            token = deterministic_token("ego_pose", self.reader.scene_key, frame_id)
            self.ego_pose_tokens_by_frame[frame_id] = token
            poses.append({
                "token": token,
                "timestamp": self.reader.frame_timestamp_us(frame_id),
                "translation": translation,
                "rotation": rotation,
            })
        self.data["ego_pose"] = poses

    def _create_sample_data_and_calibrated_sensors(self):
        sample_data_records = []
        calibrated_records = []
        per_channel_records = defaultdict(list)

        for frame_id in self.reader.frames:
            timestamp = self.reader.frame_timestamp_us(frame_id)
            sample_token = self.sample_tokens_by_frame[frame_id]
            ego_pose_token = self.ego_pose_tokens_by_frame[frame_id]

            for channel in SUPPORTED_CHANNELS:
                sensor_world = self.reader.sensor_world_matrix(frame_id, channel)
                ego_row = self.reader.ego_by_frame[frame_id]
                ego_world_matrix = build_ego_world_matrix(ego_row)
                sensor_to_ego = np.linalg.inv(ego_world_matrix) @ sensor_world
                translation, rotation = sensor_matrix_to_nuscenes_translation_rotation(
                    sensor_to_ego,
                    sensor_modality="lidar" if channel == SUPPORTED_LIDAR else "camera",
                )
                calibrated_token = deterministic_token("calibrated_sensor", self.reader.scene_key, frame_id, channel)
                calibrated_records.append({
                    "token": calibrated_token,
                    "sensor_token": self.sensor_tokens[channel],
                    "translation": translation,
                    "rotation": rotation,
                    "camera_intrinsic": self._camera_intrinsic(channel),
                })

                filename = self._copy_sample_file(frame_id, channel, timestamp)
                sample_data_token = deterministic_token("sample_data", self.reader.scene_key, frame_id, channel)
                record = {
                    "token": sample_data_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": calibrated_token,
                    "timestamp": timestamp,
                    "fileformat": "bin" if channel == SUPPORTED_LIDAR else "jpg",
                    "is_key_frame": True,
                    "height": 0 if channel == SUPPORTED_LIDAR else 900,
                    "width": 0 if channel == SUPPORTED_LIDAR else 1600,
                    "filename": filename,
                    "prev": "",
                    "next": "",
                }
                sample_data_records.append(record)
                per_channel_records[channel].append(record)

        for channel, records in per_channel_records.items():
            for index, record in enumerate(records):
                if index > 0:
                    record["prev"] = records[index - 1]["token"]
                if index < len(records) - 1:
                    record["next"] = records[index + 1]["token"]

        self.data["calibrated_sensor"] = calibrated_records
        self.data["sample_data"] = sample_data_records

    def _camera_intrinsic(self, channel):
        if channel == SUPPORTED_LIDAR:
            return []
        return self.reader.calibration_by_sensor[channel]["camera_intrinsic"]

    def _copy_sample_file(self, frame_id, channel, timestamp):
        if channel == SUPPORTED_LIDAR:
            src = self.reader.lidar_path(frame_id)
            dst_name = "{}-{}-{}-{}.bin".format(self.reader.scene_prefix, frame_id, channel, timestamp)
            dst = self.output_dir / "samples" / channel / dst_name
            points = np.fromfile(str(src), dtype=np.float32).reshape(-1, 4)
            ring = np.zeros((points.shape[0], 1), dtype=np.float32)
            np.hstack((points, ring)).astype(np.float32).tofile(str(dst))
        else:
            src = self.reader.image_path(frame_id, channel)
            dst_name = "{}-{}-{}-{}.jpg".format(self.reader.scene_prefix, frame_id, channel, timestamp)
            dst = self.output_dir / "samples" / channel / dst_name
            shutil.copy2(src, dst)
        return str(dst.relative_to(self.output_dir))

    def _create_sample_annotations(self):
        annotations = []
        annotations_by_instance = defaultdict(list)

        for frame_id in self.reader.frames:
            frame = self.reader.label_by_frame[frame_id]
            sample_token = self.sample_tokens_by_frame[frame_id]
            timestamp = self.reader.frame_timestamp_us(frame_id)
            del timestamp

            for obj in frame["objects"]:
                category_name = obj["class"]
                if category_name not in self.category_tokens:
                    continue

                instance_token = deterministic_token("instance", self.reader.scene_key, obj["object_id"])
                annotation_token = deterministic_token(
                    "sample_annotation",
                    self.reader.scene_key,
                    frame_id,
                    obj["object_id"],
                    category_name,
                )
                annotation = {
                    "token": annotation_token,
                    "sample_token": sample_token,
                    "instance_token": instance_token,
                    "visibility_token": str(obj.get("occlusion", 4)),
                    "attribute_tokens": self._attribute_tokens(obj),
                    "translation": [
                        float(obj["location"][0]),
                        -float(obj["location"][1]),
                        float(obj["location"][2]),
                    ],
                    "size": [
                        float(obj["dimensions"][1]),
                        float(obj["dimensions"][0]),
                        float(obj["dimensions"][2]),
                    ],
                    "rotation": object_rotation_to_nuscenes_quaternion(obj["rotation"]),
                    "prev": "",
                    "next": "",
                    "num_lidar_pts": int(obj.get("num_lidar_pts", 0)),
                    "num_radar_pts": 0,
                }
                annotations.append(annotation)
                annotations_by_instance[instance_token].append(annotation)
                self._update_instance_state(obj, category_name, instance_token, annotation_token)

        for instance_token, records in annotations_by_instance.items():
            ordered = sorted(
                records,
                key=lambda record: next(
                    sample["timestamp"] for sample in self.data["sample"] if sample["token"] == record["sample_token"]
                ),
            )
            for index, record in enumerate(ordered):
                if index > 0:
                    record["prev"] = ordered[index - 1]["token"]
                if index < len(ordered) - 1:
                    record["next"] = ordered[index + 1]["token"]

        self.data["sample_annotation"] = annotations

    def _update_instance_state(self, obj, category_name, instance_token, annotation_token):
        state = self.instance_state.setdefault(
            instance_token,
            {
                "token": instance_token,
                "category_token": self.category_tokens[category_name],
                "nbr_annotations": 0,
                "first_annotation_token": annotation_token,
                "last_annotation_token": annotation_token,
            },
        )
        state["nbr_annotations"] += 1
        if state["nbr_annotations"] == 1:
            state["first_annotation_token"] = annotation_token
        state["last_annotation_token"] = annotation_token

    def _create_instances(self):
        self.data["instance"] = list(self.instance_state.values())

    def _attribute_name(self, obj):
        class_name = obj.get("class")
        behavior_name = obj.get("behavior_name")
        if class_name == "Pedestrian" and behavior_name:
            return "pedestrian.behavior.{}".format(behavior_name)

        speed = obj.get("speed")
        if speed is None:
            return None

        speed = float(speed)
        if class_name == "Pedestrian":
            return "pedestrian.moving" if speed >= PED_SPEED_THRESHOLD else "pedestrian.standing"
        if class_name in {"Car", "Truck", "Bus", "Van", "Bicycle", "Motorcycle"}:
            return "vehicle.moving" if speed >= VEHICLE_SPEED_THRESHOLD else "vehicle.stopped"
        return None

    def _attribute_tokens(self, obj):
        name = self._attribute_name(obj)
        if name is None:
            return []
        return [self.attribute_tokens[name]]

    def _write_tables(self):
        for table_name, payload in self.data.items():
            write_json(self.version_dir / "{}.json".format(table_name), payload)

    def _merge_with_existing(self):
        existing = self.existing_data
        if self.replace_sequence:
            existing = self._remove_sequence_from_existing(existing)

        merged = self._empty_tables()
        for table_name in TABLE_NAMES:
            merged[table_name] = list(existing.get(table_name, []))

        self._merge_token_table(merged, "visibility", self.data["visibility"], allow_existing=True)
        self._merge_token_table(merged, "sensor", self.data["sensor"], allow_existing=True)
        self._merge_token_table(merged, "category", self.data["category"], allow_existing=True)
        self._merge_token_table(merged, "attribute", self.data["attribute"], allow_existing=True)
        self._merge_maps(merged, self.data["map"])

        for table_name in [
            "log",
            "scene",
            "sample",
            "ego_pose",
            "calibrated_sensor",
            "sample_data",
            "sample_annotation",
            "instance",
        ]:
            self._merge_token_table(merged, table_name, self.data[table_name], allow_existing=False)

        self.data = merged

    def _remove_sequence_from_existing(self, existing):
        scene_token = self._scene_token()
        log_token = self._log_token()
        cleaned = self._empty_tables()
        for table_name in TABLE_NAMES:
            cleaned[table_name] = list(existing.get(table_name, []))

        removed_sample_tokens = {
            record["token"] for record in cleaned["sample"] if record.get("scene_token") == scene_token
        }
        removed_sample_data = [
            record for record in cleaned["sample_data"] if record.get("sample_token") in removed_sample_tokens
        ]
        removed_ego_pose_tokens = {record["ego_pose_token"] for record in removed_sample_data}
        removed_calibrated_sensor_tokens = {record["calibrated_sensor_token"] for record in removed_sample_data}
        removed_annotations = [
            record for record in cleaned["sample_annotation"] if record.get("sample_token") in removed_sample_tokens
        ]
        removed_instance_tokens = {record["instance_token"] for record in removed_annotations}

        cleaned["scene"] = [record for record in cleaned["scene"] if record.get("token") != scene_token]
        cleaned["log"] = [record for record in cleaned["log"] if record.get("token") != log_token]
        cleaned["sample"] = [
            record for record in cleaned["sample"] if record.get("token") not in removed_sample_tokens
        ]
        cleaned["sample_data"] = [
            record for record in cleaned["sample_data"] if record.get("sample_token") not in removed_sample_tokens
        ]
        cleaned["ego_pose"] = [
            record for record in cleaned["ego_pose"] if record.get("token") not in removed_ego_pose_tokens
        ]
        cleaned["calibrated_sensor"] = [
            record
            for record in cleaned["calibrated_sensor"]
            if record.get("token") not in removed_calibrated_sensor_tokens
        ]
        cleaned["sample_annotation"] = [
            record
            for record in cleaned["sample_annotation"]
            if record.get("sample_token") not in removed_sample_tokens
        ]
        cleaned["instance"] = [
            record for record in cleaned["instance"] if record.get("token") not in removed_instance_tokens
        ]

        for map_record in cleaned["map"]:
            map_record["log_tokens"] = [
                token for token in map_record.get("log_tokens", []) if token != log_token
            ]
        cleaned["map"] = [record for record in cleaned["map"] if record.get("log_tokens")]
        return cleaned

    def _merge_token_table(self, merged, table_name, records, allow_existing):
        token_to_record = {record["token"]: record for record in merged[table_name]}
        for record in records:
            token = record["token"]
            if token in token_to_record:
                if allow_existing:
                    continue
                raise RuntimeError(
                    "Duplicate token while appending table {}: {}. "
                    "Use --replace-sequence if you are re-exporting the same sequence.".format(
                        table_name,
                        token,
                    )
                )
            merged[table_name].append(record)
            token_to_record[token] = record

    def _merge_maps(self, merged, records):
        token_to_record = {record["token"]: record for record in merged["map"]}
        for record in records:
            token = record["token"]
            if token not in token_to_record:
                merged["map"].append(record)
                token_to_record[token] = record
                continue

            existing_record = token_to_record[token]
            log_tokens = list(existing_record.get("log_tokens", []))
            for log_token in record.get("log_tokens", []):
                if log_token not in log_tokens:
                    log_tokens.append(log_token)
            existing_record["log_tokens"] = log_tokens


def main():
    args = parse_args()
    reader = SequenceReader(args.sequence_dir)
    exporter = NuScenesExporter(
        reader,
        args.output_dir,
        args.version,
        append=args.append,
        replace_sequence=args.replace_sequence,
    )
    exporter.export()
    print(
        {
            "sequence_dir": str(reader.sequence_dir),
            "scene_name": reader.scene_name,
            "output_dir": str(Path(args.output_dir).resolve()),
            "version": args.version,
            "valid_frames": len(reader.frames),
            "total_frames": len(reader.all_labels),
            "append": bool(args.append),
            "replace_sequence": bool(args.replace_sequence),
        }
    )


if __name__ == "__main__":
    main()
