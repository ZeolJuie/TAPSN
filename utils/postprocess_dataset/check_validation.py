#!/usr/bin/env python3
"""Summarize TAPSN per-sequence frame alignment.

The script accepts one positional input path: either a TAPSN dataset root
containing map directories or a single map root such as Town01_Opt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable

import numpy as np
from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_common import (
    build_sensor_world_matrix,
    load_csv_index,
    load_sensor_pose_index,
    normalize_frame_id,
    resolve_sensor_pose_frame,
    sensor_record_index,
    transform_points,
)
from utils.postprocess_dataset.common import (
    count_points_in_oriented_box,
    read_semantic_lidar_points_xyz,
)


MAP_NAMES = [
    "Town01_Opt",
    "Town02_Opt",
    "Town03_Opt",
    "Town04_Opt",
    "Town05_Opt",
    "Town10HD_Opt",
    "Town15",
]
REFERENCE_SENSOR_NAME = "CAM_FRONT"
REFERENCE_SENSOR_RELPATH = Path("sampling/image/CAM_FRONT")
SEMANTIC_LIDAR_SENSOR_NAME = "LIDAR_TOP_SEMANTIC"
MIN_VALID_FRAMES = 20
MAX_EGO_ROUTE_DISTANCE_M = 15.0
CATEGORY_COMPLETE = "complete"
CATEGORY_BOUNDARY_MISS = "boundary_miss"
CATEGORY_INVALID = "invalid"
FRAME_RE = re.compile(r"\d+")
CATALOG_CACHE: dict[Path, Any] = {}
EMPTY_POINTS_WORLD = np.empty((0, 3), dtype=np.float64)


@dataclass(frozen=True)
class SequenceRecord:
    map_name: str
    sequence_dir: Path

    @property
    def sequence_id(self) -> str:
        return self.sequence_dir.name


@dataclass(frozen=True)
class StreamSpec:
    name: str
    relpath: Path
    kind: str
    is_sensor: bool


@dataclass(frozen=True)
class FrameStats:
    spec: StreamSpec
    exists: bool
    count: int
    first: int | None
    last: int | None
    frames: frozenset[int]
    continuous: bool
    step: int | None


@dataclass(frozen=True)
class BoundaryTrim:
    stream_name: str
    start_missing: int
    end_missing: int
    is_sensor: bool


@dataclass(frozen=True)
class PedBoundaryResult:
    frame: int | None
    info: str


@dataclass
class SequenceSummary:
    map_name: str
    sequence_id: str
    category: str
    stream_count: int
    ref_count: int
    ref_range: str
    usable_count: int
    valid_count: int
    usable_range: str
    ego_route_distance_m: float | None
    ped_boundary_frame: int | None
    ped_boundary_info: str
    bad_streams: list[str]
    issue_counts: Counter[str]
    sequence_dir: Path
    deleted: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check TAPSN sequence frame count, range, and stream alignment."
    )
    parser.add_argument(
        "input_path",
        help="TAPSN dataset root or map root, e.g. TAPSN/Town10HD_Opt.",
    )
    return parser.parse_args()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def is_sequence_dir(path: Path) -> bool:
    return any(
        (path / marker).exists()
        for marker in (
            Path("sampling/log.json"),
            Path("sampling/labels.json"),
            Path("manifest.json"),
            Path("config/case.json"),
        )
    )


def sequence_dirs_from_map_root(map_root: Path) -> list[Path]:
    if not map_root.exists():
        return []
    return sorted(
        (path for path in map_root.iterdir() if path.is_dir() and is_sequence_dir(path)),
        key=natural_key,
    )


def iter_map_dirs(dataset_root: Path) -> Iterable[Path]:
    yielded = set()
    for map_name in MAP_NAMES:
        map_dir = dataset_root / map_name
        if map_dir.is_dir():
            yielded.add(map_dir.name)
            yield map_dir

    for map_dir in sorted(
        (path for path in dataset_root.iterdir() if path.is_dir() and path.name not in yielded),
        key=natural_key,
    ):
        yield map_dir


def discover_sequences(input_path: Path) -> tuple[str, list[SequenceRecord]]:
    root = input_path.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Input path does not exist: {}".format(root))

    if is_sequence_dir(root):
        return "sequence", [SequenceRecord(map_name=root.parent.name, sequence_dir=root)]

    map_sequences = sequence_dirs_from_map_root(root)
    if map_sequences:
        return (
            "map root",
            [SequenceRecord(map_name=root.name, sequence_dir=seq_dir) for seq_dir in map_sequences],
        )

    records: list[SequenceRecord] = []
    for map_dir in iter_map_dirs(root):
        for seq_dir in sequence_dirs_from_map_root(map_dir):
            records.append(SequenceRecord(map_name=map_dir.name, sequence_dir=seq_dir))

    if records:
        return "dataset root", records
    raise RuntimeError("No TAPSN sequence directories found under {}".format(root))


def sensor_relpath(sensor_type: str, sensor_name: str) -> Path | None:
    type_name = sensor_type.lower()
    if type_name.startswith("sensor."):
        type_name = type_name[len("sensor.") :]

    if "camera" in type_name and "semantic" in type_name:
        return Path("sampling/semantic") / sensor_name
    if "camera" in type_name and "depth" in type_name:
        return Path("sampling/depth") / sensor_name
    if "camera" in type_name:
        return Path("sampling/image") / sensor_name
    if "lidar" in type_name and "semantic" in type_name:
        return Path("sampling/velodyne_semantic") / sensor_name
    if "lidar" in type_name:
        return Path("sampling/velodyne") / sensor_name
    if "radar" in type_name:
        return Path("sampling/radar") / sensor_name
    if "imu" in type_name:
        return Path("sampling/imu/imu.csv")
    if "gnss" in type_name:
        return Path("sampling/gnss/gnss.csv")
    return None


def kind_from_relpath(relpath: Path) -> str:
    suffix = relpath.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    return "dir"


def specs_from_calibration(sequence_dir: Path) -> list[StreamSpec]:
    payload = read_json(sequence_dir / "sampling/calibration.json", default={})
    sensors = payload.get("sensors") if isinstance(payload, dict) else None
    if not isinstance(sensors, list):
        return []

    specs: list[StreamSpec] = []
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        sensor_name = str(sensor.get("name", "")).strip()
        sensor_type = str(sensor.get("type", "")).strip()
        if not sensor_name or not sensor_type:
            continue
        relpath = sensor_relpath(sensor_type, sensor_name)
        if relpath is None:
            continue
        specs.append(
            StreamSpec(
                name=sensor_name,
                relpath=relpath,
                kind=kind_from_relpath(relpath),
                is_sensor=True,
            )
        )
    return specs


def discover_existing_sensor_specs(sequence_dir: Path) -> list[StreamSpec]:
    sampling_dir = sequence_dir / "sampling"
    specs: list[StreamSpec] = []
    for subdir in (
        "image",
        "depth",
        "semantic",
        "velodyne",
        "velodyne_semantic",
        "radar",
    ):
        modality_dir = sampling_dir / subdir
        if not modality_dir.exists():
            continue
        for sensor_dir in sorted(
            (path for path in modality_dir.iterdir() if path.is_dir()),
            key=natural_key,
        ):
            specs.append(
                StreamSpec(
                    name=sensor_dir.name,
                    relpath=sensor_dir.relative_to(sequence_dir),
                    kind="dir",
                    is_sensor=True,
                )
            )

    for name, relpath in (
        ("IMU_BODY", Path("sampling/imu/imu.csv")),
        ("GNSS_BODY", Path("sampling/gnss/gnss.csv")),
    ):
        if (sequence_dir / relpath).exists():
            specs.append(StreamSpec(name=name, relpath=relpath, kind="csv", is_sensor=True))
    return specs


def metadata_specs() -> list[StreamSpec]:
    return [
        StreamSpec("labels.json", Path("sampling/labels.json"), "json", False),
        StreamSpec("ego.csv", Path("sampling/ego.csv"), "csv", False),
        StreamSpec("sensor_pose.csv", Path("sampling/sensor_pose.csv"), "csv", False),
    ]


def build_stream_specs(sequence_dir: Path) -> list[StreamSpec]:
    specs: list[StreamSpec] = []
    seen_relpaths: set[Path] = set()

    for spec in (
        specs_from_calibration(sequence_dir)
        + discover_existing_sensor_specs(sequence_dir)
        + metadata_specs()
    ):
        if spec.relpath in seen_relpaths:
            continue
        seen_relpaths.add(spec.relpath)
        specs.append(spec)
    return specs


def parse_frame_id(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def frame_id_from_filename(path: Path) -> int | None:
    if path.stem.isdigit():
        return int(path.stem)
    match = FRAME_RE.search(path.stem)
    if match is None:
        return None
    return int(match.group(0))


def frames_from_dir(path: Path) -> tuple[bool, list[int]]:
    if not path.exists():
        return False, []
    if not path.is_dir():
        return True, []

    frames: list[int] = []
    for child in path.iterdir():
        if not child.is_file() or child.name.startswith("."):
            continue
        frame_id = frame_id_from_filename(child)
        if frame_id is not None:
            frames.append(frame_id)
    return True, frames


def frames_from_csv(path: Path) -> tuple[bool, list[int]]:
    if not path.exists():
        return False, []
    if not path.is_file():
        return True, []

    frames: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "frame" not in reader.fieldnames:
            return True, []
        for row in reader:
            frame_id = parse_frame_id(row.get("frame"))
            if frame_id is not None:
                frames.append(frame_id)
    return True, frames


def frames_from_labels_json(path: Path) -> tuple[bool, list[int]]:
    if not path.exists():
        return False, []
    labels = read_json(path, default=[])
    if not isinstance(labels, list):
        return True, []

    frames: list[int] = []
    for frame_record in labels:
        if not isinstance(frame_record, dict):
            continue
        frame_id = parse_frame_id(
            frame_record.get("frame_id", frame_record.get("frame"))
        )
        if frame_id is not None:
            frames.append(frame_id)
    return True, frames


def last_ego_xy(sequence_dir: Path) -> tuple[float, float] | None:
    ego_path = sequence_dir / "sampling/ego.csv"
    if not ego_path.exists():
        return None

    last_xy: tuple[float, float] | None = None
    with ego_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None
        for row in reader:
            x = safe_float(row.get("location_x"))
            y = safe_float(row.get("location_y"))
            if x is None or y is None:
                continue
            last_xy = (x, y)
    return last_xy


def iter_route_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        routes = payload.get("routes")
        if isinstance(routes, list):
            for route in routes:
                if isinstance(route, dict):
                    yield route
        for value in payload.values():
            yield from iter_route_records(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from iter_route_records(item)


def find_route_record(catalog_path: Path, route_id: str) -> tuple[dict[str, Any] | None, bool]:
    catalog_key = catalog_path.resolve()
    if catalog_key not in CATALOG_CACHE:
        CATALOG_CACHE[catalog_key] = read_json(catalog_key, default={})
    catalog = CATALOG_CACHE[catalog_key]
    route_by_id: dict[str, dict[str, Any]] = {}
    for route in iter_route_records(catalog):
        value = route.get("route_id")
        if isinstance(value, str) and value not in route_by_id:
            route_by_id[value] = route

    if route_id in route_by_id:
        return route_by_id[route_id], False
    if route_id.endswith("__reverse"):
        base_route_id = route_id[: -len("__reverse")]
        if base_route_id in route_by_id:
            return route_by_id[base_route_id], True
    return None, False


def route_destination_xy(route: dict[str, Any], reverse: bool) -> tuple[float, float] | None:
    key_order = ("spawn_location", "path_waypoints") if reverse else ("destination", "path_waypoints")
    for key in key_order:
        value = route.get(key)
        if key == "path_waypoints" and isinstance(value, list) and value:
            point = value[0] if reverse else value[-1]
        else:
            point = value
        xy = point_xy(point)
        if xy is not None:
            return xy
    return None


def point_xy(point: Any) -> tuple[float, float] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    x = safe_float(point[0])
    y = safe_float(point[1])
    if x is None or y is None:
        return None
    return x, y


def expected_route_destination_xy(sequence_dir: Path) -> tuple[float, float] | None:
    case_payload = read_json(sequence_dir / "config/case.json", default={})
    if not isinstance(case_payload, dict):
        return None

    skeleton = case_payload.get("skeleton")
    if not isinstance(skeleton, dict):
        return None
    main_route = skeleton.get("main_route")
    if not isinstance(main_route, dict):
        return None
    route_id = main_route.get("route_id")
    if not isinstance(route_id, str) or not route_id:
        return None

    catalog_path_value = skeleton.get("catalog_path")
    if not isinstance(catalog_path_value, str) or not catalog_path_value:
        return None
    catalog_path = Path(catalog_path_value)
    if not catalog_path.is_absolute():
        catalog_path = (sequence_dir / catalog_path).resolve()

    route_record, reverse = find_route_record(catalog_path, route_id)
    if route_record is None:
        return None
    return route_destination_xy(route_record, reverse)


def ego_route_distance_m(sequence_dir: Path) -> float | None:
    ego_xy = last_ego_xy(sequence_dir)
    target_xy = expected_route_destination_xy(sequence_dir)
    if ego_xy is None or target_xy is None:
        return None
    return math.hypot(ego_xy[0] - target_xy[0], ego_xy[1] - target_xy[1])


def resolve_object_id(obj: dict[str, Any]) -> str | None:
    for key in ("object_id", "id", "track_id"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return None


def is_pedestrian_object(obj: dict[str, Any]) -> bool:
    label = str(obj.get("class", obj.get("type", ""))).lower()
    return label == "pedestrian"


def collect_last_pedestrian_records(
    labels: list[Any],
) -> dict[str, tuple[int, dict[str, Any]]]:
    last_records: dict[str, tuple[int, dict[str, Any]]] = {}
    for frame_record in labels:
        if not isinstance(frame_record, dict):
            continue
        frame_id = parse_frame_id(frame_record.get("frame_id", frame_record.get("frame")))
        objects = frame_record.get("objects")
        if frame_id is None or not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict) or not is_pedestrian_object(obj):
                continue
            object_id = resolve_object_id(obj)
            if object_id is None:
                continue
            previous = last_records.get(object_id)
            if previous is None or frame_id > previous[0]:
                last_records[object_id] = (frame_id, obj)
    return last_records


def load_semantic_points_world(
    sequence_dir: Path,
    frame_id: int,
    ego_by_frame: dict[str, dict[str, str]],
    sensor_pose_by_frame: dict[str, dict[str, dict[str, str]]],
    calibration_by_sensor: dict[str, dict[str, Any]],
) -> np.ndarray:
    frame_key = normalize_frame_id(frame_id)
    ego_row = ego_by_frame.get(frame_key)
    calibration_record = calibration_by_sensor.get(SEMANTIC_LIDAR_SENSOR_NAME)
    if ego_row is None or calibration_record is None:
        return EMPTY_POINTS_WORLD

    lidar_path = (
        sequence_dir
        / "sampling"
        / "velodyne_semantic"
        / SEMANTIC_LIDAR_SENSOR_NAME
        / "{}.bin".format(frame_key)
    )
    if not lidar_path.exists():
        return EMPTY_POINTS_WORLD

    pose_frame_id = resolve_sensor_pose_frame(frame_key, sensor_pose_by_frame)
    frame_sensor_pose = sensor_pose_by_frame.get(pose_frame_id, {}) if pose_frame_id is not None else {}
    sensor_world = build_sensor_world_matrix(
        ego_row=ego_row,
        sensor_name=SEMANTIC_LIDAR_SENSOR_NAME,
        calibration_record=calibration_record,
        frame_sensor_pose=frame_sensor_pose,
    )
    points_lidar, _ = read_semantic_lidar_points_xyz(lidar_path)
    return transform_points(sensor_world, points_lidar)


def object_lidar_points(
    sequence_dir: Path,
    obj: dict[str, Any],
    frame_id: int,
    points_world_by_frame: dict[int, np.ndarray],
    ego_by_frame: dict[str, dict[str, str]],
    sensor_pose_by_frame: dict[str, dict[str, dict[str, str]]],
    calibration_by_sensor: dict[str, dict[str, Any]],
) -> int:
    existing = parse_frame_id(obj.get("num_lidar_pts"))
    if existing is not None:
        return existing

    if frame_id not in points_world_by_frame:
        points_world_by_frame[frame_id] = load_semantic_points_world(
            sequence_dir=sequence_dir,
            frame_id=frame_id,
            ego_by_frame=ego_by_frame,
            sensor_pose_by_frame=sensor_pose_by_frame,
            calibration_by_sensor=calibration_by_sensor,
        )
    points_world = points_world_by_frame[frame_id]
    if points_world.size == 0:
        return 0
    try:
        return count_points_in_oriented_box(points_world, obj)
    except (KeyError, TypeError, ValueError):
        return 0


def ped_boundary_from_semantic_lidar(sequence_dir: Path) -> PedBoundaryResult:
    sampling_dir = sequence_dir / "sampling"
    labels = read_json(sampling_dir / "labels.json", default=[])
    if not isinstance(labels, list):
        return PedBoundaryResult(None, "no labels")

    last_records = collect_last_pedestrian_records(labels)
    if not last_records:
        return PedBoundaryResult(None, "no ped")

    try:
        ego_by_frame = load_csv_index(sampling_dir / "ego.csv")
    except (FileNotFoundError, KeyError):
        ego_by_frame = {}

    try:
        sensor_pose_by_frame = (
            load_sensor_pose_index(sampling_dir / "sensor_pose.csv")
            if (sampling_dir / "sensor_pose.csv").exists()
            else {}
        )
    except (FileNotFoundError, KeyError):
        sensor_pose_by_frame = {}

    calibration_payload = read_json(sampling_dir / "calibration.json", default={})
    try:
        calibration_by_sensor = sensor_record_index(calibration_payload)
    except (KeyError, TypeError):
        calibration_by_sensor = {}

    candidates: list[tuple[int, str, int]] = []
    points_world_by_frame: dict[int, np.ndarray] = {}
    for object_id, (frame_id, obj) in last_records.items():
        points = object_lidar_points(
            sequence_dir=sequence_dir,
            obj=obj,
            frame_id=frame_id,
            points_world_by_frame=points_world_by_frame,
            ego_by_frame=ego_by_frame,
            sensor_pose_by_frame=sensor_pose_by_frame,
            calibration_by_sensor=calibration_by_sensor,
        )
        if points > 0:
            candidates.append((frame_id, object_id, points))

    if not candidates:
        return PedBoundaryResult(None, "none")

    frame_id, object_id, points = min(candidates, key=lambda item: (item[0], item[1]))
    return PedBoundaryResult(
        frame=frame_id,
        info="ped_{} pts={}".format(object_id, points),
    )


def collect_frame_stats(sequence_dir: Path, spec: StreamSpec) -> FrameStats:
    path = sequence_dir / spec.relpath
    if spec.kind == "dir":
        exists, frames = frames_from_dir(path)
    elif spec.kind == "csv":
        exists, frames = frames_from_csv(path)
    elif spec.kind == "json":
        exists, frames = frames_from_labels_json(path)
    else:
        exists, frames = False, []

    unique_frames = sorted(set(frames))
    count = len(unique_frames)
    first = unique_frames[0] if unique_frames else None
    last = unique_frames[-1] if unique_frames else None
    step = None
    continuous = True
    if count > 1:
        diffs = [right - left for left, right in zip(unique_frames, unique_frames[1:])]
        unique_diffs = set(diffs)
        continuous = len(unique_diffs) == 1 and next(iter(unique_diffs)) > 0
        step = diffs[0] if continuous else None

    return FrameStats(
        spec=spec,
        exists=exists,
        count=count,
        first=first,
        last=last,
        frames=frozenset(unique_frames),
        continuous=continuous,
        step=step,
    )


def select_reference(stats: list[FrameStats]) -> FrameStats | None:
    preferred = [
        stat
        for stat in stats
        if stat.spec.name == REFERENCE_SENSOR_NAME
        or stat.spec.relpath == REFERENCE_SENSOR_RELPATH
    ]
    for stat in preferred:
        if stat.count > 0:
            return stat

    for stat in stats:
        if stat.spec.name == "labels.json" and stat.count > 0:
            return stat

    for stat in stats:
        if stat.count > 0:
            return stat

    return preferred[0] if preferred else (stats[0] if stats else None)


def compare_to_reference(stat: FrameStats, reference: FrameStats | None) -> str | None:
    if not stat.exists:
        return "missing"
    if stat.count == 0:
        return "empty"
    if reference is None or reference.count == 0:
        return "no_reference"
    if stat.count != reference.count:
        return "count_mismatch"
    if stat.first != reference.first or stat.last != reference.last:
        return "range_mismatch"
    if stat.frames != reference.frames:
        return "frame_mismatch"
    if not stat.continuous:
        return "non_contiguous"
    return None


def common_frames(stats: list[FrameStats]) -> frozenset[int]:
    frame_sets = [stat.frames if stat.exists else frozenset() for stat in stats]
    if not frame_sets:
        return frozenset()
    common = set(frame_sets[0])
    for frames in frame_sets[1:]:
        common &= frames
    return frozenset(common)


def sorted_contiguous(frames: frozenset[int]) -> bool:
    ordered = sorted(frames)
    if len(ordered) <= 1:
        return True
    diffs = {right - left for left, right in zip(ordered, ordered[1:])}
    return len(diffs) == 1 and next(iter(diffs)) > 0


def boundary_trim_against_reference(
    stat: FrameStats,
    reference: FrameStats | None,
) -> BoundaryTrim | None:
    if (
        reference is None
        or reference.count == 0
        or not stat.exists
        or stat.count == 0
        or not stat.continuous
        or stat.frames - reference.frames
    ):
        return None

    missing = sorted(reference.frames - stat.frames)
    if not missing:
        return BoundaryTrim(stat.spec.name, 0, 0, stat.spec.is_sensor)

    ref_ordered = sorted(reference.frames)
    ref_index = {frame: index for index, frame in enumerate(ref_ordered)}
    missing_indices = sorted(ref_index[frame] for frame in missing)
    start_missing = 0
    while start_missing in missing_indices:
        start_missing += 1

    end_missing = 0
    last_index = len(ref_ordered) - 1
    missing_index_set = set(missing_indices)
    while last_index - end_missing in missing_index_set:
        end_missing += 1

    if start_missing + end_missing != len(missing_indices):
        return None
    return BoundaryTrim(
        stream_name=stat.spec.name,
        start_missing=start_missing,
        end_missing=end_missing,
        is_sensor=stat.spec.is_sensor,
    )


def format_boundary_trim(trim: BoundaryTrim) -> str:
    parts = []
    if trim.start_missing:
        parts.append("start={}".format(trim.start_missing))
    if trim.end_missing:
        parts.append("end={}".format(trim.end_missing))
    if not parts:
        return trim.stream_name
    return "{}({})".format(trim.stream_name, ",".join(parts))


def classify_sequence(
    stats: list[FrameStats],
    reference: FrameStats | None,
    issue_counts: Counter[str],
) -> tuple[str, list[str]]:
    usable_frames = common_frames(stats)
    if (
        reference is None
        or reference.count < MIN_VALID_FRAMES
        or len(usable_frames) < MIN_VALID_FRAMES
    ):
        bad_streams = [
            format_bad_sensor(stat, reference)
            for stat in stats
            if stat.spec.is_sensor
            and (not stat.exists or stat.count < MIN_VALID_FRAMES)
        ]
        return CATEGORY_INVALID, bad_streams

    if not issue_counts:
        return CATEGORY_COMPLETE, []

    trims: list[BoundaryTrim] = []
    for stat in stats:
        trim = boundary_trim_against_reference(stat, reference)
        if trim is None:
            return CATEGORY_INVALID, [
                format_bad_sensor(bad_stat, reference)
                for bad_stat in stats
                if compare_to_reference(bad_stat, reference) is not None
                and bad_stat.spec.is_sensor
            ]
        if trim.start_missing or trim.end_missing:
            trims.append(trim)

    if trims:
        return CATEGORY_BOUNDARY_MISS, [
            format_boundary_trim(trim) for trim in trims if trim.is_sensor
        ]
    return CATEGORY_INVALID, [
        format_bad_sensor(stat, reference)
        for stat in stats
        if compare_to_reference(stat, reference) is not None
        and stat.spec.is_sensor
    ]


def format_bad_sensor(stat: FrameStats, reference: FrameStats | None) -> str:
    trim = boundary_trim_against_reference(stat, reference)
    if trim is not None and (trim.start_missing or trim.end_missing):
        return format_boundary_trim(trim)
    return stat.spec.name


def format_frame(frame: int | None) -> str:
    if frame is None:
        return "-"
    if frame >= 0:
        return "{:06d}".format(frame)
    return str(frame)


def format_range(first: int | None, last: int | None) -> str:
    if first is None or last is None:
        return "-"
    return "{}-{}".format(format_frame(first), format_frame(last))


def format_frame_set_range(frames: frozenset[int]) -> str:
    if not frames:
        return "-"
    ordered = sorted(frames)
    return format_range(ordered[0], ordered[-1])


def format_distance(distance_m: float | None) -> str:
    if distance_m is None or not math.isfinite(distance_m):
        return "-"
    return "{:.2f}".format(distance_m)


def format_boundary_frame(frame: int | None) -> str:
    return format_frame(frame)


def valid_frames_for_sequence(
    stats: list[FrameStats],
    category: str,
    ped_boundary_frame: int | None,
) -> frozenset[int]:
    if category == CATEGORY_INVALID:
        return frozenset()

    frames = set(common_frames(stats))
    if ped_boundary_frame is not None:
        frames = {frame for frame in frames if frame <= ped_boundary_frame}
    if len(frames) < MIN_VALID_FRAMES:
        return frozenset()
    return frozenset(frames)


def write_frame_valid(sequence_dir: Path, valid_frames: frozenset[int]) -> int:
    labels_path = sequence_dir / "sampling/labels.json"
    labels = read_json(labels_path, default=[])
    if not isinstance(labels, list):
        return 0

    valid_count = 0
    for frame_record in labels:
        if not isinstance(frame_record, dict):
            continue
        frame_id = parse_frame_id(frame_record.get("frame_id", frame_record.get("frame")))
        frame_valid = frame_id in valid_frames if frame_id is not None else False
        frame_record["frame_valid"] = frame_valid
        if frame_valid:
            valid_count += 1

    write_json(labels_path, labels)
    return valid_count


def summarize_sequence(record: SequenceRecord) -> SequenceSummary:
    specs = build_stream_specs(record.sequence_dir)
    stats = [collect_frame_stats(record.sequence_dir, spec) for spec in specs]
    reference = select_reference(stats)

    issue_counts: Counter[str] = Counter()
    for stat in stats:
        issue = compare_to_reference(stat, reference)
        if issue is None:
            continue
        issue_counts[issue] += 1

    category, bad_streams = classify_sequence(stats, reference, issue_counts)
    usable_frames = common_frames(stats)
    ped_boundary = ped_boundary_from_semantic_lidar(record.sequence_dir)
    ego_distance = ego_route_distance_m(record.sequence_dir)
    if ego_distance is not None and math.isfinite(ego_distance) and ego_distance > MAX_EGO_ROUTE_DISTANCE_M:
        category = CATEGORY_INVALID
        valid_frames = frozenset()
    else:
        valid_frames = valid_frames_for_sequence(stats, category, ped_boundary.frame)
    valid_count = write_frame_valid(record.sequence_dir, valid_frames)
    logger.info("{} valid_count={}", record.sequence_dir, valid_count)
    return SequenceSummary(
        map_name=record.map_name,
        sequence_id=record.sequence_id,
        category=category,
        stream_count=len(stats),
        ref_count=reference.count if reference is not None else 0,
        ref_range=format_range(
            reference.first if reference is not None else None,
            reference.last if reference is not None else None,
        ),
        usable_count=len(usable_frames),
        valid_count=valid_count,
        usable_range=format_frame_set_range(usable_frames)
        if sorted_contiguous(usable_frames)
        else "non-contiguous",
        ego_route_distance_m=ego_distance,
        ped_boundary_frame=ped_boundary.frame,
        ped_boundary_info=ped_boundary.info,
        bad_streams=bad_streams,
        issue_counts=issue_counts,
        sequence_dir=record.sequence_dir,
    )


def wrap_cell(value: str, width: int) -> list[str]:
    text = str(value)
    if not text:
        return [""]
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        wrapped = wrap(
            raw_line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    max_widths: dict[str, int] | None = None,
) -> None:
    max_widths = max_widths or {}
    string_rows = [[str(value) for value in row] for row in rows]
    widths: list[int] = []
    for index, header in enumerate(headers):
        width = len(header)
        for row in string_rows:
            width = max(width, len(row[index]))
        if header in max_widths:
            width = min(width, max_widths[header])
        widths.append(width)

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    print()
    print(title)
    print(border())
    print(
        "| "
        + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        + " |"
    )
    print(border())
    for row in string_rows:
        wrapped_cells = [wrap_cell(row[index], widths[index]) for index in range(len(headers))]
        row_height = max(len(cell) for cell in wrapped_cells)
        for line_index in range(row_height):
            print(
                "| "
                + " | ".join(
                    (
                        wrapped_cells[index][line_index]
                        if line_index < len(wrapped_cells[index])
                        else ""
                    ).ljust(widths[index])
                    for index in range(len(headers))
                )
                + " |"
            )
    print(border())


def print_header(input_path: Path, scope: str, summaries: list[SequenceSummary]) -> None:
    map_names = sorted({summary.map_name for summary in summaries})
    print("Input root: {}".format(input_path.expanduser().resolve()))
    print("Scope: {}".format(scope))
    print("Maps: {} ({})".format(len(map_names), ", ".join(map_names)))
    print("Sequences: {}".format(len(summaries)))
    print(
        "Reference stream: {} ({})".format(
            REFERENCE_SENSOR_NAME,
            REFERENCE_SENSOR_RELPATH,
        )
    )
    print("Minimum valid frames: {}".format(MIN_VALID_FRAMES))


def format_issue_note(issue_counts: Counter[str]) -> str:
    if not issue_counts:
        return "-"

    labels = [
        ("missing", "missing"),
        ("empty", "empty"),
        ("count_mismatch", "count mismatch"),
        ("range_mismatch", "range mismatch"),
        ("frame_mismatch", "frame mismatch"),
        ("non_contiguous", "non-contiguous"),
        ("no_reference", "no reference"),
    ]
    parts = []
    for key, label in labels:
        count = issue_counts.get(key, 0)
        if count:
            parts.append("{}={}".format(label, count))
    return ", ".join(parts) if parts else "-"


def print_sequence_summary(summaries: list[SequenceSummary]) -> None:
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary.map_name,
                summary.sequence_id,
                summary.category,
                summary.stream_count,
                summary.ref_count,
                summary.ref_range,
                summary.usable_count,
                summary.valid_count,
                summary.usable_range,
                format_distance(summary.ego_route_distance_m),
                format_boundary_frame(summary.ped_boundary_frame),
                summary.ped_boundary_info,
                ", ".join(summary.bad_streams) if summary.bad_streams else "-",
                format_issue_note(summary.issue_counts),
            ]
        )

    print_table(
        "Sequence Alignment Summary",
        [
            "Map",
            "Seq",
            "Category",
            "Streams",
            "Ref Count",
            "Ref Range",
            "Usable",
            "Valid Count",
            "Usable Range",
            "Ego-Route(m)",
            "Ped Boundary",
            "Ped Boundary Info",
            # "Bad Streams",
            # "Notes",
        ],
        rows,
        max_widths={"Ped Boundary Info": 24, "Bad Streams": 56, "Notes": 48},
    )


def delete_invalid_sequences(summaries: list[SequenceSummary]) -> int:
    deleted_count = 0
    for summary in summaries:
        if summary.valid_count != 0:
            continue
        if not summary.sequence_dir.exists():
            summary.deleted = True
            deleted_count += 1
            continue
        shutil.rmtree(summary.sequence_dir)
        summary.deleted = True
        logger.info("{} deleted", summary.sequence_dir)
        deleted_count += 1
    return deleted_count


def print_final_summary(summaries: list[SequenceSummary]) -> None:
    issue_totals: Counter[str] = Counter()
    category_totals = Counter(summary.category for summary in summaries)
    ped_boundary_count = sum(1 for summary in summaries if summary.ped_boundary_frame is not None)
    deleted_count = sum(1 for summary in summaries if summary.deleted)
    for summary in summaries:
        issue_totals.update(summary.issue_counts)

    rows = [
        ["Total sequences", len(summaries)],
        ["Complete sequences", category_totals[CATEGORY_COMPLETE]],
        ["Boundary-miss sequences", category_totals[CATEGORY_BOUNDARY_MISS]],
        ["Invalid sequences", category_totals[CATEGORY_INVALID]],
        ["Ped-boundary sequences", ped_boundary_count],
        ["Deleted sequences", deleted_count],
        ["Missing streams", issue_totals["missing"]],
        ["Empty streams", issue_totals["empty"]],
        ["Count mismatches", issue_totals["count_mismatch"]],
        ["Range mismatches", issue_totals["range_mismatch"]],
        ["Frame-set mismatches", issue_totals["frame_mismatch"]],
        ["Non-contiguous streams", issue_totals["non_contiguous"]],
        ["No-reference streams", issue_totals["no_reference"]],
    ]
    print_table("Final Summary", ["Metric", "Value"], rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    scope, records = discover_sequences(input_path)
    summaries = [summarize_sequence(record) for record in records]
    delete_invalid_sequences(summaries)

    print_header(input_path, scope, summaries)
    print_sequence_summary(summaries)
    print_final_summary(summaries)


if __name__ == "__main__":
    main()
