#!/usr/bin/env python3
"""Statistics and visualization entry point for TAPSN dataset cases.

The script is intentionally parameter-free: dataset roots, maps, and counting
rules are fixed in code so paper statistics remain reproducible.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from loguru import logger
except ModuleNotFoundError:  # Keep the script runnable in minimal envs.
    import logging

    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    class _FallbackLogger:
        def __init__(self) -> None:
            self._logger = logging.getLogger("tapsn_dataset_statistics")

        def info(self, message: str, *args: Any) -> None:
            self._logger.info(message.format(*args))

        def warning(self, message: str, *args: Any) -> None:
            self._logger.warning(message.format(*args))

    logger = _FallbackLogger()

DATASET_ROOT = Path(os.environ.get("TAPSN_ROOT", "TAPSN"))
MAP_NAMES = [
    "Town01_Opt",
    "Town02_Opt",
    "Town03_Opt",
    "Town04_Opt",
    "Town05_Opt",
    "Town06",
    "Town07",
    "Town10HD_Opt",
    "Town12",
    "Town13",
    "Town15",
]
REFERENCE_SENSOR_SUBDIR = Path("sampling/image/CAM_FRONT")
OUTPUT_DIR = Path(os.environ.get("TAPSN_ANALYSIS_DIR", "analysis/tapsn_all_maps"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapsn_matplotlib")
WEATHER_PARAMETER_SPECS = [
    ("sun_altitude_angle", "Sun Altitude (deg)"),
    ("cloudiness", "Cloudiness"),
    ("precipitation", "Precipitation"),
    ("wetness", "Wetness"),
    ("fog_density", "Fog Density"),
    ("wind_intensity", "Wind Intensity"),
]
WEATHER_PARALLEL_LABELS = {
    "sun_altitude_angle": "Sun\nAlt.",
    "cloudiness": "Cloud",
    "precipitation": "Rain",
    "wetness": "Wet",
    "fog_density": "Fog",
    "wind_intensity": "Wind",
}
EGO_MOTION_OUTPUT_PREFIX = "ego_motion_statistics_panel"
SENSOR_JITTER_OUTPUT_PREFIX = "sensor_pose_motion_jitter"
JITTER_SENSOR_NAME = "LIDAR_TOP_SEMANTIC"
JITTER_SAMPLE_RATE_HZ = 2.0
OBSTACLE_POLAR_OUTPUT_PREFIX = "obstacles_polar_distribution"
SPATIAL_OCCLUSION_MAX = 2
SPATIAL_PANEL_GROUPS = [
    ("pedestrians", "Pedestrians", 20.0),
    ("vehicles", "Vehicles", 40.0),
    ("obstacles", "Obstacles", 40.0),
]
PAPER_DPI = 300
IEEE_COLUMN_WIDTH_IN = 3.45
IEEE_TEXT_WIDTH_IN = 7.16
INCLUDE_MAIN_TITLES = False
PAPER_COLORS = {
    "vehicle": "#2F6F9F",
    "pedestrian": "#9A4F7A",
    "obstacle": "#5F8F4E",
    "barrier": "#A67C52",
    "cyclist": "#D18A2F",
    "traffic": "#B65A4A",
    "weather": "#4C9A96",
    "route": "#6F7F8F",
    "route_length": "#A66558",
    "motion_speed": "#2F6F9F",
    "motion_accel": "#B65A4A",
    "motion_yaw": "#5F8F4E",
    "dark": "#263845",
    "grid": "#B8C0C8",
    "text": "#20262E",
    "note_edge": "#D5DADF",
}
EGO_MOTION_FONT_RC = {
    "font.size": 9.4,
    "axes.labelsize": 9.4,
    "axes.titlesize": 10.0,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
}
EGO_MOTION_TITLE_SIZE = 10.5
EGO_MOTION_LABEL_SIZE = 9.6
EGO_MOTION_TICK_SIZE = 8.7
EGO_MOTION_NOTE_SIZE = 7.3
EGO_MOTION_LEGEND_SIZE = 7.5
ROUTE_TYPE_LABELS = {
    "sidewalk_through": "Sidewalk\nThrough",
    "junction_turn": "Junction\nTurn",
    "unmarked_junction_cross": "Unmarked\nJunction Cross",
    "crosswalk_cross": "Crosswalk\nCross",
    "junction_through": "Junction\nThrough",
}
BEHAVIOR_PROFILE_ORDER = [
    "steady_walk",
    "slow_walk",
    "fast_pass",
    "cautious_cross",
    "normal_cross",
    "assertive_cross",
    "ignore_gap_cross",
]
BEHAVIOR_PROFILE_LABELS = {
    "steady_walk": "Steady",
    "slow_walk": "Slow",
    "fast_pass": "Fast",
    "cautious_cross": "Cautious",
    "normal_cross": "Normal",
    "assertive_cross": "Assertive",
    "ignore_gap_cross": "Ignore-gap",
}
EGO_MOTION_VALID_RANGES = {
    "speed": (0.0, 3.0),
    "acceleration": (-5.0, 5.0),
    "yaw_rate": (-120.0, 120.0),
}
SPATIAL_GROUP_CLASS_MAP = {
    "pedestrians": {"Pedestrian"},
    "vehicles": {"Car", "Truck", "Bus", "Van"},
    "two-Wheeled_vehicles": {"Bicycle", "Motorcycle"},
    "obstacles": {"Static", "Poles", "Other", "Dynamic"},
    "fences": {"Fences", "GuardRail"},
}
OBJECT_GROUP_OUTPUT_PREFIX = "stats_sqrt_robust"
OBJECT_GROUP_CLASS_GROUP = {
    "Vehicles": ["Car", "Truck", "Bus", "Van"],
    "Cyclists": ["Bicycle", "Motorcycle"],
    "Pedestrians": ["Pedestrian"],
    "Traffic Fac.": ["TrafficLight", "TrafficSigns"],
    "Obstacles": ["Static", "Other", "Poles", "Dynamic"],
    "Barriers": ["Fences", "GuardRail"],
}
OBJECT_GROUP_SUB_COLORS = {
    "Vehicles": {
        "Car": "#2F6F9F",
        "Truck": "#245879",
        "Bus": "#4F8CAB",
        "Van": "#7FA9BD",
    },
    "Cyclists": {
        "Bicycle": "#D18A2F",
        "Motorcycle": "#A96E28",
    },
    "Pedestrians": {"Pedestrian": "#9A4F7A"},
    "Traffic Fac.": {
        "TrafficLight": "#B65A4A",
        "TrafficSigns": "#8F483E",
    },
    "Obstacles": {
        "Static": "#5F8F4E",
        "Other": "#496F3E",
        "Poles": "#8CB37E",
        "Dynamic": "#A8C97A",
    },
    "Barriers": {
        "Fences": "#A67C52",
        "GuardRail": "#7C5A3C",
    },
}


@dataclass(frozen=True)
class CasePaths:
    case_dir: Path
    manifest: Path
    case_json: Path
    scenario_toml: Path
    sampling_log: Path
    calibration: Path
    labels: Path
    ego_csv: Path
    sensor_pose_csv: Path
    pedestrian_csv: Path
    vehicle_csv: Path
    obstacles_csv: Path
    run_summary: Path
    run_result: Path

    @property
    def reference_sensor_dir(self) -> Path:
        return self.case_dir / REFERENCE_SENSOR_SUBDIR

    @property
    def case_id(self) -> str:
        return self.case_dir.name


@dataclass(frozen=True)
class ValidLabelInfo:
    frames: list[dict[str, Any]]
    normalized_frame_ids: set[str]
    source_frame_ids: set[str]


@dataclass
class CaseRecord:
    paths: CasePaths
    manifest: dict[str, Any]
    case_json: dict[str, Any]
    sampling_log: dict[str, Any]
    _valid_label_info_cache: ValidLabelInfo | None = field(default=None, init=False, repr=False)

    @property
    def case_id(self) -> str:
        return self.paths.case_id

    @property
    def map_name(self) -> str:
        return (
            self.case_json.get("skeleton", {}).get("map_name")
            or self.sampling_log.get("location")
            or self.paths.case_dir.parent.name
        )

    @property
    def is_valid_sampled(self) -> bool:
        if not self.paths.labels.exists() or not self.paths.ego_csv.exists():
            return False
        duration = self.duration_sec
        return self.valid_frame_count > 0 and duration is not None and duration > 0

    @property
    def sensor_count(self) -> int | None:
        sensor_count = safe_int(self.manifest.get("sensor_count"))
        if sensor_count is not None:
            return sensor_count
        calibration = read_json(self.paths.calibration, default={})
        sensors = calibration.get("sensors")
        return len(sensors) if isinstance(sensors, list) else None

    @property
    def frame_num(self) -> int | None:
        return self.valid_frame_count

    @property
    def duration_sec(self) -> float | None:
        duration = duration_from_ego_csv(self.paths.ego_csv, self.valid_frame_ids)
        if duration is not None:
            return duration
        return safe_float(self.sampling_log.get("duration"))

    @property
    def valid_label_info(self) -> ValidLabelInfo:
        if self._valid_label_info_cache is None:
            self._valid_label_info_cache = load_valid_label_info(self.paths.labels)
        return self._valid_label_info_cache

    @property
    def valid_frame_count(self) -> int:
        return len(self.valid_label_info.frames)

    @property
    def valid_frame_ids(self) -> set[str]:
        return self.valid_label_info.normalized_frame_ids


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_frame_id(value: Any) -> str:
    frame = safe_int(value)
    if frame is None:
        return str(value)
    return str(frame)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def load_valid_label_info(labels_path: Path) -> ValidLabelInfo:
    labels = read_json(labels_path, default=[])
    if not isinstance(labels, list):
        return ValidLabelInfo(frames=[], normalized_frame_ids=set(), source_frame_ids=set())

    missing = [
        normalize_frame_id(frame.get("frame_id"))
        for frame in labels
        if isinstance(frame, dict) and "frame_valid" not in frame
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"{labels_path} is missing frame_valid on {len(missing)} frames, examples {preview}"
        )

    non_bool = [
        normalize_frame_id(frame.get("frame_id"))
        for frame in labels
        if isinstance(frame, dict)
        and "frame_valid" in frame
        and not isinstance(frame["frame_valid"], bool)
    ]
    if non_bool:
        preview = ", ".join(non_bool[:5])
        raise RuntimeError(
            f"{labels_path} has non-boolean frame_valid on {len(non_bool)} frames, examples {preview}"
        )

    valid_indices = [
        index
        for index, frame in enumerate(labels)
        if isinstance(frame, dict) and frame.get("frame_valid") is True
    ]
    if not valid_indices:
        return ValidLabelInfo(frames=[], normalized_frame_ids=set(), source_frame_ids=set())

    contiguous_indices = list(range(valid_indices[0], valid_indices[-1] + 1))
    if valid_indices != contiguous_indices:
        raise RuntimeError(f"{labels_path} frame_valid=true frames are not contiguous")

    frames = [labels[index] for index in valid_indices]
    normalized_frame_ids = {
        normalize_frame_id(frame.get("frame_id"))
        for frame in frames
        if isinstance(frame, dict)
    }
    source_frame_ids = {
        str(frame.get("frame_id"))
        for frame in frames
        if isinstance(frame, dict)
    }
    return ValidLabelInfo(
        frames=frames,
        normalized_frame_ids=normalized_frame_ids,
        source_frame_ids=source_frame_ids,
    )


def duration_from_ego_csv(ego_csv_path: Path, valid_frame_ids: set[str]) -> float | None:
    if not ego_csv_path.exists() or not valid_frame_ids:
        return None
    timestamps = []
    for row in iter_csv_rows(ego_csv_path) or []:
        frame = normalize_frame_id(row.get("frame"))
        if frame not in valid_frame_ids:
            continue
        timestamp = safe_float(row.get("timestamp"))
        if timestamp is not None and math.isfinite(timestamp):
            timestamps.append(timestamp)
    if len(timestamps) < 2:
        return None
    return max(timestamps) - min(timestamps)


def iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def make_case_paths(case_dir: Path) -> CasePaths:
    return CasePaths(
        case_dir=case_dir,
        manifest=case_dir / "manifest.json",
        case_json=case_dir / "config" / "case.json",
        scenario_toml=case_dir / "config" / "scenario.toml",
        sampling_log=case_dir / "sampling" / "log.json",
        calibration=case_dir / "sampling" / "calibration.json",
        labels=case_dir / "sampling" / "labels.json",
        ego_csv=case_dir / "sampling" / "ego.csv",
        sensor_pose_csv=case_dir / "sampling" / "sensor_pose.csv",
        pedestrian_csv=case_dir / "simulation" / "pedestrian.csv",
        vehicle_csv=case_dir / "simulation" / "vehicle.csv",
        obstacles_csv=case_dir / "simulation" / "obstacles.csv",
        run_summary=case_dir / "simulation" / "run_summary.json",
        run_result=case_dir / "logs" / "run_result.json",
    )


def discover_cases(map_root: Path) -> list[CaseRecord]:
    if not map_root.exists():
        raise FileNotFoundError(f"Map root does not exist: {map_root}")

    records: list[CaseRecord] = []
    for case_dir in sorted(p for p in map_root.iterdir() if p.is_dir()):
        paths = make_case_paths(case_dir)
        if not (
            paths.case_json.exists()
            or paths.manifest.exists()
            or paths.sampling_log.exists()
        ):
            continue
        records.append(
            CaseRecord(
                paths=paths,
                manifest=read_json(paths.manifest, default={}),
                case_json=read_json(paths.case_json, default={}),
                sampling_log=read_json(paths.sampling_log, default={}),
            )
        )
    return records


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            try:
                total += filepath.stat().st_size
            except OSError:
                continue
    return total


def count_label_annotations(labels_path: Path) -> tuple[int, Counter[str]]:
    return count_label_annotations_in_frames(load_valid_label_info(labels_path).frames)


def count_label_annotations_in_frames(labels: list[dict[str, Any]]) -> tuple[int, Counter[str]]:
    total = 0
    class_counts: Counter[str] = Counter()
    for frame in labels:
        objects = frame.get("objects", [])
        if not isinstance(objects, list):
            continue
        total += len(objects)
        for obj in objects:
            if isinstance(obj, dict):
                class_counts[str(obj.get("class", "Unknown"))] += 1
    return total, class_counts


def summarize_counter(counter: Counter[str], top_n: int | None = None) -> str:
    items = counter.most_common(top_n)
    if not items:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in items)


def count_sensor_frames(sensor_dir: Path) -> int:
    if not sensor_dir.exists():
        return 0
    return sum(1 for p in sensor_dir.iterdir() if p.is_file())


def parse_ego_ped_id(ego_selection_path: Path, manifest_path: Path) -> int | None:
    for path in (ego_selection_path, manifest_path):
        payload = read_json(path, default={})
        resolved = payload.get("resolved_ego") if isinstance(payload, dict) else None
        if not isinstance(resolved, dict):
            continue
        ped_name = resolved.get("ped_name")
        if isinstance(ped_name, str) and ped_name.startswith("ped_"):
            suffix = ped_name.split("_", 1)[1]
            try:
                return int(suffix)
            except ValueError:
                continue
    return None


def load_all_cases() -> list[CaseRecord]:
    all_records: list[CaseRecord] = []
    for map_name in MAP_NAMES:
        map_root = DATASET_ROOT / map_name
        logger.info(f"Scanning map root: {map_root}")
        records = discover_cases(map_root)
        logger.info(f"Discovered {len(records)} cases for {map_name}")
        all_records.extend(records)
    logger.info(f"Discovered {len(all_records)} cases in total")
    return all_records


def find_sensor_count(records: list[CaseRecord]) -> tuple[int | None, list[str]]:
    for record in records:
        calibration = read_json(record.paths.calibration, default={})
        sensors = calibration.get("sensors")
        if isinstance(sensors, list) and sensors:
            names = [str(s.get("name", "Unknown")) for s in sensors if isinstance(s, dict)]
            return len(sensors), names
    return None, []


def build_dataset_scale_overview(records: list[CaseRecord]) -> dict[str, Any]:
    logger.info("Building Dataset Scale Overview")
    maps = Counter(record.map_name for record in records)
    case_count = len(records)

    total_frames = 0
    missing_valid_frames = []
    for record in records:
        frame_count = record.valid_frame_count
        if frame_count == 0:
            missing_valid_frames.append(record.case_id)
        total_frames += frame_count

    if missing_valid_frames:
        logger.warning(
            "No frame_valid=true labels for {} cases: {}",
            len(missing_valid_frames),
            ", ".join(missing_valid_frames[:10])
            + ("..." if len(missing_valid_frames) > 10 else ""),
        )

    total_duration = 0.0
    missing_duration = []
    for record in records:
        duration = record.duration_sec
        if duration is None or not math.isfinite(duration):
            missing_duration.append(record.case_id)
            continue
        total_duration += duration

    if missing_duration:
        logger.warning(
            "Missing sampling duration for {} cases: {}",
            len(missing_duration),
            ", ".join(missing_duration[:10])
            + ("..." if len(missing_duration) > 10 else ""),
        )

    sensor_count, sensor_names = find_sensor_count(records)
    if sensor_count is None:
        logger.warning("No usable calibration.json found for sensor count")

    total_annotations = 0
    object_counts: Counter[str] = Counter()
    for record in records:
        annotation_count, class_counts = count_label_annotations_in_frames(
            record.valid_label_info.frames
        )
        total_annotations += annotation_count
        object_counts.update(class_counts)

    logger.info("Computing storage size")
    storage_bytes = sum(directory_size_bytes(record.paths.case_dir) for record in records)

    return {
        "maps": maps,
        "case_count": case_count,
        "reference_sensor": str(REFERENCE_SENSOR_SUBDIR),
        "total_frames": total_frames,
        "total_duration_sec": total_duration,
        "sensor_count": sensor_count,
        "sensor_names": sensor_names,
        "total_annotations": total_annotations,
        "object_counts": object_counts,
        "storage_bytes": storage_bytes,
        "missing_valid_frame_cases": missing_valid_frames,
        "missing_duration_cases": missing_duration,
    }


def format_duration(seconds: float) -> str:
    minutes = seconds / 60.0
    hours = seconds / 3600.0
    return f"{seconds:,.2f} s ({minutes:,.2f} min / {hours:,.2f} h)"


def format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def print_table(title: str, rows: list[tuple[str, str]]) -> None:
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    title_width = len(title) + 2
    inner_width = max(label_width + value_width + 5, title_width)

    print()
    print("+" + "-" * inner_width + "+")
    print("| " + title.ljust(inner_width - 2) + " |")
    print("+" + "-" * (label_width + 2) + "+" + "-" * (inner_width - label_width - 3) + "+")
    for label, value in rows:
        print(
            "| "
            + label.ljust(label_width)
            + " | "
            + value.ljust(inner_width - label_width - 5)
            + " |"
        )
    print("+" + "-" * (label_width + 2) + "+" + "-" * (inner_width - label_width - 3) + "+")


def apply_ieee_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": PAPER_DPI,
            "savefig.dpi": PAPER_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "text.color": PAPER_COLORS["text"],
            "axes.labelcolor": PAPER_COLORS["text"],
            "xtick.color": PAPER_COLORS["text"],
            "ytick.color": PAPER_COLORS["text"],
        }
    )


def set_optional_title(ax: Any, title: str, *, pad: float = 5.0) -> None:
    if INCLUDE_MAIN_TITLES:
        ax.set_title(title, pad=pad)


def add_panel_title(ax: Any, title: str, *, pad: float = 4.0) -> None:
    ax.set_title(title, pad=pad)


def format_route_type_label(route_type: str) -> str:
    return ROUTE_TYPE_LABELS.get(route_type, route_type.replace("_", " ").title())


def format_behavior_profile_label(behavior: str) -> str:
    return BEHAVIOR_PROFILE_LABELS.get(behavior, behavior.replace("_", "\n").title())


def print_dataset_scale_overview(summary: dict[str, Any]) -> None:
    maps = summary["maps"]
    rows = [
        ("Dataset root", str(DATASET_ROOT)),
        ("Maps", f"{len(maps)} ({', '.join(maps.keys())})"),
        ("Cases", format_int(summary["case_count"])),
        ("Reference sensor", summary["reference_sensor"]),
        ("Valid frames", format_int(summary["total_frames"])),
        ("Total duration", format_duration(summary["total_duration_sec"])),
        ("Sensor count", format_int(summary["sensor_count"])),
        ("Annotations", format_int(summary["total_annotations"])),
        ("Storage size", human_bytes(summary["storage_bytes"])),
    ]
    print_table("Dataset Scale Overview", rows)

    sensor_names = summary["sensor_names"]
    if sensor_names:
        print()
        print("Sensors")
        print("  " + ", ".join(sensor_names))

    object_counts = summary["object_counts"]
    if object_counts:
        print()
        print("Top annotation classes")
        print("  " + summarize_counter(object_counts, top_n=12))

    missing_reference = summary["missing_valid_frame_cases"]
    missing_duration = summary["missing_duration_cases"]
    if missing_reference or missing_duration:
        print()
        print("Notes")
        if missing_reference:
            print(
                "  "
                + f"{len(missing_reference)} cases have no frame_valid=true labels"
            )
        if missing_duration:
            print(f"  {len(missing_duration)} cases have no sampling duration")


def run_dataset_scale_overview() -> list[CaseRecord]:
    records = load_all_cases()
    summary = build_dataset_scale_overview(records)
    print_dataset_scale_overview(summary)
    return records


def collect_route_type_coverage(records: list[CaseRecord]) -> list[dict[str, Any]]:
    logger.info("Building Route Type Coverage")
    stats: dict[str, dict[str, Any]] = {}
    missing_route_type = []
    missing_route_length = []
    missing_behavior = []

    for record in records:
        skeleton = record.case_json.get("skeleton", {})
        main_route = skeleton.get("main_route") or {}
        plan = record.case_json.get("plan", {})
        route_type = main_route.get("route_type")
        route_length = safe_float(main_route.get("route_length_m"))
        behavior = plan.get("ego_behavior")

        if not route_type:
            missing_route_type.append(record.case_id)
            route_type = "Unknown"
        if route_length is None:
            missing_route_length.append(record.case_id)
            route_length = 0.0
        if not behavior:
            missing_behavior.append(record.case_id)
            behavior = "Unknown"

        bucket = stats.setdefault(
            str(route_type),
            {
                "route_type": str(route_type),
                "count": 0,
                "total_length_m": 0.0,
                "lengths": [],
                "behavior_counts": Counter(),
            },
        )
        bucket["count"] += 1
        bucket["total_length_m"] += route_length
        bucket["lengths"].append(route_length)
        bucket["behavior_counts"][str(behavior)] += 1

    if missing_route_type:
        logger.warning(
            "Missing main route type for {} cases: {}",
            len(missing_route_type),
            ", ".join(missing_route_type[:10])
            + ("..." if len(missing_route_type) > 10 else ""),
        )
    if missing_route_length:
        logger.warning(
            "Missing main route length for {} cases: {}",
            len(missing_route_length),
            ", ".join(missing_route_length[:10])
            + ("..." if len(missing_route_length) > 10 else ""),
        )
    if missing_behavior:
        logger.warning(
            "Missing ego behavior for {} cases: {}",
            len(missing_behavior),
            ", ".join(missing_behavior[:10])
            + ("..." if len(missing_behavior) > 10 else ""),
        )

    rows = []
    total_cases = sum(item["count"] for item in stats.values())
    for item in stats.values():
        count = item["count"]
        lengths = item["lengths"]
        rows.append(
            {
                "route_type": item["route_type"],
                "count": count,
                "ratio": count / total_cases if total_cases else 0.0,
                "total_length_m": item["total_length_m"],
                "mean_length_m": item["total_length_m"] / count if count else 0.0,
                "min_length_m": min(lengths) if lengths else 0.0,
                "max_length_m": max(lengths) if lengths else 0.0,
                "behavior_counts": item["behavior_counts"],
            }
        )

    return sorted(rows, key=lambda row: (-row["count"], row["route_type"]))


def print_route_type_coverage(rows: list[dict[str, Any]]) -> None:
    table_rows = [
        (
            row["route_type"],
            (
                f"count={row['count']:,}, "
                f"ratio={row['ratio'] * 100:.1f}%, "
                f"total={row['total_length_m']:.1f} m, "
                f"mean={row['mean_length_m']:.1f} m"
            ),
        )
        for row in rows
    ]
    print_table("Route Type Coverage", table_rows)


def plot_route_type_coverage(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        logger.warning("No route type rows to plot")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    output_dir.mkdir(parents=True, exist_ok=True)
    route_types = [format_route_type_label(row["route_type"]) for row in rows]
    observed_behaviors = {
        behavior
        for row in rows
        for behavior, count in row["behavior_counts"].items()
        if count > 0
    }
    behaviors = [
        behavior for behavior in BEHAVIOR_PROFILE_ORDER if behavior in observed_behaviors
    ]
    behaviors.extend(sorted(observed_behaviors - set(behaviors)))
    if not behaviors:
        logger.warning("No behavior counts to plot for route coverage")
        return

    x_values = []
    y_values = []
    counts = []
    for route_idx, row in enumerate(rows):
        behavior_counts = row["behavior_counts"]
        for behavior_idx, behavior in enumerate(behaviors):
            count = behavior_counts.get(behavior, 0)
            if count <= 0:
                continue
            x_values.append(route_idx)
            y_values.append(behavior_idx)
            counts.append(count)

    apply_ieee_plot_style()

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, IEEE_COLUMN_WIDTH_IN), dpi=PAPER_DPI)
    max_count = max(counts) if counts else 1
    sizes = [32.0 + 250.0 * (count / max_count) ** 0.75 for count in counts]
    cmap = LinearSegmentedColormap.from_list(
        "route_behavior_bubble",
        ["#D7E8E6", PAPER_COLORS["weather"], PAPER_COLORS["dark"]],
    )
    scatter = ax.scatter(
        x_values,
        y_values,
        s=sizes,
        c=counts,
        cmap=cmap,
        vmin=0,
        vmax=max_count,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )

    for x_pos, y_pos, count in zip(x_values, y_values, counts):
        if count >= max(2, max_count * 0.12):
            ax.text(
                x_pos,
                y_pos,
                str(count),
                ha="center",
                va="center",
                color="white",
                fontsize=5.4,
                fontweight="bold",
                zorder=4,
            )

    ax.set_xticks(range(len(route_types)))
    ax.set_xticklabels(route_types, rotation=25, ha="right")
    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels([format_behavior_profile_label(behavior) for behavior in behaviors])
    ax.set_xlim(-0.5, len(route_types) - 0.5)
    ax.set_ylim(-0.5, len(behaviors) - 0.5)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", linewidth=0.5, color=PAPER_COLORS["grid"], alpha=0.45)
    ax.tick_params(axis="both", length=0)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Cases", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)

    fig.tight_layout()

    png_path = output_dir / "route_type_coverage_dual_axis.png"
    pdf_path = output_dir / "route_type_coverage_dual_axis.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved route type coverage figure: {png_path}")
    logger.info(f"Saved route type coverage figure: {pdf_path}")


def run_route_type_coverage(records: list[CaseRecord]) -> None:
    rows = collect_route_type_coverage(records)
    print_route_type_coverage(rows)
    plot_route_type_coverage(rows, OUTPUT_DIR)


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "n": n,
        "min": min(values),
        "mean": mean,
        "max": max(values),
        "std": math.sqrt(variance),
    }


def collect_weather_parameter_distributions(
    records: list[CaseRecord],
) -> tuple[list[dict[str, Any]], list[str]]:
    logger.info("Building Weather Diversity Statistics")
    missing_weather_cases = []
    series_map: dict[str, list[float]] = {key: [] for key, _ in WEATHER_PARAMETER_SPECS}

    for record in records:
        weather = record.sampling_log.get("weather", {})
        params = weather.get("parameters", {}) if isinstance(weather, dict) else {}
        if not isinstance(params, dict) or not params:
            missing_weather_cases.append(record.case_id)
            continue

        for key, _ in WEATHER_PARAMETER_SPECS:
            value = safe_float(params.get(key))
            if value is not None and math.isfinite(value):
                series_map[key].append(value)

    if missing_weather_cases:
        logger.warning(
            "Missing weather.parameters for {} cases: {}",
            len(missing_weather_cases),
            ", ".join(missing_weather_cases[:10])
            + ("..." if len(missing_weather_cases) > 10 else ""),
        )

    rows = []
    for key, label in WEATHER_PARAMETER_SPECS:
        values = series_map[key]
        summary = summarize_numeric(values)
        rows.append(
            {
                "key": key,
                "label": label,
                "values": values,
                **summary,
            }
        )
    return rows, missing_weather_cases


def print_weather_parameter_summary(rows: list[dict[str, Any]]) -> None:
    table_rows = [
        (
            row["label"],
            (
                f"n={int(row['n'])}, "
                f"min={row['min']:.2f}, "
                f"mean={row['mean']:.2f}, "
                f"max={row['max']:.2f}, "
                f"std={row['std']:.2f}"
            ),
        )
        for row in rows
    ]
    print_table("Weather Diversity Summary", table_rows)


def collect_weather_case_vectors(records: list[CaseRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    logger.info("Building Weather Case Vectors")
    rows: list[dict[str, Any]] = []
    missing_weather_cases = []
    keys = [key for key, _ in WEATHER_PARAMETER_SPECS]

    for record in records:
        weather = record.sampling_log.get("weather", {})
        params = weather.get("parameters", {}) if isinstance(weather, dict) else {}
        if not isinstance(params, dict) or not params:
            missing_weather_cases.append(record.case_id)
            continue

        values = []
        missing_value = False
        for key in keys:
            value = safe_float(params.get(key))
            if value is None or not math.isfinite(value):
                missing_value = True
                break
            values.append(value)
        if missing_value:
            missing_weather_cases.append(record.case_id)
            continue

        rows.append(
            {
                "case_id": record.case_id,
                "map_name": record.map_name,
                "values": values,
            }
        )

    if missing_weather_cases:
        logger.warning(
            "Missing complete weather vectors for {} cases: {}",
            len(missing_weather_cases),
            ", ".join(missing_weather_cases[:10])
            + ("..." if len(missing_weather_cases) > 10 else ""),
        )
    return rows, missing_weather_cases


def kmeans_weather_clusters(values: Any, cluster_count: int = 4) -> Any:
    import numpy as np

    if len(values) == 0:
        return np.asarray([], dtype=int)
    actual_clusters = min(cluster_count, len(values))
    if actual_clusters <= 1:
        return np.zeros(len(values), dtype=int)

    centers = values[np.linspace(0, len(values) - 1, actual_clusters, dtype=int)].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(40):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_idx in range(actual_clusters):
            members = values[labels == cluster_idx]
            if len(members) > 0:
                centers[cluster_idx] = members.mean(axis=0)
    return labels


def choose_highlight_weather_cluster(values_norm: Any, labels: Any) -> int:
    import numpy as np

    if len(labels) == 0:
        return 0
    cluster_scores = []
    for cluster_idx in sorted(set(labels.tolist())):
        members = values_norm[labels == cluster_idx]
        if len(members) == 0:
            continue
        mean = members.mean(axis=0)
        severity = mean[1:].mean() + (1.0 - mean[0]) * 0.35
        cluster_scores.append((severity, cluster_idx))
    if not cluster_scores:
        return 0
    return max(cluster_scores)[1]


def plot_weather_parallel_coordinates(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    if not rows:
        logger.warning("No weather case vectors to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_ieee_plot_style()

    keys = [key for key, _ in WEATHER_PARAMETER_SPECS]
    labels = [WEATHER_PARALLEL_LABELS.get(key, label) for key, label in WEATHER_PARAMETER_SPECS]
    values = np.asarray([row["values"] for row in rows], dtype=float)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    values_norm = (values - mins) / spans

    order = np.lexsort(tuple(values_norm[:, idx] for idx in reversed(range(values_norm.shape[1]))))
    values_norm = values_norm[order]
    cluster_labels = kmeans_weather_clusters(values_norm, cluster_count=4)
    highlight_cluster = choose_highlight_weather_cluster(values_norm, cluster_labels)
    highlight_mask = cluster_labels == highlight_cluster

    x = np.arange(len(keys), dtype=float)
    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, IEEE_COLUMN_WIDTH_IN), dpi=PAPER_DPI)

    background_segments = [np.column_stack([x, row]) for row in values_norm[~highlight_mask]]
    if background_segments:
        ax.add_collection(
            LineCollection(
                background_segments,
                colors=PAPER_COLORS["grid"],
                linewidths=0.45,
                alpha=0.28,
                zorder=1,
            )
        )

    highlight_segments = [np.column_stack([x, row]) for row in values_norm[highlight_mask]]
    if highlight_segments:
        ax.add_collection(
            LineCollection(
                highlight_segments,
                colors=PAPER_COLORS["weather"],
                linewidths=0.75,
                alpha=0.70,
                zorder=3,
            )
        )

    for axis_idx, xpos in enumerate(x):
        ax.vlines(xpos, 0.0, 1.0, color=PAPER_COLORS["text"], linewidth=0.7, zorder=2)
        tick_values = [mins[axis_idx], (mins[axis_idx] + maxs[axis_idx]) / 2.0, maxs[axis_idx]]
        tick_positions = (np.asarray(tick_values) - mins[axis_idx]) / spans[axis_idx]
        for tick_pos, tick_value in zip(tick_positions, tick_values):
            ax.hlines(
                tick_pos,
                xpos - 0.035,
                xpos + 0.035,
                color=PAPER_COLORS["text"],
                linewidth=0.45,
                zorder=4,
            )
            ax.text(
                xpos + 0.045,
                tick_pos,
                f"{tick_value:.0f}",
                ha="left",
                va="center",
                fontsize=5.2,
                color=PAPER_COLORS["text"],
                zorder=4,
            )

    if highlight_segments:
        highlight_mean = values_norm[highlight_mask].mean(axis=0)
        ax.plot(x, highlight_mean, color=PAPER_COLORS["dark"], linewidth=1.25, zorder=5)

    highlight_count = int(highlight_mask.sum())
    ax.text(
        0.0,
        1.04,
        f"highlight: n={highlight_count}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            alpha=0.82,
            edgecolor=PAPER_COLORS["note_edge"],
        ),
        zorder=6,
    )

    ax.set_xlim(x[0] - 0.15, x[-1] + 0.45)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    png_path = output_dir / "weather_parameter_distributions.png"
    pdf_path = output_dir / "weather_parameter_distributions.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved weather parallel-coordinates figure: {png_path}")
    logger.info(f"Saved weather parallel-coordinates figure: {pdf_path}")


def _gaussian_smooth_histogram(
    counts: Any,
    sigma_bins: float = 1.0,
) -> Any:
    import numpy as np

    if len(counts) == 0:
        return counts
    radius = max(1, int(math.ceil(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts, kernel, mode="same")
    if len(smoothed) == len(counts):
        return smoothed
    if len(smoothed) > len(counts):
        start = (len(smoothed) - len(counts)) // 2
        return smoothed[start : start + len(counts)]
    padded = np.zeros(len(counts), dtype=float)
    padded[: len(smoothed)] = smoothed
    return padded


def plot_weather_parameter_distributions(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_ieee_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(IEEE_TEXT_WIDTH_IN, 4.25), dpi=PAPER_DPI)
    axes = axes.flatten()
    hist_color = PAPER_COLORS["weather"]
    curve_color = PAPER_COLORS["dark"]

    for ax, row in zip(axes, rows):
        values = row["values"]
        label = row["label"]
        if not values:
            add_panel_title(ax, label)
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        values_np = np.asarray(values, dtype=float)
        bin_count = min(12, max(6, int(round(math.sqrt(len(values_np))))))
        counts, edges, _ = ax.hist(
            values_np,
            bins=bin_count,
            color=hist_color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        smooth_counts = _gaussian_smooth_histogram(counts, sigma_bins=1.0)
        ax.plot(centers, smooth_counts, color=curve_color, linewidth=1.3, zorder=3)

        add_panel_title(ax, label)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color=PAPER_COLORS["grid"], alpha=0.55, zorder=0)
        ax.set_ylabel("Cases")
        stat_text = (
            f"n={int(row['n'])}\n"
            f"mean={row['mean']:.1f}\n"
            f"std={row['std']:.1f}"
        )
        ax.text(
            0.97,
            0.95,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                alpha=0.82,
                edgecolor=PAPER_COLORS["note_edge"],
            ),
        )

    fig.tight_layout()

    png_path = output_dir / "weather_parameter_distributions.png"
    pdf_path = output_dir / "weather_parameter_distributions.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved weather diversity figure: {png_path}")
    logger.info(f"Saved weather diversity figure: {pdf_path}")


def run_weather_diversity_statistics(records: list[CaseRecord]) -> None:
    rows, _ = collect_weather_parameter_distributions(records)
    print_weather_parameter_summary(rows)
    case_rows, _ = collect_weather_case_vectors(records)
    plot_weather_parallel_coordinates(case_rows, OUTPUT_DIR)


def unwrap_degrees(angle_deg: float, prev_angle_deg: float) -> float:
    delta = angle_deg - prev_angle_deg
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return prev_angle_deg + delta


def collect_ego_motion_samples(records: list[CaseRecord]) -> dict[str, list[float]]:
    logger.info("Building Ego Motion Statistics")
    speed_samples: list[float] = []
    acceleration_samples: list[float] = []
    yaw_rate_samples: list[float] = []
    missing_ego_csv = []

    for record in records:
        valid_frame_ids = record.valid_frame_ids
        if not record.paths.ego_csv.exists():
            missing_ego_csv.append(record.case_id)
            continue

        rows = list(iter_csv_rows(record.paths.ego_csv) or [])
        if not rows:
            continue

        prev_time = None
        prev_speed = None
        prev_yaw_unwrapped = None
        for row in rows:
            if normalize_frame_id(row.get("frame")) not in valid_frame_ids:
                continue
            vx = safe_float(row.get("velocity_x"))
            vy = safe_float(row.get("velocity_y"))
            vz = safe_float(row.get("velocity_z"))
            timestamp = safe_float(row.get("timestamp"))
            yaw = safe_float(row.get("rotation_yaw"))
            if None in (vx, vy, vz, timestamp, yaw):
                continue

            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            speed_samples.append(speed)

            if prev_yaw_unwrapped is None:
                yaw_unwrapped = yaw
            else:
                yaw_unwrapped = unwrap_degrees(yaw, prev_yaw_unwrapped)

            if prev_time is not None and timestamp > prev_time:
                dt = timestamp - prev_time
                if prev_speed is not None:
                    acceleration_samples.append((speed - prev_speed) / dt)
                if prev_yaw_unwrapped is not None:
                    yaw_rate_samples.append((yaw_unwrapped - prev_yaw_unwrapped) / dt)

            prev_time = timestamp
            prev_speed = speed
            prev_yaw_unwrapped = yaw_unwrapped

    if missing_ego_csv:
        logger.warning(
            "Missing ego.csv for {} cases: {}",
            len(missing_ego_csv),
            ", ".join(missing_ego_csv[:10]) + ("..." if len(missing_ego_csv) > 10 else ""),
        )

    return {
        "speed": speed_samples,
        "acceleration": acceleration_samples,
        "yaw_rate": yaw_rate_samples,
    }


def summarize_motion_metric(values: list[float]) -> dict[str, float]:
    import numpy as np

    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def filter_motion_metric(values: list[float], key: str) -> tuple[list[float], int]:
    low, high = EGO_MOTION_VALID_RANGES[key]
    filtered = [value for value in values if low <= value <= high]
    return filtered, len(values) - len(filtered)


def filter_motion_samples(
    samples: dict[str, list[float]],
) -> tuple[dict[str, list[float]], dict[str, int]]:
    filtered_samples: dict[str, list[float]] = {}
    removed_counts: dict[str, int] = {}
    for key, values in samples.items():
        if key in EGO_MOTION_VALID_RANGES:
            filtered_values, removed = filter_motion_metric(values, key)
        else:
            filtered_values = values
            removed = 0
        filtered_samples[key] = filtered_values
        removed_counts[key] = removed
    return filtered_samples, removed_counts


def print_ego_motion_summary(samples: dict[str, list[float]]) -> None:
    rows = []
    filtered_samples, _ = filter_motion_samples(samples)
    for key, label in (
        ("speed", "Speed (m/s)"),
        ("acceleration", "Acceleration (m/s^2)"),
        ("yaw_rate", "Yaw Rate (deg/s)"),
    ):
        stats = summarize_motion_metric(filtered_samples[key])
        rows.append(
            (
                label,
                (
                    # f"n={int(stats['n'])}, "
                    # f"removed={removed_counts[key]}, "
                    f"mean={stats['mean']:.2f}, "
                    f"median={stats['median']:.2f}, "
                    f"p95={stats['p95']:.2f}, "
                    f"max={stats['max']:.2f}, "
                    f"std={stats['std']:.2f}"
                ),
            )
        )
    print_table("Ego Motion Summary", rows)


def clip_for_plot(values: list[float], low_pct: float = 0.5, high_pct: float = 99.5) -> list[float]:
    import numpy as np

    if not values:
        return values
    arr = np.asarray(values, dtype=float)
    low = np.percentile(arr, low_pct)
    high = np.percentile(arr, high_pct)
    clipped = arr[(arr >= low) & (arr <= high)]
    return clipped.tolist()


def style_compact_panel_axis(ax: Any) -> None:
    ax.grid(True, which="major", color="#D9DEE3", linewidth=0.45, zorder=0)
    ax.grid(True, which="minor", color="#EDF0F2", linewidth=0.25, zorder=0)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=2.5, pad=1.5, labelsize=EGO_MOTION_TICK_SIZE)


def world_to_ego(object_world_loc: list[float], ego_world_loc: list[float], ego_yaw_deg: float) -> list[float]:
    dx = object_world_loc[0] - ego_world_loc[0]
    dy = object_world_loc[1] - ego_world_loc[1]
    dz = object_world_loc[2] - ego_world_loc[2]
    ego_yaw_rad = math.radians(ego_yaw_deg)
    cos_yaw = math.cos(-ego_yaw_rad)
    sin_yaw = math.sin(-ego_yaw_rad)
    x_ego = dx * cos_yaw - dy * sin_yaw
    y_ego = dx * sin_yaw + dy * cos_yaw
    z_ego = dz
    return [x_ego, y_ego, z_ego]


def calculate_relative_angle_and_distance(object_ego_loc: list[float]) -> tuple[float, float]:
    x, y, _ = object_ego_loc
    distance = math.hypot(x, y)
    relative_angle_deg = math.degrees(math.atan2(y, x))
    if relative_angle_deg < 0:
        relative_angle_deg += 360.0
    return relative_angle_deg, distance


def plot_ego_motion_statistics(samples: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_ieee_plot_style()
    filtered_samples, removed_counts = filter_motion_samples(samples)
    with plt.rc_context(EGO_MOTION_FONT_RC):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(IEEE_COLUMN_WIDTH_IN, 4.75),
            dpi=PAPER_DPI,
            constrained_layout=True,
        )
    specs = [
        ("speed", "(a) Speed", "Speed (m/s)", PAPER_COLORS["motion_speed"]),
        ("acceleration", "(b) Acceleration", "Acceleration (m/s$^2$)", PAPER_COLORS["motion_accel"]),
        ("yaw_rate", "(c) Yaw Rate", "Yaw Rate (deg/s)", PAPER_COLORS["motion_yaw"]),
    ]

    legend_handles = [
        Line2D([0], [0], color=PAPER_COLORS["dark"], linewidth=0.95, label="Mean"),
        Line2D(
            [0],
            [0],
            color=PAPER_COLORS["dark"],
            linewidth=0.85,
            linestyle=(0, (3.0, 2.0)),
            label="P95",
        ),
    ]

    for ax, (key, title, xlabel, color) in zip(axes, specs):
        values = filtered_samples[key]
        stats = summarize_motion_metric(values)
        if not values:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            add_panel_title(ax, title)
            continue

        arr = np.asarray(clip_for_plot(values), dtype=float)
        if arr.size == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            add_panel_title(ax, title)
            continue

        bins = min(42, max(18, int(round(math.sqrt(len(arr))))))
        q05, q95 = np.percentile(arr, [5, 95])
        counts, edges, _ = ax.hist(
            arr,
            bins=bins,
            density=True,
            color=color,
            alpha=0.76,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        smooth = _gaussian_smooth_histogram(counts, sigma_bins=1.15)
        ax.axvspan(q05, q95, color=color, alpha=0.08, linewidth=0, zorder=1)
        ax.plot(centers, smooth, color=PAPER_COLORS["dark"], linewidth=1.18, zorder=3)
        ax.axvline(stats["mean"], color=PAPER_COLORS["dark"], linewidth=0.95, zorder=4)
        ax.axvline(
            stats["p95"],
            color=PAPER_COLORS["dark"],
            linewidth=0.85,
            linestyle=(0, (3.0, 2.0)),
            alpha=0.82,
            zorder=4,
        )
        ax.set_title(title, loc="left", pad=2.0, fontsize=EGO_MOTION_TITLE_SIZE)
        ax.set_xlabel(xlabel, fontsize=EGO_MOTION_LABEL_SIZE)
        ax.set_ylabel("Density", fontsize=EGO_MOTION_LABEL_SIZE)
        style_compact_panel_axis(ax)

        if key == "speed":
            ax.set_xlim(left=0.0)
        else:
            x_abs = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
            ax.set_xlim(-x_abs * 1.04, x_abs * 1.04)

        stat_text = (
            f"n={int(stats['n'])}\n"
            f"mean={stats['mean']:.2f}\n"
            f"p95={stats['p95']:.2f}"
        )
        ax.text(
            0.97,
            0.95,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=EGO_MOTION_NOTE_SIZE,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                alpha=0.90,
                edgecolor=PAPER_COLORS["note_edge"],
                linewidth=0.55,
            ),
        )
        ax.legend(handles=legend_handles, frameon=False, loc="upper left", 
                  bbox_to_anchor=(0.0, 0.84), handlelength=1.25, 
                  labelspacing=0.2, borderaxespad=0.0, 
                  fontsize=EGO_MOTION_LEGEND_SIZE)

    legend_handles = [
        Line2D([0], [0], color=PAPER_COLORS["dark"], linewidth=0.95, label="Mean"),
        Line2D(
            [0],
            [0],
            color=PAPER_COLORS["dark"],
            linewidth=0.85,
            linestyle=(0, (3.0, 2.0)),
            label="P95",
        ),
    ]

    png_path = output_dir / f"{EGO_MOTION_OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{EGO_MOTION_OUTPUT_PREFIX}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ego motion figure: {png_path}")
    logger.info(f"Saved ego motion figure: {pdf_path}")


def compute_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def load_sensor_jitter_series(record: CaseRecord, sensor_name: str) -> dict[str, list[float] | float]:
    frames: list[float] = []
    z_offsets = []
    pitch_offsets = []
    ego_speeds = []
    valid_frame_ids = record.valid_frame_ids
    for row in iter_csv_rows(record.paths.sensor_pose_csv) or []:
        if row.get("sensor_name") != sensor_name:
            continue
        frame = safe_int(row.get("frame"))
        z_offset = safe_float(row.get("gait_z_offset"))
        pitch_offset = safe_float(row.get("gait_pitch_offset_deg"))
        ego_speed = safe_float(row.get("ego_speed"))
        if None in (frame, z_offset, pitch_offset, ego_speed):
            continue
        if normalize_frame_id(frame) not in valid_frame_ids:
            continue
        frames.append(float(frame))
        z_offsets.append(z_offset)
        pitch_offsets.append(pitch_offset)
        ego_speeds.append(ego_speed)
    sample_count = len(frames)
    sample_interval = 1.0 / JITTER_SAMPLE_RATE_HZ
    time_seconds = [index * sample_interval for index in range(sample_count)]
    return {
        "frames": frames,
        "time_seconds": time_seconds,
        "z_offset": z_offsets,
        "pitch_offset": pitch_offsets,
        "ego_speed": ego_speeds,
        "sample_count": float(sample_count),
        "duration_seconds": max(sample_count - 1, 0) * sample_interval,
        "speed_mean": float(sum(ego_speeds) / sample_count) if sample_count else 0.0,
        "speed_std": compute_std(ego_speeds),
        "z_std": compute_std(z_offsets),
        "pitch_std": compute_std(pitch_offsets),
    }


def score_jitter_series(series: dict[str, list[float] | float]) -> float:
    sample_count = int(series.get("sample_count", 0.0))
    if sample_count < 100:
        return -1.0

    speed_mean = float(series.get("speed_mean", 0.0))
    speed_std = float(series.get("speed_std", 0.0))
    z_std = float(series.get("z_std", 0.0))
    pitch_std = float(series.get("pitch_std", 0.0))

    length_score = min(sample_count, 180) / 180.0
    # Prefer a normal walking sequence, rather than a mostly standing case.
    speed_mean_score = max(0.0, 1.0 - min(abs(speed_mean - 1.1) / 0.5, 1.0))
    # Some variation in ego speed helps the gait signal read naturally.
    speed_std_score = min(speed_std / 0.35, 1.0)
    # Vertical bobbing is the main signal we want to show.
    z_score = min(z_std / 0.02, 1.0)
    # Pitch offset is optional; some cases appear to have it fixed at zero.
    pitch_score = min(pitch_std / 0.5, 1.0)

    return (
        0.30 * length_score
        + 0.25 * speed_mean_score
        + 0.20 * speed_std_score
        + 0.20 * z_score
        + 0.05 * pitch_score
    )


def choose_representative_jitter_case(
    records: list[CaseRecord], sensor_name: str
) -> tuple[CaseRecord | None, dict[str, list[float] | float] | None, float]:
    best_record = None
    best_series = None
    best_score = -1.0
    for record in records:
        if not record.paths.sensor_pose_csv.exists():
            continue
        series = load_sensor_jitter_series(record, sensor_name)
        if not series["frames"]:
            continue
        score = score_jitter_series(series)
        if score > best_score:
            best_record = record
            best_series = series
            best_score = score
    return best_record, best_series, best_score


def plot_sensor_pose_motion_jitter(records: list[CaseRecord], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_ieee_plot_style()
    record, series, score = choose_representative_jitter_case(records, JITTER_SENSOR_NAME)
    if record is None or series is None:
        logger.warning("No usable sensor_pose.csv found for jitter plot")
        return

    logger.info(
        "Selected representative jitter case: {}/{} ({}) with {} samples, duration={:.1f}s, score={:.3f}",
        record.map_name,
        record.case_id,
        JITTER_SENSOR_NAME,
        int(series["sample_count"]),
        float(series["duration_seconds"]),
        score,
    )
    if not series["time_seconds"]:
        logger.warning("Selected jitter case has no usable rows after parsing")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(EGO_MOTION_FONT_RC):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(IEEE_TEXT_WIDTH_IN, 4.05),
            dpi=PAPER_DPI,
            sharex=True,
            constrained_layout=True,
        )
    x = series["time_seconds"]
    jitter_specs = [
        (
            axes[0],
            series["z_offset"],
            "(a) Vertical Offset",
            "Offset (m)",
            PAPER_COLORS["motion_speed"],
        ),
        (
            axes[1],
            series["pitch_offset"],
            "(b) Pitch Offset",
            "Pitch (deg)",
            PAPER_COLORS["motion_accel"],
        ),
        (
            axes[2],
            series["ego_speed"],
            "(c) Ego Speed",
            "Speed (m/s)",
            PAPER_COLORS["motion_yaw"],
        ),
    ]

    for ax, values, title, ylabel, color in jitter_specs:
        ax.plot(x, values, color=color, linewidth=1.18, zorder=3)
        ax.fill_between(x, values, 0, color=color, alpha=0.08, linewidth=0, zorder=1)
        ax.set_title(title, loc="left", pad=2.0, fontsize=EGO_MOTION_TITLE_SIZE)
        ax.set_ylabel(ylabel, fontsize=EGO_MOTION_LABEL_SIZE)
        style_compact_panel_axis(ax)

    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=PAPER_COLORS["motion_speed"],
                linewidth=1.2,
                label=f"{JITTER_SENSOR_NAME}, {float(series['duration_seconds']):.1f}s",
            )
        ],
        frameon=False,
        loc="upper right",
        borderaxespad=0.1,
        handlelength=1.35,
        fontsize=EGO_MOTION_LEGEND_SIZE,
    )
    axes[2].set_xlabel("Time Since Start (s)", fontsize=EGO_MOTION_LABEL_SIZE)

    png_path = output_dir / f"{SENSOR_JITTER_OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{SENSOR_JITTER_OUTPUT_PREFIX}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sensor jitter figure: {png_path}")
    logger.info(f"Saved sensor jitter figure: {pdf_path}")


def collect_ego_centric_spatial_objects(
    records: list[CaseRecord],
) -> tuple[list[dict[str, Any]], Counter[str], list[str], list[str]]:
    logger.info("Building Ego-Centric Spatial Distribution")
    rows: list[dict[str, Any]] = []
    grouped_counts: Counter[str] = Counter()
    missing_labels = []
    missing_ego = []

    for record in records:
        if not record.paths.labels.exists():
            missing_labels.append(f"{record.map_name}/{record.case_id}")
            continue
        if not record.paths.ego_csv.exists():
            missing_ego.append(f"{record.map_name}/{record.case_id}")
            continue

        ego_frame_map: dict[str, dict[str, Any]] = {}
        for ego_row in iter_csv_rows(record.paths.ego_csv) or []:
            frame = normalize_frame_id(ego_row.get("frame"))
            x = safe_float(ego_row.get("location_x"))
            y = safe_float(ego_row.get("location_y"))
            z = safe_float(ego_row.get("location_z"))
            yaw = safe_float(ego_row.get("rotation_yaw"))
            if frame is None or None in (x, y, z, yaw):
                continue
            ego_frame_map[frame] = {"location": [x, y, z], "yaw": yaw}

        for frame in record.valid_label_info.frames:
            if not isinstance(frame, dict):
                continue
            frame_id = normalize_frame_id(frame.get("frame_id"))
            ego_state = ego_frame_map.get(frame_id)
            if ego_state is None:
                continue
            ego_world_loc = ego_state["location"]
            ego_yaw = ego_state["yaw"]
            objects = frame.get("objects", [])
            if not isinstance(objects, list):
                continue

            for group_name, class_names in SPATIAL_GROUP_CLASS_MAP.items():
                grouped = [
                    obj
                    for obj in objects
                    if isinstance(obj, dict)
                    and obj.get("class") in class_names
                    and safe_int(obj.get("occlusion")) is not None
                    and safe_int(obj.get("occlusion")) <= SPATIAL_OCCLUSION_MAX
                ]
                grouped_counts[group_name] += len(grouped)
                for obj in grouped:
                    obj_loc = obj.get("location")
                    if not (
                        isinstance(obj_loc, list)
                        and len(obj_loc) >= 3
                        and all(isinstance(v, (int, float)) for v in obj_loc[:3])
                    ):
                        continue
                    object_ego_loc = world_to_ego(obj_loc[:3], ego_world_loc, ego_yaw)
                    angle, distance = calculate_relative_angle_and_distance(object_ego_loc)
                    rows.append(
                        {
                            "map_name": record.map_name,
                            "case_id": record.case_id,
                            "frame_id": frame_id,
                            "angle": angle % 360.0,
                            "distance": distance,
                            "class": group_name,
                        }
                    )

    if missing_labels:
        logger.warning(
            "Missing labels.json for {} cases: {}",
            len(missing_labels),
            ", ".join(missing_labels[:10]) + ("..." if len(missing_labels) > 10 else ""),
        )
    if missing_ego:
        logger.warning(
            "Missing ego.csv for {} cases in spatial statistic: {}",
            len(missing_ego),
            ", ".join(missing_ego[:10]) + ("..." if len(missing_ego) > 10 else ""),
        )

    return rows, grouped_counts, missing_labels, missing_ego


def print_spatial_summary(rows: list[dict[str, Any]], grouped_counts: Counter[str]) -> None:
    near_field_count = sum(1 for row in rows if row["distance"] <= 10.0)
    ratio = (near_field_count / len(rows) * 100.0) if rows else 0.0
    summary_rows = [
        ("Frame-object samples", format_int(len(rows))),
        ("Near-field (0-10m)", f"{near_field_count:,} ({ratio:.1f}%)"),
        ("Occlusion levels", f"0-{SPATIAL_OCCLUSION_MAX}"),
        ("Grouped counts", summarize_counter(grouped_counts)),
    ]
    print_table("Ego-Centric Spatial Summary", summary_rows)


def make_polar_colormap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.96, 0.98, 0.98, 0.0),
        (0.80, 0.91, 0.90, 0.85),
        (0.36, 0.68, 0.66, 0.95),
        (0.16, 0.45, 0.52, 1.0),
    ]
    return LinearSegmentedColormap.from_list("ieee_teal", colors, N=256)


def make_polar_norm(frequency: Any) -> Any:
    import numpy as np
    from matplotlib.colors import PowerNorm

    positive = frequency[frequency > 0]
    if positive.size == 0:
        return PowerNorm(gamma=1.35, vmin=0.0, vmax=1.0)
    return PowerNorm(
        gamma=1.35,
        vmin=0.0,
        vmax=float(np.quantile(positive, 0.98)),
    )


def draw_polar_heatmap(
    ax: Any,
    frequency: Any,
    angle_edges: Any,
    distance_edges: Any,
    title: str,
    norm: Any,
    cmap: Any,
) -> Any:
    import numpy as np

    theta = np.radians(angle_edges)
    max_distance = float(distance_edges[-1])
    mesh = ax.pcolormesh(
        theta,
        distance_edges,
        frequency.T,
        norm=norm,
        cmap=cmap,
        shading="auto",
        edgecolors="face",
        linewidth=0,
        rasterized=True,
    )
    ax.set_ylim(0.0, max_distance)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    radial_ticks = [tick for tick in (5, 10, 20, 30, 40) if tick <= max_distance]
    ax.set_rticks(radial_ticks)
    ax.set_yticklabels([f"{int(tick)}m" for tick in radial_ticks], fontsize=5.6)
    ax.set_rlabel_position(22)
    angles = [0, 90, 180, 270]
    labels = ["Front", "Right", "Rear", "Left"]
    ax.set_xticks(np.radians(angles))
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.tick_params(axis="x", pad=-1)
    ax.tick_params(axis="y", pad=0)
    ax.grid(True, color=PAPER_COLORS["grid"], linestyle="--", linewidth=0.4, alpha=0.55, zorder=10)
    for gridline in ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines():
        gridline.set_clip_path(ax.patch)
        gridline.set_clip_on(True)

    add_panel_title(ax, title, pad=6)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color(PAPER_COLORS["grid"])
    ax.spines["polar"].set_linewidth(0.5)
    return mesh


def plot_polar_heatmap_panel(
    group_frequencies: list[tuple[str, Any, Any]],
    angle_edges: Any,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    apply_ieee_plot_style()
    plt.rcParams["axes.linewidth"] = 0.7

    cmap = make_polar_colormap()

    fig = plt.figure(figsize=(IEEE_TEXT_WIDTH_IN, 2.35), dpi=PAPER_DPI)
    width_ratios = []
    for _ in group_frequencies:
        width_ratios.extend([1.0, 0.045])
    grid = GridSpec(
        1,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.30,
    )

    for index, (title, frequency, distance_edges) in enumerate(group_frequencies):
        ax = fig.add_subplot(grid[0, index * 2], projection="polar")
        cax = fig.add_subplot(grid[0, index * 2 + 1])
        norm = make_polar_norm(frequency)
        mesh = draw_polar_heatmap(ax, frequency, angle_edges, distance_edges, title, norm, cmap)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.solids.set_edgecolor("face")
        cbar.set_label("Frame-Object Count", labelpad=6, fontsize=6.3)
        cbar.ax.tick_params(labelsize=5.8, length=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)


def plot_spatial_distribution_panel(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import numpy as np

    if not rows:
        logger.warning("No spatial object rows found for spatial polar plot")
        return

    angle_bins = np.linspace(0.0, 360.0, 19)
    group_frequencies = []
    angle_edges = angle_bins
    missing_groups = []

    for group_key, group_label, max_distance in SPATIAL_PANEL_GROUPS:
        distance_bins = np.linspace(0.0, max_distance, 9)
        group_rows = [
            row
            for row in rows
            if row["class"] == group_key
            and 0.0 <= float(row["distance"]) <= max_distance
        ]
        if not group_rows:
            missing_groups.append(group_label)
        frequency, angle_edges, distance_edges = np.histogram2d(
            [float(row["angle"]) for row in group_rows],
            [float(row["distance"]) for row in group_rows],
            bins=[angle_bins, distance_bins],
        )
        group_frequencies.append((group_label, frequency, distance_edges))

    if missing_groups:
        logger.warning(
            "No spatial rows remained after distance filtering for groups: {}",
            ", ".join(missing_groups),
        )
    if not any(frequency.sum() > 0 for _, frequency, _ in group_frequencies):
        logger.warning("No spatial rows remain after distance filtering")
        return

    png_path = output_dir / f"{OBSTACLE_POLAR_OUTPUT_PREFIX}.png"
    plot_polar_heatmap_panel(
        group_frequencies,
        angle_edges,
        output_path=png_path,
    )
    logger.info(f"Saved spatial polar panel: {png_path}")
    logger.info(f"Saved spatial polar panel: {png_path.with_suffix('.pdf')}")


def run_ego_centric_spatial_distribution(records: list[CaseRecord]) -> None:
    rows, grouped_counts, _, _ = collect_ego_centric_spatial_objects(records)
    print_spatial_summary(rows, grouped_counts)
    plot_spatial_distribution_panel(rows, OUTPUT_DIR)


def collect_object_group_counts(records: list[CaseRecord]) -> tuple[dict[str, Counter[str]], list[str]]:
    logger.info("Building Object Group Counts")
    class_to_group = {
        cls: group for group, classes in OBJECT_GROUP_CLASS_GROUP.items() for cls in classes
    }
    all_classes = set(class_to_group.keys())
    grouped_counts: dict[str, Counter[str]] = {
        group: Counter() for group in OBJECT_GROUP_CLASS_GROUP.keys()
    }
    missing_labels = []

    for record in records:
        if not record.paths.labels.exists():
            missing_labels.append(f"{record.map_name}/{record.case_id}")
            continue
        for frame in record.valid_label_info.frames:
            if not isinstance(frame, dict):
                continue
            objects = frame.get("objects", [])
            if not isinstance(objects, list):
                continue
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                cls = obj.get("class")
                occlusion = safe_int(obj.get("occlusion"))
                if cls not in all_classes or occlusion is None or occlusion >= 3:
                    continue
                grouped_counts[class_to_group[str(cls)]][str(cls)] += 1

    if missing_labels:
        logger.warning(
            "Missing labels.json for {} cases in object-group statistic: {}",
            len(missing_labels),
            ", ".join(missing_labels[:10]) + ("..." if len(missing_labels) > 10 else ""),
        )

    return grouped_counts, missing_labels


def print_object_group_summary(grouped_counts: dict[str, Counter[str]]) -> None:
    rows = []
    for group, classes in OBJECT_GROUP_CLASS_GROUP.items():
        total = sum(grouped_counts[group].values())
        parts = [f"{cls}={grouped_counts[group].get(cls, 0):,}" for cls in classes]
        rows.append((group, f"total={total:,}; " + ", ".join(parts)))
    print_table("Object Group Summary", rows)


def plot_object_group_counts(grouped_counts: dict[str, Counter[str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker

    groups = list(OBJECT_GROUP_CLASS_GROUP.keys())
    totals = [sum(grouped_counts[group].values()) for group in groups]
    if max(totals, default=0) <= 0:
        logger.warning("No grouped object counts found for object-group plot")
        return

    apply_ieee_plot_style()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, 3.95), dpi=PAPER_DPI)
    ax.set_yscale("function", functions=(lambda x: x**0.5, lambda x: x**2))
    ax.yaxis.set_major_locator(
        ticker.FixedLocator([0, 1000, 4000, 8000, 20000, 40000, 80000, 120000])
    )

    def y_fmt(x: float, _: Any) -> str:
        if x == 0:
            return "0"
        if x >= 1000:
            return f"{int(x / 1000)}k"
        return f"{int(x)}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    x = np.arange(len(groups))
    bar_width = 0.65
    max_visual_height = math.sqrt(max(totals))

    for i, group in enumerate(groups):
        total_group = sum(grouped_counts[group].values())
        bottom = 0
        sub_classes = [
            cls
            for cls in OBJECT_GROUP_CLASS_GROUP[group]
            if grouped_counts[group].get(cls, 0) > 0
        ]
        sub_classes.sort(key=lambda cls: grouped_counts[group].get(cls, 0), reverse=True)

        small_items = []
        for cls in sub_classes:
            count = grouped_counts[group][cls]
            vis_h = (math.sqrt(bottom + count) - math.sqrt(bottom)) / max_visual_height
            if vis_h <= 0.04:
                small_items.append(cls)
            bottom += count

        bottom = 0
        small_item_idx = 0
        for cls in sub_classes:
            count = grouped_counts[group][cls]
            color = OBJECT_GROUP_SUB_COLORS[group][cls]
            ratio_in_group = count / total_group if total_group > 0 else 0.0
            ax.bar(
                x[i],
                count,
                bottom=bottom,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=cls,
            )

            label_text = f"{ratio_in_group:.1%}"
            visual_height_ratio = (math.sqrt(bottom + count) - math.sqrt(bottom)) / max_visual_height
            center_y = bottom + count / 2.0
            if visual_height_ratio > 0.04:
                ax.text(
                    x[i],
                    center_y,
                    label_text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=5.8,
                    fontweight="bold",
                )
            elif ratio_in_group > 0.001:
                is_last_cols = i >= len(groups) - 3
                direction = -1 if is_last_cols else 1
                anchor_x = x[i] + (bar_width / 2 * direction)
                anchor_y = center_y
                offset_x_points = 12 * direction
                if len(small_items) == 2:
                    offset_y_points = 12
                else:
                    offset_y_points = 12
                small_item_idx += 1
                ax.annotate(
                    label_text,
                    xy=(anchor_x, anchor_y),
                    xytext=(offset_x_points, offset_y_points),
                    textcoords="offset points",
                    ha="right" if is_last_cols else "left",
                    va="center",
                    fontsize=5.8,
                    color=PAPER_COLORS["text"],
                    annotation_clip=False,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=PAPER_COLORS["text"],
                        lw=0.5,
                        shrinkA=0,
                        shrinkB=2,
                    ),
                )
            bottom += count

        if total_group > 0:
            ax.text(
                x[i],
                bottom * 1.05,
                f"{int(total_group):,}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=PAPER_COLORS["text"],
                fontweight="bold",
            )

    ax.set_ylabel("Object Count (sqrt scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color=PAPER_COLORS["grid"], alpha=0.45)

    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for group in groups:
        for cls in OBJECT_GROUP_CLASS_GROUP[group]:
            if cls in labels:
                idx = labels.index(cls)
                by_label[cls] = handles[idx]

    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()
    png_path = output_dir / f"{OBJECT_GROUP_OUTPUT_PREFIX}.png"
    pdf_path = output_dir / f"{OBJECT_GROUP_OUTPUT_PREFIX}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved object-group figure: {png_path}")
    logger.info(f"Saved object-group figure: {pdf_path}")


def run_object_group_counts(records: list[CaseRecord]) -> None:
    grouped_counts, _ = collect_object_group_counts(records)
    print_object_group_summary(grouped_counts)
    plot_object_group_counts(grouped_counts, OUTPUT_DIR)


def run_ego_motion_statistics(records: list[CaseRecord]) -> None:
    samples = collect_ego_motion_samples(records)
    print_ego_motion_summary(samples)
    plot_ego_motion_statistics(samples, OUTPUT_DIR)
    plot_sensor_pose_motion_jitter(records, OUTPUT_DIR)


def audit_dataset(records: list[CaseRecord], include_storage: bool) -> dict[str, Any]:
    file_presence = Counter()
    maps = Counter()
    scene_kinds = Counter()
    route_types = Counter()
    ego_behaviors = Counter()
    weather_regimes = Counter()
    sun_buckets = Counter()
    precipitation_regimes = Counter()

    total_frames = 0
    total_duration = 0.0
    sensor_counts: list[int] = []
    total_annotations = 0
    object_counts: Counter[str] = Counter()

    for record in records:
        maps[record.map_name] += 1
        paths = record.paths
        for name, path in (
            ("manifest", paths.manifest),
            ("case_json", paths.case_json),
            ("sampling_log", paths.sampling_log),
            ("calibration", paths.calibration),
            ("labels", paths.labels),
            ("ego_csv", paths.ego_csv),
            ("sensor_pose_csv", paths.sensor_pose_csv),
            ("pedestrian_csv", paths.pedestrian_csv),
            ("vehicle_csv", paths.vehicle_csv),
            ("obstacles_csv", paths.obstacles_csv),
            ("run_summary", paths.run_summary),
            ("run_result", paths.run_result),
        ):
            if path.exists():
                file_presence[name] += 1

        skeleton = record.case_json.get("skeleton", {})
        scene_kinds[str(skeleton.get("scene_kind", "Unknown"))] += 1
        main_route = skeleton.get("main_route") or {}
        route_types[str(main_route.get("route_type", "Unknown"))] += 1

        plan = record.case_json.get("plan", {})
        ego_behaviors[str(plan.get("ego_behavior", "Unknown"))] += 1

        weather = record.sampling_log.get("weather", {})
        if isinstance(weather, dict):
            weather_regimes[str(weather.get("regime") or weather.get("preset") or "Unknown")] += 1
            sun_buckets[str(weather.get("sun_bucket", "Unknown"))] += 1
            precipitation_regimes[
                str(
                    weather.get("precipitation_regime")
                    or weather.get("precipitation_bucket")
                    or "Unknown"
                )
            ] += 1

        if record.frame_num is not None:
            total_frames += record.frame_num
        if record.duration_sec is not None and math.isfinite(record.duration_sec):
            total_duration += record.duration_sec
        if record.sensor_count is not None:
            sensor_counts.append(record.sensor_count)

        annotation_count, class_counts = count_label_annotations_in_frames(
            record.valid_label_info.frames
        )
        total_annotations += annotation_count
        object_counts.update(class_counts)

    storage_bytes = (
        sum(directory_size_bytes(r.paths.case_dir) for r in records)
        if include_storage
        else None
    )

    return {
        "cases": len(records),
        "valid_sampled_cases": sum(r.is_valid_sampled for r in records),
        "maps": maps,
        "file_presence": file_presence,
        "scene_kinds": scene_kinds,
        "route_types": route_types,
        "ego_behaviors": ego_behaviors,
        "weather_regimes": weather_regimes,
        "sun_buckets": sun_buckets,
        "precipitation_regimes": precipitation_regimes,
        "total_frames": total_frames,
        "total_duration_sec": total_duration,
        "sensor_counts": sensor_counts,
        "total_annotations": total_annotations,
        "object_counts": object_counts,
        "storage_bytes": storage_bytes,
    }


def print_audit(summary: dict[str, Any]) -> None:
    sensor_counts = summary["sensor_counts"]
    unique_sensor_counts = sorted(set(sensor_counts))
    print("Dataset audit")
    print(f"  cases: {summary['cases']}")
    print(f"  valid_sampled_cases: {summary['valid_sampled_cases']}")
    print(f"  maps: {summarize_counter(summary['maps'])}")
    print(f"  total_frames: {summary['total_frames']}")
    print(f"  total_duration_sec: {summary['total_duration_sec']:.3f}")
    if sensor_counts:
        print(
            "  sensor_count_per_case: "
            f"min={min(sensor_counts)}, max={max(sensor_counts)}, "
            f"unique={unique_sensor_counts}"
        )
    else:
        print("  sensor_count_per_case: -")
    print(f"  total_annotations: {summary['total_annotations']}")
    storage_bytes = summary["storage_bytes"]
    if storage_bytes is not None:
        print(f"  storage_size: {human_bytes(storage_bytes)}")
    print()
    print("File availability")
    print(f"  {summarize_counter(summary['file_presence'])}")
    print()
    print("Semantic preview")
    print(f"  scene_kinds: {summarize_counter(summary['scene_kinds'])}")
    print(f"  route_types: {summarize_counter(summary['route_types'])}")
    print(f"  ego_behaviors: {summarize_counter(summary['ego_behaviors'])}")
    print(f"  weather_regimes: {summarize_counter(summary['weather_regimes'], top_n=12)}")
    print(f"  sun_buckets: {summarize_counter(summary['sun_buckets'])}")
    print(
        "  precipitation_regimes: "
        f"{summarize_counter(summary['precipitation_regimes'])}"
    )
    print()
    print("Object preview")
    print(f"  {summarize_counter(summary['object_counts'], top_n=20)}")


PENDING_STATISTICS = [
    "dataset_scale_overview",
    "route_type_coverage",
    "weather_continuous_diversity",
    "background_flow_analysis",
    "ego_motion_statistics_panel",
    "ego_centric_obstacle_spatial_distribution",
    "object_group_counts",
]


def print_pending() -> None:
    print("Pending statistic nodes:")
    for name in PENDING_STATISTICS:
        print(f"  - {name}")
    print()
    print("Each node should be implemented after confirming:")
    print("  - case filtering rules")
    print("  - exact counting fields")
    print("  - aggregation level")
    print("  - visual encoding and output formats")


def main() -> None:
    records = run_dataset_scale_overview()
    # run_route_type_coverage(records)
    # run_weather_diversity_statistics(records)
    run_ego_motion_statistics(records)
    # run_ego_centric_spatial_distribution(records)
    # run_object_group_counts(records)


if __name__ == "__main__":
    main()
