#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nuscenes_converter.common import standard_scene_name_from_ids  # noqa: E402


SAMPLE_RE = re.compile(r"^sample_\d+$")
STANDARD_SCENE_RE = re.compile(r"^scene-(\d{4})$")
LEGACY_SCENE_RE = re.compile(r"^(?:scene[-_])?(\d{2})[-_](\d+)$")
TOWN_RE = re.compile(r"^Town(\d+)")


@dataclass(frozen=True)
class RenameTask:
    src_dir: Path
    dst_dir: Path


def infer_standard_scene_name(sample_dir: Path) -> str:
    ancestors = [sample_dir, *sample_dir.parents]

    for path in ancestors:
        if STANDARD_SCENE_RE.fullmatch(path.name):
            return path.name

    for path in ancestors:
        match = LEGACY_SCENE_RE.fullmatch(path.name)
        if match is not None:
            map_id, seq_id = match.group(1), match.group(2)
            return standard_scene_name_from_ids(map_id, seq_id)

    for idx, path in enumerate(ancestors[:-1]):
        if not path.name.isdigit():
            continue
        town_match = TOWN_RE.match(ancestors[idx + 1].name)
        if town_match is None:
            continue
        map_id = town_match.group(1)
        seq_id = path.name
        return standard_scene_name_from_ids(map_id, seq_id)

    raise RuntimeError(f"Cannot infer scene name from: {sample_dir}")


def collect_tasks(src_root: Path, dst_root: Path) -> list[RenameTask]:
    tasks = []
    seen_dst = {}

    for labels_path in sorted(src_root.rglob("labels.npz")):
        sample_dir = labels_path.parent
        if not SAMPLE_RE.fullmatch(sample_dir.name):
            continue

        scene_name = infer_standard_scene_name(sample_dir)
        dst_dir = dst_root / scene_name / sample_dir.name
        task = RenameTask(src_dir=sample_dir, dst_dir=dst_dir)

        other = seen_dst.get(dst_dir)
        if other is not None and other != sample_dir:
            raise RuntimeError(
                f"Destination collision:\n  {other}\n  {sample_dir}\n-> {dst_dir}"
            )
        seen_dst[dst_dir] = sample_dir
        tasks.append(task)

    return tasks


def prune_empty_parents(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    while current.exists() and current != stop_dir:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def apply_task(task: RenameTask, mode: str, src_root: Path, skip_existing: bool) -> str:
    if task.src_dir.resolve() == task.dst_dir.resolve():
        return "already_standard"

    if task.dst_dir.exists():
        if skip_existing:
            return "skipped_existing"
        raise FileExistsError(f"Destination already exists: {task.dst_dir}")

    task.dst_dir.parent.mkdir(parents=True, exist_ok=True)

    if mode == "move":
        shutil.move(str(task.src_dir), str(task.dst_dir))
        prune_empty_parents(task.src_dir.parent, src_root)
        return "moved"

    if mode == "copy":
        shutil.copytree(task.src_dir, task.dst_dir)
        return "copied"

    raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename OCC GT to scene-0xxx/sample_xxxxxx layout."
    )
    parser.add_argument("--src-root", required=True, help="Dataset root / sequence root / occ root")
    parser.add_argument("--dst-root", required=True, help="Target gts root")
    parser.add_argument("--mode", choices=["move", "copy"], default="move")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    tasks = collect_tasks(src_root, dst_root)
    if not tasks:
        print(f"No OCC sample directories found under {src_root}")
        return 0

    print(f"Found {len(tasks)} OCC sample directories")
    stats = {
        "moved": 0,
        "copied": 0,
        "skipped_existing": 0,
        "already_standard": 0,
    }

    for task in tasks:
        print(f"{task.src_dir} -> {task.dst_dir}")
        if args.dry_run:
            continue
        status = apply_task(task, args.mode, src_root, args.skip_existing)
        stats[status] += 1

    if args.dry_run:
        print("Dry run only. No changes were made.")
        return 0

    print(
        "Done. moved={moved} copied={copied} skipped_existing={skipped_existing} already_standard={already_standard}".format(
            **stats
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
