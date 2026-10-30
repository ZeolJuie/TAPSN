"""Command-line entry point for prompt-driven scenario generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scenariogen.catalog import load_catalogs, summarize_catalogs
from scenariogen.compiler import compile_case
from scenariogen.executor import execute_case
from scenariogen.models import UserRequest
from scenariogen.planner import DeepSeekPlanner, OpenAIPlanner
from scenariogen.resolver import resolve_cases


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scenariogen",
        description="Generate executable CARLA cases from one natural-language request.",
    )
    parser.add_argument("prompt", nargs="?", help="Natural-language scenario request.")
    parser.add_argument("--prompt-file", type=Path, help="Read the request from a text file.")
    parser.add_argument(
        "--catalog",
        action="append",
        required=True,
        help="Path to walkable_cells.json. Repeat for multiple maps.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=None, help="Override the requested case count.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--provider",
        choices=("deepseek", "openai"),
        default="deepseek",
        help="LLM provider (default: deepseek).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to the provider default or its model environment variable.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint. Defaults to the provider endpoint or its base URL environment variable.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--dry-run", action="store_true", help="Stop after the scene plan.")
    parser.add_argument("--execute", action="store_true", help="Run each compiled case in CARLA.")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--wall-time-limit", type=float, default=None)
    parser.add_argument("--conda-env", default="carla")
    parser.add_argument("--no-conda", action="store_true", help="Use the current Python interpreter.")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--no-sampling", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    prompt = _resolve_prompt(args)
    if prompt is None:
        return 2

    try:
        return _run(args, prompt)
    except Exception as exc:
        print(f"[scenariogen] error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, prompt: str) -> int:
    catalogs = load_catalogs([Path(path) for path in args.catalog])
    resource_summary = summarize_catalogs(catalogs)
    _log(
        "catalogs "
        + ", ".join(
            f"{item.map_name}(scenes={item.scene_count}, routes={item.route_count})"
            for item in resource_summary.maps
        )
    )

    planner_class = OpenAIPlanner if args.provider == "openai" else DeepSeekPlanner
    planner = planner_class(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.timeout,
        max_attempts=args.max_attempts,
    )
    plan = planner.plan(
        UserRequest(prompt=prompt, count=args.count, seed=args.seed),
        resource_summary,
    )
    _log(
        f"plan accepted source={plan.planner_source} count={plan.count} "
        f"topology={plan.topology} ego={plan.ego.get('behavior')}"
    )

    if args.dry_run:
        _print_payload({"status": "dry_run", "prompt": prompt, "plan": plan.to_dict(), "cases": []})
        return 0

    output_root = Path(args.output_root).resolve()
    existing_indices = scan_existing_indices(output_root)
    cases = resolve_cases(
        plan,
        catalogs,
        seed=args.seed,
        sampling_enabled=not args.no_sampling,
        existing_indices=existing_indices,
    )
    _log(f"resolved {len(cases)} cases")

    conda_env = None if args.no_conda else args.conda_env
    case_summaries: List[Dict[str, Any]] = []
    all_ok = True

    for case in cases:
        case_dir = output_root / case.map_name / case.case_id
        built = compile_case(case, case_dir=case_dir, user_prompt=prompt)
        _log(f"compiled case={case.case_id} map={case.map_name} scene={case.scene_id} dir={case_dir}")

        result = None
        if args.execute:
            result = execute_case(
                built,
                conda_env=conda_env,
                carla_host=args.carla_host,
                carla_port=args.carla_port,
                duration=args.duration,
                wall_time_limit_s=args.wall_time_limit,
            )
            ok = result.returncode == 0
            all_ok = all_ok and ok
            _log(
                f"executed case={case.case_id} returncode={result.returncode} "
                f"elapsed={result.elapsed_s:.1f}s log={result.log_path}"
            )

        case_summaries.append(
            {
                "case_id": case.case_id,
                "map_name": case.map_name,
                "scene_id": case.scene_id,
                "case_dir": str(case_dir),
                "scenario_path": str(built.scenario_path),
                "returncode": result.returncode if result is not None else None,
                "elapsed_s": result.elapsed_s if result is not None else None,
            }
        )

    if args.execute:
        status = "executed" if all_ok else "run_failed"
    else:
        status = "compiled"

    _print_payload(
        {
            "status": status,
            "prompt": prompt,
            "plan": plan.to_dict(),
            "cases": case_summaries,
        }
    )
    return 0 if all_ok else 1


def scan_existing_indices(output_root: Path) -> Dict[str, int]:
    indices: Dict[str, int] = {}
    if not output_root.exists():
        return indices

    for map_dir in output_root.iterdir():
        if not map_dir.is_dir():
            continue
        numbers = [
            int(child.name)
            for child in map_dir.iterdir()
            if child.is_dir() and re.fullmatch(r"\d{3}", child.name)
        ]
        if numbers:
            indices[map_dir.name] = max(numbers)
    return indices


def _resolve_prompt(args: argparse.Namespace) -> Optional[str]:
    if args.prompt and args.prompt_file:
        print("[scenariogen] use either a positional prompt or --prompt-file, not both", file=sys.stderr)
        return None
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    print("[scenariogen] a prompt or --prompt-file is required", file=sys.stderr)
    return None


def _log(message: str) -> None:
    print(f"[scenariogen] {message}", flush=True)


def _print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
