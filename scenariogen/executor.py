"""Execute one compiled case with run_simulation.py.

The executor builds the command, prepares the Python environment, enforces an
optional wall-clock limit, and stores the run log and result next to the case.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scenariogen.models import BuiltCase, ExecutionResult


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_RETURNCODE = 124


def execute_case(
    built_case: BuiltCase,
    *,
    conda_env: Optional[str] = "carla",
    carla_host: str = "127.0.0.1",
    carla_port: int = 2000,
    duration: Optional[float] = None,
    wall_time_limit_s: Optional[float] = None,
    extra_args: Sequence[str] = (),
) -> ExecutionResult:
    scenario_path = Path(built_case.scenario_path).resolve()
    if not scenario_path.exists():
        raise FileNotFoundError(f"scenario config not found: {scenario_path}")

    command = build_command(
        built_case,
        scenario_path=scenario_path,
        conda_env=conda_env,
        carla_host=carla_host,
        carla_port=carla_port,
        duration=duration,
        extra_args=extra_args,
    )

    logs_dir = Path(built_case.case_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (Path(built_case.case_dir) / "simulation").mkdir(parents=True, exist_ok=True)

    started = time.time()
    timed_out = False

    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=build_execution_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=wall_time_limit_s,
        )
        output = completed.stdout or ""
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = _timeout_output(exc, wall_time_limit_s)
        returncode = TIMEOUT_RETURNCODE

    elapsed_s = time.time() - started
    log_path = logs_dir / "run.log"
    log_path.write_text(output, encoding="utf-8")

    result = ExecutionResult(
        command=command,
        returncode=returncode,
        elapsed_s=elapsed_s,
        log_path=log_path,
        timed_out=timed_out,
    )
    (logs_dir / "run_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def build_command(
    built_case: BuiltCase,
    *,
    scenario_path: Path,
    conda_env: Optional[str],
    carla_host: str,
    carla_port: int,
    duration: Optional[float],
    extra_args: Sequence[str] = (),
) -> List[str]:
    if conda_env:
        command = [
            "conda",
            "run",
            "-n",
            conda_env,
            "--no-capture-output",
            "python",
            "run_simulation.py",
        ]
    else:
        command = [sys.executable, "run_simulation.py"]

    simulation_dir = Path(built_case.case_dir) / "simulation"
    command.extend(
        [
            "--scenario-config",
            str(scenario_path),
            "--carla-host",
            carla_host,
            "--carla-port",
            str(carla_port),
            "--output",
            str(simulation_dir),
            "--csv",
        ]
    )
    if duration is not None:
        command.extend(["--duration", str(duration)])
    command.extend(extra_args)
    return command


def build_execution_env() -> Dict[str, str]:
    env = os.environ.copy()

    entries = [str(REPO_ROOT)]
    carla_root = env.get("CARLA_ROOT")
    if carla_root:
        python_api = Path(carla_root) / "PythonAPI"
        for candidate in (python_api, python_api / "carla"):
            if candidate.exists():
                entries.append(str(candidate))

    existing = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    merged: List[str] = []
    for entry in entries + existing:
        if entry not in merged:
            merged.append(entry)

    env["PYTHONPATH"] = os.pathsep.join(merged)
    return env


def _timeout_output(exc: subprocess.TimeoutExpired, wall_time_limit_s: Optional[float]) -> str:
    stdout = exc.stdout or ""
    stderr = exc.stderr or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return (
        stdout
        + stderr
        + f"\n[executor] wall-clock timeout after {wall_time_limit_s}s; run_simulation.py terminated\n"
    )
