#!/usr/bin/env python3
"""Queue the Opt4 KF2-KF8 ablation across idle GPUs.

The existing ``run_opt4_model_fusion_ablation.sh`` runs one scope on one GPU
and intentionally keeps every round serial.  This scheduler keeps the same
round/selection semantics, but dispatches individual benchmark processes to a
pool of GPUs.  A GPU must report no compute process, low utilization, and low
memory use continuously for ``GPU_IDLE_SECONDS`` before a task is assigned.

Configuration is supplied through the same environment variables as the
existing runner.  By default both scopes are run, so the full matrix contains
7 stages x 2 variants x 4 systems x 5 repeats x 2 scopes = 560 processes.
The scheduler does not start, stop, or configure MPS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional


STAGES = (
    ("KF2", "gather-wigner"),
    ("KF3", "reverse-scatter"),
    ("KF4", "rmsnorm"),
    ("KF5", "gate"),
    ("KF6", "radial-mlp"),
    ("KF7", "so3-mlp"),
    ("KF8", "energy-head"),
)
STATUS_HEADER = (
    "scope",
    "variant",
    "fusion_stage",
    "model_fusions",
    "system",
    "temperature_K",
    "repeat",
    "run_name",
    "status",
    "exit_code",
    "process_wall_time_s",
    "physical_gpu",
)
IDLE_PID_RE = re.compile(r"^\s*(\d+)\s*$")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _split(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item) for item in _split(value)]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU_LIST must contain at least one unique GPU index")
    if any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU_LIST cannot contain negative GPU indices")
    return gpus


def _temperature_label(value: str) -> str:
    return format(float(value), "g")


def _scope_dir_name(scope: str) -> str:
    return scope.replace("-", "_")


def _scope_choices(value: str) -> tuple[str, ...]:
    requested = tuple(_split(value))
    if not requested:
        raise ValueError("SCOPES must contain model-only, whole-step, or both")
    if requested == ("both",):
        return ("model-only", "whole-step")
    if any(item not in {"model-only", "whole-step"} for item in requested):
        raise ValueError("SCOPES must contain model-only, whole-step, or both")
    return tuple(dict.fromkeys(requested))


def _hash_shuffle(scope: str, stage: str, system: str, temperature: str, repeat: int) -> list[str]:
    salt = f"42|{stage}|{scope}|{system}|{temperature}|{repeat}"
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "big")
    variants = ["base", "candidate"]
    random.Random(seed).shuffle(variants)
    return variants


def _write_metadata(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_status_file(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(STATUS_HEADER)


def _append_status(path: Path, values: Iterable[Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(values)


def _classify(exit_code: int, log_path: Path, result_path: Path) -> str:
    log = ""
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    if exit_code == 0 and result_path.is_file() and result_path.stat().st_size:
        return "success"
    if exit_code == 42 or re.search(r"BENCHMARK_STATUS=oom|out of memory", log, re.I):
        return "oom"
    if exit_code == 43 or re.search(r"BENCHMARK_STATUS=graph_validation_failed", log, re.I):
        return "graph_validation_failed"
    if re.search(r"BENCHMARK_STATUS=validation_failed", log, re.I):
        return "validation_failed"
    if exit_code == 45 or re.search(r"BENCHMARK_STATUS=capacity_overflow", log, re.I):
        return "capacity_overflow"
    if exit_code == 46 or re.search(r"BENCHMARK_STATUS=unsupported_fusion_config", log, re.I):
        return "unsupported_fusion_config"
    return "error"


def _query_gpu(gpu: int, memory_limit_mib: int, utilization_limit: int) -> bool:
    """Return true only when nvidia-smi proves the GPU is idle."""

    try:
        info = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = [item.strip() for item in info.stdout.strip().split(",")]
        if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
            return False
        utilization = int(fields[0])
        memory_used = int(fields[1])
        if utilization > utilization_limit or memory_used > memory_limit_mib:
            return False

        processes = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pids = [line for line in processes.stdout.splitlines() if IDLE_PID_RE.match(line)]
        return not pids
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


@dataclass
class Config:
    repo_root: Path
    root_output: Path
    checkpoint: Path
    structure_dir: Path
    baseline_dir: Optional[Path]
    baseline_steps: int
    systems: list[str]
    temperatures: list[str]
    steps: int
    warmup_steps: int
    repeats: int
    min_paired: int
    min_faster: int
    gpus: list[int]
    idle_seconds: float
    poll_seconds: float
    memory_limit_mib: int
    utilization_limit: int
    scopes: tuple[str, ...]
    python: str

    @classmethod
    def from_environment(cls) -> "Config":
        repo = Path(_env("REPO_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
        baseline_value = os.environ.get("BASELINE_DIR", "").strip()
        return cls(
            repo_root=repo,
            root_output=Path(
                _env(
                    "ROOT_OUTPUT_DIR",
                    str(repo / "example" / "md_out" / f"esen_opt4_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"),
                )
            ).resolve(),
            checkpoint=Path(_env("CHECKPOINT", str(repo / "esen_30m_oam.pt"))).resolve(),
            structure_dir=Path(
                _env("STRUCTURE_DIR", str(repo.parent / "MatRIS-09bk" / "example" / "cif_file"))
            ).resolve(),
            baseline_dir=Path(baseline_value).resolve() if baseline_value else None,
            baseline_steps=_env_int("BASELINE_STEPS", 1000),
            systems=_split(_env("SYSTEMS", "Cu32 Cu192 H2O32 H2O60")),
            temperatures=_split(_env("TEMPERATURES", "300")),
            steps=_env_int("STEPS", 1000),
            warmup_steps=_env_int("WARMUP_STEPS", 3),
            repeats=_env_int("REPEATS", 5),
            min_paired=_env_int("SELECT_MIN_PAIRED", 5),
            min_faster=_env_int("SELECT_MIN_FASTER", 4),
            gpus=_parse_gpus(_env("GPU_LIST", "0 1 2 3 4 5 6 7")),
            idle_seconds=_env_float("GPU_IDLE_SECONDS", 120.0),
            poll_seconds=_env_float("GPU_POLL_SECONDS", 10.0),
            memory_limit_mib=_env_int("GPU_IDLE_MEMORY_MIB", 1024),
            utilization_limit=_env_int("GPU_IDLE_UTILIZATION_PERCENT", 5),
            scopes=_scope_choices(_env("SCOPES", "both")),
            python=sys.executable,
        )


@dataclass
class Task:
    scope: str
    variant: str
    stage: str
    fusion_stage: str
    model_fusions: str
    system: str
    temperature: str
    repeat: int
    run_name: str
    round_dir: Path
    result_path: Path
    log_path: Path
    command: list[str]
    process: Optional[subprocess.Popen[Any]] = None
    log_handle: Any = None
    started_at: float = 0.0
    gpu: Optional[int] = None


class QueueScheduler:
    def __init__(self, config: Config, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self.active: dict[int, Task] = {}
        self.idle_since: dict[int, Optional[float]] = {
            gpu: None for gpu in config.gpus
        }
        self.stop_requested = False

    def _round_dir(self, scope: str, stage: str) -> Path:
        return self.config.root_output / _scope_dir_name(scope) / f"round_{stage}"

    def _baseline_reference(self, system: str, temperature: str, repeat: int) -> list[str]:
        if self.config.baseline_dir is None:
            return []
        name = (
            f"{system}_{_temperature_label(temperature)}K_"
            f"{self.config.baseline_steps}step_esen_baseline_r{repeat}.json"
        )
        path = self.config.baseline_dir / name
        if path.is_file() and path.stat().st_size:
            return ["--baseline-result", str(path)]
        return ["--missing-baseline-reference"]

    def _common_environment(self, gpu: int) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONHASHSEED"] = "42"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        source = str(self.config.repo_root / "src")
        old_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = source + (os.pathsep + old_pythonpath if old_pythonpath else "")
        return environment

    def _make_task(
        self,
        scope: str,
        stage: str,
        candidate_fusion: str,
        accepted: tuple[str, ...],
        system: str,
        temperature: str,
        repeat: int,
        variant: str,
        round_dir: Path,
    ) -> Task:
        if variant == "candidate":
            fusion_stage = stage
            model_fusions = ",".join((*accepted, candidate_fusion))
        else:
            fusion_stage = f"{stage}_base" if accepted else "KF0"
            model_fusions = ",".join(accepted)

        scope_suffix = _scope_dir_name(scope)
        temp_label = _temperature_label(temperature)
        run_name = (
            f"{system}_{temp_label}K_{self.config.steps}step_esen_"
            f"{scope_suffix}_{fusion_stage}_r{repeat}"
        )
        result_dir = round_dir / "results"
        log_dir = round_dir / "logs"
        result_path = result_dir / f"{run_name}.json"
        log_path = log_dir / f"{run_name}.log"
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        if scope == "model-only":
            opt4 = bool(model_fusions)
            backend = "model-cg-opt4" if opt4 else "model-cg"
            script = self.config.repo_root / "example" / "benchmark_md_gpu.py"
            args = [
                "--backend", backend,
                "--structure", str(self.config.structure_dir / f"{system}.cif"),
                "--checkpoint", str(self.config.checkpoint),
                "--system", system,
                "--output-dir", str(result_dir),
                "--run-name", run_name,
                "--steps", str(self.config.steps),
                "--warmup-steps", str(self.config.warmup_steps),
                "--temperature", temp_label,
                "--timestep", "1.0",
                "--taut", "100.0",
                "--seed", "42",
                "--repeat", str(repeat),
                "--md-dtype", "float64",
                "--cg-probe-steps", "50",
                "--cg-capacity-margin", "0.10",
                "--cg-edge-step", "256",
                "--cg-dummy-atoms", "32",
                "--cg-capture-warmup", "3",
                "--cg-replay-energy-atol", "0.0",
                "--cg-replay-force-atol", "2e-4",
                "--energy-per-atom-atol", "1e-5",
                "--force-max-atol", "2e-4",
            ]
            if opt4:
                args.extend(["--model-fusions", model_fusions, "--fusion-stage", fusion_stage])
            else:
                args.extend(["--fusion-stage", fusion_stage])
            args.extend(self._baseline_reference(system, temperature, repeat))
        else:
            opt4 = bool(model_fusions)
            backend = "whole-step-cg-opt4" if opt4 else "whole-step-cg"
            script = self.config.repo_root / "example" / (
                "benchmark_md_opt4.py" if opt4 else "benchmark_md_opt3.py"
            )
            args = [
                "--backend", backend,
                "--structure", str(self.config.structure_dir / f"{system}.cif"),
                "--checkpoint", str(self.config.checkpoint),
                "--system", system,
                "--output-dir", str(result_dir),
                "--run-name", run_name,
                "--steps", str(self.config.steps),
                "--warmup-steps", str(self.config.warmup_steps),
                "--temperature", temp_label,
                "--timestep", "1.0",
                "--taut", "100.0",
                "--seed", "42",
                "--repeat", str(repeat),
                "--probe-steps", "50",
                "--neighbor-margin", "0.10",
                "--neighbor-slot-step", "8",
                "--dummy-atoms", "32",
                "--capture-warmup", "3",
                "--max-neighbors", "300",
                "--degeneracy-tolerance", "0.01",
                "--energy-per-atom-atol", "1e-5",
                "--force-max-atol", "2e-4",
            ]
            if opt4:
                args.extend(["--model-fusions", model_fusions, "--fusion-stage", fusion_stage])
            args.extend(self._baseline_reference(system, temperature, repeat))

        return Task(
            scope=scope,
            variant=variant,
            stage=stage,
            fusion_stage=fusion_stage,
            model_fusions=model_fusions,
            system=system,
            temperature=temp_label,
            repeat=repeat,
            run_name=run_name,
            round_dir=round_dir,
            result_path=result_path,
            log_path=log_path,
            command=[self.config.python, "-u", str(script), *args],
        )

    def _make_round_tasks(
        self,
        stage: str,
        candidate_fusion: str,
        accepted: dict[str, tuple[str, ...]],
    ) -> list[Task]:
        tasks: list[Task] = []
        for scope in self.config.scopes:
            round_dir = self._round_dir(scope, stage)
            _ensure_status_file(round_dir / "run_status.tsv")
            for repeat in range(1, self.config.repeats + 1):
                for system in self.config.systems:
                    for temperature in self.config.temperatures:
                        variants = _hash_shuffle(scope, stage, system, temperature, repeat)
                        for variant in variants:
                            tasks.append(
                                self._make_task(
                                    scope,
                                    stage,
                                    candidate_fusion,
                                    accepted[scope],
                                    system,
                                    temperature,
                                    repeat,
                                    variant,
                                    round_dir,
                                )
                            )
        return tasks

    def _write_round_metadata(
        self,
        stage: str,
        candidate_fusion: str,
        accepted: dict[str, tuple[str, ...]],
        task_count: int,
    ) -> None:
        for scope in self.config.scopes:
            round_dir = self._round_dir(scope, stage)
            _write_metadata(
                round_dir / "queue_metadata.json",
                {
                    "scope": scope,
                    "stage": stage,
                    "candidate_fusion": candidate_fusion,
                    "accepted_before": list(accepted[scope]),
                    "systems": self.config.systems,
                    "temperatures": self.config.temperatures,
                    "steps": self.config.steps,
                    "repeats": self.config.repeats,
                    "gpu_list": self.config.gpus,
                    "gpu_idle_seconds": self.config.idle_seconds,
                    "task_count_all_scopes": task_count,
                    "repo_root": str(self.config.repo_root),
                    "checkpoint": str(self.config.checkpoint),
                    "baseline_dir": str(self.config.baseline_dir or ""),
                },
            )

    def _start(self, task: Task, gpu: int) -> None:
        task.log_handle = task.log_path.open("w", encoding="utf-8")
        task.started_at = time.monotonic()
        task.gpu = gpu
        task.process = subprocess.Popen(
            task.command,
            cwd=str(self.config.repo_root),
            env=self._common_environment(gpu),
            stdout=task.log_handle,
            stderr=subprocess.STDOUT,
        )
        self.active[gpu] = task
        self.idle_since[gpu] = None
        print(
            f"started gpu={gpu} scope={task.scope} variant={task.variant} "
            f"system={task.system} stage={task.fusion_stage} repeat={task.repeat} "
            f"pid={task.process.pid}",
            flush=True,
        )

    def _finish(self, gpu: int, task: Task) -> None:
        assert task.process is not None
        exit_code = task.process.returncode
        if task.log_handle is not None:
            task.log_handle.close()
        status = _classify(exit_code, task.log_path, task.result_path)
        elapsed = time.monotonic() - task.started_at
        _append_status(
            task.round_dir / "run_status.tsv",
            (
                task.scope,
                task.variant,
                task.fusion_stage,
                task.model_fusions,
                task.system,
                task.temperature,
                task.repeat,
                task.run_name,
                status,
                exit_code,
                f"{elapsed:.6f}",
                gpu,
            ),
        )
        print(
            f"finished gpu={gpu} status={status} exit={exit_code} "
            f"scope={task.scope} variant={task.variant} "
            f"system={task.system} stage={task.fusion_stage} repeat={task.repeat} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        self.active.pop(gpu, None)
        self.idle_since[gpu] = None

    def _poll(self, pending: list[Task]) -> None:
        now = time.monotonic()
        for gpu, task in list(self.active.items()):
            assert task.process is not None
            if task.process.poll() is not None:
                self._finish(gpu, task)

        for gpu in self.config.gpus:
            if gpu in self.active:
                continue
            if _query_gpu(
                gpu,
                memory_limit_mib=self.config.memory_limit_mib,
                utilization_limit=self.config.utilization_limit,
            ):
                if self.idle_since[gpu] is None:
                    self.idle_since[gpu] = now
            else:
                self.idle_since[gpu] = None

        for gpu in self.config.gpus:
            if not pending or gpu in self.active:
                continue
            since = self.idle_since[gpu]
            if since is not None and now - since >= self.config.idle_seconds:
                self._start(pending.pop(0), gpu)

    def run_tasks(self, tasks: list[Task]) -> None:
        pending = list(tasks)
        print(
            f"queue: {len(tasks)} tasks, GPUs={self.config.gpus}, "
            f"idle_window={self.config.idle_seconds:.0f}s, "
            f"poll={self.config.poll_seconds:.0f}s",
            flush=True,
        )
        while pending or self.active:
            if self.stop_requested:
                raise KeyboardInterrupt
            self._poll(pending)
            if pending or self.active:
                time.sleep(self.config.poll_seconds)

    def _select_scope(
        self,
        scope: str,
        stage: str,
        candidate_fusion: str,
        accepted_before: tuple[str, ...],
    ) -> tuple[str, ...]:
        round_dir = self._round_dir(scope, stage)
        base_stage = f"{stage}_base" if accepted_before else "KF0"
        selection_path = self.config.root_output / _scope_dir_name(scope) / f"{stage}_selection.json"
        command = [
            self.config.python,
            str(self.config.repo_root / "example" / "select_opt4_model_fusions.py"),
            "--input-dir", str(round_dir),
            "--scope", scope,
            "--base-stage", base_stage,
            "--candidate-stage", stage,
            "--candidate-fusion", candidate_fusion,
            "--accepted-before", ",".join(accepted_before),
            "--min-paired-repeats", str(self.config.min_paired),
            "--min-faster-directions", str(self.config.min_faster),
            "--output", str(selection_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.config.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout.rstrip(), flush=True)
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Selector did not write {selection_path}: {exc}") from exc
        accepted = tuple(str(item) for item in selection.get("accepted_after", []))
        print(
            f"selection scope={scope} stage={stage} accepted={list(accepted)} "
            f"selector_exit={completed.returncode}",
            flush=True,
        )
        return accepted

    def run(self) -> None:
        self.config.root_output.mkdir(parents=True, exist_ok=True)
        _write_metadata(
            self.config.root_output / "queue_metadata.json",
            {
                "scopes": self.config.scopes,
                "stages": [stage for stage, _ in STAGES],
                "systems": self.config.systems,
                "temperatures": self.config.temperatures,
                "steps": self.config.steps,
                "repeats": self.config.repeats,
                "expected_task_count": len(self.config.scopes)
                * len(STAGES)
                * 2
                * len(self.config.systems)
                * len(self.config.temperatures)
                * self.config.repeats,
                "gpus": self.config.gpus,
                "gpu_idle_seconds": self.config.idle_seconds,
                "gpu_poll_seconds": self.config.poll_seconds,
                "gpu_idle_memory_mib": self.config.memory_limit_mib,
                "gpu_idle_utilization_percent": self.config.utilization_limit,
                "repo_root": str(self.config.repo_root),
            },
        )
        accepted = {scope: tuple() for scope in self.config.scopes}
        final_stage = {scope: "KF0" for scope in self.config.scopes}
        for stage, candidate_fusion in STAGES:
            tasks = self._make_round_tasks(stage, candidate_fusion, accepted)
            self._write_round_metadata(stage, candidate_fusion, accepted, len(tasks))
            self.run_tasks(tasks)
            for scope in self.config.scopes:
                before = accepted[scope]
                accepted[scope] = self._select_scope(
                    scope, stage, candidate_fusion, accepted[scope]
                )
                if accepted[scope] != before:
                    final_stage[scope] = stage

        for scope, fusions in accepted.items():
            scope_root = self.config.root_output / _scope_dir_name(scope)
            _write_metadata(
                scope_root / "accepted_fusions.json",
                {
                    "scope": scope,
                    "accepted_fusions": list(fusions),
                    "final_stage": final_stage[scope],
                    "repeats": self.config.repeats,
                    "policy": "energy/force-vs-baseline errors are telemetry only",
                },
            )
        print(f"completed queue: {self.config.root_output}", flush=True)

    def stop(self) -> None:
        self.stop_requested = True
        for task in self.active.values():
            if task.process is not None and task.process.poll() is None:
                task.process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the expected queue size and configuration without launching tasks",
    )
    args = parser.parse_args()
    try:
        config = Config.from_environment()
    except (ValueError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    expected = (
        len(config.scopes)
        * len(STAGES)
        * 2
        * len(config.systems)
        * len(config.temperatures)
        * config.repeats
    )
    print(
        f"expected benchmark processes={expected}; scopes={config.scopes}; "
        f"GPUs={config.gpus}; idle_window={config.idle_seconds:.0f}s",
        flush=True,
    )
    if args.dry_run:
        return 0

    scheduler = QueueScheduler(config)

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; terminating active tasks", file=sys.stderr, flush=True)
        scheduler.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        scheduler.run()
    except KeyboardInterrupt:
        scheduler.stop()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
