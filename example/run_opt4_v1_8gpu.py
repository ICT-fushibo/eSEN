#!/usr/bin/env python3
"""Poll idle GPUs and benchmark the accepted Opt4 KF9 v1 candidate.

Each scope runs an interleaved base/candidate comparison on the same matrix:

* model-only: Opt2 ``model-cg`` vs KF9 ``model-cg-opt4``;
* whole-step: Opt3 ``whole-step-cg`` vs Opt4 v1
  ``whole-step-cg-opt4`` with ``rmsnorm,so2-epilogue``.

The scheduler starts one fresh Python process per task.  A GPU must be free,
below the configured utilization/memory limits, and remain so for the idle
window before a task is assigned.  It does not start, stop, or configure MPS.
Numerical validation is recorded by the benchmark JSON and is not used as a
performance-task gate; Matbench is the later correctness validation path.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
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
from typing import Any, Iterable, Optional


DEFAULT_SYSTEMS = (
    "Cu32",
    "Cu64",
    "Cu192",
    "Cu512",
    "Cu1024",
    "H2O32",
    "H2O60",
    "H2O192",
    "H2O512",
    "H2O1024",
)
SCOPES = ("model-only", "whole-step")
STATUS_HEADER = (
    "scope",
    "variant",
    "base_stage",
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
PID_RE = re.compile(r"^\s*(\d+)\s*$")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _split(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item) for item in _split(value)]
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU_LIST must contain unique non-negative GPU indices")
    return gpus


def _scope_choices(value: str) -> tuple[str, ...]:
    requested = _split(value)
    if requested == ["both"]:
        return SCOPES
    if not requested or any(item not in SCOPES for item in requested):
        raise ValueError("SCOPES must contain model-only, whole-step, or both")
    return tuple(dict.fromkeys(requested))


def _temperature_label(value: str) -> str:
    return format(float(value), "g")


def _capacity_policy(value: str) -> str:
    if value not in {"uniform", "species", "atom"}:
        raise ValueError(
            "neighbor capacity policy must be uniform, species, or atom"
        )
    return value


def _shuffle_variants(scope: str, system: str, temperature: str, repeat: int) -> list[str]:
    salt = f"42|opt4-v1|{scope}|{system}|{temperature}|{repeat}"
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "big")
    variants = ["base", "candidate"]
    random.Random(seed).shuffle(variants)
    return variants


def _write_json(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_status(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(STATUS_HEADER)


def _append_status(path: Path, values: Iterable[Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(values)


def _classify(exit_code: int, log_path: Path, result_path: Path) -> str:
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log = ""
    result: dict[str, Any] = {}
    if result_path.is_file() and result_path.stat().st_size:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
    if result:
        if result.get("capacity_overflow") is True:
            return "capacity_overflow"
        if result.get("graph_invariants_pass") is False:
            return "graph_validation_failed"
        # Energy/force reference differences are telemetry for this short
        # performance matrix.  Preserve the distinction in status while
        # keeping a healthy result eligible for later timing analysis.
        if exit_code == 0:
            return "success"
        if result.get("performance_sample_eligible") is True:
            return "validation_failed"
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
    """Return true only when nvidia-smi proves that a GPU is idle."""

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
        if int(fields[0]) > utilization_limit or int(fields[1]) > memory_limit_mib:
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
        return not any(PID_RE.match(line) for line in processes.stdout.splitlines())
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
    gpus: list[int]
    idle_seconds: float
    poll_seconds: float
    idle_memory_mib: int
    idle_utilization_percent: int
    scopes: tuple[str, ...]
    model_base_fusions: str
    model_candidate_fusions: str
    whole_base_fusions: str
    whole_candidate_fusions: str
    model_base_stage: str
    model_candidate_stage: str
    whole_base_stage: str
    whole_candidate_stage: str
    model_base_capacity_policy: str
    model_candidate_capacity_policy: str
    whole_base_capacity_policy: str
    whole_candidate_capacity_policy: str
    run_kind: str
    status_filename: str
    resume: bool
    python: str

    @classmethod
    def from_environment(cls) -> "Config":
        repo = Path(_env("REPO_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
        baseline = os.environ.get("BASELINE_DIR", "").strip()
        model_base_fusions = _env("MODEL_BASE_FUSIONS", "")
        whole_base_fusions = _env("WHOLE_BASE_FUSIONS", "")
        return cls(
            repo_root=repo,
            root_output=Path(
                _env(
                    "ROOT_OUTPUT_DIR",
                    str(repo / "example" / "md_out" / f"opt4_v1_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"),
                )
            ).resolve(),
            checkpoint=Path(_env("CHECKPOINT", str(repo / "esen_30m_oam.pt"))).resolve(),
            structure_dir=Path(
                _env("STRUCTURE_DIR", str(repo.parent / "MatRIS-09bk" / "example" / "cif_file"))
            ).resolve(),
            baseline_dir=Path(baseline).resolve() if baseline else None,
            baseline_steps=_env_int("BASELINE_STEPS", 100),
            systems=_split(_env("SYSTEMS", " ".join(DEFAULT_SYSTEMS))),
            temperatures=_split(_env("TEMPERATURES", "300 800")),
            steps=_env_int("STEPS", 100),
            warmup_steps=_env_int("WARMUP_STEPS", 3),
            repeats=_env_int("REPEATS", 3),
            gpus=_parse_gpus(_env("GPU_LIST", "0 1 2 3 4 5 6 7")),
            idle_seconds=_env_float("GPU_IDLE_SECONDS", 120.0),
            poll_seconds=_env_float("GPU_POLL_SECONDS", 10.0),
            idle_memory_mib=_env_int("GPU_IDLE_MEMORY_MIB", 1024),
            idle_utilization_percent=_env_int("GPU_IDLE_UTILIZATION_PERCENT", 5),
            scopes=_scope_choices(_env("SCOPES", "both")),
            model_base_fusions=model_base_fusions,
            model_candidate_fusions=_env("MODEL_CANDIDATE_FUSIONS", "so2-epilogue"),
            whole_base_fusions=whole_base_fusions,
            whole_candidate_fusions=_env(
                "WHOLE_CANDIDATE_FUSIONS", "rmsnorm,so2-epilogue"
            ),
            model_base_stage=_env(
                "MODEL_BASE_STAGE", "MODEL_BASE" if model_base_fusions else "OPT2"
            ),
            model_candidate_stage=_env("MODEL_CANDIDATE_STAGE", "OPT4V1"),
            whole_base_stage=_env(
                "WHOLE_BASE_STAGE", "WHOLE_BASE" if whole_base_fusions else "OPT3"
            ),
            whole_candidate_stage=_env("WHOLE_CANDIDATE_STAGE", "OPT4V1"),
            model_base_capacity_policy=_capacity_policy(
                _env("MODEL_BASE_NEIGHBOR_CAPACITY_POLICY", "uniform")
            ),
            model_candidate_capacity_policy=_capacity_policy(
                _env("MODEL_CANDIDATE_NEIGHBOR_CAPACITY_POLICY", "uniform")
            ),
            whole_base_capacity_policy=_capacity_policy(
                _env("WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY", "uniform")
            ),
            whole_candidate_capacity_policy=_capacity_policy(
                _env("WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY", "uniform")
            ),
            run_kind=_env("RUN_KIND", "opt4_v1_kf9_formal_performance"),
            status_filename=_env("STATUS_FILENAME", "v1_status.tsv"),
            resume=_env("RESUME", "1") not in {"0", "false", "False"},
            python=sys.executable,
        )


@dataclass
class Task:
    scope: str
    variant: str
    base_stage: str
    fusion_stage: str
    model_fusions: str
    neighbor_capacity_policy: str
    system: str
    temperature: str
    repeat: int
    run_name: str
    result_path: Path
    log_path: Path
    status_path: Path
    command: list[str]
    process: Optional[subprocess.Popen[Any]] = None
    log_handle: Any = None
    started_at: float = 0.0
    gpu: Optional[int] = None


class Scheduler:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.active: dict[int, Task] = {}
        self.idle_since: dict[int, Optional[float]] = {gpu: None for gpu in config.gpus}
        self.stop_requested = False

    def _scope_root(self, scope: str) -> Path:
        return self.config.root_output / scope.replace("-", "_")

    def _reference_args(self, system: str, temperature: str, repeat: int) -> list[str]:
        if self.config.baseline_dir is None:
            return []
        path = self.config.baseline_dir / (
            f"{system}_{_temperature_label(temperature)}K_"
            f"{self.config.baseline_steps}step_esen_baseline_r{repeat}.json"
        )
        return ["--baseline-result", str(path)] if path.is_file() else ["--missing-baseline-reference"]

    def _task_command(
        self,
        scope: str,
        variant: str,
        system: str,
        temperature: str,
        repeat: int,
        output_dir: Path,
        run_name: str,
        base_stage: str,
        fusion_stage: str,
        model_fusions: str,
        neighbor_capacity_policy: str,
    ) -> list[str]:
        if scope == "model-only":
            script = self.config.repo_root / "example" / "benchmark_md_gpu.py"
            backend = "model-cg-opt4" if model_fusions else "model-cg"
            args = [
                "--backend", backend,
                "--structure", str(self.config.structure_dir / f"{system}.cif"),
                "--checkpoint", str(self.config.checkpoint),
                "--system", system,
                "--output-dir", str(output_dir),
                "--run-name", run_name,
                "--steps", str(self.config.steps),
                "--warmup-steps", str(self.config.warmup_steps),
                "--temperature", _temperature_label(temperature),
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
        else:
            script = self.config.repo_root / "example" / "benchmark_md_opt4.py"
            backend = "whole-step-cg-opt4" if model_fusions else "whole-step-cg"
            args = [
                "--backend", backend,
                "--structure", str(self.config.structure_dir / f"{system}.cif"),
                "--checkpoint", str(self.config.checkpoint),
                "--system", system,
                "--output-dir", str(output_dir),
                "--run-name", run_name,
                "--steps", str(self.config.steps),
                "--warmup-steps", str(self.config.warmup_steps),
                "--temperature", _temperature_label(temperature),
                "--timestep", "1.0",
                "--taut", "100.0",
                "--seed", "42",
                "--repeat", str(repeat),
                "--probe-steps", "50",
                "--neighbor-margin", "0.10",
                "--neighbor-slot-step", "8",
                "--neighbor-capacity-policy", neighbor_capacity_policy,
                "--dummy-atoms", "32",
                "--capture-warmup", "3",
                "--max-neighbors", "300",
                "--degeneracy-tolerance", "0.01",
                "--replay-energy-atol", "0.0",
                "--replay-force-atol", "2e-4",
                "--energy-per-atom-atol", "1e-5",
                "--force-max-atol", "2e-4",
            ]
        if model_fusions:
            args.extend(["--model-fusions", model_fusions, "--fusion-stage", fusion_stage])
        args.extend(self._reference_args(system, temperature, repeat))
        return [self.config.python, "-u", str(script), *args]

    def _make_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        for scope in self.config.scopes:
            scope_root = self._scope_root(scope)
            result_dir = scope_root / "results"
            log_dir = scope_root / "logs"
            status_path = scope_root / "run_status.tsv"
            result_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            _ensure_status(status_path)
            if scope == "model-only":
                base_stage = self.config.model_base_stage
                candidate_stage = self.config.model_candidate_stage
                base_fusions = self.config.model_base_fusions
                candidate_fusions = self.config.model_candidate_fusions
                base_capacity_policy = self.config.model_base_capacity_policy
                candidate_capacity_policy = (
                    self.config.model_candidate_capacity_policy
                )
            else:
                base_stage = self.config.whole_base_stage
                candidate_stage = self.config.whole_candidate_stage
                base_fusions = self.config.whole_base_fusions
                candidate_fusions = self.config.whole_candidate_fusions
                base_capacity_policy = self.config.whole_base_capacity_policy
                candidate_capacity_policy = (
                    self.config.whole_candidate_capacity_policy
                )

            for repeat in range(1, self.config.repeats + 1):
                for system in self.config.systems:
                    for temperature in self.config.temperatures:
                        for variant in _shuffle_variants(scope, system, temperature, repeat):
                            is_candidate = variant == "candidate"
                            fusions = candidate_fusions if is_candidate else base_fusions
                            capacity_policy = (
                                candidate_capacity_policy
                                if is_candidate
                                else base_capacity_policy
                            )
                            stage = candidate_stage if is_candidate else base_stage
                            label = stage
                            scope_label = scope.replace("-", "_")
                            run_name = (
                                f"{system}_{_temperature_label(temperature)}K_"
                                f"{self.config.steps}step_esen_{scope_label}_{label}_r{repeat}"
                            )
                            result_path = result_dir / f"{run_name}.json"
                            log_path = log_dir / f"{run_name}.log"
                            if self.config.resume and result_path.is_file() and result_path.stat().st_size:
                                continue
                            tasks.append(
                                Task(
                                    scope=scope,
                                    variant=variant,
                                    base_stage=base_stage,
                                    fusion_stage=stage,
                                    model_fusions=fusions,
                                    neighbor_capacity_policy=capacity_policy,
                                    system=system,
                                    temperature=temperature,
                                    repeat=repeat,
                                    run_name=run_name,
                                    result_path=result_path,
                                    log_path=log_path,
                                    status_path=status_path,
                                    command=self._task_command(
                                        scope,
                                        variant,
                                        system,
                                        temperature,
                                        repeat,
                                        result_dir,
                                        run_name,
                                        base_stage,
                                        stage,
                                        fusions,
                                        capacity_policy,
                                    ),
                                )
                            )
        return tasks

    def _environment(self, gpu: int) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONHASHSEED"] = "42"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        source = str(self.config.repo_root / "src")
        old = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = source + (os.pathsep + old if old else "")
        return environment

    def _start(self, task: Task, gpu: int) -> None:
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_handle = task.log_path.open("w", encoding="utf-8")
        task.started_at = time.monotonic()
        task.gpu = gpu
        task.process = subprocess.Popen(
            task.command,
            cwd=str(self.config.repo_root),
            env=self._environment(gpu),
            stdout=task.log_handle,
            stderr=subprocess.STDOUT,
        )
        self.active[gpu] = task
        self.idle_since[gpu] = None
        print(
            f"started gpu={gpu} scope={task.scope} variant={task.variant} "
            f"system={task.system} T={task.temperature} repeat={task.repeat} "
            f"pid={task.process.pid}",
            flush=True,
        )

    def _finish(self, gpu: int, task: Task) -> None:
        assert task.process is not None
        exit_code = int(task.process.returncode)
        if task.log_handle is not None:
            task.log_handle.close()
        status = _classify(exit_code, task.log_path, task.result_path)
        elapsed = time.monotonic() - task.started_at
        _append_status(
            task.status_path,
            (
                task.scope,
                task.variant,
                task.base_stage,
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
            f"system={task.system} T={task.temperature} repeat={task.repeat} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        self.active.pop(gpu, None)
        self.idle_since[gpu] = None

    def _poll(self, pending: deque[Task]) -> None:
        now = time.monotonic()
        for gpu, task in list(self.active.items()):
            assert task.process is not None
            if task.process.poll() is not None:
                self._finish(gpu, task)

        for gpu in self.config.gpus:
            if gpu in self.active:
                continue
            idle = _query_gpu(
                gpu,
                self.config.idle_memory_mib,
                self.config.idle_utilization_percent,
            )
            if idle:
                if self.idle_since[gpu] is None:
                    self.idle_since[gpu] = now
            else:
                self.idle_since[gpu] = None

        for gpu in self.config.gpus:
            if not pending or gpu in self.active:
                continue
            since = self.idle_since[gpu]
            if since is not None and now - since >= self.config.idle_seconds:
                self._start(pending.popleft(), gpu)

    def _write_metadata(self, expected_tasks: int) -> None:
        _write_json(
            self.config.root_output / "run_metadata.json",
            {
                "kind": self.config.run_kind,
                "repo_commit": subprocess.run(
                    ["git", "-C", str(self.config.repo_root), "rev-parse", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "scopes": list(self.config.scopes),
                "systems": self.config.systems,
                "temperatures": self.config.temperatures,
                "steps": self.config.steps,
                "warmup_steps": self.config.warmup_steps,
                "repeats": self.config.repeats,
                "seed": 42,
                "model_base_fusions": self.config.model_base_fusions,
                "model_candidate_fusions": self.config.model_candidate_fusions,
                "whole_base_fusions": self.config.whole_base_fusions,
                "whole_candidate_fusions": self.config.whole_candidate_fusions,
                "model_base_stage": self.config.model_base_stage,
                "model_candidate_stage": self.config.model_candidate_stage,
                "whole_base_stage": self.config.whole_base_stage,
                "whole_candidate_stage": self.config.whole_candidate_stage,
                "model_base_neighbor_capacity_policy": (
                    self.config.model_base_capacity_policy
                ),
                "model_candidate_neighbor_capacity_policy": (
                    self.config.model_candidate_capacity_policy
                ),
                "whole_base_neighbor_capacity_policy": (
                    self.config.whole_base_capacity_policy
                ),
                "whole_candidate_neighbor_capacity_policy": (
                    self.config.whole_candidate_capacity_policy
                ),
                "gpus": self.config.gpus,
                "gpu_idle_seconds": self.config.idle_seconds,
                "gpu_poll_seconds": self.config.poll_seconds,
                "gpu_idle_memory_mib": self.config.idle_memory_mib,
                "gpu_idle_utilization_percent": self.config.idle_utilization_percent,
                "expected_task_count": expected_tasks,
                "baseline_dir": str(self.config.baseline_dir or ""),
                "checkpoint": str(self.config.checkpoint),
                "structure_dir": str(self.config.structure_dir),
                "numerical_validation_policy": "telemetry_only; Matbench is the correctness path",
                "mps_policy": "unchanged; scheduler only sets CUDA_VISIBLE_DEVICES",
            },
        )

    def _write_summary(self) -> None:
        rows: list[tuple[str, str, int]] = []
        for scope in self.config.scopes:
            path = self._scope_root(scope) / "run_status.tsv"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    rows.append((row["scope"], row["status"], 1))
        counts = Counter((scope, status) for scope, status, _ in rows)
        summary_path = self.config.root_output / self.config.status_filename
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("scope", "status", "count"))
            for (scope, status), count in sorted(counts.items()):
                writer.writerow((scope, status, count))

    def run(self, dry_run: bool = False) -> int:
        self.config.root_output.mkdir(parents=True, exist_ok=True)
        tasks = self._make_tasks()
        expected = (
            len(self.config.scopes)
            * 2
            * len(self.config.systems)
            * len(self.config.temperatures)
            * self.config.repeats
        )
        self._write_metadata(expected)
        print(
            f"{self.config.run_kind} queue: pending={len(tasks)} "
            f"expected_total={expected} "
            f"scopes={self.config.scopes} GPUs={self.config.gpus} "
            f"idle_window={self.config.idle_seconds:.0f}s",
            flush=True,
        )
        if dry_run:
            return 0

        pending: deque[Task] = deque(tasks)
        while pending or self.active:
            if self.stop_requested:
                break
            self._poll(pending)
            if pending or self.active:
                time.sleep(self.config.poll_seconds)
        self._write_summary()
        if self.stop_requested:
            print(
                f"{self.config.run_kind} queue interrupted; "
                "active tasks were terminated",
                file=sys.stderr,
            )
            return 130
        print(
            f"{self.config.run_kind} queue completed: {self.config.root_output}",
            flush=True,
        )
        return 0

    def stop(self) -> None:
        self.stop_requested = True
        for task in self.active.values():
            if task.process is not None and task.process.poll() is None:
                task.process.terminate()


def main() -> int:
    dry_run = "--dry-run" in sys.argv[1:]
    try:
        config = Config.from_environment()
    except (ValueError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    scheduler = Scheduler(config)

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; terminating active tasks", file=sys.stderr, flush=True)
        scheduler.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return scheduler.run(dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
