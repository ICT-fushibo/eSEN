#!/usr/bin/env python3
"""Run one Matbench system per job on GPUs idle for a sustained interval.

Each job runs the requested baseline/Opt1/Opt2/Opt3/Opt4 set on one GPU so
per-system speedups remain same-GPU comparisons.  Different systems run in
parallel.  The scheduler never starts, stops, or changes CUDA MPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from statistics import median
from typing import Any, Optional


BACKEND_CHOICES = ("baseline", "opt1", "opt2", "opt3", "opt4")
IDLE_PID_RE = re.compile(r"^\s*(\d+)\s*$")
METRIC_KEYS = ("rdf_error", "adf_error", "vdos_error")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes"}


def _split(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _parse_gpus(value: str) -> list[int]:
    result = [int(item) for item in _split(value)]
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise ValueError("GPU_LIST must contain unique non-negative GPU indices")
    return result


def _discover_systems(reference_h5: Path) -> list[str]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Matbench queue discovery requires h5py") from exc
    with h5py.File(reference_h5, "r") as handle:
        return sorted(
            name for name, value in handle.items() if isinstance(value, h5py.Group)
        )


def _gpu_is_idle(gpu: int, memory_limit_mib: int, utilization_limit: int) -> bool:
    try:
        info = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        )
        fields = [item.strip() for item in info.stdout.strip().split(",")]
        if len(fields) != 2:
            return False
        if int(fields[0]) > utilization_limit or int(fields[1]) > memory_limit_mib:
            return False
        processes = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        )
        return not any(IDLE_PID_RE.match(line) for line in processes.stdout.splitlines())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


@dataclass
class Config:
    repo_root: Path
    save_dir: Path
    reference_h5: Path
    checkpoint: Path
    matbench_repo: Path
    systems: list[str]
    backends: list[str]
    steps: int
    record_interval: int
    probe_steps: int
    neighbor_margin: float
    neighbor_slot_step: int
    edge_step: int
    dummy_atoms: int
    capture_warmup: int
    max_neighbors: int
    degeneracy_tolerance: float
    statistics: bool
    gpus: list[int]
    idle_seconds: float
    poll_seconds: float
    memory_limit_mib: int
    utilization_limit: int
    retry_failed: bool
    python: str

    @classmethod
    def from_environment(cls) -> "Config":
        repo = Path(_env("REPO_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
        reference = Path(
            _env(
                "REFERENCE_H5",
                str(repo.parent / "matbench-discovery-data" / "2026-06-29-dynamat-v1.0-reference-trajectories.h5"),
            )
        ).resolve()
        requested_systems = _split(_env("SYSTEMS"))
        backends = _split(_env("BACKENDS", "baseline opt1 opt2 opt3"))
        unknown = sorted(set(backends) - set(BACKEND_CHOICES))
        if unknown:
            raise ValueError("unknown Matbench backend(s): " + ", ".join(unknown))
        save_value = _env("MATBENCH_SAVE_DIR", _env("SAVE_DIR", _env("ROOT_OUTPUT_DIR")))
        if not save_value:
            save_value = str(
                repo / "example" / "md_out" / f"esen_matbench_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        config = cls(
            repo_root=repo,
            save_dir=Path(save_value).resolve(),
            reference_h5=reference,
            checkpoint=Path(_env("CHECKPOINT", str(repo / "esen_30m_oam.pt"))).resolve(),
            matbench_repo=Path(
                _env("MATBENCH_REPO", str(repo.parent / "matbench-discovery"))
            ).resolve(),
            systems=requested_systems or _discover_systems(reference),
            backends=backends,
            steps=_env_int("STEPS", 80_000),
            record_interval=_env_int("RECORD_INTERVAL", 10),
            probe_steps=_env_int("PROBE_STEPS", 50),
            neighbor_margin=_env_float("NEIGHBOR_MARGIN", 0.10),
            neighbor_slot_step=_env_int("NEIGHBOR_SLOT_STEP", 8),
            edge_step=_env_int("EDGE_STEP", 256),
            dummy_atoms=_env_int("DUMMY_ATOMS", 32),
            capture_warmup=_env_int("CAPTURE_WARMUP", 3),
            max_neighbors=_env_int("MAX_NEIGHBORS", 300),
            degeneracy_tolerance=_env_float("DEGENERACY_TOLERANCE", 0.01),
            statistics=_env_bool("STATISTICS", True),
            gpus=_parse_gpus(_env("GPU_LIST", "0 1 2 3 4 5 6 7")),
            idle_seconds=_env_float("GPU_IDLE_SECONDS", 120.0),
            poll_seconds=_env_float("GPU_POLL_SECONDS", 10.0),
            memory_limit_mib=_env_int("GPU_IDLE_MEMORY_MIB", 1024),
            utilization_limit=_env_int("GPU_IDLE_UTILIZATION_PERCENT", 5),
            retry_failed=_env_bool("RETRY_FAILED", False),
            python=sys.executable,
        )
        if not config.reference_h5.is_file():
            raise FileNotFoundError(config.reference_h5)
        if not config.checkpoint.is_file():
            raise FileNotFoundError(config.checkpoint)
        if config.steps < 1 or config.steps % config.record_interval:
            raise ValueError("STEPS must be positive and divisible by RECORD_INTERVAL")
        return config


@dataclass
class Task:
    system: str
    output_dir: Path
    log_path: Path
    report_path: Path
    process: Optional[subprocess.Popen[Any]] = None
    log_handle: Any = None
    gpu: Optional[int] = None
    started_at: float = 0.0


class Scheduler:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.active: dict[int, Task] = {}
        self.idle_since: dict[int, Optional[float]] = {
            gpu: None for gpu in config.gpus
        }
        self.stop_requested = False
        self.status_paths = (
            config.save_dir / "run_status.tsv",
            config.save_dir / "queue_status.tsv",
        )

    def _existing_terminal(self, task: Task) -> bool:
        if not task.report_path.is_file():
            return False
        try:
            report = json.loads(task.report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        rows = {row.get("backend"): row for row in report.get("runs", [])}
        if any(backend not in rows for backend in self.config.backends):
            return False
        all_success = all(rows[backend].get("status") == "success" for backend in self.config.backends)
        return all_success or not self.config.retry_failed

    def tasks(self) -> list[Task]:
        tasks = []
        for system in self.config.systems:
            output = self.config.save_dir / "systems" / system
            task = Task(
                system=system,
                output_dir=output,
                log_path=self.config.save_dir / "logs" / f"{system}.log",
                report_path=output / "matbench_esen_report.json",
            )
            if not self._existing_terminal(task):
                tasks.append(task)
        random.Random(0).shuffle(tasks)
        return tasks

    def _environment(self, gpu: int) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONHASHSEED"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        additions = [
            str(self.config.repo_root / "src"),
            str(self.config.matbench_repo),
            str(self.config.repo_root.parent),
        ]
        old = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(additions + ([old] if old else []))
        return env

    def _command(self, task: Task, gpu: int) -> list[str]:
        command = [
            self.config.python,
            "-u",
            str(self.config.repo_root / "example" / "run_esen_matbench.py"),
            "--backend", *self.config.backends,
            "--reference-h5", str(self.config.reference_h5),
            "--checkpoint", str(self.config.checkpoint),
            "--matbench-repo", str(self.config.matbench_repo),
            "--systems", task.system,
            "--save-dir", str(task.output_dir),
            "--gpu", str(gpu),
            "--steps", str(self.config.steps),
            "--record-interval", str(self.config.record_interval),
            "--seed", "0",
            "--probe-steps", str(self.config.probe_steps),
            "--neighbor-margin", str(self.config.neighbor_margin),
            "--neighbor-slot-step", str(self.config.neighbor_slot_step),
            "--edge-step", str(self.config.edge_step),
            "--dummy-atoms", str(self.config.dummy_atoms),
            "--capture-warmup", str(self.config.capture_warmup),
            "--max-neighbors", str(self.config.max_neighbors),
            "--degeneracy-tolerance", str(self.config.degeneracy_tolerance),
            "--overwrite",
        ]
        if not self.config.statistics:
            command.append("--no-statistics")
        return command

    def _start(self, task: Task, gpu: int) -> None:
        task.output_dir.mkdir(parents=True, exist_ok=True)
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_handle = task.log_path.open("w", encoding="utf-8")
        task.gpu = gpu
        task.started_at = time.monotonic()
        task.process = subprocess.Popen(
            self._command(task, gpu),
            cwd=str(self.config.repo_root),
            env=self._environment(gpu),
            stdout=task.log_handle,
            stderr=subprocess.STDOUT,
        )
        self.active[gpu] = task
        self.idle_since[gpu] = None
        print(f"started gpu={gpu} system={task.system} pid={task.process.pid}", flush=True)

    def _finish(self, gpu: int, task: Task) -> None:
        assert task.process is not None
        code = int(task.process.returncode)
        if task.log_handle is not None:
            task.log_handle.close()
        status = "error"
        completed_backends = 0
        try:
            report = json.loads(task.report_path.read_text(encoding="utf-8"))
            rows = [row for row in report.get("runs", []) if row.get("backend") in self.config.backends]
            completed_backends = sum(row.get("status") == "success" for row in rows)
            status = "success" if completed_backends == len(self.config.backends) else "partial"
        except (OSError, json.JSONDecodeError):
            pass
        elapsed = time.monotonic() - task.started_at
        values = [
            task.system,
            status,
            code,
            completed_backends,
            gpu,
            f"{elapsed:.6f}",
            task.output_dir,
        ]
        for status_path in self.status_paths:
            with status_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(values)
        print(
            f"finished gpu={gpu} system={task.system} status={status} "
            f"backends={completed_backends}/{len(self.config.backends)} elapsed={elapsed:.1f}s",
            flush=True,
        )
        self.active.pop(gpu, None)
        self.idle_since[gpu] = None

    def run(self, tasks: list[Task]) -> None:
        pending = list(tasks)
        while pending or self.active:
            if self.stop_requested:
                raise KeyboardInterrupt
            now = time.monotonic()
            for gpu, task in list(self.active.items()):
                assert task.process is not None
                if task.process.poll() is not None:
                    self._finish(gpu, task)
            for gpu in self.config.gpus:
                if gpu in self.active:
                    continue
                if _gpu_is_idle(gpu, self.config.memory_limit_mib, self.config.utilization_limit):
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
            if pending or self.active:
                time.sleep(self.config.poll_seconds)

    def stop(self) -> None:
        self.stop_requested = True
        for task in self.active.values():
            if task.process is not None and task.process.poll() is None:
                task.process.terminate()


def _aggregate(config: Config) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    published: dict[str, Any] = {}
    protocol: dict[str, Any] = {}
    for system in config.systems:
        root = config.save_dir / "systems" / system
        report_path = root / "matbench_esen_report.json"
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        protocol = protocol or report.get("protocol", {})
        published = published or report.get("published_esen_30m_oam", {})
        baseline = next(
            (
                row.get("rollout_wall_time_s")
                for row in report.get("runs", [])
                if row.get("backend") == "baseline" and row.get("status") == "success"
            ),
            None,
        )
        for row in report.get("runs", []):
            if row.get("backend") not in config.backends:
                continue
            current = row.get("rollout_wall_time_s")
            row = dict(row)
            row["speedup_vs_baseline"] = (
                float(baseline) / float(current)
                if baseline is not None and current not in (None, 0)
                else None
            )
            runs.append(row)
        metric_path = root / "matbench_esen_metrics.tsv"
        if metric_path.is_file():
            with metric_path.open(encoding="utf-8", newline="") as handle:
                metrics.extend(csv.DictReader(handle, delimiter="\t"))

    speedups: dict[str, Any] = {}
    for backend in config.backends:
        values = [
            float(row["speedup_vs_baseline"])
            for row in runs
            if row.get("backend") == backend and row.get("speedup_vs_baseline") is not None
        ]
        speedups[backend] = {
            "n_systems": len(values),
            "geomean_speedup_vs_baseline": (
                math.exp(sum(math.log(value) for value in values) / len(values))
                if values else None
            ),
            "median_speedup_vs_baseline": median(values) if values else None,
        }
    metric_aggregate: dict[str, Any] = {}
    for backend in config.backends:
        selected = [row for row in metrics if row.get("backend") == backend]
        summary: dict[str, Any] = {"n_systems": len(selected)}
        for key in METRIC_KEYS:
            values = []
            for row in selected:
                try:
                    value = float(row.get(key, ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            summary[key] = sum(values) / len(values) if values else None
        metric_aggregate[backend] = summary
    expected = len(config.systems) * len(config.backends)
    report = {
        "schema": 1,
        "benchmark": "matbench-dynamat-v1.0-queued",
        "save_dir": str(config.save_dir),
        "protocol": protocol,
        "systems": config.systems,
        "backends": config.backends,
        "runs": runs,
        "speedup_summary": speedups,
        "public_metrics": metric_aggregate,
        "published_esen_30m_oam": published,
        "complete_matrix": len(runs) == expected and all(row.get("status") == "success" for row in runs),
        "same_gpu_comparison": "All backends for each system ran in one queued process on one physical GPU.",
    }
    _write_json(config.save_dir / "matbench_esen_queue_report.json", report)
    with (config.save_dir / "matbench_esen_speedups.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["system", "backend", "seconds_per_step", "steps_per_second", "speedup_vs_baseline", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in runs:
            writer.writerow({key: row.get(key) for key in fields})
    lines = [
        "# eSEN Matbench queued report", "",
        f"Complete matrix: **{report['complete_matrix']}**", "",
        "Each system keeps baseline and Opt1-Opt3 on the same physical GPU.", "",
        "| backend | systems | geomean speedup vs baseline | median speedup | RDF error | ADF error | vDOS error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend in config.backends:
        speed = speedups[backend]
        metric = metric_aggregate[backend]
        lines.append(
            f"| {backend} | {speed['n_systems']} | {speed['geomean_speedup_vs_baseline'] or ''} | "
            f"{speed['median_speedup_vs_baseline'] or ''} | {metric['rdf_error'] or ''} | "
            f"{metric['adf_error'] or ''} | {metric['vdos_error'] or ''} |"
        )
    (config.save_dir / "matbench_esen_queue_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        config = Config.from_environment()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    config.save_dir.mkdir(parents=True, exist_ok=True)
    status_header = [
        "system", "status", "exit_code", "successful_backends",
        "physical_gpu", "process_wall_time_s", "output_dir",
    ]
    for status in (config.save_dir / "run_status.tsv", config.save_dir / "queue_status.tsv"):
        if not status.is_file():
            with status.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(status_header)
    scheduler = Scheduler(config)
    tasks = scheduler.tasks()
    _write_json(
        config.save_dir / "queue_metadata.json",
        {
            "systems": config.systems,
            "backends": config.backends,
            "steps": config.steps,
            "record_interval": config.record_interval,
            "gpu_list": config.gpus,
            "gpu_idle_seconds": config.idle_seconds,
            "gpu_poll_seconds": config.poll_seconds,
            "pending_tasks": len(tasks),
            "save_dir": str(config.save_dir),
            "reference_h5": str(config.reference_h5),
            "checkpoint": str(config.checkpoint),
        },
    )
    print(
        f"Matbench queue: tasks={len(tasks)} systems={len(config.systems)} "
        f"backends={config.backends} GPUs={config.gpus} save={config.save_dir}",
        flush=True,
    )
    if args.dry_run:
        return 0

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; stopping active jobs", file=sys.stderr, flush=True)
        scheduler.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        scheduler.run(tasks)
    except KeyboardInterrupt:
        scheduler.stop()
        return 130
    report = _aggregate(config)
    print(f"Matbench queued results: {config.save_dir}", flush=True)
    return 0 if report["complete_matrix"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
