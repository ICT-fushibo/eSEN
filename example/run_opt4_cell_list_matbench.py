#!/usr/bin/env python3
"""Poll GPUs for paired Opt4-v5 dense versus CELL1 Matbench runs.

Each task keeps one base/candidate pair on the same physical GPU.  The order is
deterministically shuffled per system, scope, and repeat.  Opt4 v5 remains the
base: model-only runs without ROB1, while whole-step enables ROB1 for both
variants.  CELL1 changes only the fixed-shape candidate generator.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
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
from typing import Any


FOCUS_SYSTEMS = (
    "anthracene_293K_Sharma_S",
    "bulkLiMgAlZnSn_900K_J_Schmidt_VASP",
)
FORMAL_SYSTEMS = (
    "anthracene_293K_Sharma_S",
    "tetracene_295K_Sharma_S",
    "MAPbBr3_300K_Ivor_VASP",
    "bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP",
    "TiSe2_400K_Ivor_VASP",
    "CsSnI3_500K_Ivor_VASP",
    "bulkAu_1500K_Kapil",
    "bulkLiMgAlZnSn_900K_J_Schmidt_VASP",
)
SCOPES = ("model-only", "whole-step")
V5_MODEL_FUSIONS = (
    "so2-epilogue,so2-gate-bridge,so2-block-gemm,"
    "so2-prepare-backward-reduce"
)
V5_WHOLE_FUSIONS = f"rmsnorm,{V5_MODEL_FUSIONS}"
PID_RE = re.compile(r"^\s*(\d+)\s*$")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _split(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _parse_gpus(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in _split(value))
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise ValueError("GPU_LIST must contain unique non-negative GPU ids")
    return result


def _variant_order(scope: str, system: str, repeat: int) -> list[str]:
    digest = hashlib.sha256(
        f"CELL1|{scope}|{system}|{repeat}".encode()
    ).digest()
    result = ["base", "candidate"]
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def _gpu_is_idle(gpu: int, memory_limit: int, utilization_limit: int) -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        if len(values) != 2:
            return False
        if int(values[0]) > utilization_limit or int(values[1]) > memory_limit:
            return False
        processes = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits",
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
    repo: Path
    output: Path
    reference_h5: Path
    matbench_repo: Path
    checkpoint: Path
    phase: str
    systems: tuple[str, ...]
    scopes: tuple[str, ...]
    steps: int
    repeats: int
    record_interval: int
    probe_steps: int
    gpus: tuple[int, ...]
    idle_seconds: float
    poll_seconds: float
    idle_memory_mib: int
    idle_utilization: int
    retry_failed: bool
    python: str

    @classmethod
    def from_environment(cls) -> "Config":
        repo = Path(
            _env("REPO_ROOT", str(Path(__file__).resolve().parents[1]))
        ).resolve()
        phase = _env("CELL1_PHASE", "ablation").lower()
        if phase not in {"smoke", "ablation", "formal"}:
            raise ValueError("CELL1_PHASE must be smoke, ablation, or formal")
        defaults = {
            "smoke": (FOCUS_SYSTEMS, 30, 1, 0.0),
            "ablation": (FOCUS_SYSTEMS, 100, 3, 120.0),
            "formal": (FORMAL_SYSTEMS, 1000, 3, 120.0),
        }[phase]
        systems = tuple(_split(_env("SYSTEMS"))) or defaults[0]
        scopes = tuple(_split(_env("SCOPES", "model-only whole-step")))
        if not scopes or any(scope not in SCOPES for scope in scopes):
            raise ValueError("SCOPES must contain model-only and/or whole-step")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = Path(
            _env(
                "ROOT_OUTPUT_DIR",
                str(repo / "example" / "md_out" / f"opt4_cell1_{phase}_{stamp}"),
            )
        ).resolve()
        config = cls(
            repo=repo,
            output=output,
            reference_h5=Path(_env("REFERENCE_H5")).resolve(),
            matbench_repo=Path(_env("MATBENCH_REPO")).resolve(),
            checkpoint=Path(
                _env("CHECKPOINT", str(repo / "esen_30m_oam.pt"))
            ).resolve(),
            phase=phase,
            systems=systems,
            scopes=tuple(dict.fromkeys(scopes)),
            steps=_env_int("STEPS", defaults[1]),
            repeats=_env_int("REPEATS", defaults[2]),
            record_interval=_env_int("RECORD_INTERVAL", 10),
            probe_steps=_env_int("PROBE_STEPS", 50),
            gpus=_parse_gpus(_env("GPU_LIST", "0 1 2 3 4 5 6 7")),
            idle_seconds=_env_float("GPU_IDLE_SECONDS", defaults[3]),
            poll_seconds=_env_float("GPU_POLL_SECONDS", 10.0),
            idle_memory_mib=_env_int("GPU_IDLE_MEMORY_MIB", 1024),
            idle_utilization=_env_int("GPU_IDLE_UTILIZATION_PERCENT", 5),
            retry_failed=_env("RETRY_FAILED", "0").lower()
            in {"1", "true", "yes"},
            python=_env("PYTHON", sys.executable),
        )
        for path in (config.reference_h5, config.checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not config.matbench_repo.is_dir():
            raise FileNotFoundError(config.matbench_repo)
        if config.steps < 1 or config.steps % config.record_interval:
            raise ValueError("STEPS must be positive and divisible by RECORD_INTERVAL")
        if config.repeats < 1 or config.probe_steps < 0:
            raise ValueError("REPEATS must be positive and PROBE_STEPS non-negative")
        return config


@dataclass
class PairTask:
    scope: str
    system: str
    repeat: int
    variants: list[str]
    index: int = 0
    gpu: int | None = None
    process: subprocess.Popen[Any] | None = None
    log_handle: Any = None
    started: float = 0.0
    variant_started: float = 0.0
    failures: int = 0
    statuses: list[dict[str, Any]] = field(default_factory=list)

    @property
    def variant(self) -> str:
        return self.variants[self.index]


class Scheduler:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.active: dict[int, PairTask] = {}
        self.idle_since = {gpu: None for gpu in config.gpus}
        self.stop_requested = False
        self.status_path = config.output / "cell1_status.tsv"

    def result_path(self, task: PairTask, variant: str) -> Path:
        return (
            self.config.output
            / "runs"
            / task.scope
            / task.system
            / f"repeat_{task.repeat}"
            / variant
            / "runs"
            / "opt4"
            / f"{task.system}.json"
        )

    def output_dir(self, task: PairTask, variant: str) -> Path:
        return self.result_path(task, variant).parents[2]

    def log_path(self, task: PairTask, variant: str) -> Path:
        return (
            self.config.output
            / "logs"
            / task.scope
            / f"{task.system}_r{task.repeat}_{variant}.log"
        )

    def _healthy_existing(self, task: PairTask, variant: str) -> bool:
        path = self.result_path(task, variant)
        if not path.is_file():
            return False
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        terminal = row.get("status") == "success"
        return terminal or not self.config.retry_failed

    def tasks(self) -> list[PairTask]:
        result = []
        for scope in self.config.scopes:
            for system in self.config.systems:
                for repeat in range(1, self.config.repeats + 1):
                    variants = _variant_order(scope, system, repeat)
                    if all(self._healthy_existing(
                        PairTask(scope, system, repeat, variants), variant
                    ) for variant in variants):
                        continue
                    result.append(PairTask(scope, system, repeat, variants))
        random.Random(42).shuffle(result)
        return result

    def _command(self, task: PairTask, variant: str, gpu: int) -> list[str]:
        builder = "dense" if variant == "base" else "cell-list"
        command = [
            self.config.python,
            "-u",
            str(self.config.repo / "example" / "run_esen_matbench.py"),
            "--backend", "opt4",
            "--reference-h5", str(self.config.reference_h5),
            "--checkpoint", str(self.config.checkpoint),
            "--matbench-repo", str(self.config.matbench_repo),
            "--systems", task.system,
            "--save-dir", str(self.output_dir(task, variant)),
            "--gpu", str(gpu),
            "--steps", str(self.config.steps),
            "--record-interval", str(self.config.record_interval),
            "--seed", "0",
            "--probe-steps", str(self.config.probe_steps),
            "--neighbor-margin", "0.10",
            "--neighbor-slot-step", "8",
            "--dummy-atoms", "32",
            "--capture-warmup", "3",
            "--max-neighbors", "300",
            "--degeneracy-tolerance", "0.01",
            "--opt4-model-fusions", (
                V5_MODEL_FUSIONS
                if task.scope == "model-only"
                else V5_WHOLE_FUSIONS
            ),
            "--opt4-fusion-stage", (
                "OPT4V5_FP32_ROB1"
                if variant == "base" and task.scope == "whole-step"
                else (
                    "OPT4V5_FP32"
                    if variant == "base"
                    else "CELL1_FP32"
                )
            ),
            "--opt4-execution-scope", task.scope,
            "--opt4-neighbor-builder", builder,
            "--opt4-neighbor-capacity-policy", "auto-safe",
            "--neighbor-auto-min-reduction", "0.05",
            "--neighbor-auto-guard-slots", "1",
            "--cell-list-bin-capacity", "0",
            "--cell-list-bin-margin", "0.25",
            "--cell-list-bin-step", "8",
            "--rob1-window-steps", "0",
            "--rob1-max-retries", "2",
            "--no-statistics",
            "--no-offline-stress",
            "--overwrite",
        ]
        command.append("--rob1" if task.scope == "whole-step" else "--no-rob1")
        return command

    def _environment(self, gpu: int) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONHASHSEED"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        paths = [
            str(self.config.repo / "src"),
            str(self.config.matbench_repo),
            str(self.config.repo.parent),
        ]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    def _start_variant(self, task: PairTask, gpu: int) -> None:
        variant = task.variant
        output = self.output_dir(task, variant)
        output.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path(task, variant)
        log.parent.mkdir(parents=True, exist_ok=True)
        task.log_handle = log.open("w", encoding="utf-8")
        task.variant_started = time.monotonic()
        task.process = subprocess.Popen(
            self._command(task, variant, gpu),
            cwd=str(self.config.repo),
            env=self._environment(gpu),
            stdout=task.log_handle,
            stderr=subprocess.STDOUT,
        )
        print(
            f"started gpu={gpu} scope={task.scope} system={task.system} "
            f"repeat={task.repeat} variant={variant} pid={task.process.pid}",
            flush=True,
        )

    def _start_pair(self, task: PairTask, gpu: int) -> None:
        task.gpu = gpu
        task.started = time.monotonic()
        self.active[gpu] = task
        self.idle_since[gpu] = None
        self._start_variant(task, gpu)

    def _finish_variant(self, task: PairTask) -> None:
        assert task.process is not None and task.gpu is not None
        variant = task.variant
        code = int(task.process.returncode)
        if task.log_handle is not None:
            task.log_handle.close()
        status = "error"
        try:
            row = json.loads(
                self.result_path(task, variant).read_text(encoding="utf-8")
            )
            status = str(row.get("status", "error"))
        except (OSError, json.JSONDecodeError):
            pass
        if code or status != "success":
            task.failures += 1
        elapsed = time.monotonic() - task.variant_started
        record = {
            "scope": task.scope,
            "variant": variant,
            "neighbor_builder": "dense" if variant == "base" else "cell-list",
            "system": task.system,
            "repeat": task.repeat,
            "status": status,
            "exit_code": code,
            "physical_gpu": task.gpu,
            "process_wall_time_s": f"{elapsed:.6f}",
            "result": self.result_path(task, variant),
        }
        task.statuses.append(record)
        with self.status_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(record), delimiter="\t", lineterminator="\n"
            )
            writer.writerow(record)
        print(
            f"finished gpu={task.gpu} scope={task.scope} system={task.system} "
            f"repeat={task.repeat} variant={variant} status={status} exit={code}",
            flush=True,
        )

    def _finish_pair(self, gpu: int, task: PairTask) -> None:
        print(
            f"pair complete gpu={gpu} scope={task.scope} system={task.system} "
            f"repeat={task.repeat} failures={task.failures} "
            f"elapsed={time.monotonic() - task.started:.1f}s",
            flush=True,
        )
        self.active.pop(gpu, None)
        self.idle_since[gpu] = None

    def run(self, pending: list[PairTask]) -> int:
        while pending or self.active:
            if self.stop_requested:
                raise KeyboardInterrupt
            now = time.monotonic()
            for gpu, task in list(self.active.items()):
                assert task.process is not None
                if task.process.poll() is None:
                    continue
                self._finish_variant(task)
                task.index += 1
                if task.index < len(task.variants):
                    self._start_variant(task, gpu)
                else:
                    self._finish_pair(gpu, task)
            for gpu in self.config.gpus:
                if gpu in self.active:
                    continue
                if _gpu_is_idle(
                    gpu,
                    self.config.idle_memory_mib,
                    self.config.idle_utilization,
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
                    self._start_pair(pending.pop(0), gpu)
            if pending or self.active:
                time.sleep(self.config.poll_seconds)
        failures = 0
        for scope in self.config.scopes:
            for system in self.config.systems:
                for repeat in range(1, self.config.repeats + 1):
                    task = PairTask(
                        scope, system, repeat,
                        _variant_order(scope, system, repeat),
                    )
                    for variant in ("base", "candidate"):
                        path = self.result_path(task, variant)
                        try:
                            row = json.loads(path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            failures += 1
                            continue
                        failures += int(row.get("status") != "success")
        return int(failures > 0)

    def stop(self) -> None:
        self.stop_requested = True
        for task in self.active.values():
            if task.process is not None and task.process.poll() is None:
                task.process.terminate()


def _write_metadata(config: Config, pending: int) -> None:
    payload = {
        "experiment": "CELL1_fixed_shape_gpu_cell_list",
        "base": "Opt4 v5 dense fixed builder",
        "candidate": "Opt4 v5 CELL1 fixed cell-list builder",
        "phase": config.phase,
        "systems": list(config.systems),
        "scopes": list(config.scopes),
        "steps": config.steps,
        "repeats": config.repeats,
        "record_interval": config.record_interval,
        "probe_steps": config.probe_steps,
        "gpus": list(config.gpus),
        "pending_pairs": pending,
        "reference_h5": str(config.reference_h5),
        "checkpoint": str(config.checkpoint),
        "output": str(config.output),
    }
    (config.output / "cell1_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    try:
        config = Config.from_environment()
    except (OSError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    config.output.mkdir(parents=True, exist_ok=True)
    scheduler = Scheduler(config)
    if not scheduler.status_path.is_file():
        with scheduler.status_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
                [
                    "scope", "variant", "neighbor_builder", "system", "repeat",
                    "status", "exit_code", "physical_gpu", "process_wall_time_s",
                    "result",
                ]
            )
    tasks = scheduler.tasks()
    _write_metadata(config, len(tasks))
    print(f"CELL1 output: {config.output}")
    print(f"pending paired tasks: {len(tasks)}")

    def stop(_signum, _frame):
        scheduler.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return scheduler.run(tasks)
    except KeyboardInterrupt:
        scheduler.stop()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
