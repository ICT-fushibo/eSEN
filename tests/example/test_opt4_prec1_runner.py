from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


EXAMPLE_DIR = Path(__file__).parents[2] / "example"
PREC1_SCRIPT = EXAMPLE_DIR / "run_opt4_prec1_8gpu.py"


def _load_prec1():
    spec = importlib.util.spec_from_file_location("run_opt4_prec1_8gpu", PREC1_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prec1_smoke_tasks_only_change_tf32(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in (
        "SYSTEMS",
        "TEMPERATURES",
        "STEPS",
        "REPEATS",
        "WARMUP_STEPS",
        "SCOPES",
        "GPU_IDLE_SECONDS",
        "MODEL_BASE_FUSIONS",
        "MODEL_CANDIDATE_FUSIONS",
        "WHOLE_BASE_FUSIONS",
        "WHOLE_CANDIDATE_FUSIONS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PREC1_PHASE", "smoke")
    monkeypatch.setenv("ROOT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("GPU_LIST", "0 1")
    monkeypatch.setenv("CHECKPOINT", str(tmp_path / "checkpoint.pt"))
    monkeypatch.setenv("STRUCTURE_DIR", str(tmp_path / "structures"))
    module = _load_prec1()
    module._set_prec1_defaults()
    runner = importlib.import_module("run_opt4_v1_8gpu")
    config = runner.Config.from_environment()
    tasks = runner.Scheduler(config)._make_tasks()

    assert len(tasks) == 16
    assert sum(task.variant == "base" for task in tasks) == 8
    assert sum(task.variant == "candidate" for task in tasks) == 8
    for task in tasks:
        command_mode = task.command[task.command.index("--tf32-mode") + 1]
        assert command_mode == task.tf32_mode
        assert (task.variant == "candidate") == (task.tf32_mode == "on")

    by_case = {}
    for task in tasks:
        key = (task.scope, task.system, task.temperature, task.repeat)
        by_case.setdefault(key, []).append(task)
    for pair in by_case.values():
        assert len(pair) == 2
        assert pair[0].model_fusions == pair[1].model_fusions
        assert (
            pair[0].neighbor_capacity_policy
            == pair[1].neighbor_capacity_policy
        )
