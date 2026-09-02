from __future__ import annotations

import importlib
from pathlib import Path
import sys


EXAMPLE_DIR = Path(__file__).parents[2] / "example"


def _load(name: str, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_cell1_runner_pairs_dense_and_cell_list_with_scope_specific_rob1(
    tmp_path, monkeypatch
):
    reference = tmp_path / "reference.h5"
    checkpoint = tmp_path / "checkpoint.pt"
    matbench = tmp_path / "matbench-discovery"
    reference.touch()
    checkpoint.touch()
    matbench.mkdir()
    for name, value in {
        "REPO_ROOT": str(EXAMPLE_DIR.parent),
        "ROOT_OUTPUT_DIR": str(tmp_path / "out"),
        "REFERENCE_H5": str(reference),
        "CHECKPOINT": str(checkpoint),
        "MATBENCH_REPO": str(matbench),
        "CELL1_PHASE": "smoke",
        "SYSTEMS": "anthracene_293K_Sharma_S",
        "SCOPES": "model-only whole-step",
        "GPU_LIST": "0 1",
    }.items():
        monkeypatch.setenv(name, value)
    module = _load("run_opt4_cell_list_matbench", monkeypatch)
    config = module.Config.from_environment()
    scheduler = module.Scheduler(config)
    tasks = scheduler.tasks()
    assert len(tasks) == 2

    for task in tasks:
        base = scheduler._command(task, "base", 0)
        candidate = scheduler._command(task, "candidate", 0)
        assert base[base.index("--opt4-neighbor-builder") + 1] == "dense"
        assert candidate[candidate.index("--opt4-neighbor-builder") + 1] == "cell-list"
        assert base[base.index("--opt4-execution-scope") + 1] == task.scope
        assert ("--rob1" in base) == (task.scope == "whole-step")
        assert ("--rob1" in candidate) == (task.scope == "whole-step")
        fusions = base[base.index("--opt4-model-fusions") + 1]
        assert fusions.startswith("rmsnorm,") == (task.scope == "whole-step")


def test_matbench_cell1_cli_defaults_preserve_dense_whole_step(tmp_path, monkeypatch):
    module = _load("run_esen_matbench", monkeypatch)
    args = module.parse_args(["--save-dir", str(tmp_path)])
    assert args.opt4_execution_scope == "whole-step"
    assert args.opt4_neighbor_builder == "dense"
    assert args.cell_list_bin_capacity == 0


def test_matbench_rejects_rob1_for_model_only(tmp_path, monkeypatch):
    module = _load("run_esen_matbench", monkeypatch)
    try:
        module.parse_args(
            [
                "--save-dir", str(tmp_path),
                "--backend", "opt4",
                "--opt4-execution-scope", "model-only",
                "--rob1",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("model-only ROB1 must be rejected")
