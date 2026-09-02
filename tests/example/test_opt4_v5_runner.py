from __future__ import annotations

import importlib
from pathlib import Path


EXAMPLE_DIR = Path(__file__).parents[2] / "example"


def test_opt4_v5_defaults_keep_v4_mask_and_enable_rob1_only_for_whole_candidate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in (
        "ROOT_OUTPUT_DIR",
        "SYSTEMS",
        "TEMPERATURES",
        "STEPS",
        "REPEATS",
        "WARMUP_STEPS",
        "SCOPES",
        "GPU_LIST",
        "GPU_IDLE_SECONDS",
        "WHOLE_BASE_ROB1",
        "WHOLE_CANDIDATE_ROB1",
        "MODEL_BASE_STAGE",
        "MODEL_CANDIDATE_STAGE",
        "MODEL_BASE_FUSIONS",
        "MODEL_CANDIDATE_FUSIONS",
        "WHOLE_BASE_STAGE",
        "WHOLE_CANDIDATE_STAGE",
        "WHOLE_BASE_FUSIONS",
        "WHOLE_CANDIDATE_FUSIONS",
        "MODEL_BASE_TF32_MODE",
        "MODEL_CANDIDATE_TF32_MODE",
        "WHOLE_BASE_TF32_MODE",
        "WHOLE_CANDIDATE_TF32_MODE",
        "MODEL_BASE_NEIGHBOR_CAPACITY_POLICY",
        "MODEL_CANDIDATE_NEIGHBOR_CAPACITY_POLICY",
        "WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY",
        "WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPT4_V5_PHASE", "smoke")
    monkeypatch.setenv("ROOT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("GPU_LIST", "0")
    monkeypatch.setenv("CHECKPOINT", str(tmp_path / "checkpoint.pt"))
    monkeypatch.setenv("STRUCTURE_DIR", str(tmp_path / "structures"))

    module = importlib.import_module("run_opt4_v5_8gpu")
    module._set_v5_defaults()
    runner = importlib.import_module("run_opt4_v1_8gpu")
    config = runner.Config.from_environment()
    tasks = runner.Scheduler(config)._make_tasks()

    assert len(tasks) == 16
    whole_candidates = [
        task for task in tasks
        if task.scope == "whole-step" and task.variant == "candidate"
    ]
    whole_bases = [
        task for task in tasks
        if task.scope == "whole-step" and task.variant == "base"
    ]
    model_tasks = [task for task in tasks if task.scope == "model-only"]
    assert whole_candidates and whole_bases and model_tasks
    assert all(task.rob1 for task in whole_candidates)
    assert all(not task.rob1 for task in whole_bases + model_tasks)
    assert all(
        task.neighbor_capacity_policy == "auto-safe"
        for task in whole_candidates + whole_bases
    )
    assert all("--rob1" in task.command for task in whole_candidates)
    assert all("--rob1" not in task.command for task in whole_bases + model_tasks)
    model_fusions = (
        "so2-epilogue,so2-gate-bridge,so2-block-gemm,"
        "so2-prepare-backward-reduce"
    )
    whole_fusions = f"rmsnorm,{model_fusions}"
    assert all(task.model_fusions == model_fusions for task in model_tasks)
    assert all(
        task.model_fusions == whole_fusions
        for task in whole_candidates + whole_bases
    )
