"""CPU-only CLI/selector regressions; no fairchem/CUDA import required."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[3] / "example"
sys.path.insert(0, str(EXAMPLE))
import run_esen_matbench as matbench
import run_opt4_runtime_ablation as runner


def config(phase="ablation", experiment="nhc"):
    return dict(
        phase=phase,
        experiment=experiment,
        steps=1000 if phase == "ablation" else 30,
        repeats=3 if phase == "ablation" else 1,
        systems=runner.SYSTEMS,
        gpu="2",
        reference_h5="reference.h5",
        checkpoint="checkpoint.pt",
        matbench_repo="matbench",
    )


def valid_row(cfg, variant):
    experiment = "none" if variant == "base" else cfg["experiment"]
    return dict(
        status="success",
        exit_code=0,
        opt4_runtime_experiment=experiment,
        nhc_implementation="FusedMatbenchNHCIntegrator"
        if experiment == "nhc"
        else "MatbenchNHCIntegrator",
        graph_invariants_pass=True,
        initial_graph_validation_pass=True,
        seconds_per_step=0.1,
        peak_reserved_gib=1.0,
        trajectory_frames=cfg["steps"] // 10 + 1,
        graph_stats=dict(
            rob1_enabled=True,
            fixed_builder_backend="dense",
            rob1_initial_capacity_policy="auto-safe",
            rob1_committed_replays=cfg["steps"] + 1,
            rob1_rollback_count=0,
            cuda_graph_recovery_capture_count=0,
            rob1_unrecovered_overflows=0,
            rob1_snapshot_addresses_stable=True,
            rob1_window_status_mode="compact"
            if experiment == "rob-compact"
            else "full",
            shared_geometry_blocks=10 if experiment == "shared-geometry" else 0,
        ),
    )


@pytest.mark.parametrize("experiment", runner.EXPERIMENTS)
@pytest.mark.parametrize("variant", ["base", "candidate"])
def test_runner_commands_pin_v5_and_one_experiment(experiment, variant):
    cfg = config(experiment=experiment)
    cmd = runner.command(cfg, variant, runner.SYSTEMS[0], Path("run"))
    args = matbench.parse_args(cmd[3:])
    assert args.backend == ["opt4"] and args.rob1
    assert args.opt4_neighbor_capacity_policy == "auto-safe"
    assert args.opt4_neighbor_builder == "dense"
    assert args.opt4_execution_scope == "whole-step"
    assert args.opt4_runtime_experiment == ("none" if variant == "base" else experiment)
    assert args.steps == 1000 and args.record_interval == 10
    assert not args.statistics and not args.offline_stress
    assert args.timestep_fs == 0.25 and args.thermostat_time_fs == 25
    assert not runner.run_errors(valid_row(cfg, variant), cfg, variant)


def test_legacy_default_unchanged():
    args = matbench.parse_args(["--save-dir", "run"])
    assert args.opt4_runtime_experiment == "none"
    assert args.rob1 is False


@pytest.mark.parametrize(
    "extra",
    [
        ["--no-rob1"],
        ["--opt4-neighbor-builder", "cell-list"],
        ["--backend", "baseline", "opt4"],
        ["--opt4-neighbor-capacity-policy", "elastic"],
        ["--opt4-runtime-experiment", "nhc,rob-compact"],
    ],
)
def test_combination_or_wrong_baseline_rejected(extra):
    cmd = runner.command(config(), "candidate", runner.SYSTEMS[0], Path("run"))
    with pytest.raises(SystemExit):
        matbench.parse_args(cmd[3:] + extra)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "capacity_overflow"),
        ("exit_code", 45),
        ("seconds_per_step", float("nan")),
        ("opt4_runtime_experiment", "none"),
        ("graph_invariants_pass", False),
    ],
)
def test_failed_or_mislabeled_results_do_not_pass(field, value):
    cfg = config()
    row = valid_row(cfg, "candidate")
    row[field] = value
    assert runner.run_errors(row, cfg, "candidate")


def pairs(cfg, speedup):
    return [
        dict(system=system, errors=[], speedup=speedup, reserved_increase_gib=0.0)
        for system in cfg["systems"]
        for _ in range(cfg["repeats"])
    ]


def test_smoke_not_a_performance_acceptance():
    cfg = config("smoke")
    result = runner.summarize(pairs(cfg, 0.8), cfg, [])
    assert result["accepted"] and not result["performance_ok"]


def test_ablation_acceptance_and_rejection_are_fail_closed():
    cfg = config()
    good = pairs(cfg, 1.03)
    assert runner.summarize(good, cfg, [])["accepted"]
    for bad in (good[:-1], pairs(cfg, 0.99)):
        assert not runner.summarize(bad, cfg, [])["accepted"]
    bad = copy.deepcopy(good)
    bad[0]["errors"] = ["capacity_overflow"]
    bad[0]["speedup"] = None
    result = runner.summarize(bad, cfg, [])
    assert not result["accepted"] and result["geomean_speedup"] is None
    bad = pairs(cfg, 1.05)
    for row in bad:
        if row["system"] == cfg["systems"][0]:
            row["speedup"] = 0.985
    assert not runner.summarize(bad, cfg, [])["accepted"]
    for row in bad:
        row["speedup"] = 1.05
        row["reserved_increase_gib"] = 1.1
    assert not runner.summarize(bad, cfg, [])["accepted"]


def test_fingerprint_tracks_steps_source_and_independent_experiment():
    cfg = config()
    for key, value in [
        ("steps", 80000),
        ("experiment", "rob-compact"),
        ("gpu", "0"),
        ("source_hashes", {"kernel.py": "changed"}),
    ]:
        changed = {**cfg, key: value}
        assert runner.fingerprint(changed) != runner.fingerprint(cfg)


def test_ablation_requires_matching_smoke(tmp_path, monkeypatch):
    cfg = config()
    smoke = {**cfg, "phase": "smoke", "steps": 30, "repeats": 1}
    runner.write_json(tmp_path / "manifest.json", {"config": smoke})
    monkeypatch.setattr(runner, "select", lambda *_: 0)
    runner.validate_smoke(tmp_path, cfg)
    with pytest.raises(ValueError, match="differs"):
        runner.validate_smoke(tmp_path, {**cfg, "experiment": "rob-compact"})
    monkeypatch.setattr(runner, "select", lambda *_: 1)
    with pytest.raises(ValueError, match="not passed"):
        runner.validate_smoke(tmp_path, cfg)


@pytest.mark.parametrize("damage", [None, "capacity", "frame", "force", "exit"])
def test_pair_reads_actual_hdf5_and_rejects_invalid_results(tmp_path, damage):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    cfg = config("smoke")
    system = cfg["systems"][0]
    pair = {
        "base": "base",
        "candidate": "candidate",
        "exit_codes": {"base": 0, "candidate": 0},
    }
    for variant in ("base", "candidate"):
        row = valid_row(cfg, variant)
        row["atoms"] = 2
        row["graph_stats"]["rob1_initial_capacities"] = [80, 80]
        if variant == "candidate" and damage == "capacity":
            row["graph_stats"]["rob1_initial_capacities"] = [84, 84]
        runner.write_json(tmp_path / variant / "runs/opt4" / (system + ".json"), row)
        path = tmp_path / variant / "trajectories/opt4" / (system + ".h5")
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as f:
            f["md_step"] = [
                0,
                10,
                20,
                20 if variant == "candidate" and damage == "frame" else 30,
            ]
            for key in ("positions", "momenta", "forces"):
                values = np.zeros((4, 2, 3))
                if variant == "candidate" and damage == "force" and key == "forces":
                    values[0] = 1.0
                f[key] = values
            f["energy"] = np.zeros(4)
    if damage == "exit":
        pair["exit_codes"]["candidate"] = 1
    result = runner.inspect_pair(tmp_path, pair, cfg, system)
    assert bool(result["errors"]) == (damage is not None)
    assert (result["speedup"] is None) == (damage is not None)
