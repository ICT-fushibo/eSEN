#!/usr/bin/env python3
"""Three INDEPENDENT Matbench experiments against frozen Opt4 v5.

One invocation owns one GPU and one experiment. Paired AB/BA process runs,
fingerprinted resume (whole pairs only), explicit protocol, fail-closed checks.
No statistics/stress pass or formal matrix is launched by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_esen_matbench import OPT4_V4_MODEL_FUSIONS

REPO = Path(__file__).resolve().parents[1]
SYSTEMS = ["anthracene_293K_Sharma_S", "bulkLiMgAlZnSn_900K_J_Schmidt_VASP"]
EXPERIMENTS = ("nhc", "rob-compact", "shared-geometry")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_smoke(smoke_root, config):
    smoke_root = Path(smoke_root)
    smoke = read_json(smoke_root / "manifest.json")["config"]
    expected = {**config, "phase": "smoke", "steps": 30, "repeats": 1}
    if smoke != expected:
        raise ValueError(
            "Smoke protocol/source/data/GPU differs; rerun smoke with current files"
        )
    # Revalidate the actual paired records, not merely a cached accepted flag.
    if select(smoke_root, smoke) != 0:
        raise ValueError("Smoke has not passed; ablation is blocked")


def command(config, variant, system, destination):
    return [
        sys.executable,
        "-u",
        str(REPO / "example/run_esen_matbench.py"),
        "--backend",
        "opt4",
        "--systems",
        system,
        "--reference-h5",
        config["reference_h5"],
        "--checkpoint",
        config["checkpoint"],
        "--matbench-repo",
        config["matbench_repo"],
        "--gpu",
        config["gpu"],
        "--save-dir",
        str(destination),
        "--steps",
        str(config["steps"]),
        "--record-interval",
        "10",
        "--timestep-fs",
        "0.25",
        "--thermostat-time-fs",
        "25",
        "--seed",
        "0",
        "--probe-steps",
        "50",
        "--neighbor-margin",
        "0.10",
        "--neighbor-slot-step",
        "8",
        "--edge-step",
        "256",
        "--dummy-atoms",
        "32",
        "--capture-warmup",
        "3",
        "--max-neighbors",
        "300",
        "--degeneracy-tolerance",
        "0.01",
        "--opt4-model-fusions",
        OPT4_V4_MODEL_FUSIONS,
        "--opt4-fusion-stage",
        "OPT4V5_FP32_ROB1",
        "--opt4-execution-scope",
        "whole-step",
        "--opt4-neighbor-builder",
        "dense",
        "--opt4-neighbor-capacity-policy",
        "auto-safe",
        "--neighbor-auto-min-reduction",
        "0.05",
        "--neighbor-auto-guard-slots",
        "1",
        "--rob1",
        "--rob1-window-steps",
        "10",
        "--rob1-max-retries",
        "2",
        "--cap2-test-capacity-limit",
        "0",
        "--opt4-runtime-experiment",
        "none" if variant == "base" else config["experiment"],
        "--no-statistics",
        "--no-offline-stress",
    ]


def run_errors(row, config, variant):
    errors = []
    expected = "none" if variant == "base" else config["experiment"]
    graph = row.get("graph_stats", {})
    checks = {
        "successful_exit": row.get("status") == "success" and row.get("exit_code") == 0,
        "correct_experiment": row.get("opt4_runtime_experiment") == expected,
        "nhc_implementation": row.get("nhc_implementation")
        == (
            "FusedMatbenchNHCIntegrator"
            if expected == "nhc"
            else "MatbenchNHCIntegrator"
        ),
        "graph_invariants": row.get("graph_invariants_pass") is True,
        "initial_force_validation": row.get("initial_graph_validation_pass") is True,
        "rob1_enabled": graph.get("rob1_enabled") is True,
        "dense": graph.get("fixed_builder_backend") == "dense",
        "auto_safe": graph.get("rob1_initial_capacity_policy") == "auto-safe",
        "committed": graph.get("rob1_committed_replays") == config["steps"] + 1,
        "no_rollback": graph.get("rob1_rollback_count") == 0,
        "no_recovery_capture": graph.get("cuda_graph_recovery_capture_count") == 0,
        "no_unrecovered_overflow": graph.get("rob1_unrecovered_overflows") == 0,
        "fixed_snapshot": graph.get("rob1_snapshot_addresses_stable") is True,
        "status_mode": graph.get("rob1_window_status_mode")
        == ("compact" if expected == "rob-compact" else "full"),
        "shared_geometry": (graph.get("shared_geometry_blocks", 0) > 0)
        == (expected == "shared-geometry"),
        "frames": row.get("trajectory_frames") == config["steps"] // 10 + 1,
        "finite_time": math.isfinite(row.get("seconds_per_step", float("nan")))
        and row.get("seconds_per_step", 0) > 0,
        "finite_memory": math.isfinite(row.get("peak_reserved_gib", float("nan"))),
    }
    errors.extend(key for key, ok in checks.items() if not ok)
    return errors


def inspect_pair(root, pair, config, system):
    import h5py
    import numpy as np

    errors, rows, frames = [], {}, {}
    for variant in ("base", "candidate"):
        run = root / pair[variant]
        row = read_json(run / "runs/opt4" / (system + ".json"))
        rows[variant] = row
        errors.extend(variant + ":" + e for e in run_errors(row, config, variant))
        if pair["exit_codes"].get(variant) != 0:
            errors.append(variant + ":process_exit")
        with h5py.File(run / "trajectories/opt4" / (system + ".h5"), "r") as handle:
            if not np.array_equal(
                handle["md_step"][:], np.arange(0, config["steps"] + 1, 10)
            ):
                errors.append(variant + ":trajectory_steps")
            for key in ("positions", "momenta", "forces", "energy"):
                if not np.isfinite(handle[key][:]).all():
                    errors.append(variant + ":nonfinite_" + key)
            frames[variant] = {
                k: handle[k][0] for k in ("positions", "forces", "energy")
            }
    base, candidate = rows["base"], rows["candidate"]
    b, c = frames["base"], frames["candidate"]
    force_error = float(np.max(np.abs(b["forces"] - c["forces"])))
    energy_error = float(np.max(np.abs(b["energy"] - c["energy"]))) / base["atoms"]
    if not np.array_equal(b["positions"], c["positions"]):
        errors.append("different_initial_positions")
    if force_error >= 2e-4 or energy_error >= 1e-5:
        errors.append("cross_variant_initial_energy_force")
    if (
        base["graph_stats"]["rob1_initial_capacities"]
        != candidate["graph_stats"]["rob1_initial_capacities"]
    ):
        errors.append("different_capacity_layout")
    return {
        "system": system,
        "errors": errors,
        "base_seconds_per_step": base["seconds_per_step"],
        "candidate_seconds_per_step": candidate["seconds_per_step"],
        "speedup": None
        if errors
        else base["seconds_per_step"] / candidate["seconds_per_step"],
        "reserved_increase_gib": candidate["peak_reserved_gib"]
        - base["peak_reserved_gib"],
        "initial_force_max_abs_error": force_error,
        "initial_energy_per_atom_abs_error": energy_error,
    }


def summarize(comparisons, config, missing):
    health = (
        not missing and len(comparisons) == len(config["systems"]) * config["repeats"]
    )
    health = health and all(not p["errors"] for p in comparisons)
    summary = []
    for system in config["systems"]:
        pairs = [
            p for p in comparisons if p["system"] == system and p["speedup"] is not None
        ]
        if len(pairs) != config["repeats"]:
            continue
        values = [p["speedup"] for p in pairs]
        summary.append(
            {
                "system": system,
                "median_speedup": statistics.median(values),
                "paired_speedups": values,
                "stable_regression_over_1pct": all(v < 1 / 1.01 for v in values),
                "median_reserved_increase_gib": statistics.median(
                    p["reserved_increase_gib"] for p in pairs
                ),
            }
        )
    gm = (
        math.exp(statistics.mean(math.log(p["median_speedup"]) for p in summary))
        if health
        else None
    )
    memory_ok = bool(
        health and all(p["median_reserved_increase_gib"] <= 1 for p in summary)
    )
    performance = bool(
        health
        and gm >= 1.01
        and not any(p["stable_regression_over_1pct"] for p in summary)
    )
    accepted = bool(
        health and memory_ok and (config["phase"] == "smoke" or performance)
    )
    return {
        "experiment": config["experiment"],
        "phase": config["phase"],
        "accepted": accepted,
        "health_ok": bool(health),
        "memory_ok": memory_ok,
        "performance_ok": performance,
        "geomean_speedup": gm,
        "systems": summary,
        "comparisons": comparisons,
        "missing_or_errors": missing,
        "policy": "Smoke gates health/memory only; ablation requires >=1.01x geometric mean, no consistent >1% system regression, <=1 GiB median reserved increase, zero rollback/recovery and identical capacities. Not a formal acceptance.",
    }


def select(root, config):
    comparisons, missing = [], []
    for repeat in range(1, config["repeats"] + 1):
        for system in config["systems"]:
            marker = root / "pairs" / f"{system}_r{repeat}.json"
            try:
                pair = read_json(marker)
                if pair["fingerprint"] != fingerprint(config):
                    raise ValueError("pair fingerprint mismatch")
                result = inspect_pair(root, pair, config, system)
                result["repeat"] = repeat
                comparisons.append(result)
            except (OSError, KeyError, ValueError) as error:
                missing.append(f"{system}/r{repeat}: {type(error).__name__}: {error}")
    result = summarize(comparisons, config, missing)
    write_json(root / "selection.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["accepted"] else 1


def wait_idle(gpu, seconds):
    if seconds <= 0:
        return
    idle_since = None
    print(f"Waiting for GPU {gpu} idle for {seconds}s", flush=True)
    while True:
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        usage = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        idle = not pids and float(usage) <= 5
        idle_since = (idle_since or time.monotonic()) if idle else None
        if idle_since is not None and time.monotonic() - idle_since >= seconds:
            return
        time.sleep(5)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=EXPERIMENTS)
    p.add_argument("--phase", choices=("smoke", "ablation"), default="smoke")
    p.add_argument("--gpu", default="0")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-h5", type=Path)
    p.add_argument("--checkpoint", type=Path, default=REPO / "esen_30m_oam.pt")
    p.add_argument(
        "--matbench-repo", type=Path, default=REPO.parent / "matbench-discovery"
    )
    p.add_argument("--systems", nargs="+", default=SYSTEMS)
    p.add_argument("--idle-seconds", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--smoke-dir",
        type=Path,
        help="Required matching successful smoke before ablation",
    )
    p.add_argument("--select-only", action="store_true")
    args = p.parse_args(argv)
    root = args.output_dir.resolve()
    manifest = root / "manifest.json"
    if args.select_only:
        return select(root, read_json(manifest)["config"])
    if not args.experiment or args.reference_h5 is None:
        p.error("--experiment and --reference-h5 are required")
    if len(args.systems) != 2 or len(set(args.systems)) != 2:
        p.error("Choose exactly one small and one large, distinct Matbench system")
    if not args.gpu.isdigit() or args.idle_seconds < 0:
        p.error("Use one physical GPU index and nonnegative idle time")
    source_paths = [
        *sorted((REPO / "src/fairchem/core/applications").glob("*.py")),
        *sorted((REPO / "src/fairchem/core/models/esen").rglob("*.py")),
        Path(__file__).resolve(),
        REPO / "example/run_esen_matbench.py",
    ]
    config = {
        "schema": 1,
        "experiment": args.experiment,
        "phase": args.phase,
        "steps": 30 if args.phase == "smoke" else 1000,
        "repeats": 1 if args.phase == "smoke" else 3,
        "gpu": args.gpu,
        "systems": args.systems,
        "reference_h5": str(args.reference_h5.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "matbench_repo": str(args.matbench_repo.resolve()),
        "fusions": OPT4_V4_MODEL_FUSIONS,
        "source_hashes": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
    }
    config["data_identity"] = {
        key: {
            "size": Path(config[key]).stat().st_size,
            "mtime_ns": Path(config[key]).stat().st_mtime_ns,
        }
        for key in ("reference_h5", "checkpoint")
    }
    config["environment"] = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "ase", "numpy", "h5py")
        },
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                args.gpu,
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    if args.phase == "ablation":
        if args.smoke_dir is None:
            p.error("Ablation requires --smoke-dir from the same experiment")
        validate_smoke(args.smoke_dir, config)
    if manifest.exists():
        if read_json(manifest)["config"] != config:
            p.error("Output fingerprint differs; use a NEW directory")
        if not args.resume:
            p.error("Existing experiment; pass --resume or use a NEW directory")
    elif root.exists() and any(root.iterdir()):
        p.error("Output is non-empty without a manifest; use a NEW directory")
    write_json(manifest, {"fingerprint": fingerprint(config), "config": config})
    env = dict(
        os.environ,
        PYTHONPATH=str(REPO / "src") + os.pathsep + str(REPO / "example"),
        PYTHONHASHSEED="42",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        NVIDIA_TF32_OVERRIDE="0",
    )
    # Each repeat swaps AB/BA; also counterbalance order across the two systems.
    for repeat in range(1, config["repeats"] + 1):
        for index, system in enumerate(config["systems"]):
            marker = root / "pairs" / f"{system}_r{repeat}.json"
            if marker.exists() and args.resume:
                try:
                    pair = read_json(marker)
                    if (
                        pair["fingerprint"] == fingerprint(config)
                        and not inspect_pair(root, pair, config, system)["errors"]
                    ):
                        print(f"resume: complete pair {system} r{repeat}", flush=True)
                        continue
                except (OSError, KeyError, ValueError):
                    pass
            wait_idle(args.gpu, args.idle_seconds)
            attempt = root / "runs" / system / f"r{repeat}_{time.time_ns()}"
            attempt.mkdir(parents=True)
            pair = {"fingerprint": fingerprint(config), "exit_codes": {}}
            order = (
                ("base", "candidate") if (repeat + index) % 2 else ("candidate", "base")
            )
            for variant in order:
                destination = attempt / variant
                cmd = command(config, variant, system, destination)
                pair[variant] = str(destination.relative_to(root))
                write_json(attempt / (variant + "_command.json"), cmd)
                print(f"START {system} r{repeat} {variant} gpu={args.gpu}", flush=True)
                with (attempt / (variant + ".log")).open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        cmd,
                        cwd=REPO,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                pair["exit_codes"][variant] = result.returncode
                print(
                    f"END {system} r{repeat} {variant} exit={result.returncode}",
                    flush=True,
                )
            write_json(marker, pair)
    return select(root, config)


if __name__ == "__main__":
    raise SystemExit(main())
