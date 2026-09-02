#!/usr/bin/env python3
"""Evaluate paired Opt4-v5 dense versus CELL1 Matbench results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


FOCUS_SYSTEMS = (
    "anthracene_293K_Sharma_S",
    "bulkLiMgAlZnSn_900K_J_Schmidt_VASP",
)
SCOPES = ("model-only", "whole-step")


def _load_results(root: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    pattern = "runs/*/*/repeat_*/*/runs/opt4/*.json"
    for path in root.glob(pattern):
        relative = path.relative_to(root).parts
        scope = relative[1]
        system = relative[2]
        repeat = int(relative[3].removeprefix("repeat_"))
        variant = relative[4]
        result[(scope, system, repeat, variant)] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return result


def _healthy(row: dict[str, Any], expected_builder: str) -> tuple[bool, list[str]]:
    reasons = []
    stats = row.get("graph_stats", {})
    if row.get("status") != "success":
        reasons.append(f"status={row.get('status')}")
    if row.get("initial_graph_validation_pass") is not True:
        reasons.append("initial_graph_validation_failed")
    if row.get("graph_invariants_pass") is not True:
        reasons.append("graph_invariants_failed")
    if stats.get("fixed_builder_backend") != expected_builder:
        reasons.append(
            f"builder={stats.get('fixed_builder_backend')} expected={expected_builder}"
        )
    if stats.get("cuda_graph_capacity_misses", 0) != 0:
        reasons.append("capacity_miss")
    if stats.get("cell_list_bin_overflow_replays", 0) != 0:
        reasons.append("bin_overflow")
    if stats.get("cuda_graph_hit_rate") != 1.0:
        reasons.append("graph_hit_rate")
    if stats.get("cuda_graph_replay_output_addresses_stable", True) is not True:
        reasons.append("unstable_addresses")
    return not reasons, reasons


def _geomean(values: list[float]) -> float | None:
    return (
        math.exp(sum(math.log(value) for value in values) / len(values))
        if values
        else None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--phase", choices=("smoke", "ablation", "formal"), default="ablation"
    )
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-small-regression", type=float, default=0.01)
    parser.add_argument("--maximum-memory-increase-gib", type=float, default=1.0)
    parser.add_argument("--minimum-candidate-reduction", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata_path = args.input_dir / "cell1_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    systems = tuple(metadata.get("systems", FOCUS_SYSTEMS))
    scopes = tuple(metadata.get("scopes", SCOPES))
    repeats = int(metadata.get("repeats", 1 if args.phase == "smoke" else 3))
    rows = _load_results(args.input_dir)
    missing = []
    health_failures = []
    comparisons = []
    grouped_speedups: dict[str, list[float]] = defaultdict(list)
    grouped_system_speedups: dict[tuple[str, str], list[float]] = defaultdict(list)
    memory_ok = True
    candidate_reductions = []

    for scope in scopes:
        for system in systems:
            for repeat in range(1, repeats + 1):
                key_base = (scope, system, repeat, "base")
                key_candidate = (scope, system, repeat, "candidate")
                if key_base not in rows or key_candidate not in rows:
                    missing.append(
                        {
                            "scope": scope,
                            "system": system,
                            "repeat": repeat,
                            "missing": [
                                variant
                                for variant, key in (
                                    ("base", key_base),
                                    ("candidate", key_candidate),
                                )
                                if key not in rows
                            ],
                        }
                    )
                    continue
                base = rows[key_base]
                candidate = rows[key_candidate]
                base_ok, base_reasons = _healthy(base, "dense")
                candidate_ok, candidate_reasons = _healthy(candidate, "cell-list")
                if not base_ok or not candidate_ok:
                    health_failures.append(
                        {
                            "scope": scope,
                            "system": system,
                            "repeat": repeat,
                            "base": base_reasons,
                            "candidate": candidate_reasons,
                        }
                    )
                base_time = float(base.get("rollout_wall_time_s", math.nan))
                candidate_time = float(
                    candidate.get("rollout_wall_time_s", math.nan)
                )
                speedup = (
                    base_time / candidate_time
                    if base_time > 0 and candidate_time > 0
                    else math.nan
                )
                if math.isfinite(speedup):
                    grouped_speedups[scope].append(speedup)
                    grouped_system_speedups[(scope, system)].append(speedup)
                memory_increase = float(
                    candidate.get("peak_reserved_gib", math.nan)
                ) - float(base.get("peak_reserved_gib", math.nan))
                pair_memory_ok = bool(
                    math.isfinite(memory_increase)
                    and memory_increase <= args.maximum_memory_increase_gib
                )
                memory_ok &= pair_memory_ok
                stats = candidate.get("graph_stats", {})
                candidate_reduction = stats.get("cell_list_candidate_reduction")
                if candidate_reduction is not None:
                    candidate_reductions.append(float(candidate_reduction))
                comparisons.append(
                    {
                        "scope": scope,
                        "system": system,
                        "repeat": repeat,
                        "base_seconds_per_step": base.get("seconds_per_step"),
                        "candidate_seconds_per_step": candidate.get(
                            "seconds_per_step"
                        ),
                        "speedup": speedup,
                        "candidate_faster": bool(speedup > 1.0),
                        "peak_reserved_increase_gib": memory_increase,
                        "memory_ok": pair_memory_ok,
                        "cell_list_grid_shape": stats.get("cell_list_grid_shape"),
                        "cell_list_bin_capacity": stats.get(
                            "cell_list_bin_capacity"
                        ),
                        "cell_list_candidate_reduction": candidate_reduction,
                    }
                )

    scope_summary = {}
    performance_ok = True
    for scope in scopes:
        speedups = grouped_speedups[scope]
        geomean = _geomean(speedups)
        scope_ok = bool(
            geomean is not None and geomean >= args.minimum_speedup
        )
        if args.phase != "smoke":
            performance_ok &= scope_ok
        scope_summary[scope] = {
            "pairs": len(speedups),
            "geomean_speedup": geomean,
            "median_speedup": median(speedups) if speedups else None,
            "minimum_speedup": min(speedups) if speedups else None,
            "accepted": scope_ok,
        }

    focus_consistency = {}
    consistency_ok = True
    for scope in scopes:
        for system in systems:
            values = grouped_system_speedups[(scope, system)]
            faster = sum(value > 1.0 for value in values)
            small_regression_ok = not (
                system == FOCUS_SYSTEMS[0]
                and len(values) == repeats
                and all(
                    value < 1.0 - args.maximum_small_regression
                    for value in values
                )
            )
            focus_gate = bool(
                len(values) == repeats
                and (
                    args.phase == "smoke"
                    or system not in FOCUS_SYSTEMS
                    or faster == repeats
                )
                and small_regression_ok
            )
            if args.phase == "ablation" and system in FOCUS_SYSTEMS:
                consistency_ok &= focus_gate
            focus_consistency[f"{scope}:{system}"] = {
                "pairs": len(values),
                "candidate_faster": faster,
                "geomean_speedup": _geomean(values),
                "small_regression_ok": small_regression_ok,
                "accepted": focus_gate,
            }

    health_ok = not missing and not health_failures
    candidate_reduction_ok = bool(
        candidate_reductions
        and min(candidate_reductions) >= args.minimum_candidate_reduction
    )
    accepted = bool(
        health_ok
        and memory_ok
        and (
            args.phase == "smoke"
            or (
                performance_ok
                and consistency_ok
                and candidate_reduction_ok
            )
        )
    )
    report = {
        "experiment": "CELL1_fixed_shape_gpu_cell_list",
        "phase": args.phase,
        "accepted": accepted,
        "health_ok": health_ok,
        "performance_ok": performance_ok,
        "focus_consistency_ok": consistency_ok,
        "memory_ok": memory_ok,
        "candidate_reduction_ok": candidate_reduction_ok,
        "minimum_speedup": args.minimum_speedup,
        "maximum_small_regression": args.maximum_small_regression,
        "maximum_memory_increase_gib": args.maximum_memory_increase_gib,
        "minimum_candidate_reduction": args.minimum_candidate_reduction,
        "minimum_observed_candidate_reduction": (
            min(candidate_reductions) if candidate_reductions else None
        ),
        "systems": list(systems),
        "scopes": list(scopes),
        "repeats": repeats,
        "missing": missing,
        "health_failures": health_failures,
        "scope_summary": scope_summary,
        "focus_consistency": focus_consistency,
        "comparisons": comparisons,
        "policy": (
            "Smoke gates correctness/Graph health only. Ablation/formal require "
            "both scopes to reach the speedup threshold; ablation additionally "
            "requires every focus-system candidate repeat to be faster."
        ),
    }
    output = args.output or args.input_dir / "CELL1_selection.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
