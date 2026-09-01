#!/usr/bin/env python3
"""Select CAP2+ROB1 from an interleaved Opt4 v4 whole-step experiment.

Numerical energy/force differences remain telemetry.  Acceptance is based on
paired timing, CUDA Graph/transaction integrity, compact-capacity reduction,
and reserved-memory guardrails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


NON_FATAL_STATUSES = {"success", "validation_failed"}


def _load_json_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in (root / "results").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("kernel_fusion_stage"):
            records.append(row)
    return records


def _load_status(root: Path) -> list[dict[str, str]]:
    path = root / "run_status.tsv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("system", "")),
        int(float(row.get("temperature_K", 0))),
        int(row.get("repeat", 1)),
    )


def _geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--base-stage", default="OPT4V4_CAP1_AUTO_SAFE")
    parser.add_argument("--candidate-stage", default="CAP2_ROB1")
    parser.add_argument("--focus-systems", nargs="+", default=["Cu512", "H2O192"])
    parser.add_argument("--guardrail-systems", nargs="+", default=["Cu32", "H2O32"])
    parser.add_argument("--minimum-focus-geomean-speedup", type=float, default=1.01)
    parser.add_argument("--minimum-capacity-reduction", type=float, default=0.05)
    parser.add_argument("--maximum-small-regression", type=float, default=0.01)
    parser.add_argument("--maximum-reserved-increase-gib", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input_dir / "CAP2_ROB1_selection.json"

    records = _load_json_records(args.input_dir)
    by_stage: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_stage[str(row["kernel_fusion_stage"])][_key(row)] = row
    base = by_stage[args.base_stage]
    candidate = by_stage[args.candidate_stage]
    paired_keys = sorted(set(base) & set(candidate))

    status_rows = _load_status(args.input_dir)
    relevant_status = [
        row
        for row in status_rows
        if row.get("fusion_stage") in {args.base_stage, args.candidate_stage}
    ]
    status_ok = bool(relevant_status) and all(
        row.get("status") in NON_FATAL_STATUSES for row in relevant_status
    )

    structural_failures: list[str] = []
    for key, row in base.items():
        checks = {
            "base_graph_invariants": row.get("graph_invariants_pass") is True,
            "base_capture_count": int(row.get("cuda_graph_capture_count", 0))
            == 1,
            "base_production_capture_count": int(
                row.get("cuda_graph_production_capture_count", -1)
            )
            == 0,
            "base_capacity_misses": int(
                row.get("cuda_graph_capacity_misses", -1)
            )
            == 0,
            "base_hit_rate": float(row.get("cuda_graph_hit_rate", 0.0))
            == 1.0,
        }
        for name, passed in checks.items():
            if not passed:
                structural_failures.append(f"{key}: {name}")
    for key, row in candidate.items():
        checks = {
            "graph_invariants_pass": row.get("graph_invariants_pass") is True,
            "capture_count": int(row.get("cuda_graph_capture_count", 0)) == 1,
            "production_capture_count": int(
                row.get("cuda_graph_production_capture_count", -1)
            )
            == 0,
            "capacity_misses": int(row.get("cuda_graph_capacity_misses", -1)) == 0,
            "hit_rate": float(row.get("cuda_graph_hit_rate", 0.0)) == 1.0,
            "rob1_enabled": row.get("rob1_enabled") is True,
            "rollback_count": int(row.get("rob1_rollback_count", -1)) == 0,
            "recovery_capture_count": int(
                row.get("cuda_graph_recovery_capture_count", -1)
            )
            == 0,
            "unrecovered": int(row.get("rob1_unrecovered_overflows", -1)) == 0,
            "snapshot_addresses": row.get("rob1_snapshot_addresses_stable") is True,
            "sink_mode": row.get("sink_padding_mode")
            == "distributed_dummy_self_edges",
            "sink_shift": row.get("sink_nonzero_shift_verified") is True,
            "sink_cutoff": row.get("sink_cutoff_zero_verified") is True,
        }
        for name, passed in checks.items():
            if not passed:
                structural_failures.append(f"{key}: {name}")

    grouped: dict[
        tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for key in paired_keys:
        grouped[(key[0], key[1])].append((base[key], candidate[key]))

    comparisons: list[dict[str, Any]] = []
    for (system, temperature), pairs in sorted(grouped.items()):
        pairs.sort(key=lambda pair: int(pair[0].get("repeat", 1)))
        ratios = [
            float(base_row["seconds_per_step"])
            / float(candidate_row["seconds_per_step"])
            for base_row, candidate_row in pairs
        ]
        reductions = [
            (
                float(base_row["neighbor_edge_capacity"])
                - float(candidate_row["cuda_graph_initial_edge_capacity"])
            )
            / float(base_row["neighbor_edge_capacity"])
            for base_row, candidate_row in pairs
        ]
        reserved_deltas = [
            float(candidate_row["peak_reserved_gib"])
            - float(base_row["peak_reserved_gib"])
            for base_row, candidate_row in pairs
        ]
        comparisons.append(
            {
                "system": system,
                "temperature_K": temperature,
                "paired_repeats": len(pairs),
                "base_seconds_per_step": median(
                    float(pair[0]["seconds_per_step"]) for pair in pairs
                ),
                "candidate_seconds_per_step": median(
                    float(pair[1]["seconds_per_step"]) for pair in pairs
                ),
                "median_paired_speedup": median(ratios),
                "paired_speedups": ratios,
                "directions_faster": sum(value > 1.0 for value in ratios),
                "initial_capacity_reduction_median": median(reductions),
                "initial_capacity_reductions": reductions,
                "peak_reserved_increase_gib_median": median(reserved_deltas),
                "reserved_increase_over_limit_directions": sum(
                    value > args.maximum_reserved_increase_gib
                    for value in reserved_deltas
                ),
            }
        )

    focus = [row for row in comparisons if row["system"] in args.focus_systems]
    guards = [
        row for row in comparisons if row["system"] in args.guardrail_systems
    ]
    focus_present = {row["system"] for row in focus} == set(args.focus_systems)
    guard_present = {row["system"] for row in guards} == set(args.guardrail_systems)
    focus_speedup = _geomean(
        [float(row["median_paired_speedup"]) for row in focus]
    )
    focus_timing_ok = bool(focus_present) and all(
        row["paired_repeats"] == args.repeats
        and row["directions_faster"] == args.repeats
        for row in focus
    )
    guardrail_ok = bool(guard_present) and all(
        not (
            row["median_paired_speedup"] < 1.0 - args.maximum_small_regression
            and row["directions_faster"] == 0
        )
        for row in guards
    )
    capacity_ok = any(
        row["initial_capacity_reduction_median"]
        >= args.minimum_capacity_reduction
        for row in focus
    )
    memory_ok = all(
        row["reserved_increase_over_limit_directions"] < args.repeats
        for row in comparisons
    )
    expected_keys = {
        (system, 300, repeat)
        for system in [*args.focus_systems, *args.guardrail_systems]
        for repeat in range(1, args.repeats + 1)
    }
    coverage_ok = set(base) == set(candidate) == expected_keys
    accepted = bool(
        coverage_ok
        and status_ok
        and not structural_failures
        and focus_timing_ok
        and focus_speedup >= args.minimum_focus_geomean_speedup
        and guardrail_ok
        and capacity_ok
        and memory_ok
    )
    result = {
        "experiment": "CAP2_elastic_sink_ROB1",
        "accepted": accepted,
        "base_stage": args.base_stage,
        "candidate_stage": args.candidate_stage,
        "focus_systems": args.focus_systems,
        "guardrail_systems": args.guardrail_systems,
        "focus_geomean_speedup": focus_speedup,
        "coverage_ok": coverage_ok,
        "status_ok": status_ok,
        "structural_ok": not structural_failures,
        "focus_timing_ok": focus_timing_ok,
        "guardrail_ok": guardrail_ok,
        "capacity_reduction_ok": capacity_ok,
        "reserved_memory_ok": memory_ok,
        "structural_failures": structural_failures,
        "comparisons": comparisons,
        "policy": (
            "Energy and force errors are telemetry only. Normal timing runs "
            "must not roll back or recapture; forced-overflow recovery is "
            "validated separately by the smoke runner."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
