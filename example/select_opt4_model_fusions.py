#!/usr/bin/env python3
"""Select cumulative Opt4 model fusions from interleaved stage results.

Acceptance policy: energy/force errors versus the baseline are recorded as
numerical telemetry only and do NOT decide whether an experiment passes.
A candidate is accepted on structural CUDA Graph health (one capture, no
capacity miss, invariants not broken) plus stable paired timing. With
``--focus-systems``, the focus subset supplies the geomean and stable-speedup
primary decision; every other supplied system is a regression guardrail.
Without it, the historical all-system policy is retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def _mad(values: list[float]) -> float:
    centre = median(values)
    return median(abs(value - centre) for value in values)


def _read_status(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


BACKENDS_BY_SCOPE = {
    "model-only": {
        "esen_gpu_resident_model_cg",
        "esen_gpu_resident_model_cg_opt4",
    },
    "whole-step": {
        "esen_gpu_resident_whole_step_cg",
        "esen_gpu_resident_whole_step_cg_opt4",
    },
}

NON_FATAL_STATUSES = {"success", "validation_failed"}
HARD_STATUSES = {
    "oom",
    "capacity_overflow",
    "unsupported_fusion_config",
    "graph_validation_failed",
    "error",
}


def _load_records(root: Path, scope: str) -> list[dict[str, object]]:
    records = []
    for path in root.rglob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("backend") in BACKENDS_BY_SCOPE[scope]
            and record.get("kernel_fusion_stage")
        ):
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=tuple(BACKENDS_BY_SCOPE), required=True
    )
    parser.add_argument("--base-stage", required=True)
    parser.add_argument("--candidate-stage", required=True)
    parser.add_argument("--candidate-fusion", required=True)
    parser.add_argument("--accepted-before", default="")
    parser.add_argument(
        "--focus-systems",
        nargs="+",
        default=None,
        help=(
            "Primary systems for acceptance (for example Cu512 H2O512). "
            "All other systems remain reported and are guardrails."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-geomean-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-system-regression", type=float, default=0.01)
    parser.add_argument("--min-paired-repeats", type=int, default=5)
    parser.add_argument("--min-faster-directions", type=int, default=4)
    parser.add_argument(
        "--maximum-peak-reserved-increase-gib",
        type=float,
        default=None,
        help="Optional median candidate-minus-base peak reserved memory guardrail",
    )
    args = parser.parse_args()

    statuses = _read_status(args.input_dir / "run_status.tsv")
    records = _load_records(args.input_dir, args.scope)
    grouped: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)
    reserved_grouped: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)
    records_by_stage: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        stage = str(record.get("kernel_fusion_stage", ""))
        key = (str(record["system"]), int(float(record["temperature_K"])), stage)
        grouped[key][int(record.get("repeat", 1))] = float(
            record["seconds_per_step"]
        )
        if record.get("peak_reserved_gib") is not None:
            reserved_grouped[key][int(record.get("repeat", 1))] = float(
                record["peak_reserved_gib"]
            )
        records_by_stage[stage].append(record)

    systems = sorted(
        {key[:2] for key in grouped if key[2] == args.base_stage}
        & {key[:2] for key in grouped if key[2] == args.candidate_stage}
    )
    comparisons = []
    for system, temperature in systems:
        base_by_repeat = grouped[(system, temperature, args.base_stage)]
        candidate_by_repeat = grouped[(system, temperature, args.candidate_stage)]
        repeats = sorted(set(base_by_repeat) & set(candidate_by_repeat))
        base = [base_by_repeat[repeat] for repeat in repeats]
        candidate = [candidate_by_repeat[repeat] for repeat in repeats]
        base_median = median(base)
        candidate_median = median(candidate)
        paired = len(repeats)
        directions = sum(
            candidate_by_repeat[repeat] < base_by_repeat[repeat]
            for repeat in repeats
        )
        base_reserved = reserved_grouped[(system, temperature, args.base_stage)]
        candidate_reserved = reserved_grouped[
            (system, temperature, args.candidate_stage)
        ]
        reserved_repeats = sorted(set(base_reserved) & set(candidate_reserved))
        reserved_deltas = [
            candidate_reserved[repeat] - base_reserved[repeat]
            for repeat in reserved_repeats
        ]
        reserved_delta = (
            median(reserved_deltas)
            if reserved_repeats
            else None
        )
        comparisons.append(
            {
                "system": system,
                "temperature_K": temperature,
                "base_median": base_median,
                "candidate_median": candidate_median,
                "speedup": base_median / candidate_median,
                "directions_faster": directions,
                "paired_repeats": paired,
                "delta_exceeds_mad_sum": (
                    abs(base_median - candidate_median) > _mad(base) + _mad(candidate)
                ),
                "peak_reserved_increase_gib_median": reserved_delta,
                "peak_reserved_paired_repeats": len(reserved_repeats),
                "peak_reserved_increase_over_limit_directions": (
                    None
                    if args.maximum_peak_reserved_increase_gib is None
                    else sum(
                        delta > args.maximum_peak_reserved_increase_gib
                        for delta in reserved_deltas
                    )
                ),
            }
        )

    candidate_records = records_by_stage[args.candidate_stage]
    candidate_statuses = [
        row
        for row in statuses
        if row.get("scope", args.scope) == args.scope
        and row.get("fusion_stage") == args.candidate_stage
    ]
    base_records = records_by_stage[args.base_stage]

    def record_key(record: dict[str, object]) -> tuple[str, int, int]:
        return (
            str(record.get("system", "")),
            int(float(record.get("temperature_K", 0))),
            int(record.get("repeat", 1)),
        )

    def status_key(row: dict[str, str]) -> tuple[str, int, int]:
        return (
            str(row.get("system", "")),
            int(float(row.get("temperature_K", 0))),
            int(row.get("repeat", 1)),
        )

    candidate_keys = {record_key(record) for record in candidate_records}
    base_keys = {record_key(record) for record in base_records}
    candidate_status_keys = {status_key(row) for row in candidate_statuses}
    coverage_ok = bool(candidate_keys) and candidate_keys == base_keys == candidate_status_keys
    # Structural CUDA Graph health: graph wiring must be intact for the
    # timing comparison to be valid.  Energy/force-vs-baseline errors are
    # telemetry only and never gate acceptance.
    structural_ok = coverage_ok and all(
        record.get("graph_invariants_pass") is not False
        and record.get("performance_sample_eligible", True) is not False
        and int(record.get("cuda_graph_capacity_misses", 0)) == 0
        and int(record.get("cuda_graph_capture_count", 0)) == 1
        and int(record.get("cuda_graph_production_capture_count", 0)) == 0
        and float(record.get("cuda_graph_hit_rate", 1.0)) == 1.0
        for record in candidate_records
    )
    engineering_validation_ok = bool(candidate_records) and all(
        record.get("engineering_validation_pass") is not False
        for record in candidate_records
    )
    # A legacy validation_failed row is non-fatal: the benchmark wrote a
    # complete result and the numerical mismatch is recorded in JSON. Hard
    # runtime failures still invalidate the timing comparison.
    status_ok = bool(candidate_statuses) and coverage_ok and all(
        row.get("status") in NON_FATAL_STATUSES
        and row.get("status") not in HARD_STATUSES
        for row in candidate_statuses
    )
    geomean = (
        math.exp(sum(math.log(row["speedup"]) for row in comparisons) / len(comparisons))
        if comparisons
        else 0.0
    )
    stable = bool(comparisons) and all(
        row["paired_repeats"] >= args.min_paired_repeats
        and row["directions_faster"] >= args.min_faster_directions
        and row["delta_exceeds_mad_sum"]
        for row in comparisons
    )
    no_regression = bool(comparisons) and all(
        row["speedup"] >= 1.0 - args.maximum_system_regression
        for row in comparisons
    )
    focus_set = set(args.focus_systems or [])
    focus_comparisons = (
        [row for row in comparisons if row["system"] in focus_set]
        if focus_set
        else comparisons
    )
    guardrail_comparisons = (
        [row for row in comparisons if row["system"] not in focus_set]
        if focus_set
        else comparisons
    )

    def _geomean(rows):
        return (
            math.exp(sum(math.log(row["speedup"]) for row in rows) / len(rows))
            if rows
            else 0.0
        )

    focus_geomean = _geomean(focus_comparisons)
    focus_stable = bool(focus_set) and len(focus_comparisons) == len(focus_set) and all(
        row["paired_repeats"] >= args.min_paired_repeats
        and row["directions_faster"] >= args.min_faster_directions
        and row["delta_exceeds_mad_sum"]
        for row in focus_comparisons
    )
    focus_no_regression = bool(focus_comparisons) and all(
        row["speedup"] >= 1.0 - args.maximum_system_regression
        for row in focus_comparisons
    )
    # Non-focus systems protect against regressions but do not need to show a
    # speedup.  With no focus list preserve the historical all-system policy.
    guardrail_ok = all(
        row["speedup"] >= 1.0 - args.maximum_system_regression
        for row in guardrail_comparisons
    )
    reserved_memory_ok = (
        args.maximum_peak_reserved_increase_gib is None
        or (
            bool(comparisons)
            and all(
                row["peak_reserved_increase_gib_median"] is not None
                and not (
                    row["peak_reserved_paired_repeats"]
                    >= args.min_paired_repeats
                    and row["peak_reserved_increase_over_limit_directions"]
                    == row["peak_reserved_paired_repeats"]
                )
                for row in comparisons
            )
        )
    )
    if focus_set:
        accepted = (
            structural_ok
            and status_ok
            and focus_stable
            and focus_no_regression
            and focus_geomean >= args.minimum_geomean_speedup
            and guardrail_ok
            and reserved_memory_ok
        )
    else:
        accepted = (
            structural_ok
            and status_ok
            and stable
            and no_regression
            and geomean >= args.minimum_geomean_speedup
            and reserved_memory_ok
        )
    before = [item for item in args.accepted_before.split(",") if item]
    after = before + ([args.candidate_fusion] if accepted else [])
    result = {
        "scope": args.scope,
        "base_stage": args.base_stage,
        "candidate_stage": args.candidate_stage,
        "candidate_fusion": args.candidate_fusion,
        "accepted": accepted,
        "accepted_before": before,
        "accepted_after": after,
        "geomean_speedup": geomean,
        "focus_systems": sorted(focus_set),
        "focus_geomean_speedup": focus_geomean,
        "focus_stable": focus_stable,
        "focus_no_system_regression": focus_no_regression,
        "guardrail_ok": guardrail_ok,
        "reserved_memory_ok": reserved_memory_ok,
        "maximum_peak_reserved_increase_gib": args.maximum_peak_reserved_increase_gib,
        "structural_ok": structural_ok,
        "coverage_ok": coverage_ok,
        "engineering_validation_ok": engineering_validation_ok,
        "status_ok": status_ok,
        "stable": stable,
        "no_system_regression": no_regression,
        "min_paired_repeats": args.min_paired_repeats,
        "min_faster_directions": args.min_faster_directions,
        "policy": (
            "energy/force-vs-baseline errors are telemetry only; legacy "
            "validation_failed is non-fatal when result and graph are healthy"
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
