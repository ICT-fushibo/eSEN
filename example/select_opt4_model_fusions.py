#!/usr/bin/env python3
"""Select cumulative Opt4 model fusions from interleaved stage results.

Acceptance policy: energy/force errors versus the baseline are recorded as
numerical telemetry only and do NOT decide whether an experiment passes.
A candidate is accepted on structural CUDA Graph health (one capture, no
capacity miss, invariants not broken) plus stable paired timing: at least
``--min-faster-directions`` of ``--min-paired-repeats`` repeats faster per
(system, temperature), delta beyond the MAD sum, no system regression beyond
``--maximum-system-regression``, and a geomean speedup of at least
``--minimum-geomean-speedup``.
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-geomean-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-system-regression", type=float, default=0.01)
    parser.add_argument("--min-paired-repeats", type=int, default=5)
    parser.add_argument("--min-faster-directions", type=int, default=4)
    args = parser.parse_args()

    statuses = _read_status(args.input_dir / "run_status.tsv")
    records = _load_records(args.input_dir, args.scope)
    grouped: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)
    records_by_stage: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        stage = str(record.get("kernel_fusion_stage", ""))
        key = (str(record["system"]), int(float(record["temperature_K"])), stage)
        grouped[key][int(record.get("repeat", 1))] = float(
            record["seconds_per_step"]
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
            }
        )

    candidate_records = records_by_stage[args.candidate_stage]
    candidate_statuses = [
        row
        for row in statuses
        if row.get("scope", args.scope) == args.scope
        and row.get("fusion_stage") == args.candidate_stage
    ]
    # Structural CUDA Graph health: graph wiring must be intact for the
    # timing comparison to be valid.  Energy/force-vs-baseline errors are
    # telemetry only and never gate acceptance.
    structural_ok = bool(candidate_records) and all(
        record.get("graph_invariants_pass") is not False
        and int(record.get("cuda_graph_capacity_misses", 0)) == 0
        and int(record.get("cuda_graph_capture_count", 0)) == 1
        for record in candidate_records
    )
    engineering_validation_ok = bool(candidate_records) and all(
        record.get("engineering_validation_pass") is not False
        for record in candidate_records
    )
    status_ok = bool(candidate_statuses) and all(
        row.get("status") == "success" for row in candidate_statuses
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
    accepted = (
        structural_ok
        and stable
        and no_regression
        and geomean >= args.minimum_geomean_speedup
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
        "structural_ok": structural_ok,
        "engineering_validation_ok": engineering_validation_ok,
        "status_ok": status_ok,
        "stable": stable,
        "no_system_regression": no_regression,
        "min_paired_repeats": args.min_paired_repeats,
        "min_faster_directions": args.min_faster_directions,
        "policy": "energy/force-vs-baseline errors are telemetry only",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
