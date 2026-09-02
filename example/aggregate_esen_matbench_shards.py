#!/usr/bin/env python3
"""Aggregate fixed-GPU Matbench backend shards and baseline alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRICS = (
    "rdf_error",
    "adf_error",
    "vdos_error",
    "pressure_mae",
    "pressure_wasserstein",
    "pressure_error",
)
ALIGNMENT_LIMITS = {
    "rdf_error": 0.1,
    "adf_error": 0.1,
    "vdos_error": 0.5,
    "pressure_mae": 0.02,
    "pressure_wasserstein": 0.02,
    "pressure_error": 2.0,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--backends", nargs="+", default=("baseline", "opt1", "opt2", "opt3", "opt4")
    )
    parser.add_argument("--systems", nargs="+", required=True)
    args = parser.parse_args()

    runs: dict[tuple[str, str], dict[str, Any]] = {}
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for backend in args.backends:
        shard = args.root / backend
        for system in args.systems:
            run_path = shard / "runs" / backend / f"{system}.json"
            if not run_path.is_file():
                missing.append(str(run_path))
                continue
            runs[(backend, system)] = _read_json(run_path)
        metric_path = shard / "matbench_esen_metrics.tsv"
        if metric_path.is_file():
            with metric_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    metrics[(backend, row["system"])] = row

    speed_rows = []
    alignment_rows = []
    for system in args.systems:
        baseline = runs.get(("baseline", system), {})
        baseline_time = baseline.get("seconds_per_step")
        baseline_metric = metrics.get(("baseline", system), {})
        for backend in args.backends:
            run = runs.get((backend, system), {})
            current = run.get("seconds_per_step")
            speedup = (
                float(baseline_time) / float(current)
                if baseline_time not in (None, 0) and current not in (None, 0)
                else None
            )
            speed_rows.append(
                {
                    "system": system,
                    "backend": backend,
                    "status": run.get("status", "missing"),
                    "seconds_per_step": current,
                    "steps_per_second": run.get("steps_per_second"),
                    "speedup_vs_baseline": speedup,
                    "physical_alignment_pass": None,
                }
            )
            if backend == "baseline":
                continue
            row: dict[str, Any] = {"system": system, "backend": backend}
            passes = []
            current_metric = metrics.get((backend, system), {})
            for key in METRICS:
                try:
                    base_value = float(baseline_metric[key])
                    value = float(current_metric[key])
                    delta = value - base_value
                except (KeyError, TypeError, ValueError):
                    base_value = value = delta = None
                limit = ALIGNMENT_LIMITS[key]
                passed = delta is not None and abs(delta) <= limit
                passes.append(passed)
                row[f"baseline_{key}"] = base_value
                row[key] = value
                row[f"delta_{key}"] = delta
                row[f"limit_{key}"] = limit
                row[f"{key}_aligned"] = passed
            row["physical_alignment_pass"] = all(passes)
            alignment_rows.append(row)
            for speed_row in reversed(speed_rows):
                if speed_row["system"] == system and speed_row["backend"] == backend:
                    speed_row["physical_alignment_pass"] = all(passes)
                    break

    speed_path = args.root / "matbench_5backend_speedups.tsv"
    with speed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(speed_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(speed_rows)
    alignment_path = args.root / "matbench_5backend_physical_alignment.tsv"
    fields = list(dict.fromkeys(key for row in alignment_rows for key in row))
    with alignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(alignment_rows)

    speedup_summary = {}
    for backend in args.backends:
        values = [
            float(row["speedup_vs_baseline"])
            for row in speed_rows
            if row["backend"] == backend
            and row["speedup_vs_baseline"] is not None
        ]
        speedup_summary[backend] = {
            "systems": len(values),
            "geomean_speedup_vs_baseline": (
                math.exp(sum(math.log(value) for value in values) / len(values))
                if values
                else None
            ),
        }

    report = {
        "experiment": "Matbench_10k_baseline_Opt1_Opt2_Opt3_Opt4v5",
        "root": str(args.root.resolve()),
        "systems": args.systems,
        "backends": args.backends,
        "missing": missing,
        "speedups": speed_rows,
        "speedup_summary": speedup_summary,
        "physical_alignment": alignment_rows,
        "alignment_limits": ALIGNMENT_LIMITS,
        "complete": not missing,
        "all_runs_success": not missing and all(
            row.get("status") == "success" for row in runs.values()
        ),
        "all_optimized_backends_physically_aligned": bool(
            alignment_rows
            and all(row["physical_alignment_pass"] for row in alignment_rows)
        ),
        "timing_note": (
            "Backends were intentionally pinned to different physical GPUs; "
            "speedups assume the GPUs have equivalent clocks and load."
        ),
    }
    output = args.root / "matbench_5backend_aggregate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return (
        0
        if report["complete"]
        and report["all_runs_success"]
        and report["all_optimized_backends_physically_aligned"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
