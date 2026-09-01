#!/usr/bin/env python3
"""Compare a CAP2/ROB1 bulkCu pilot with the frozen Opt4 v4 result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_run(root: Path, system: str) -> dict[str, Any]:
    path = root / "runs" / "opt4" / f"{system}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(root: Path, system: str) -> dict[str, str]:
    path = root / "matbench_esen_metrics.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [
        row
        for row in rows
        if row.get("backend") == "opt4" and row.get("system") == system
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Opt4 metric row for {system} in {path}")
    if matches[0].get("metric_status") != "computed":
        raise RuntimeError(f"Metrics are not complete in {path}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--system", default="bulkCu_1000K_Kapil")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base_run = _load_run(args.base_dir, args.system)
    candidate_run = _load_run(args.candidate_dir, args.system)
    base_metrics = _load_metrics(args.base_dir, args.system)
    candidate_metrics = _load_metrics(args.candidate_dir, args.system)
    thresholds = {
        "rdf_error": 0.1,
        "adf_error": 0.1,
        "vdos_error": 0.5,
        "pressure_mae": 0.02,
        "pressure_wasserstein": 0.02,
        "pressure_error": 2.0,
    }
    metric_deltas = {
        name: float(candidate_metrics[name]) - float(base_metrics[name])
        for name in thresholds
    }
    physics_ok = all(
        abs(metric_deltas[name]) <= limit
        for name, limit in thresholds.items()
    )
    stats = candidate_run.get("graph_stats", {})
    graph_ok = bool(
        candidate_run.get("status") == "success"
        and candidate_run.get("graph_invariants_pass") is True
        and stats.get("rob1_unrecovered_overflows") == 0
        and stats.get("rob1_committed_physical_steps") == 10_000
        and stats.get("cuda_graph_committed_replays") == 10_001
        and stats.get("rob1_snapshot_addresses_stable") is True
    )
    base_seconds = float(base_run["seconds_per_step"])
    candidate_seconds = float(candidate_run["seconds_per_step"])
    result = {
        "experiment": "Opt4_v4_CAP1_auto_safe_vs_CAP2_sink_ROB1_bulkCu_10k",
        "accepted": bool(physics_ok and graph_ok),
        "system": args.system,
        "base_dir": str(args.base_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "base_seconds_per_step": base_seconds,
        "candidate_seconds_per_step": candidate_seconds,
        "speedup": base_seconds / candidate_seconds,
        "physics_ok": physics_ok,
        "graph_ok": graph_ok,
        "metric_deltas_candidate_minus_base": metric_deltas,
        "absolute_metric_thresholds": thresholds,
        "rollback_count": stats.get("rob1_rollback_count"),
        "recovery_capture_count": stats.get(
            "cuda_graph_recovery_capture_count"
        ),
        "initial_edge_capacity": stats.get("cuda_graph_initial_edge_capacity"),
        "final_edge_capacity": stats.get("cuda_graph_final_edge_capacity"),
        "promotion_history": stats.get("cap2_promotion_history", []),
    }
    output = args.output or args.candidate_dir / "CAP2_ROB1_10k_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
