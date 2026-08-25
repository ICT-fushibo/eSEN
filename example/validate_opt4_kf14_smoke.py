#!/usr/bin/env python3
"""Validate compile, replacement, and graph metadata from KF14 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    expected = int(metadata["expected_task_count"]) // 2
    failures: list[str] = []
    records = []
    for path in root.glob("*/results/*_KF14_FP32_r*.json"):
        try:
            records.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: unreadable JSON: {exc}")
    if len(records) != expected:
        failures.append(f"expected {expected} KF14 results, found {len(records)}")
    for path, data in records:
        checks = {
            "replacement_count": data.get(
                "model_fusion_so2_prepare_backward_reduce_replacements"
            ) == 10,
            "backward_kernel_count": data.get(
                "model_fusion_so2_prepare_backward_reduce_kernel_count"
            ) == 10,
            "kernel_version": data.get(
                "model_fusion_so2_prepare_backward_reduce_version"
            ) == "opt4-model-fusion-v8-so2-prepare-backward-reduce",
            "capture_count": data.get("cuda_graph_capture_count") == 1,
            "production_capture_count": data.get(
                "cuda_graph_production_capture_count"
            ) == 0,
            "production_replays": data.get("cuda_graph_production_replays") == 2,
            "capacity_misses": data.get("cuda_graph_capacity_misses") == 0,
            "hit_rate": data.get("cuda_graph_hit_rate") == 1.0,
            "graph_invariants": data.get("graph_invariants_pass") is True,
            "tf32": data.get("tf32") is False,
            "tf32_mode": data.get("tf32_mode_requested") == "off",
            "tf32_verified": data.get("tf32_config_verified") is True,
        }
        bad = [name for name, passed in checks.items() if not passed]
        if bad:
            failures.append(f"{path.name}: failed {', '.join(bad)}")
    if failures:
        raise SystemExit("KF14 smoke failed:\n" + "\n".join(failures))
    print(f"KF14 smoke passed: {len(records)} candidate results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
