#!/usr/bin/env python3
"""Validate compile/capture metadata from a polling KF13 smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "tf32"), required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    expected = int(metadata["expected_task_count"]) // 2
    stage = "KF13_FP32" if args.precision == "fp32" else "KF13_PREC1_TF32"
    enabled = args.precision == "tf32"
    failures: list[str] = []
    records = []
    for path in root.glob(f"*/results/*_{stage}_r*.json"):
        try:
            records.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: unreadable JSON: {exc}")
    if len(records) != expected:
        failures.append(f"expected {expected} KF13 results, found {len(records)}")
    for path, data in records:
        checks = {
            "replacement_count": data.get(
                "model_fusion_so3_weight_cache_replacements"
            ) == 20,
            "expanded_weight_count": data.get(
                "model_fusion_so3_weight_cache_expanded_weight_count"
            ) == 20,
            "cache_bytes": data.get("model_fusion_so3_weight_cache_bytes")
            == 20 * 16 * 128 * 128 * 4,
            "cache_version": data.get("model_fusion_so3_weight_cache_version")
            == "opt4-model-fusion-v7-so3-weight-cache",
            "capture_count": data.get("cuda_graph_capture_count") == 1,
            "production_capture_count": data.get(
                "cuda_graph_production_capture_count"
            ) == 0,
            "production_replays": data.get("cuda_graph_production_replays") == 2,
            "capacity_misses": data.get("cuda_graph_capacity_misses") == 0,
            "hit_rate": data.get("cuda_graph_hit_rate") == 1.0,
            "graph_invariants": data.get("graph_invariants_pass") is True,
            "tf32": data.get("tf32") is enabled,
            "tf32_mode": data.get("tf32_mode_requested")
            == ("on" if enabled else "off"),
            "tf32_verified": data.get("tf32_config_verified") is True,
        }
        bad = [name for name, passed in checks.items() if not passed]
        if bad:
            failures.append(f"{path.name}: failed {', '.join(bad)}")
    if failures:
        raise SystemExit("KF13 smoke failed:\n" + "\n".join(failures))
    print(
        f"KF13 {args.precision} smoke passed: {len(records)} candidate results"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
