#!/usr/bin/env python3
"""Gate KF16 model integration on large-system operator microbenchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SYSTEMS = ("Cu512", "H2O192")
MODES = ("forward", "forward-backward")
VARIANTS = ("base", "tiled")


def _close_sequence(
    actual: list[float],
    expected: list[float],
    *,
    rtol: float,
    atol: float,
) -> bool:
    return len(actual) == len(expected) and all(
        math.isfinite(a)
        and math.isfinite(e)
        and abs(a - e) <= atol + rtol * abs(e)
        for a, e in zip(actual, expected)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--maximum-reserved-increase-gib", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--atol", type=float, default=2e-3)
    args = parser.parse_args()

    root = args.input_dir.resolve()
    records: dict[tuple[str, str, str], dict[str, object]] = {}
    missing: list[str] = []
    for system in SYSTEMS:
        for mode in MODES:
            for variant in VARIANTS:
                path = root / "results" / f"{system}_{mode}_{variant}.json"
                try:
                    records[(system, mode, variant)] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    missing.append(f"{path}: {exc}")

    comparisons: list[dict[str, object]] = []
    correctness_ok = not missing
    graph_ok = not missing
    memory_ok = not missing
    primary_ok = not missing
    for system in SYSTEMS:
        for mode in MODES:
            if missing:
                continue
            base = records[(system, mode, "base")]
            candidate = records[(system, mode, "tiled")]
            base_ms = float(base["median_ms_per_replay"])
            candidate_ms = float(candidate["median_ms_per_replay"])
            speedup = base_ms / candidate_ms
            outputs_ok = _close_sequence(
                list(candidate.get("output_checksums", [])),
                list(base.get("output_checksums", [])),
                rtol=args.rtol,
                atol=args.atol,
            )
            gradients_ok = mode == "forward" or _close_sequence(
                list(candidate.get("gradient_checksums", [])),
                list(base.get("gradient_checksums", [])),
                rtol=args.rtol,
                atol=args.atol,
            )
            pair_graph_ok = bool(base.get("addresses_stable")) and bool(
                candidate.get("addresses_stable")
            )
            reserved_increase = float(
                candidate.get("capture_reserved_delta_gib", 0.0)
            ) - float(base.get("capture_reserved_delta_gib", 0.0))
            pair_memory_ok = reserved_increase <= args.maximum_reserved_increase_gib
            pair_primary_ok = (
                speedup >= args.minimum_speedup
                if mode == "forward-backward"
                else speedup >= 1.0
            )
            correctness_ok &= outputs_ok and gradients_ok
            graph_ok &= pair_graph_ok
            memory_ok &= pair_memory_ok
            primary_ok &= pair_primary_ok
            comparisons.append(
                {
                    "system": system,
                    "mode": mode,
                    "base_ms_per_replay": base_ms,
                    "candidate_ms_per_replay": candidate_ms,
                    "speedup": speedup,
                    "outputs_ok": outputs_ok,
                    "gradients_ok": gradients_ok,
                    "addresses_stable": pair_graph_ok,
                    "capture_reserved_increase_gib": reserved_increase,
                    "memory_ok": pair_memory_ok,
                    "performance_ok": pair_primary_ok,
                }
            )

    accepted = correctness_ok and graph_ok and memory_ok and primary_ok
    result = {
        "experiment": "KF16_wigner_so2_tiled_backward_operator_gate",
        "accepted": accepted,
        "candidate_fusion": "wigner-so2-tiled-backward",
        "systems": list(SYSTEMS),
        "minimum_forward_backward_speedup": args.minimum_speedup,
        "maximum_reserved_increase_gib": args.maximum_reserved_increase_gib,
        "correctness_ok": correctness_ok,
        "graph_ok": graph_ok,
        "memory_ok": memory_ok,
        "performance_ok": primary_ok,
        "missing": missing,
        "comparisons": comparisons,
        "policy": (
            "Model smoke/ablation is allowed only when both Cu512 and H2O192 "
            "forward-backward operator speedups reach the configured threshold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
