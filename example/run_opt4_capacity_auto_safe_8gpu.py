#!/usr/bin/env python3
"""Poll idle GPUs for CAP1-auto-safe on top of the KF12 candidate."""

from __future__ import annotations

import os
from pathlib import Path
import time


def _set_defaults() -> None:
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "ROOT_OUTPUT_DIR",
        str(
            repo
            / "example"
            / "md_out"
            / f"opt4_kf12_cap1_safe_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ["SCOPES"] = "whole-step"
    # Keep known OOM-only Cu1024/H2O512+ cases out of paired acceptance.
    # They remain available as explicit probes via the SYSTEMS override.
    os.environ.setdefault("SYSTEMS", "Cu32 H2O32 H2O60 H2O192")
    os.environ.setdefault("TEMPERATURES", "300 800")
    os.environ.setdefault("STEPS", "100")
    os.environ.setdefault("REPEATS", "3")
    fusions = (
        "rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm"
    )
    os.environ["WHOLE_BASE_STAGE"] = "KF12"
    os.environ["WHOLE_CANDIDATE_STAGE"] = "KF12CAP1SAFE"
    os.environ["WHOLE_BASE_FUSIONS"] = fusions
    os.environ["WHOLE_CANDIDATE_FUSIONS"] = fusions
    os.environ["WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"
    os.environ["WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "auto-safe"
    os.environ.setdefault("NEIGHBOR_AUTO_MIN_REDUCTION", "0.05")
    os.environ.setdefault("NEIGHBOR_AUTO_GUARD_SLOTS", "1")
    os.environ.setdefault("WHOLE_PROBE_STEPS", "100")
    os.environ.setdefault("RUN_KIND", "opt4_kf12_cap1_auto_safe_ablation")
    os.environ.setdefault("STATUS_FILENAME", "cap1_auto_safe_status.tsv")


if __name__ == "__main__":
    _set_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
