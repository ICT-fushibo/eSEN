#!/usr/bin/env python3
"""Poll idle GPUs for KF12 while holding CAP1-auto-safe constant."""

from __future__ import annotations

import os
from pathlib import Path
import time


def _set_kf12_defaults() -> None:
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "ROOT_OUTPUT_DIR",
        str(
            repo
            / "example"
            / "md_out"
            / f"opt4_kf12_ablation_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ.setdefault("SYSTEMS", "Cu32 Cu512 H2O32 H2O192")
    os.environ.setdefault("TEMPERATURES", "300")
    os.environ.setdefault("STEPS", "100")
    os.environ.setdefault("REPEATS", "3")
    os.environ.setdefault("SCOPES", "both")
    os.environ.setdefault("MODEL_BASE_STAGE", "OPT4V2")
    os.environ.setdefault("MODEL_CANDIDATE_STAGE", "KF12")
    os.environ.setdefault(
        "MODEL_BASE_FUSIONS", "so2-epilogue,so2-gate-bridge"
    )
    os.environ.setdefault(
        "MODEL_CANDIDATE_FUSIONS",
        "so2-epilogue,so2-gate-bridge,so2-block-gemm",
    )
    os.environ.setdefault("WHOLE_BASE_STAGE", "OPT4V2CAP1SAFE")
    os.environ.setdefault("WHOLE_CANDIDATE_STAGE", "KF12CAP1SAFE")
    os.environ.setdefault(
        "WHOLE_BASE_FUSIONS", "rmsnorm,so2-epilogue,so2-gate-bridge"
    )
    os.environ.setdefault(
        "WHOLE_CANDIDATE_FUSIONS",
        "rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm",
    )
    # CAP1-auto-safe is held constant in both whole-step variants.
    os.environ.setdefault("WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY", "auto-safe")
    os.environ.setdefault(
        "WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY", "auto-safe"
    )
    os.environ.setdefault("NEIGHBOR_AUTO_MIN_REDUCTION", "0.05")
    os.environ.setdefault("NEIGHBOR_AUTO_GUARD_SLOTS", "1")
    os.environ.setdefault("WHOLE_PROBE_STEPS", "100")
    os.environ.setdefault("RUN_KIND", "opt4_v2_cap1_auto_safe_kf12_ablation")
    os.environ.setdefault("STATUS_FILENAME", "kf12_status.tsv")


if __name__ == "__main__":
    _set_kf12_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
