#!/usr/bin/env python3
"""Poll idle GPUs for the combined KF12 plus CAP1-auto-safe matrix."""

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
            / f"opt4_kf12_cap1_safe_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ.setdefault("SCOPES", "both")
    os.environ.setdefault(
        "SYSTEMS",
        "Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024",
    )
    os.environ.setdefault("TEMPERATURES", "300 800")
    os.environ.setdefault("STEPS", "100")
    os.environ.setdefault("REPEATS", "3")
    model_base = "so2-epilogue,so2-gate-bridge"
    model_candidate = f"{model_base},so2-block-gemm"
    whole_base = f"rmsnorm,{model_base}"
    whole_candidate = f"rmsnorm,{model_candidate}"
    os.environ["MODEL_BASE_STAGE"] = "OPT4V2"
    os.environ["MODEL_CANDIDATE_STAGE"] = "KF12"
    os.environ["MODEL_BASE_FUSIONS"] = model_base
    os.environ["MODEL_CANDIDATE_FUSIONS"] = model_candidate
    os.environ["MODEL_BASE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"
    os.environ["MODEL_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"
    os.environ["WHOLE_BASE_STAGE"] = "OPT4V2"
    os.environ["WHOLE_CANDIDATE_STAGE"] = "KF12CAP1SAFE"
    os.environ["WHOLE_BASE_FUSIONS"] = whole_base
    os.environ["WHOLE_CANDIDATE_FUSIONS"] = whole_candidate
    os.environ["WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"
    os.environ["WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "auto-safe"
    os.environ.setdefault("NEIGHBOR_AUTO_MIN_REDUCTION", "0.05")
    os.environ.setdefault("NEIGHBOR_AUTO_GUARD_SLOTS", "1")
    os.environ.setdefault("WHOLE_PROBE_STEPS", "100")
    os.environ.setdefault("RUN_KIND", "opt4_kf12_cap1_auto_safe_combined")
    os.environ.setdefault("STATUS_FILENAME", "kf12_cap1_safe_status.tsv")


if __name__ == "__main__":
    _set_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
