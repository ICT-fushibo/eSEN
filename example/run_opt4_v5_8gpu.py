#!/usr/bin/env python3
"""Poll idle GPUs for Opt4 v5: Opt4 v4 kernels plus ROB1 safety.

Opt4 v5 keeps the accepted Opt4 v4 model-fusion mask and CAP1-auto-safe
neighbor allocation.  ROB1 is enabled only for the whole-step candidate; it
does not enable CAP2's compact initial allocation.  A transaction is retried
only when the fixed-capacity builder reports a real overflow.
"""

from __future__ import annotations

import os
from pathlib import Path
import time


V4_MODEL_FUSIONS = (
    "so2-epilogue,so2-gate-bridge,so2-block-gemm,"
    "so2-prepare-backward-reduce"
)
V4_WHOLE_FUSIONS = f"rmsnorm,{V4_MODEL_FUSIONS}"


def _set_v5_defaults() -> None:
    repo = Path(__file__).resolve().parents[1]
    phase = os.environ.get("OPT4_V5_PHASE", "formal").strip().lower()
    if phase not in {"smoke", "ablation", "formal"}:
        raise ValueError("OPT4_V5_PHASE must be smoke, ablation, or formal")
    defaults = {
        "smoke": {
            "systems": "Cu32 Cu512 H2O32 H2O192",
            "temperatures": "300",
            "steps": "1",
            "repeats": "1",
            "idle_seconds": "0",
        },
        "ablation": {
            "systems": "Cu32 Cu512 H2O32 H2O192",
            "temperatures": "300",
            "steps": "100",
            "repeats": "3",
            "idle_seconds": "120",
        },
        "formal": {
            "systems": (
                "Cu32 Cu64 Cu192 Cu512 Cu1024 "
                "H2O32 H2O60 H2O192 H2O512 H2O1024"
            ),
            "temperatures": "300 800",
            "steps": "100",
            "repeats": "3",
            "idle_seconds": "120",
        },
    }[phase]

    os.environ.setdefault(
        "ROOT_OUTPUT_DIR",
        str(
            repo
            / "example"
            / "md_out"
            / f"opt4_v5_{phase}_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ.setdefault("SCOPES", "both")
    os.environ.setdefault("SYSTEMS", defaults["systems"])
    os.environ.setdefault("TEMPERATURES", defaults["temperatures"])
    os.environ.setdefault("STEPS", defaults["steps"])
    os.environ.setdefault("REPEATS", defaults["repeats"])
    os.environ.setdefault("WARMUP_STEPS", "3")
    os.environ.setdefault("GPU_IDLE_SECONDS", defaults["idle_seconds"])

    # Model-only is unchanged by ROB1.  Whole-step compares v4 against v5;
    # both use CAP1-auto-safe, and only the candidate enables transactions.
    os.environ.setdefault("MODEL_BASE_STAGE", "OPT4V4_FP32")
    os.environ.setdefault("MODEL_CANDIDATE_STAGE", "OPT4V5_FP32_ROB1")
    os.environ.setdefault("MODEL_BASE_FUSIONS", V4_MODEL_FUSIONS)
    os.environ.setdefault("MODEL_CANDIDATE_FUSIONS", V4_MODEL_FUSIONS)
    os.environ.setdefault("MODEL_BASE_TF32_MODE", "off")
    os.environ.setdefault("MODEL_CANDIDATE_TF32_MODE", "off")
    os.environ.setdefault("MODEL_BASE_NEIGHBOR_CAPACITY_POLICY", "uniform")
    os.environ.setdefault("MODEL_CANDIDATE_NEIGHBOR_CAPACITY_POLICY", "uniform")

    os.environ.setdefault("WHOLE_BASE_STAGE", "OPT4V4_FP32")
    os.environ.setdefault("WHOLE_CANDIDATE_STAGE", "OPT4V5_FP32_ROB1")
    os.environ.setdefault("WHOLE_BASE_FUSIONS", V4_WHOLE_FUSIONS)
    os.environ.setdefault("WHOLE_CANDIDATE_FUSIONS", V4_WHOLE_FUSIONS)
    os.environ.setdefault("WHOLE_BASE_TF32_MODE", "off")
    os.environ.setdefault("WHOLE_CANDIDATE_TF32_MODE", "off")
    os.environ.setdefault("WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY", "auto-safe")
    os.environ.setdefault("WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY", "auto-safe")
    os.environ.setdefault("WHOLE_BASE_ROB1", "0")
    os.environ.setdefault("WHOLE_CANDIDATE_ROB1", "1")
    os.environ.setdefault("ROB1_WINDOW_STEPS", "10")
    os.environ.setdefault("ROB1_MAX_RETRIES", "2")
    os.environ.setdefault("NEIGHBOR_AUTO_MIN_REDUCTION", "0.05")
    os.environ.setdefault("NEIGHBOR_AUTO_GUARD_SLOTS", "1")
    os.environ.setdefault("WHOLE_PROBE_STEPS", "100")
    os.environ.setdefault("RUN_KIND", f"opt4_v5_rob1_{phase}")
    os.environ.setdefault("STATUS_FILENAME", "v5_status.tsv")


if __name__ == "__main__":
    _set_v5_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
