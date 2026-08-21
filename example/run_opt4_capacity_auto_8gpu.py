#!/usr/bin/env python3
"""Poll idle GPUs for Opt4 v2 uniform-vs-CAP1-auto ablation."""

from __future__ import annotations

import os
from pathlib import Path
import time


def _set_capacity_auto_defaults() -> None:
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "ROOT_OUTPUT_DIR",
        str(
            repo
            / "example"
            / "md_out"
            / f"opt4_capacity_auto_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ["SCOPES"] = "whole-step"
    os.environ.setdefault("SYSTEMS", "Cu32 Cu512 H2O32 H2O192")
    os.environ.setdefault("TEMPERATURES", "300")
    os.environ.setdefault("STEPS", "100")
    os.environ.setdefault("REPEATS", "3")
    os.environ.setdefault("NEIGHBOR_AUTO_MIN_REDUCTION", "0.05")
    os.environ["WHOLE_BASE_STAGE"] = "OPT4V2"
    os.environ["WHOLE_CANDIDATE_STAGE"] = "CAP1AUTO"
    fusions = "rmsnorm,so2-epilogue,so2-gate-bridge"
    os.environ["WHOLE_BASE_FUSIONS"] = fusions
    os.environ["WHOLE_CANDIDATE_FUSIONS"] = fusions
    os.environ["WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"
    os.environ["WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "auto"
    os.environ["RUN_KIND"] = "opt4_v2_auto_capacity_ablation"
    os.environ["STATUS_FILENAME"] = "capacity_auto_status.tsv"


if __name__ == "__main__":
    _set_capacity_auto_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
