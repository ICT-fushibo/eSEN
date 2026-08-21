#!/usr/bin/env python3
"""Run the accepted KF11 confirmation matrix with the idle-GPU scheduler."""

from __future__ import annotations

import os
from pathlib import Path
import time


def _set_kf11_defaults() -> None:
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "ROOT_OUTPUT_DIR",
        str(
            repo
            / "example"
            / "md_out"
            / f"opt4_kf11_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ.setdefault(
        "SYSTEMS",
        "Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024",
    )
    os.environ.setdefault("TEMPERATURES", "300 800")
    os.environ.setdefault("STEPS", "100")
    os.environ.setdefault("REPEATS", "3")
    os.environ.setdefault("SCOPES", "both")
    os.environ.setdefault("MODEL_BASE_STAGE", "OPT4V2")
    os.environ.setdefault("MODEL_CANDIDATE_STAGE", "KF11")
    os.environ.setdefault(
        "MODEL_BASE_FUSIONS", "so2-epilogue,so2-gate-bridge"
    )
    os.environ.setdefault(
        "MODEL_CANDIDATE_FUSIONS",
        "so2-epilogue,so2-gate-bridge,wigner-so2-bridge",
    )
    os.environ.setdefault("WHOLE_BASE_STAGE", "OPT4V2")
    os.environ.setdefault("WHOLE_CANDIDATE_STAGE", "KF11")
    os.environ.setdefault(
        "WHOLE_BASE_FUSIONS", "rmsnorm,so2-epilogue,so2-gate-bridge"
    )
    os.environ.setdefault(
        "WHOLE_CANDIDATE_FUSIONS",
        "rmsnorm,so2-epilogue,so2-gate-bridge,wigner-so2-bridge",
    )
    os.environ.setdefault("RUN_KIND", "opt4_kf11_formal_performance")
    os.environ.setdefault("STATUS_FILENAME", "kf11_status.tsv")


if __name__ == "__main__":
    _set_kf11_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
