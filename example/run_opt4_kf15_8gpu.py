#!/usr/bin/env python3
"""Poll idle GPUs for KF15 hybrid Wigner/SO2 experiments.

KF15 is isolated against frozen Opt4-v4 FP32.  It keeps KF11's fused producer
forward, saves the two standard BMM intermediates, and restores KF14/cuBLAS
backward.  The generic scheduler provides resumability and interleaved A/B
ordering; numerical differences remain telemetry rather than timing gates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time


V4_MODEL_FUSIONS = (
    "so2-epilogue,so2-gate-bridge,so2-block-gemm,"
    "so2-prepare-backward-reduce"
)
V4_WHOLE_FUSIONS = f"rmsnorm,{V4_MODEL_FUSIONS}"
KF15_FUSION = "wigner-so2-hybrid"
KF15_MODEL_FUSIONS = f"{V4_MODEL_FUSIONS},{KF15_FUSION}"
KF15_WHOLE_FUSIONS = f"{V4_WHOLE_FUSIONS},{KF15_FUSION}"


def _selection_name(scope: str) -> str:
    return f"KF15_{scope.replace('-', '_')}_selection.json"


def _set_kf15_defaults() -> None:
    repo = Path(__file__).resolve().parents[1]
    phase = os.environ.get("KF15_PHASE", "ablation").strip().lower()
    if phase not in {"smoke", "ablation", "formal"}:
        raise ValueError("KF15_PHASE must be smoke, ablation, or formal")
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
            / f"opt4_kf15_{phase}_8gpu_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ.setdefault("SCOPES", "both")
    os.environ.setdefault("SYSTEMS", defaults["systems"])
    os.environ.setdefault("TEMPERATURES", defaults["temperatures"])
    os.environ.setdefault("STEPS", defaults["steps"])
    os.environ.setdefault("REPEATS", defaults["repeats"])
    os.environ.setdefault("WARMUP_STEPS", "3")
    os.environ.setdefault("GPU_IDLE_SECONDS", defaults["idle_seconds"])

    os.environ["MODEL_BASE_STAGE"] = "OPT4V4_FP32"
    os.environ["MODEL_CANDIDATE_STAGE"] = "KF15_FP32"
    os.environ["MODEL_BASE_FUSIONS"] = V4_MODEL_FUSIONS
    os.environ["MODEL_CANDIDATE_FUSIONS"] = KF15_MODEL_FUSIONS
    os.environ["MODEL_BASE_TF32_MODE"] = "off"
    os.environ["MODEL_CANDIDATE_TF32_MODE"] = "off"
    os.environ["MODEL_BASE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"
    os.environ["MODEL_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"

    os.environ["WHOLE_BASE_STAGE"] = "OPT4V4_FP32"
    os.environ["WHOLE_CANDIDATE_STAGE"] = "KF15_FP32"
    os.environ["WHOLE_BASE_FUSIONS"] = V4_WHOLE_FUSIONS
    os.environ["WHOLE_CANDIDATE_FUSIONS"] = KF15_WHOLE_FUSIONS
    os.environ["WHOLE_BASE_TF32_MODE"] = "off"
    os.environ["WHOLE_CANDIDATE_TF32_MODE"] = "off"
    os.environ["WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY"] = "auto-safe"
    os.environ["WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "auto-safe"
    os.environ.setdefault("NEIGHBOR_AUTO_MIN_REDUCTION", "0.05")
    os.environ.setdefault("NEIGHBOR_AUTO_GUARD_SLOTS", "1")
    os.environ.setdefault("WHOLE_PROBE_STEPS", "100")
    os.environ.setdefault("RUN_KIND", f"opt4_v4_kf15_{phase}")
    os.environ.setdefault("STATUS_FILENAME", "kf15_status.tsv")

    if phase == "formal":
        selection_value = os.environ.get("KF15_SELECTION_DIR", "").strip()
        if not selection_value:
            raise ValueError(
                "KF15_PHASE=formal requires KF15_SELECTION_DIR from the "
                "completed ablation selector"
            )
        selection_dir = Path(selection_value).resolve()
        scopes_value = os.environ.get("SCOPES", "both").replace(",", " ")
        scopes = (
            ("model-only", "whole-step")
            if scopes_value.strip() == "both"
            else tuple(scopes_value.split())
        )
        for scope in scopes:
            path = selection_dir / _selection_name(scope)
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read KF15 selection: {path}") from exc
            if not (
                result.get("accepted") is True
                and result.get("precision_configuration_ok") is True
                and result.get("expected_tf32_mode") == "off"
                and result.get("candidate_fusion") == KF15_FUSION
            ):
                raise ValueError(f"KF15 was not accepted for {scope}: {path}")
        os.environ["KF15_SELECTION_DIR"] = str(selection_dir)


if __name__ == "__main__":
    _set_kf15_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
