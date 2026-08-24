#!/usr/bin/env python3
"""Poll idle GPUs for KF13 Frozen-SO3-Weight Cache experiments.

``KF13_PRECISION=fp32`` isolates KF13 on the frozen Opt4 v3 FP32 baseline.
``KF13_PRECISION=tf32`` measures the interaction with PREC1 while keeping TF32
enabled on both sides.  ``KF13_PHASE`` selects smoke, ablation, or formal
matrix defaults.  The generic scheduler preserves interleaved A/B ordering,
resume behavior, and telemetry-only numerical validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time


V3_MODEL_FUSIONS = "so2-epilogue,so2-gate-bridge,so2-block-gemm"
V3_WHOLE_FUSIONS = f"rmsnorm,{V3_MODEL_FUSIONS}"
KF13_MODEL_FUSIONS = f"{V3_MODEL_FUSIONS},so3-weight-cache"
KF13_WHOLE_FUSIONS = f"{V3_WHOLE_FUSIONS},so3-weight-cache"


def _selection_name(precision: str, scope: str) -> str:
    return f"KF13_{precision}_{scope.replace('-', '_')}_selection.json"


def _set_kf13_defaults() -> None:
    repo = Path(__file__).resolve().parents[1]
    phase = os.environ.get("KF13_PHASE", "ablation").strip().lower()
    precision = os.environ.get("KF13_PRECISION", "fp32").strip().lower()
    if phase not in {"smoke", "ablation", "formal"}:
        raise ValueError("KF13_PHASE must be smoke, ablation, or formal")
    if precision not in {"fp32", "tf32"}:
        raise ValueError("KF13_PRECISION must be fp32 or tf32")
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
            / f"opt4_kf13_{precision}_{phase}_8gpu_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.environ.setdefault("SCOPES", "both")
    os.environ.setdefault("SYSTEMS", defaults["systems"])
    os.environ.setdefault("TEMPERATURES", defaults["temperatures"])
    os.environ.setdefault("STEPS", defaults["steps"])
    os.environ.setdefault("REPEATS", defaults["repeats"])
    os.environ.setdefault("WARMUP_STEPS", "3")
    os.environ.setdefault("GPU_IDLE_SECONDS", defaults["idle_seconds"])

    tf32_mode = "on" if precision == "tf32" else "off"
    if precision == "tf32":
        model_base_stage = "PREC1_TF32"
        model_candidate_stage = "KF13_PREC1_TF32"
        whole_base_stage = "PREC1_TF32"
        whole_candidate_stage = "KF13_PREC1_TF32"
    else:
        model_base_stage = "OPT4V3_FP32"
        model_candidate_stage = "KF13_FP32"
        whole_base_stage = "OPT4V3_FP32"
        whole_candidate_stage = "KF13_FP32"

    os.environ["MODEL_BASE_STAGE"] = model_base_stage
    os.environ["MODEL_CANDIDATE_STAGE"] = model_candidate_stage
    os.environ["MODEL_BASE_FUSIONS"] = V3_MODEL_FUSIONS
    os.environ["MODEL_CANDIDATE_FUSIONS"] = KF13_MODEL_FUSIONS
    os.environ["MODEL_BASE_TF32_MODE"] = tf32_mode
    os.environ["MODEL_CANDIDATE_TF32_MODE"] = tf32_mode
    os.environ["MODEL_BASE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"
    os.environ["MODEL_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "uniform"

    os.environ["WHOLE_BASE_STAGE"] = whole_base_stage
    os.environ["WHOLE_CANDIDATE_STAGE"] = whole_candidate_stage
    os.environ["WHOLE_BASE_FUSIONS"] = V3_WHOLE_FUSIONS
    os.environ["WHOLE_CANDIDATE_FUSIONS"] = KF13_WHOLE_FUSIONS
    os.environ["WHOLE_BASE_TF32_MODE"] = tf32_mode
    os.environ["WHOLE_CANDIDATE_TF32_MODE"] = tf32_mode
    os.environ["WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY"] = "auto-safe"
    os.environ["WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"] = "auto-safe"
    os.environ.setdefault("NEIGHBOR_AUTO_MIN_REDUCTION", "0.05")
    os.environ.setdefault("NEIGHBOR_AUTO_GUARD_SLOTS", "1")
    os.environ.setdefault("WHOLE_PROBE_STEPS", "100")
    os.environ.setdefault("RUN_KIND", f"opt4_v3_kf13_{precision}_{phase}")
    os.environ.setdefault("STATUS_FILENAME", f"kf13_{precision}_status.tsv")
    os.environ["KF13_PRECISION"] = precision

    if phase == "formal":
        selection_value = os.environ.get("KF13_SELECTION_DIR", "").strip()
        if not selection_value:
            raise ValueError(
                "KF13_PHASE=formal requires KF13_SELECTION_DIR from the "
                "matching FP32 or TF32 ablation selector"
            )
        selection_dir = Path(selection_value).resolve()
        scopes_value = os.environ.get("SCOPES", "both").replace(",", " ")
        scopes = (
            ("model-only", "whole-step")
            if scopes_value.strip() == "both"
            else tuple(scopes_value.split())
        )
        for scope in scopes:
            path = selection_dir / _selection_name(precision, scope)
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read KF13 selection: {path}") from exc
            expected_mode = "on" if precision == "tf32" else "off"
            if not (
                result.get("accepted") is True
                and result.get("precision_configuration_ok") is True
                and result.get("expected_tf32_mode") == expected_mode
                and result.get("candidate_fusion") == "so3-weight-cache"
            ):
                raise ValueError(f"KF13 was not accepted for {scope}: {path}")
        os.environ["KF13_SELECTION_DIR"] = str(selection_dir)


if __name__ == "__main__":
    _set_kf13_defaults()
    from run_opt4_v1_8gpu import main

    raise SystemExit(main())
