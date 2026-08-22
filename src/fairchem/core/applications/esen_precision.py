"""Precision-policy helpers for isolated eSEN performance experiments.

The production benchmarks remain strict FP32 by default.  Callers must opt in
to TF32 explicitly before model construction and CUDA Graph capture.
"""

from __future__ import annotations

from typing import Any


TF32_POLICY_VERSION = "opt4-precision-v1-tf32"
TF32_MODES = ("off", "on")


def _normalize_tf32_mode(mode: str) -> tuple[str, bool, str]:
    normalized = str(mode).strip().lower()
    if normalized not in TF32_MODES:
        raise ValueError(
            f"TF32 mode must be one of {', '.join(TF32_MODES)}, got {mode!r}"
        )
    enabled = normalized == "on"
    matmul_precision = "high" if enabled else "highest"
    return normalized, enabled, matmul_precision


def verify_tf32(torch_module: Any, mode: str) -> dict[str, object]:
    """Read back and verify the active process-wide TF32 policy."""

    normalized, enabled, matmul_precision = _normalize_tf32_mode(mode)
    actual_matmul = bool(torch_module.backends.cuda.matmul.allow_tf32)
    actual_cudnn = bool(torch_module.backends.cudnn.allow_tf32)
    actual_precision = str(torch_module.get_float32_matmul_precision())
    verified = (
        actual_matmul == enabled
        and actual_cudnn == enabled
        and actual_precision == matmul_precision
    )
    if not verified:
        raise RuntimeError(
            "PyTorch TF32 configuration did not take effect: "
            f"requested={normalized}, matmul={actual_matmul}, "
            f"cudnn={actual_cudnn}, precision={actual_precision}"
        )
    return {
        "tf32": enabled,
        "tf32_mode_requested": normalized,
        "tf32_matmul_allowed": actual_matmul,
        "tf32_cudnn_allowed": actual_cudnn,
        "float32_matmul_precision": actual_precision,
        "tf32_config_verified": verified,
        "precision_policy_version": TF32_POLICY_VERSION,
    }


def configure_tf32(torch_module: Any, mode: str) -> dict[str, object]:
    """Configure and verify the process-wide PyTorch TF32 policy.

    A fresh process is used for every benchmark sample, so changing these
    process-wide flags cannot leak between the interleaved base and candidate.
    """

    _, enabled, matmul_precision = _normalize_tf32_mode(mode)
    torch_module.set_float32_matmul_precision(matmul_precision)
    torch_module.backends.cuda.matmul.allow_tf32 = enabled
    torch_module.backends.cudnn.allow_tf32 = enabled
    return verify_tf32(torch_module, mode)
