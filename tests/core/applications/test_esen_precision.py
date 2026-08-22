from __future__ import annotations

from types import SimpleNamespace

import pytest

from fairchem.core.applications.esen_precision import configure_tf32, verify_tf32


class _FakeTorch:
    def __init__(self) -> None:
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False),
        )
        self._precision = "highest"

    def set_float32_matmul_precision(self, value: str) -> None:
        self._precision = value

    def get_float32_matmul_precision(self) -> str:
        return self._precision


@pytest.mark.parametrize(
    ("mode", "enabled", "precision"),
    (("off", False, "highest"), ("on", True, "high")),
)
def test_configure_tf32_is_explicit_and_verified(
    mode: str, enabled: bool, precision: str
) -> None:
    torch = _FakeTorch()
    metadata = configure_tf32(torch, mode)
    assert metadata["tf32"] is enabled
    assert metadata["tf32_mode_requested"] == mode
    assert metadata["tf32_matmul_allowed"] is enabled
    assert metadata["tf32_cudnn_allowed"] is enabled
    assert metadata["float32_matmul_precision"] == precision
    assert metadata["tf32_config_verified"] is True


def test_configure_tf32_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="TF32 mode"):
        configure_tf32(_FakeTorch(), "auto")


def test_verify_tf32_detects_a_runtime_policy_change() -> None:
    torch = _FakeTorch()
    configure_tf32(torch, "on")
    torch.backends.cuda.matmul.allow_tf32 = False
    with pytest.raises(RuntimeError, match="did not take effect"):
        verify_tf32(torch, "on")
