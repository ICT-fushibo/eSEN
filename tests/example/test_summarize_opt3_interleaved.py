from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "example"
    / "summarize_opt3_interleaved.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_opt3_interleaved", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_telemetry_parser_handles_nvidia_smi_headers(tmp_path: Path):
    path = tmp_path / "telemetry.csv"
    path.write_text(
        "timestamp, pstate, clocks.current.sm [MHz], "
        "clocks.current.memory [MHz], power.draw [W], temperature.gpu, "
        "utilization.gpu [%], memory.used [MiB]\n"
        "2026/08/12 10:00:00.000, P0, 1980 MHz, 2619 MHz, "
        "410.0 W, 58, 99 %, 1234 MiB\n"
        "2026/08/12 10:00:00.200, P0, 1980 MHz, 2619 MHz, "
        "420.0 W, 59, 100 %, 1234 MiB\n",
        encoding="utf-8",
    )

    result = SUMMARY.summarize_telemetry(path)

    assert result["telemetry_samples"] == 2
    assert result["pstates"] == "P0"
    assert result["sm_clock_mhz_median"] == 1980
    assert result["memory_clock_mhz_median"] == 2619
    assert result["power_w_median"] == 415
    assert result["temperature_c_max"] == 59
    assert result["gpu_utilization_pct_median"] == 99.5


def test_mad_uses_median_absolute_deviation():
    assert abs(SUMMARY.mad([1.0, 1.1, 10.0]) - 0.1) < 1e-12
