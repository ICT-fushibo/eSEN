from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "example"
    / "analyze_opt3_profiling.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_opt3_profiling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def test_nsys_parser_excludes_cuda_api_summary(tmp_path: Path):
    path = (
        tmp_path
        / "Cu192_300K_20step_whole-step-cg_nsys_node.stats.csv"
    )
    path.write_text(
        "** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):\n"
        '"Time (%)","Total Time (ns)","Instances","Name"\n'
        '"100.0","2000000","20","model_kernel"\n'
        "** CUDA API Summary (cuda_api_sum):\n"
        '"Time (%)","Total Time (ns)","Num Calls","Name"\n'
        '"100.0","9000000","20","cudaGraphLaunch"\n',
        encoding="utf-8",
    )

    raw, families = ANALYZER.parse_nsys_kernel_files(tmp_path)

    assert [row["kernel"] for row in raw] == ["model_kernel"]
    assert raw[0]["time_ms_per_step"] == 0.1
    assert len(families) == 1


def test_nsys_graph_trace_reports_span_active_and_gap(tmp_path: Path):
    path = (
        tmp_path
        / "H2O60_300K_20step_force-eval-cg_nsys_graph.gpu_trace.csv"
    )
    path.write_text(
        '"Start (ns)","Duration (ns)","Name"\n'
        '"0","1000000","graph 1"\n'
        '"2000000","1000000","graph 2"\n',
        encoding="utf-8",
    )

    rows = ANALYZER.parse_nsys_graph_traces(tmp_path)

    assert len(rows) == 1
    assert rows[0]["gpu_span_ms_per_step"] == "0.150000"
    assert rows[0]["gpu_active_ms_per_step"] == "0.100000"
    assert rows[0]["gpu_gap_ms_per_step"] == "0.050000"


def test_noisy_group_requests_new_repeat_numbers_without_overwrite():
    records = [
        {
            "profile_kind": "timing",
            "system": "Cu32",
            "temperature_K": 300,
            "backend": "whole-step-cg",
            "repeat": repeat,
            "seconds_per_step": value,
            "peak_reserved_gib": 1.0,
        }
        for repeat, value in ((1, 1.0), (2, 1.2), (5, 0.9))
    ]

    _, _, reruns, _ = ANALYZER.timing_summary(records, [])

    assert [row["repeat"] for row in reruns] == [6, 7, 8, 9]
