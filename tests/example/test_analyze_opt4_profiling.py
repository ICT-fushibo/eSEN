from __future__ import annotations

import csv
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys


SCRIPT = Path(__file__).parents[2] / "example" / "analyze_opt4_profiling.py"
SPEC = importlib.util.spec_from_file_location("analyze_opt4_profiling", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


def _write_graph_sqlite(path: Path, durations_ns: list[int]) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "(start INTEGER NOT NULL, end INTEGER NOT NULL)"
        )
        start = 0
        for duration in durations_ns:
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?)",
                (start, start + duration),
            )
            start += duration + 1000
        connection.commit()


def _write_node_sqlite(path: Path, kernels: list[tuple[str, int, int]]) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER NOT NULL, end INTEGER NOT NULL, demangledName INTEGER NOT NULL)"
        )
        start = 0
        for name_id, (name, count, duration_ns) in enumerate(kernels, 1):
            connection.execute("INSERT INTO StringIds VALUES (?, ?)", (name_id, name))
            for _ in range(count):
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                    (start, start + duration_ns, name_id),
                )
                start += duration_ns + 100
        connection.commit()


def _write_profile(
    root: Path,
    *,
    variant: str,
    stage: str,
    mode: str,
    graph_durations: list[int] | None = None,
    kernels: list[tuple[str, int, int]] | None = None,
) -> tuple[str, str]:
    stem = f"Cu512_300K_1step_whole_step_{stage}_nsys_{mode}"
    sqlite_path = root / "sqlite" / f"{stem}.sqlite"
    if graph_durations is not None:
        _write_graph_sqlite(sqlite_path, graph_durations)
    else:
        _write_node_sqlite(sqlite_path, kernels or [])
    result = {
        "kernel_fusion_stage": stage,
        "cuda_graph_production_replays": 2,
        "graph_invariants_pass": True,
        "cuda_graph_capacity_misses": 0,
        "neighbor_edge_capacity": 900,
        "neighbor_uniform_edge_capacity": 1000,
        "neighbor_capacity_reduction_vs_uniform": 0.1,
        "neighbor_capacity_policy_effective": (
            "uniform" if variant == "base" else "atom-safe"
        ),
        "model_fusion_so2_block_gemm_convolution_replacements": (
            0 if variant == "base" else 20
        ),
        "model_fusion_so2_block_gemm_linear_replacements": (
            0 if variant == "base" else 40
        ),
    }
    (root / "results" / f"{stem}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    return stem, f"/tmp/{stem}.nsys-rep"


def test_kernel_family_recognizes_opt4_fusions():
    assert ANALYZER.kernel_family("_so2_block_epilogue_kernel") == "so2_block"
    assert ANALYZER.kernel_family("_so2_gate_bridge_kernel") == "so2_gate_bridge"
    assert ANALYZER.kernel_family("sm90_xmma_gemm_kernel") == "gemm_bmm"


def test_analyze_reads_graph_and_node_sqlite(tmp_path: Path):
    (tmp_path / "sqlite").mkdir()
    (tmp_path / "results").mkdir()
    rows = []
    for variant, stage in (("base", "OPT4V2"), ("candidate", "OPT4V3")):
        stem, report = _write_profile(
            tmp_path,
            variant=variant,
            stage=stage,
            mode="graph",
            graph_durations=(
                [10_000_000, 12_000_000]
                if variant == "base"
                else [8_000_000, 9_000_000]
            ),
        )
        rows.append(("whole-step", variant, "Cu512", "300", "graph", report))
        stem, report = _write_profile(
            tmp_path,
            variant=variant,
            stage=stage,
            mode="node",
            kernels=(
                [("old_elementwise", 8, 1_000_000)]
                if variant == "base"
                else [("_so2_block_epilogue_kernel", 4, 1_000_000)]
            ),
        )
        rows.append(("whole-step", variant, "Cu512", "300", "node", report))

    with (tmp_path / "profile_status.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("scope", "variant", "system", "temperature_K", "mode", "status", "exit_code", "report")
        )
        for scope, variant, system, temperature, mode, report in rows:
            writer.writerow((scope, variant, system, temperature, mode, "success", 0, report))

    assert ANALYZER.analyze(tmp_path, tmp_path, "OPT4V2", "OPT4V3")
    payload = json.loads((tmp_path / "profile_analysis.json").read_text())
    assert payload["profiling_complete"] is True
    comparison = payload["comparisons"][0]
    assert comparison["graph_duration_speedup"] == 11.0 / 8.5
    assert comparison["kernel_instances_removed_per_replay"] == 2.0
    assert comparison["candidate_capacity_policy_effective"] == "atom-safe"
    assert comparison["candidate_so2_block_convolution_replacements"] == 20
