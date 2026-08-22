#!/usr/bin/env python3
"""Analyze Opt4 Nsight Systems graph/node confirmation traces.

The analyzer reads the SQLite exports instead of parsing localized, version-
dependent ``nsys stats`` banners.  Graph-mode traces provide replay duration;
node-mode traces provide kernel counts and family time.  Profiler measurements
are diagnostic only and are never mixed into seconds/step performance results.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import statistics
from typing import Any, Iterable


@dataclass(frozen=True)
class ProfileKey:
    scope: str
    variant: str
    system: str
    temperature_k: str
    mode: str


def kernel_family(name: str) -> str:
    lowered = name.lower()
    rules = (
        ("so2_block", r"so2_block|block_gate_bridge"),
        ("so2_gate_bridge", r"so2_gate_bridge"),
        ("so2_prepare", r"so2_prepare"),
        ("so2_epilogue", r"so2_epilogue"),
        ("rmsnorm", r"rmsnorm"),
        ("gemm_bmm", r"gemm|sgemm|matmul|bmm|cutlass|xmma"),
        ("topk_sort_select", r"topk|sort|radix|select"),
        ("scatter_reduce", r"scatter|indexadd|index_add|reduce"),
        ("gather_index", r"gather|indexselect|index_select|index_elementwise"),
        ("elementwise", r"elementwise|vectorized|pointwise"),
        ("copy_fill", r"copy|memcpy|fill|zero"),
    )
    for family, pattern in rules:
        if re.search(pattern, lowered):
            return family
    if lowered.startswith("_") or "triton" in lowered:
        return "triton_other"
    return "other"


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _graph_durations(sqlite_path: Path) -> list[int]:
    with closing(sqlite3.connect(sqlite_path)) as connection:
        if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
            return []
        return [
            int(end) - int(start)
            for start, end in connection.execute(
                "SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                "WHERE end >= start ORDER BY start"
            )
        ]


def _kernel_rows(sqlite_path: Path) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(sqlite_path)) as connection:
        if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
            return []
        rows = connection.execute(
            "SELECT strings.value, COUNT(*), SUM(kernels.end - kernels.start) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels "
            "JOIN StringIds AS strings ON strings.id = kernels.demangledName "
            "WHERE kernels.end >= kernels.start GROUP BY strings.value"
        )
        return [
            {"kernel": str(name), "instances": int(count), "total_ns": int(total)}
            for name, count, total in rows
        ]


def _report_stem(report: str) -> str:
    name = PurePosixPath(report.replace("\\", "/")).name
    suffix = ".nsys-rep"
    return name[: -len(suffix)] if name.endswith(suffix) else Path(name).stem


def _find_sqlite(root: Path, stem: str) -> Path | None:
    candidates = (
        root / "sqlite" / f"{stem}.sqlite",
        root / "reports" / f"{stem}.sqlite",
        root / f"{stem}.sqlite",
    )
    return next((path for path in candidates if path.is_file()), None)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_status(root: Path) -> list[dict[str, str]]:
    path = root / "profile_status.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"profile status not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    # Resume runs append a replacement status after an earlier failure.  The
    # final status for a logical trace is authoritative.
    latest: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["scope"], row["variant"], row["system"],
            row["temperature_K"], row["mode"],
        )
        latest[key] = row
    return list(latest.values())


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _safe_ratio(base: float, candidate: float) -> float | None:
    return base / candidate if base > 0.0 and candidate > 0.0 else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def analyze(root: Path, output_dir: Path, base_stage: str, candidate_stage: str) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_rows = _read_status(root)
    trace_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    exact_kernel_rows: dict[ProfileKey, list[dict[str, Any]]] = {}
    failures: list[str] = []

    for status in status_rows:
        key = ProfileKey(
            scope=status["scope"],
            variant=status["variant"],
            system=status["system"],
            temperature_k=status["temperature_K"],
            mode=status["mode"],
        )
        stem = _report_stem(status.get("report", ""))
        if status.get("status") != "success":
            failures.append(f"{key}: profiler status={status.get('status')}")
            continue
        sqlite_path = _find_sqlite(root, stem)
        if sqlite_path is None:
            failures.append(f"{key}: SQLite export missing for {stem}")
            continue
        result = _load_json(root / "results" / f"{stem}.json")
        if not result:
            failures.append(f"{key}: benchmark JSON missing for {stem}")

        replays = int(result.get("cuda_graph_production_replays", 0) or 0)
        stage = str(result.get("kernel_fusion_stage") or (
            base_stage if key.variant == "base" else candidate_stage
        ))
        graph_durations = _graph_durations(sqlite_path)
        kernels = _kernel_rows(sqlite_path)
        exact_kernel_rows[key] = kernels
        kernel_instances = sum(int(row["instances"]) for row in kernels)
        kernel_total_ns = sum(int(row["total_ns"]) for row in kernels)
        if key.mode == "graph" and replays and len(graph_durations) != replays:
            failures.append(
                f"{key}: graph traces={len(graph_durations)} but production replays={replays}"
            )
        if result and result.get("graph_invariants_pass") is not True:
            failures.append(f"{key}: graph invariants did not pass")
        if result and int(result.get("cuda_graph_capacity_misses", 0) or 0) != 0:
            failures.append(f"{key}: CUDA Graph capacity miss recorded")
        require_kf12 = (
            candidate_stage.upper() == "OPT4V3"
            or "so2-block-gemm" in str(result.get("model_fusions", ""))
        )
        if key.variant == "candidate" and result and require_kf12:
            convolution_replacements = int(
                result.get("model_fusion_so2_block_gemm_convolution_replacements", 0) or 0
            )
            linear_replacements = int(
                result.get("model_fusion_so2_block_gemm_linear_replacements", 0) or 0
            )
            if (convolution_replacements, linear_replacements) != (20, 40):
                failures.append(
                    f"{key}: expected KF12 replacement counts 20/40, got "
                    f"{convolution_replacements}/{linear_replacements}"
                )

        trace_rows.append(
            {
                "scope": key.scope,
                "variant": key.variant,
                "stage": stage,
                "system": key.system,
                "temperature_K": key.temperature_k,
                "mode": key.mode,
                "production_replays": replays,
                "graph_trace_instances": len(graph_durations),
                "graph_duration_median_ms": (
                    statistics.median(graph_durations) / 1e6
                    if graph_durations else ""
                ),
                "graph_duration_mean_ms": (
                    statistics.fmean(graph_durations) / 1e6
                    if graph_durations else ""
                ),
                "graph_duration_min_ms": min(graph_durations) / 1e6 if graph_durations else "",
                "graph_duration_max_ms": max(graph_durations) / 1e6 if graph_durations else "",
                "kernel_instances": kernel_instances,
                "kernel_instances_per_replay": kernel_instances / replays if replays else "",
                "kernel_total_ms": kernel_total_ns / 1e6,
                "kernel_ms_per_replay": kernel_total_ns / 1e6 / replays if replays else "",
                "seconds_per_step": result.get("seconds_per_step", ""),
                "peak_reserved_gib": result.get("peak_reserved_gib", ""),
                "edge_capacity": result.get(
                    "neighbor_edge_capacity", result.get("cuda_graph_edge_capacity", "")
                ),
                "uniform_edge_capacity": result.get("neighbor_uniform_edge_capacity", ""),
                "capacity_reduction_vs_uniform": result.get(
                    "neighbor_capacity_reduction_vs_uniform", ""
                ),
                "capacity_policy_effective": result.get(
                    "neighbor_capacity_policy_effective", ""
                ),
                "graph_invariants_pass": result.get("graph_invariants_pass", ""),
                "capacity_misses": result.get("cuda_graph_capacity_misses", ""),
                "so2_block_convolution_replacements": result.get(
                    "model_fusion_so2_block_gemm_convolution_replacements", 0
                ),
                "so2_block_linear_replacements": result.get(
                    "model_fusion_so2_block_gemm_linear_replacements", 0
                ),
                "sqlite": str(sqlite_path),
            }
        )

        if key.mode == "node" and kernels:
            grouped: dict[str, dict[str, float]] = defaultdict(
                lambda: {"instances": 0.0, "total_ns": 0.0}
            )
            for row in kernels:
                family = kernel_family(str(row["kernel"]))
                grouped[family]["instances"] += int(row["instances"])
                grouped[family]["total_ns"] += int(row["total_ns"])
            for family, values in sorted(grouped.items()):
                family_rows.append(
                    {
                        "scope": key.scope,
                        "variant": key.variant,
                        "stage": stage,
                        "system": key.system,
                        "temperature_K": key.temperature_k,
                        "family": family,
                        "production_replays": replays,
                        "instances": int(values["instances"]),
                        "instances_per_replay": values["instances"] / replays if replays else "",
                        "total_time_ms": values["total_ns"] / 1e6,
                        "time_ms_per_replay": values["total_ns"] / 1e6 / replays if replays else "",
                        "percent_of_kernel_time": (
                            100.0 * values["total_ns"] / kernel_total_ns
                            if kernel_total_ns else 0.0
                        ),
                    }
                )

    trace_map = {
        (row["scope"], row["system"], row["temperature_K"], row["variant"], row["mode"]): row
        for row in trace_rows
    }
    comparisons: list[dict[str, Any]] = []
    comparison_keys = sorted(
        {
            (row["scope"], row["system"], row["temperature_K"])
            for row in trace_rows
        }
    )
    for scope, system, temperature in comparison_keys:
        base_graph = trace_map.get((scope, system, temperature, "base", "graph"))
        candidate_graph = trace_map.get((scope, system, temperature, "candidate", "graph"))
        base_node = trace_map.get((scope, system, temperature, "base", "node"))
        candidate_node = trace_map.get((scope, system, temperature, "candidate", "node"))
        if not all((base_graph, candidate_graph, base_node, candidate_node)):
            failures.append(
                f"{scope}/{system}/{temperature}: "
                "incomplete base/candidate graph/node quartet"
            )
            continue
        base_graph_ms = base_graph["graph_duration_median_ms"]
        candidate_graph_ms = candidate_graph["graph_duration_median_ms"]
        if base_graph_ms == "" or candidate_graph_ms == "":
            failures.append(f"{scope}/{system}/{temperature}: CUDA graph trace durations missing")
        base_key = ProfileKey(scope, "base", system, temperature, "node")
        candidate_key = ProfileKey(scope, "candidate", system, temperature, "node")
        base_exact = {row["kernel"]: row for row in exact_kernel_rows.get(base_key, [])}
        candidate_exact = {
            row["kernel"]: row for row in exact_kernel_rows.get(candidate_key, [])
        }
        replays = int(base_node["production_replays"] or 0)
        deltas: list[tuple[float, str]] = []
        for name in base_exact.keys() | candidate_exact.keys():
            base_ns = int(base_exact.get(name, {}).get("total_ns", 0))
            candidate_ns = int(candidate_exact.get(name, {}).get("total_ns", 0))
            deltas.append(
                ((base_ns - candidate_ns) / 1e6 / replays if replays else 0.0, name)
            )
        saved = sorted((value for value in deltas if value[0] > 0.0), reverse=True)[:3]
        comparisons.append(
            {
                "scope": scope,
                "system": system,
                "temperature_K": temperature,
                "base_stage": base_graph["stage"],
                "candidate_stage": candidate_graph["stage"],
                "base_graph_median_ms": base_graph_ms,
                "candidate_graph_median_ms": candidate_graph_ms,
                "graph_duration_speedup": (
                    _safe_ratio(float(base_graph_ms), float(candidate_graph_ms))
                    if base_graph_ms != "" and candidate_graph_ms != "" else ""
                ),
                "base_kernel_instances_per_replay": base_node[
                    "kernel_instances_per_replay"
                ],
                "candidate_kernel_instances_per_replay": candidate_node[
                    "kernel_instances_per_replay"
                ],
                "kernel_instances_removed_per_replay": (
                    float(base_node["kernel_instances_per_replay"])
                    - float(candidate_node["kernel_instances_per_replay"])
                    if base_node["kernel_instances_per_replay"] != ""
                    and candidate_node["kernel_instances_per_replay"] != ""
                    else ""
                ),
                "base_kernel_ms_per_replay": base_node["kernel_ms_per_replay"],
                "candidate_kernel_ms_per_replay": candidate_node["kernel_ms_per_replay"],
                "kernel_time_speedup": (
                    _safe_ratio(
                        float(base_node["kernel_ms_per_replay"]),
                        float(candidate_node["kernel_ms_per_replay"]),
                    )
                    if base_node["kernel_ms_per_replay"] != ""
                    and candidate_node["kernel_ms_per_replay"] != ""
                    else ""
                ),
                "candidate_edge_capacity": candidate_graph["edge_capacity"],
                "candidate_uniform_edge_capacity": candidate_graph["uniform_edge_capacity"],
                "candidate_capacity_reduction_vs_uniform": candidate_graph[
                    "capacity_reduction_vs_uniform"
                ],
                "candidate_capacity_policy_effective": candidate_graph[
                    "capacity_policy_effective"
                ],
                "candidate_so2_block_convolution_replacements": candidate_graph[
                    "so2_block_convolution_replacements"
                ],
                "candidate_so2_block_linear_replacements": candidate_graph[
                    "so2_block_linear_replacements"
                ],
                "top_saved_kernel_1": saved[0][1] if len(saved) > 0 else "",
                "top_saved_kernel_1_ms_per_replay": saved[0][0] if len(saved) > 0 else "",
                "top_saved_kernel_2": saved[1][1] if len(saved) > 1 else "",
                "top_saved_kernel_2_ms_per_replay": saved[1][0] if len(saved) > 1 else "",
                "top_saved_kernel_3": saved[2][1] if len(saved) > 2 else "",
                "top_saved_kernel_3_ms_per_replay": saved[2][0] if len(saved) > 2 else "",
            }
        )

    trace_fields = (
        "scope", "variant", "stage", "system", "temperature_K", "mode",
        "production_replays", "graph_trace_instances", "graph_duration_median_ms",
        "graph_duration_mean_ms", "graph_duration_min_ms", "graph_duration_max_ms",
        "kernel_instances", "kernel_instances_per_replay", "kernel_total_ms",
        "kernel_ms_per_replay", "seconds_per_step", "peak_reserved_gib",
        "edge_capacity", "uniform_edge_capacity", "capacity_reduction_vs_uniform",
        "capacity_policy_effective", "graph_invariants_pass", "capacity_misses",
        "so2_block_convolution_replacements", "so2_block_linear_replacements", "sqlite",
    )
    family_fields = (
        "scope", "variant", "stage", "system", "temperature_K", "family",
        "production_replays", "instances", "instances_per_replay", "total_time_ms",
        "time_ms_per_replay", "percent_of_kernel_time",
    )
    comparison_fields = (
        "scope", "system", "temperature_K", "base_stage", "candidate_stage",
        "base_graph_median_ms", "candidate_graph_median_ms", "graph_duration_speedup",
        "base_kernel_instances_per_replay", "candidate_kernel_instances_per_replay",
        "kernel_instances_removed_per_replay", "base_kernel_ms_per_replay",
        "candidate_kernel_ms_per_replay", "kernel_time_speedup",
        "candidate_edge_capacity", "candidate_uniform_edge_capacity",
        "candidate_capacity_reduction_vs_uniform", "candidate_capacity_policy_effective",
        "candidate_so2_block_convolution_replacements",
        "candidate_so2_block_linear_replacements", "top_saved_kernel_1",
        "top_saved_kernel_1_ms_per_replay", "top_saved_kernel_2",
        "top_saved_kernel_2_ms_per_replay", "top_saved_kernel_3",
        "top_saved_kernel_3_ms_per_replay",
    )
    _write_tsv(output_dir / "profile_trace_summary.tsv", trace_rows, trace_fields)
    _write_tsv(output_dir / "kernel_summary.tsv", family_rows, family_fields)
    _write_tsv(output_dir / "profile_comparison.tsv", comparisons, comparison_fields)

    successful_statuses = sum(row.get("status") == "success" for row in status_rows)
    complete = not failures and successful_statuses == len(status_rows) and bool(comparisons)
    payload = {
        "profiling_complete": complete,
        "base_stage": base_stage,
        "candidate_stage": candidate_stage,
        "status_rows": len(status_rows),
        "successful_status_rows": successful_statuses,
        "trace_rows": len(trace_rows),
        "comparisons": comparisons,
        "failures": failures,
        "policy": "NSYS is diagnostic; no profiler time is used as seconds/step",
    }
    (output_dir / "profile_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Opt4 v3 Nsight Systems Profiling",
        "",
        f"- Complete: **{'yes' if complete else 'no'}**.",
        f"- Successful traces: {successful_statuses}/{len(status_rows)}.",
        "- Profiler timings are diagnostic and are not used as benchmark seconds/step.",
        "",
        "## CUDA Graph duration and node count",
        "",
        "| Scope | System | Base graph ms | Candidate graph ms | Graph speedup | "
        "Base nodes/replay | Candidate nodes/replay | Removed/replay | "
        "Capacity policy | Capacity reduction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in comparisons:
        reduction = row["candidate_capacity_reduction_vs_uniform"]
        reduction_text = (
            f"{100.0 * float(reduction):.2f}%" if reduction != "" else ""
        )
        lines.append(
            "| {scope} | {system} | {base} | {candidate} | {speedup}x | "
            "{base_nodes} | {candidate_nodes} | {removed} | {policy} | "
            "{reduction} |".format(
                scope=row["scope"],
                system=row["system"],
                base=_fmt(row["base_graph_median_ms"]),
                candidate=_fmt(row["candidate_graph_median_ms"]),
                speedup=_fmt(row["graph_duration_speedup"]),
                base_nodes=_fmt(row["base_kernel_instances_per_replay"], 1),
                candidate_nodes=_fmt(row["candidate_kernel_instances_per_replay"], 1),
                removed=_fmt(row["kernel_instances_removed_per_replay"], 1),
                policy=row["candidate_capacity_policy_effective"],
                reduction=reduction_text,
            )
        )
    lines.extend(["", "## Largest exact-kernel savings", ""])
    for row in comparisons:
        lines.append(f"### {row['scope']} / {row['system']}")
        lines.append("")
        for rank in range(1, 4):
            name = row[f"top_saved_kernel_{rank}"]
            value = row[f"top_saved_kernel_{rank}_ms_per_replay"]
            if name:
                lines.append(f"{rank}. `{name}`: {_fmt(value, 6)} ms/replay saved")
        lines.append("")
    if failures:
        lines.extend(["## Incomplete items", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    (output_dir / "opt4_v3_profiling.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-stage", default="OPT4V2")
    parser.add_argument("--candidate-stage", default="OPT4V3")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    output = (args.output_dir or root).resolve()
    complete = analyze(root, output, args.base_stage, args.candidate_stage)
    print(f"Opt4 profiling analysis: {output / 'opt4_v3_profiling.md'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
