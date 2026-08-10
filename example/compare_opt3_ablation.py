#!/usr/bin/env python3
"""Create the Opt3 fixed-builder and whole-step CUDA Graph ablation report."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

from compare_md_backends import SYSTEM_ORDER, load_records


Key = tuple[str, float]


def load_status(directory: Path) -> dict[Key, list[dict[str, str]]]:
    groups: dict[Key, list[dict[str, str]]] = defaultdict(list)
    path = directory / "run_status.tsv"
    if not path.is_file():
        return groups
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            groups[(row["system"], float(row["temperature_K"]))].append(row)
    return groups


def median_value(records, field):
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return median(values) if values else None


def maximum_value(records, field):
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return max(values) if values else None


def minimum_value(records, field):
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return min(values) if values else None


def ratio(left, right):
    return None if left is None or right in (None, 0) else left / right


def fmt(value, digits=6):
    return "" if value is None else f"{value:.{digits}f}"


def status_count(rows, status):
    return sum(row.get("status") == status for row in rows)


def median_process_wall(rows):
    values = [
        float(row["process_wall_time_s"])
        for row in rows
        if row["status"] in {"success", "validation_failed"}
    ]
    return median(values) if values else None


def _numeric_values(rows: list[dict[str, object]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def _write_decision_report(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fixed_attempts: int,
    whole_attempts: int,
) -> None:
    """Summarize Opt3 signals without selecting a fusion before profiling."""

    fixed_effects = _numeric_values(rows, "fixed_builder_effect_vs_opt2")
    capture_gains = _numeric_values(
        rows, "expanded_capture_speedup_fixed_to_whole"
    )
    opt3_gains = _numeric_values(rows, "opt3_incremental_speedup_vs_opt2")
    small_opt3_gains = [
        float(row["opt3_incremental_speedup_vs_opt2"])
        for row in rows
        if row.get("system") in {"Cu32", "Cu64", "H2O32", "H2O60"}
        and row.get("opt3_incremental_speedup_vs_opt2") not in (None, "")
    ]
    large_opt3_gains = [
        float(row["opt3_incremental_speedup_vs_opt2"])
        for row in rows
        if row.get("system")
        in {"Cu192", "Cu512", "Cu1024", "H2O192", "H2O512", "H2O1024"}
        and row.get("opt3_incremental_speedup_vs_opt2") not in (None, "")
    ]
    fixed_completed = sum(int(row.get("fixed_completed", 0)) for row in rows)
    whole_completed = sum(int(row.get("whole_completed", 0)) for row in rows)
    fixed_oom = sum(int(row.get("fixed_oom", 0)) for row in rows)
    whole_oom = sum(int(row.get("whole_oom", 0)) for row in rows)
    fixed_overflow = sum(
        int(row.get("fixed_capacity_overflow", 0)) for row in rows
    )
    whole_overflow = sum(
        int(row.get("whole_capacity_overflow", 0)) for row in rows
    )
    fixed_missing_reference = sum(
        int(row.get("fixed_missing_reference", 0)) for row in rows
    )
    whole_missing_reference = sum(
        int(row.get("whole_missing_reference", 0)) for row in rows
    )

    lines = [
        "# Opt3 阶段一结果判读",
        "",
        "本报告只整理 Opt3 并给出 profiling 优先级；不会在 profiling 前选定或实现 KF1。",
        "",
        "## 数据覆盖",
        "",
        f"- fixed-builder-model-CG：完成 {fixed_completed}/{fixed_attempts} 次。",
        f"- whole-step-CG：完成 {whole_completed}/{whole_attempts} 次。",
        f"- OOM：fixed={fixed_oom}，whole={whole_oom}。",
        f"- 容量溢出：fixed={fixed_overflow}，whole={whole_overflow}。",
        "- 缺失 baseline reference："
        f"fixed={fixed_missing_reference}，whole={whole_missing_reference}。",
        "",
        "## 已观察到的信号",
        "",
    ]
    if fixed_effects:
        value = median(fixed_effects)
        lines.append(
            "- Opt2 → fixed-builder-model-CG 的中位速度比 "
            f"为 {value:.4f}×（大于 1 表示 fixed-builder 更快）。"
        )
        if value < 0.95:
            lines.append(
                "  - fixed-builder 带来超过约 5% 的中位退化；profiling 应优先拆分距离计算、mask、top-k 与 padding。"
            )
    else:
        lines.append("- 缺少 Opt2 与 fixed-builder 的可比完成记录。")
    if capture_gains:
        value = median(capture_gains)
        lines.append(
            "- fixed-builder-model-CG → whole-step-CG 的中位加速为 "
            f"{value:.4f}×。"
        )
        if value <= 1.03:
            lines.append(
                "  - 扩大捕获范围的收益不超过约 3%；profiling 应重点检查 Wigner、SO2、归一化、激活与保守力 backward。"
            )
    else:
        lines.append("- 缺少 fixed-builder 与 whole-step 的可比完成记录。")
    if opt3_gains:
        lines.append(
            "- Opt2 → whole-step-CG 的中位最终增量为 "
            f"{median(opt3_gains):.4f}×。"
        )
    if small_opt3_gains and large_opt3_gains:
        small_gain = median(small_opt3_gains)
        large_gain = median(large_opt3_gains)
        lines.append(
            "- Opt2 → Opt3：小体系中位加速 "
            f"{small_gain:.4f}×，大体系中位加速 {large_gain:.4f}×。"
        )
        if small_gain >= 1.05 and large_gain <= 1.02:
            lines.append(
                "  - 收益主要集中在小体系；profiling 应优先检查 Edgewise/Wigner/SO2 等大计算 kernel。"
            )
    if whole_oom:
        lines.append(
            "- whole-step 出现 OOM；profiling 必须同时记录 CUDA Graph 私有池增量，KF1 候选优先考虑消除大中间张量。"
        )
    if fixed_overflow or whole_overflow:
        lines.append(
            "- 存在邻居容量溢出；该配置不能进入性能聚合，应先调整 probe/capacity 参数后重测。"
        )

    representative_configs = {
        ("Cu32", "300"),
        ("Cu192", "300"),
        ("H2O32", "300"),
        ("H2O192", "300"),
    }
    comparable_configs = {
        (str(row.get("system")), str(row.get("temperature_K")))
        for row in rows
        if int(row.get("fixed_completed", 0)) > 0
        and int(row.get("whole_completed", 0)) > 0
    }
    representatives_ready = representative_configs <= comparable_configs
    ready = bool(fixed_effects and capture_gains and representatives_ready)
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            (
                "- 结果已足够支持进入代表体系 profiling；先测 Cu32、Cu192、H2O32、H2O192（300 K），再依据 kernel 时间选择 KF1。"
                if ready
                else "- 当前代表体系的可比结果不足；先补齐 Cu32、Cu192、H2O32、H2O192（300 K）的两个 Opt3 后端，再进入 profiling。"
            ),
            "- 在 Nsight 数据出来前，不实现 Triton kernel，也不把任何候选融合计入 KF1。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def completed_records(records, status_rows):
    """Exclude OOM/error/overflow records from performance aggregation."""

    if not status_rows:
        return records
    completed_names = {
        row["run_name"]
        for row in status_rows
        if row["status"] in {"success", "validation_failed"}
    }
    return [
        record
        for record in records
        if str(record.get("run_name", "")) in completed_names
    ]


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("No records.\n", encoding="utf-8")
        return
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(key, "")) for key in headers) + " |"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--opt1-dir", type=Path, required=True)
    parser.add_argument("--opt2-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--whole-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    datasets = {
        "baseline": load_records(
            args.baseline_dir, "esen_ocpcalculator_eager"
        ),
        "opt1": load_records(args.opt1_dir, "esen_gpu_resident_eager"),
        "opt2": load_records(args.opt2_dir, "esen_gpu_resident_model_cg"),
        "fixed": load_records(
            args.fixed_dir, "esen_gpu_resident_fixed_builder_model_cg"
        ),
        "whole": load_records(
            args.whole_dir, "esen_gpu_resident_whole_step_cg"
        ),
    }
    statuses = {
        "baseline": load_status(args.baseline_dir),
        "opt1": load_status(args.opt1_dir),
        "opt2": load_status(args.opt2_dir),
        "fixed": load_status(args.fixed_dir),
        "whole": load_status(args.whole_dir),
    }
    keys = set()
    for groups in (*datasets.values(), *statuses.values()):
        keys.update(groups)
    if not keys:
        raise SystemExit("No Opt3 ablation records found")

    summary_rows: list[dict[str, object]] = []
    for key in sorted(
        keys, key=lambda item: (SYSTEM_ORDER.get(item[0], 999), item[1])
    ):
        system, temperature = key
        seconds = {
            name: median_value(
                completed_records(
                    groups.get(key, []), statuses[name].get(key, [])
                ),
                "seconds_per_step",
            )
            for name, groups in datasets.items()
        }
        row: dict[str, object] = {
            "system": system,
            "temperature_K": f"{temperature:g}",
            "baseline_s_per_step": fmt(seconds["baseline"], 9),
            "opt1_s_per_step": fmt(seconds["opt1"], 9),
            "opt2_s_per_step": fmt(seconds["opt2"], 9),
            "fixed_builder_model_cg_s_per_step": fmt(seconds["fixed"], 9),
            "whole_step_cg_s_per_step": fmt(seconds["whole"], 9),
            "fixed_builder_effect_vs_opt2": fmt(
                ratio(seconds["opt2"], seconds["fixed"]), 4
            ),
            "expanded_capture_speedup_fixed_to_whole": fmt(
                ratio(seconds["fixed"], seconds["whole"]), 4
            ),
            "opt3_incremental_speedup_vs_opt2": fmt(
                ratio(seconds["opt2"], seconds["whole"]), 4
            ),
            "opt3_total_speedup_vs_opt1": fmt(
                ratio(seconds["opt1"], seconds["whole"]), 4
            ),
            "opt3_total_speedup_vs_baseline": fmt(
                ratio(seconds["baseline"], seconds["whole"]), 4
            ),
        }
        for name in ("fixed", "whole"):
            status_rows = statuses[name].get(key, [])
            records = completed_records(
                datasets[name].get(key, []), status_rows
            )
            row[f"{name}_completed"] = len(records)
            row[f"{name}_success"] = status_count(status_rows, "success")
            row[f"{name}_validation_failed"] = status_count(
                status_rows, "validation_failed"
            )
            row[f"{name}_oom"] = status_count(status_rows, "oom")
            row[f"{name}_capacity_overflow"] = status_count(
                status_rows, "capacity_overflow"
            )
            row[f"{name}_error"] = status_count(status_rows, "error")
            row[f"{name}_process_wall_time_s"] = fmt(
                median_process_wall(status_rows), 6
            )
            row[f"{name}_setup_wall_time_s"] = fmt(
                median_value(records, "setup_wall_time_s"), 6
            )
            row[f"{name}_capture_wall_time_s"] = fmt(
                median_value(records, "cuda_graph_capture_wall_time_s"), 6
            )
            row[f"{name}_peak_allocated_gib"] = fmt(
                maximum_value(records, "peak_allocated_gib"), 6
            )
            row[f"{name}_peak_reserved_gib"] = fmt(
                maximum_value(records, "peak_reserved_gib"), 6
            )
            row[f"{name}_capture_device_delta_gib"] = fmt(
                maximum_value(records, "capture_total_device_used_delta_gib"),
                6,
            )
            row[f"{name}_graph_pool_device_delta_gib"] = fmt(
                maximum_value(
                    records, "cuda_graph_capture_device_used_delta_gib"
                ),
                6,
            )
            row[f"{name}_graph_pool_allocated_delta_gib"] = fmt(
                maximum_value(
                    records, "cuda_graph_capture_allocated_delta_gib"
                ),
                6,
            )
            row[f"{name}_graph_pool_reserved_delta_gib"] = fmt(
                maximum_value(
                    records, "cuda_graph_capture_reserved_delta_gib"
                ),
                6,
            )
            row[f"{name}_edge_capacity"] = fmt(
                maximum_value(records, "cuda_graph_edge_capacity"), 0
            )
            row[f"{name}_candidate_universe_size"] = fmt(
                maximum_value(
                    records, "fixed_builder_candidate_universe_size"
                ),
                0,
            )
            row[f"{name}_candidates_per_atom"] = fmt(
                maximum_value(records, "fixed_builder_candidates_per_atom"),
                0,
            )
            row[f"{name}_probe_max_neighbors_per_atom"] = fmt(
                maximum_value(records, "probe_max_neighbors_per_atom"), 0
            )
            row[f"{name}_real_edges_min"] = fmt(
                minimum_value(records, "cuda_graph_min_real_edges"), 0
            )
            row[f"{name}_real_edges_max"] = fmt(
                maximum_value(records, "cuda_graph_max_real_edges"), 0
            )
            row[f"{name}_padding_fraction_max"] = fmt(
                maximum_value(records, "cuda_graph_max_padding_fraction"), 6
            )
            row[f"{name}_raw_neighbors_max"] = fmt(
                maximum_value(records, "fixed_builder_max_raw_neighbors"), 0
            )
            row[f"{name}_included_neighbors_max"] = fmt(
                maximum_value(
                    records, "fixed_builder_max_included_neighbors"
                ),
                0,
            )
            row[f"{name}_first_overflow_step_min"] = fmt(
                minimum_value(records, "fixed_builder_first_overflow_step"),
                0,
            )
            row[f"{name}_capture_count_max"] = fmt(
                maximum_value(records, "cuda_graph_capture_count"), 0
            )
            row[f"{name}_production_replays_min"] = fmt(
                minimum_value(records, "cuda_graph_production_replays"), 0
            )
            row[f"{name}_capacity_misses_max"] = fmt(
                maximum_value(records, "cuda_graph_capacity_misses"), 0
            )
            row[f"{name}_graph_hit_rate_min"] = fmt(
                minimum_value(records, "cuda_graph_hit_rate"), 6
            )
            missing_status_rows = sum(
                status_row.get("baseline_reference_status") == "missing"
                for status_row in status_rows
            )
            row[f"{name}_missing_reference"] = (
                missing_status_rows
                if any(
                    "baseline_reference_status" in status_row
                    for status_row in status_rows
                )
                else sum(
                    record.get("baseline_reference_status") == "missing"
                    for record in records
                )
            )
            row[f"{name}_initial_force_error_max"] = (
                ""
                if (
                    value := maximum_value(
                        records,
                        "initial_eager_force_max_abs_error_eV_per_A",
                    )
                )
                is None
                else f"{value:.12e}"
            )
            for step in (1, 50, 100, 1000):
                total_error = maximum_value(
                    records, f"energy_abs_error_step_{step}_eV"
                )
                atom_error = maximum_value(
                    records, f"energy_abs_error_step_{step}_eV_per_atom"
                )
                row[f"{name}_energy_error_step{step}_eV_max"] = (
                    "" if total_error is None else f"{total_error:.12e}"
                )
                row[f"{name}_energy_error_step{step}_eV_per_atom_max"] = (
                    "" if atom_error is None else f"{atom_error:.12e}"
                )
        summary_rows.append(row)

    detailed_rows: list[dict[str, object]] = []
    for name in ("fixed", "whole"):
        for key, records in datasets[name].items():
            status_by_run = {
                row["run_name"]: row for row in statuses[name].get(key, [])
            }
            for record in sorted(
                records, key=lambda item: int(item.get("repeat", 1))
            ):
                run_status = status_by_run.get(
                    str(record.get("run_name", ""))
                )
                row = {
                    "backend": name,
                    "system": key[0],
                    "temperature_K": f"{key[1]:g}",
                    "repeat": record.get("repeat"),
                    "seconds_per_step": record.get("seconds_per_step"),
                    "md_wall_time_s": record.get("md_wall_time_s"),
                    "setup_wall_time_s": record.get("setup_wall_time_s"),
                    "capture_wall_time_s": record.get(
                        "cuda_graph_capture_wall_time_s"
                    ),
                    "process_wall_time_s": (
                        run_status.get("process_wall_time_s")
                        if run_status is not None
                        else record.get("process_wall_time_s")
                    ),
                    "peak_allocated_gib": record.get("peak_allocated_gib"),
                    "peak_reserved_gib": record.get("peak_reserved_gib"),
                    "capture_device_delta_gib": record.get(
                        "capture_total_device_used_delta_gib"
                    ),
                    "graph_pool_device_delta_gib": record.get(
                        "cuda_graph_capture_device_used_delta_gib"
                    ),
                    "graph_pool_allocated_delta_gib": record.get(
                        "cuda_graph_capture_allocated_delta_gib"
                    ),
                    "graph_pool_reserved_delta_gib": record.get(
                        "cuda_graph_capture_reserved_delta_gib"
                    ),
                    "edge_capacity": record.get("cuda_graph_edge_capacity"),
                    "candidate_universe_size": record.get(
                        "fixed_builder_candidate_universe_size"
                    ),
                    "candidates_per_atom": record.get(
                        "fixed_builder_candidates_per_atom"
                    ),
                    "probe_max_neighbors_per_atom": record.get(
                        "probe_max_neighbors_per_atom"
                    ),
                    "real_edges_min": record.get(
                        "cuda_graph_min_real_edges"
                    ),
                    "real_edges_max": record.get(
                        "cuda_graph_max_real_edges"
                    ),
                    "padding_fraction_max": record.get(
                        "cuda_graph_max_padding_fraction"
                    ),
                    "raw_neighbors_max": record.get(
                        "fixed_builder_max_raw_neighbors"
                    ),
                    "included_neighbors_max": record.get(
                        "fixed_builder_max_included_neighbors"
                    ),
                    "first_overflow_step": record.get(
                        "fixed_builder_first_overflow_step"
                    ),
                    "capture_count": record.get("cuda_graph_capture_count"),
                    "production_replays": record.get(
                        "cuda_graph_production_replays"
                    ),
                    "capacity_misses": record.get(
                        "cuda_graph_capacity_misses"
                    ),
                    "graph_hit_rate": record.get("cuda_graph_hit_rate"),
                    "initial_force_error_eV_per_A": record.get(
                        "initial_eager_force_max_abs_error_eV_per_A"
                    ),
                    "engineering_validation_status": record.get(
                        "engineering_validation_status"
                    ),
                    "capacity_overflow": record.get("capacity_overflow"),
                }
                for step in (1, 50, 100, 1000):
                    row[f"energy_error_step{step}_eV"] = record.get(
                        f"energy_abs_error_step_{step}_eV"
                    )
                    row[f"energy_error_step{step}_eV_per_atom"] = record.get(
                        f"energy_abs_error_step_{step}_eV_per_atom"
                    )
                detailed_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("opt3_ablation.tsv", summary_rows),
        ("opt3_runs.tsv", detailed_rows),
    ):
        write_tsv(args.output_dir / filename, rows)
    write_table(args.output_dir / "opt3_ablation.md", summary_rows)
    write_table(args.output_dir / "opt3_runs.md", detailed_rows)
    fixed_attempts = sum(len(rows) for rows in statuses["fixed"].values())
    whole_attempts = sum(len(rows) for rows in statuses["whole"].values())
    _write_decision_report(
        args.output_dir / "opt3_phase1_decision.md",
        summary_rows,
        fixed_attempts=fixed_attempts,
        whole_attempts=whole_attempts,
    )
    print((args.output_dir / "opt3_ablation.md").read_text(encoding="utf-8"))
    print(
        (args.output_dir / "opt3_phase1_decision.md").read_text(
            encoding="utf-8"
        )
    )
    print(f"Detailed runs: {args.output_dir / 'opt3_runs.tsv'}")


if __name__ == "__main__":
    main()
