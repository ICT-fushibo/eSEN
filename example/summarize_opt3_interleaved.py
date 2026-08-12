#!/usr/bin/env python3
"""Summarize the fair interleaved Opt2 -> Opt3 formal benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


BACKENDS = (
    "model-cg",
    "fixed-builder-model-cg",
    "force-eval-cg",
    "whole-step-cg",
)
BACKEND_ALIASES = {
    "esen_gpu_resident_model_cg": "model-cg",
    "model-cg": "model-cg",
    "fixed-builder-model-cg": "fixed-builder-model-cg",
    "force-eval-cg": "force-eval-cg",
    "whole-step-cg": "whole-step-cg",
}
TRANSITIONS = (
    ("opt2_to_fixed", "model-cg", "fixed-builder-model-cg"),
    ("fixed_to_force_eval", "fixed-builder-model-cg", "force-eval-cg"),
    ("force_eval_to_whole", "force-eval-cg", "whole-step-cg"),
    ("opt2_to_whole", "model-cg", "whole-step-cg"),
)
CHECKPOINTS = (1, 50, 100, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_暂无数据。_\n"
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def mad(values: Iterable[float]) -> float:
    values = list(values)
    centre = median(values)
    return median(abs(value - centre) for value in values)


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def maximum(records: list[dict[str, Any]], fields: tuple[str, ...]) -> float | None:
    values = [
        value
        for record in records
        for field in fields
        if (value := number(record.get(field))) is not None
    ]
    return max(values) if values else None


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def sci(value: float | None) -> str:
    return "" if value is None else f"{value:.12e}"


def load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in (root / "results").rglob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        backend = BACKEND_ALIASES.get(str(record.get("backend")))
        if backend is None:
            continue
        record["benchmark_backend"] = backend
        record["_path"] = str(path)
        records.append(record)
    return records


def summarize_telemetry(path: Path) -> dict[str, Any]:
    if not path.is_file() or not path.stat().st_size:
        return {}
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {"telemetry_samples": len(rows)}
    normalized_rows = [
        {key.strip(): value.strip() for key, value in row.items() if key}
        for row in rows
    ]
    for key, output in (
        ("clocks.current.sm [MHz]", "sm_clock_mhz_median"),
        ("clocks.current.memory [MHz]", "memory_clock_mhz_median"),
        ("power.draw [W]", "power_w_median"),
        ("temperature.gpu", "temperature_c_max"),
        ("utilization.gpu [%]", "gpu_utilization_pct_median"),
    ):
        values = []
        for row in normalized_rows:
            raw = row.get(key)
            if raw is None:
                continue
            parsed = number(str(raw).split()[0])
            if parsed is not None:
                values.append(parsed)
        if values:
            result[output] = max(values) if output.endswith("_max") else median(values)
    pstates = {
        value
        for row in normalized_rows
        if (value := row.get("pstate"))
    }
    if pstates:
        result["pstates"] = ",".join(sorted(pstates))
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statuses = read_tsv(args.input_dir / "run_status.tsv")
    records = load_records(args.input_dir)
    status_map = {row["run_name"]: row for row in statuses}

    run_rows = []
    for record in records:
        run_name = str(record["run_name"])
        status = status_map.get(run_name, {})
        telemetry = summarize_telemetry(Path(status.get("telemetry", "")))
        row = {
            "backend": record["benchmark_backend"],
            "system": record.get("system"),
            "temperature_K": record.get("temperature_K"),
            "repeat": record.get("repeat"),
            "status": status.get("status", "result_present"),
            "seconds_per_step": record.get("seconds_per_step"),
            "md_wall_time_s": record.get("md_wall_time_s"),
            "process_wall_time_s": record.get("process_wall_time_s"),
            "peak_reserved_gib": record.get("peak_reserved_gib"),
            "initial_force_error_eV_per_A": (
                record.get("initial_eager_force_max_abs_error_eV_per_A")
                if record["benchmark_backend"] != "model-cg"
                else record.get("cg_initial_force_max_abs_error_eV_per_A")
            ),
            "graph_invariants_pass": record.get("graph_invariants_pass"),
            "capture_count": record.get("cuda_graph_capture_count"),
            "production_replays": record.get("cuda_graph_production_replays"),
            "capacity_misses": record.get("cuda_graph_capacity_misses"),
            "graph_hit_rate": record.get("cuda_graph_hit_rate"),
        }
        for step in CHECKPOINTS:
            row[f"energy_step_{step}_eV"] = record.get(f"energy_step_{step}_eV")
            total_error = number(record.get(f"energy_abs_error_step_{step}_eV"))
            row[f"energy_abs_error_step_{step}_eV"] = total_error
            atoms = int(record.get("atoms", 0) or 0)
            row[f"energy_abs_error_step_{step}_eV_per_atom"] = (
                None if total_error is None or not atoms else total_error / atoms
            )
        row.update(telemetry)
        run_rows.append(row)
    write_tsv(args.output_dir / "opt3_interleaved_runs.tsv", run_rows)

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["system"]),
                int(float(record["temperature_K"])),
                str(record["benchmark_backend"]),
            )
        ].append(record)
    status_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in statuses:
        status_groups[
            (row["system"], int(float(row["temperature_K"])), row["backend"])
        ].append(row)

    summary_rows = []
    for key in sorted(set(grouped) | set(status_groups)):
        system, temperature, backend = key
        values = grouped.get(key, [])
        timing = [
            value
            for record in values
            if (value := number(record.get("seconds_per_step"))) is not None
        ]
        statuses_here = status_groups.get(key, [])
        row: dict[str, Any] = {
            "system": system,
            "temperature_K": temperature,
            "backend": backend,
            "attempted": len(statuses_here),
            "completed": len(values),
            "success": sum(row["status"] == "success" for row in statuses_here),
            "validation_failed": sum(
                row["status"] == "validation_failed" for row in statuses_here
            ),
            "oom": sum(row["status"] == "oom" for row in statuses_here),
            "capacity_overflow": sum(
                row["status"] == "capacity_overflow" for row in statuses_here
            ),
            "missing_reference": sum(
                row["status"] == "missing_reference" for row in statuses_here
            ),
            "error": sum(row["status"] == "error" for row in statuses_here),
            "seconds_per_step_median": fmt(median(timing) if timing else None, 9),
            "seconds_per_step_mad": fmt(mad(timing) if timing else None, 9),
            "relative_mad": fmt(
                mad(timing) / median(timing) if timing and median(timing) else None,
                6,
            ),
            "peak_reserved_gib_max": fmt(
                maximum(values, ("peak_reserved_gib",)), 6
            ),
            "initial_force_error_eV_per_A_max": sci(
                maximum(
                    values,
                    (
                        "initial_eager_force_max_abs_error_eV_per_A",
                        "cg_initial_force_max_abs_error_eV_per_A",
                    ),
                )
            ),
            "capture_count_values": ",".join(
                sorted({str(record.get("cuda_graph_capture_count")) for record in values})
            ),
            "production_replays_values": ",".join(
                sorted(
                    {str(record.get("cuda_graph_production_replays")) for record in values}
                )
            ),
            "capacity_misses_max": fmt(
                maximum(values, ("cuda_graph_capacity_misses",)), 0
            ),
            "graph_hit_rate_min": fmt(
                min(
                    value
                    for record in values
                    if (value := number(record.get("cuda_graph_hit_rate"))) is not None
                )
                if any(number(record.get("cuda_graph_hit_rate")) is not None for record in values)
                else None,
                6,
            ),
        }
        for step in CHECKPOINTS:
            row[f"energy_step_{step}_eV_median"] = fmt(
                median(
                    value
                    for record in values
                    if (value := number(record.get(f"energy_step_{step}_eV"))) is not None
                )
                if any(number(record.get(f"energy_step_{step}_eV")) is not None for record in values)
                else None,
                12,
            )
            total_error = maximum(values, (f"energy_abs_error_step_{step}_eV",))
            atoms = int(values[0].get("atoms", 0) or 0) if values else 0
            row[f"energy_error_step_{step}_eV_max"] = sci(total_error)
            row[f"energy_error_step_{step}_eV_per_atom_max"] = sci(
                None if total_error is None or not atoms else total_error / atoms
            )
        summary_rows.append(row)
    write_tsv(args.output_dir / "opt3_interleaved_summary.tsv", summary_rows)

    medians = {
        (row["system"], int(row["temperature_K"]), row["backend"]): number(
            row["seconds_per_step_median"]
        )
        for row in summary_rows
    }
    speedup_rows = []
    for system, temperature in sorted({(key[0], key[1]) for key in medians}):
        for label, before, after in TRANSITIONS:
            before_value = medians.get((system, temperature, before))
            after_value = medians.get((system, temperature, after))
            if before_value is None or after_value is None:
                continue
            speedup_rows.append(
                {
                    "system": system,
                    "temperature_K": temperature,
                    "transition": label,
                    "before_backend": before,
                    "after_backend": after,
                    "before_s_per_step": f"{before_value:.9f}",
                    "after_s_per_step": f"{after_value:.9f}",
                    "speedup_before_over_after": f"{before_value / after_value:.6f}",
                    "after_change_percent": f"{(after_value / before_value - 1) * 100:.3f}",
                }
            )
    write_tsv(args.output_dir / "opt3_interleaved_speedups.tsv", speedup_rows)

    failure_counts: dict[str, int] = defaultdict(int)
    for row in statuses:
        if row["status"] != "success":
            failure_counts[row["status"]] += 1
    report_rows = [
        {
            "system": row["system"],
            "T(K)": row["temperature_K"],
            "backend": row["backend"],
            "runs": f"{row['completed']}/{row['attempted']}",
            "s/step median": row["seconds_per_step_median"],
            "MAD": row["seconds_per_step_mad"],
            "peak GiB": row["peak_reserved_gib_max"],
            "force err max": row["initial_force_error_eV_per_A_max"],
            "E1/atom max": row["energy_error_step_1_eV_per_atom_max"],
            "E50/atom max": row["energy_error_step_50_eV_per_atom_max"],
            "E100/atom max": row["energy_error_step_100_eV_per_atom_max"],
            "E1000/atom max": row["energy_error_step_1000_eV_per_atom_max"],
        }
        for row in summary_rows
    ]
    report_speedups = [
        {
            "system": row["system"],
            "T(K)": row["temperature_K"],
            "transition": row["transition"],
            "speedup": row["speedup_before_over_after"],
            "after change": row["after_change_percent"] + "%",
        }
        for row in speedup_rows
    ]
    report = [
        "# eSEN Opt3 正式交错补测",
        "",
        "## 完成与失败",
        "",
        f"- JSON 结果：{len(records)}。",
        f"- 调度记录：{len(statuses)}。",
        "- 非成功状态："
        + (", ".join(f"{key}={value}" for key, value in sorted(failure_counts.items())) or "无"),
        "",
        "## 性能与数值",
        "",
        markdown_table(report_rows),
        "## 成对加速比",
        "",
        markdown_table(report_speedups),
        "## 判读规则",
        "",
        "- 正式性能使用无 profiler 的 `seconds_per_step` 中位数；MAD 用于判断稳定性。",
        "- `speedup > 1` 表示后一个后端更快；`after change < 0` 表示耗时下降。",
        "- 1/50 步采用 `<1e-5 eV/atom` 工程阈值；100/1000 步只报告。",
        "- 初始最大力误差采用 `<2e-4 eV/Å`。",
        "- 还需检查 capture=1、production replay=steps+1、capacity miss=0、hit rate=1。",
        "",
    ]
    (args.output_dir / "opt3_interleaved_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Report: {args.output_dir / 'opt3_interleaved_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
