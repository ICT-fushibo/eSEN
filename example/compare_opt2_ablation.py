#!/usr/bin/env python3
"""Compare opt1, opt2 static-eager control, and opt2 model CUDA Graph MD."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from compare_md_backends import SYSTEM_ORDER, format_float, load_records


BackendKey = tuple[str, float]


def load_status_rows(directory: Path) -> dict[BackendKey, list[dict[str, str]]]:
    groups: dict[BackendKey, list[dict[str, str]]] = defaultdict(list)
    path = directory / "run_status.tsv"
    if not path.is_file():
        return groups
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            groups[(row["system"], float(row["temperature_K"]))].append(row)
    return groups


def median_field(records: list[dict[str, object]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return median(values) if values else None


def max_field(records: list[dict[str, object]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return max(values) if values else None


def process_median(rows: list[dict[str, str]]) -> float | None:
    values = [
        float(row["process_wall_time_s"])
        for row in rows
        if row["status"] in {"success", "validation_failed"}
    ]
    return median(values) if values else None


def status_count(rows: list[dict[str, str]], status: str) -> int:
    return sum(row["status"] == status for row in rows)


def speedup(slower: float | None, faster: float | None) -> float | None:
    if slower is None or faster is None or faster == 0:
        return None
    return slower / faster


def validation_passed(record: dict[str, object]) -> bool:
    if record.get("numerical_validation_pass") is not None:
        return record.get("numerical_validation_pass") is True
    return record.get("energy_validation_pass") is True


def matched_checkpoint_difference(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    step: int,
) -> float | None:
    field = f"energy_step_{step}_eV"
    left_by_repeat = {
        int(record.get("repeat", 1)): record
        for record in left
        if record.get(field) is not None
    }
    right_by_repeat = {
        int(record.get("repeat", 1)): record
        for record in right
        if record.get(field) is not None
    }
    values = [
        abs(
            float(left_by_repeat[repeat][field])
            - float(right_by_repeat[repeat][field])
        )
        for repeat in set(left_by_repeat) & set(right_by_repeat)
    ]
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt1-dir", type=Path, required=True)
    parser.add_argument("--static-eager-dir", type=Path, required=True)
    parser.add_argument("--opt2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    opt1 = load_records(args.opt1_dir, "esen_gpu_resident_eager")
    static = load_records(
        args.static_eager_dir, "esen_gpu_resident_opt2_static_eager"
    )
    opt2 = load_records(args.opt2_dir, "esen_gpu_resident_model_cg")
    opt1_status = load_status_rows(args.opt1_dir)
    static_status = load_status_rows(args.static_eager_dir)
    opt2_status = load_status_rows(args.opt2_dir)
    keys = (
        set(opt1)
        | set(static)
        | set(opt2)
        | set(opt1_status)
        | set(static_status)
        | set(opt2_status)
    )
    if not keys:
        raise SystemExit("No opt1/static-eager/opt2 records found")

    rows: list[dict[str, object]] = []
    for system, temperature in sorted(
        keys, key=lambda key: (SYSTEM_ORDER.get(key[0], 999), key[1])
    ):
        key = (system, temperature)
        opt1_records = opt1.get(key, [])
        static_records = static.get(key, [])
        opt2_records = opt2.get(key, [])
        opt1_sps = median_field(opt1_records, "seconds_per_step")
        static_sps = median_field(static_records, "seconds_per_step")
        opt2_sps = median_field(opt2_records, "seconds_per_step")
        opt1_process = process_median(opt1_status.get(key, []))
        static_process = process_median(static_status.get(key, []))
        opt2_process = process_median(opt2_status.get(key, []))
        opt1_status_rows = opt1_status.get(key, [])
        static_status_rows = static_status.get(key, [])
        opt2_status_rows = opt2_status.get(key, [])

        row: dict[str, object] = {
            "system": system,
            "temperature_K": f"{temperature:g}",
            "opt1_attempted": len(opt1_status_rows),
            "opt1_completed": len(opt1_records),
            "opt1_validation_passed": sum(
                validation_passed(record) for record in opt1_records
            ),
            "opt1_validation_failed": status_count(
                opt1_status_rows, "validation_failed"
            ),
            "opt1_oom": status_count(opt1_status_rows, "oom"),
            "opt1_error": status_count(opt1_status_rows, "error"),
            "static_eager_attempted": len(static_status_rows),
            "static_eager_completed": len(static_records),
            "static_eager_validation_passed": sum(
                validation_passed(record) for record in static_records
            ),
            "static_eager_validation_failed": status_count(
                static_status_rows, "validation_failed"
            ),
            "static_eager_oom": status_count(static_status_rows, "oom"),
            "static_eager_capacity_overflow": status_count(
                static_status_rows, "capacity_overflow"
            ),
            "static_eager_error": status_count(static_status_rows, "error"),
            "opt2_attempted": len(opt2_status_rows),
            "opt2_completed": len(opt2_records),
            "opt2_validation_passed": sum(
                validation_passed(record) for record in opt2_records
            ),
            "opt2_validation_failed": status_count(
                opt2_status_rows, "validation_failed"
            ),
            "opt2_oom": status_count(opt2_status_rows, "oom"),
            "opt2_capacity_overflow": status_count(
                opt2_status_rows, "capacity_overflow"
            ),
            "opt2_error": status_count(opt2_status_rows, "error"),
            "opt1_seconds_per_step": format_float(opt1_sps, 9),
            "static_eager_seconds_per_step": format_float(static_sps, 9),
            "opt2_seconds_per_step": format_float(opt2_sps, 9),
            "static_adaptation_speedup_vs_opt1": format_float(
                speedup(opt1_sps, static_sps), 4
            ),
            "pure_cuda_graph_speedup": format_float(
                speedup(static_sps, opt2_sps), 4
            ),
            "overall_opt2_speedup_vs_opt1": format_float(
                speedup(opt1_sps, opt2_sps), 4
            ),
            "opt1_process_wall_time_s": format_float(opt1_process, 6),
            "static_eager_process_wall_time_s": format_float(static_process, 6),
            "opt2_process_wall_time_s": format_float(opt2_process, 6),
            "pure_cuda_graph_process_speedup": format_float(
                speedup(static_process, opt2_process), 4
            ),
            "overall_opt2_process_speedup_vs_opt1": format_float(
                speedup(opt1_process, opt2_process), 4
            ),
            "opt1_peak_reserved_gib": format_float(
                max_field(opt1_records, "peak_reserved_gib"), 6
            ),
            "static_eager_peak_reserved_gib": format_float(
                max_field(static_records, "peak_reserved_gib"), 6
            ),
            "opt2_peak_reserved_gib": format_float(
                max_field(opt2_records, "peak_reserved_gib"), 6
            ),
            "static_eager_original_initial_energy_error_eV_max": (
                ""
                if (
                    static_initial_error := max_field(
                        static_records,
                        "original_eager_initial_energy_abs_error_eV",
                    )
                )
                is None
                else f"{static_initial_error:.12e}"
            ),
            "opt2_original_initial_energy_error_eV_max": (
                ""
                if (
                    opt2_initial_error := max_field(
                        opt2_records,
                        "cg_initial_energy_abs_error_eV",
                    )
                )
                is None
                else f"{opt2_initial_error:.12e}"
            ),
        }
        for step in (1, 50, 100, 1000):
            static_error = max_field(
                static_records, f"energy_abs_error_step_{step}_eV"
            )
            opt2_error = max_field(
                opt2_records, f"energy_abs_error_step_{step}_eV"
            )
            direct_difference = matched_checkpoint_difference(
                static_records, opt2_records, step
            )
            row[f"static_eager_energy_error_step{step}_eV_max"] = (
                "" if static_error is None else f"{static_error:.12e}"
            )
            row[f"opt2_energy_error_step{step}_eV_max"] = (
                "" if opt2_error is None else f"{opt2_error:.12e}"
            )
            row[f"static_vs_opt2_energy_step{step}_eV_diff_max"] = (
                "" if direct_difference is None else f"{direct_difference:.12e}"
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "opt2_ablation.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    headers = list(rows[0])
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown.extend(
        "| " + " | ".join(str(row[key]) for key in headers) + " |"
        for row in rows
    )
    md_path = args.output_dir / "opt2_ablation.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"TSV ablation: {tsv_path}")
    print(f"Markdown ablation: {md_path}")


if __name__ == "__main__":
    main()
