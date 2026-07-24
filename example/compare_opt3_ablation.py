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
            row[f"{name}_peak_reserved_gib"] = fmt(
                maximum_value(records, "peak_reserved_gib"), 6
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
                    "process_wall_time_s": (
                        run_status.get("process_wall_time_s")
                        if run_status is not None
                        else record.get("process_wall_time_s")
                    ),
                    "peak_reserved_gib": record.get("peak_reserved_gib"),
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
    print((args.output_dir / "opt3_ablation.md").read_text(encoding="utf-8"))
    print(f"Detailed runs: {args.output_dir / 'opt3_runs.tsv'}")


if __name__ == "__main__":
    main()
