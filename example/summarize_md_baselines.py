#!/usr/bin/env python3
"""Summarize repeated eSEN baseline JSON records and process wall times."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


SYSTEM_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "Cu32",
            "Cu64",
            "Cu192",
            "Cu512",
            "Cu1024",
            "H2O32",
            "H2O60",
            "H2O192",
            "H2O512",
            "H2O1024",
        )
    )
}


def load_status(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def record_run_name(record: dict[str, object]) -> str:
    """Return the explicit run name, with compatibility for older JSON files."""

    if record.get("run_name"):
        return str(record["run_name"])
    backend_suffix = {
        "esen_ocpcalculator_eager": "esen_baseline",
        "esen_gpu_resident_eager": "esen_gpu_eager",
        "esen_gpu_resident_model_cg": "esen_model_cg",
    }.get(str(record["backend"]), "esen_unknown")
    return (
        f"{record['system']}_{float(record['temperature_K']):g}K_"
        f"{record['steps']}step_{backend_suffix}_r{record.get('repeat', 1)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        default="esen_ocpcalculator_eager",
        help="Only summarize JSON records produced by this backend",
    )
    parser.add_argument(
        "--report-prefix",
        default="baseline_report",
        help="Output filename prefix for the TSV and Markdown reports",
    )
    args = parser.parse_args()

    status_rows = load_status(args.input_dir / "run_status.tsv")
    completed_statuses = {"success", "validation_failed"}
    process_times = {
        row["run_name"]: float(row["process_wall_time_s"])
        for row in status_rows
        if row["status"] in completed_statuses
    }
    status_by_run = {row["run_name"]: row["status"] for row in status_rows}
    status_groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in status_rows:
        status_groups[(row["system"], float(row["temperature_K"]))].append(row)
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for path in args.input_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("backend") != args.backend:
            continue
        record.setdefault("run_name", path.stem)
        record["process_wall_time_s"] = process_times.get(path.stem)
        groups[(str(record["system"]), float(record["temperature_K"]))].append(record)

    rows: list[dict[str, object]] = []
    all_keys = set(groups) | set(status_groups)
    for system, temperature in sorted(
        all_keys, key=lambda key: (SYSTEM_ORDER.get(key[0], 999), key[1])
    ):
        records = groups[(system, temperature)]
        statuses = status_groups[(system, temperature)]
        completed_records = [
            record
            for record in records
            if not status_by_run
            or status_by_run.get(record_run_name(record)) in completed_statuses
        ]
        if not statuses:
            completed_records = records
        process_values = [
            float(record["process_wall_time_s"])
            for record in completed_records
            if record["process_wall_time_s"] is not None
        ]
        seconds_per_step_values = [
            float(record["seconds_per_step"]) for record in completed_records
        ]
        md_wall_values = [
            float(record["md_wall_time_s"]) for record in completed_records
        ]
        peak_allocated_values = [
            float(record["peak_allocated_gib"]) for record in completed_records
        ]
        peak_reserved_values = [
            float(record["peak_reserved_gib"]) for record in completed_records
        ]
        cg_setup_values = [
            float(record["cg_setup_wall_time_s"])
            for record in completed_records
            if record.get("cg_setup_wall_time_s") is not None
        ]
        cg_hit_rates = [
            float(record["cuda_graph_hit_rate"])
            for record in completed_records
            if record.get("cuda_graph_hit_rate") is not None
        ]
        energy_errors: dict[int, list[float]] = {}
        checkpoint_energies: dict[int, list[float]] = {}
        for step in (1, 50, 100, 1000):
            field = f"energy_abs_error_step_{step}_eV"
            energy_errors[step] = [
                float(record[field])
                for record in records
                if record.get(field) is not None
            ]
            energy_field = f"energy_step_{step}_eV"
            checkpoint_energies[step] = [
                float(record[energy_field])
                for record in completed_records
                if record.get(energy_field) is not None
            ]
        rows.append(
            {
                "system": system,
                "temperature_K": f"{temperature:g}",
                "atoms": records[0]["atoms"] if records else "",
                "attempted_repeats": len(statuses) if statuses else len(records),
                "completed_repeats": len(completed_records),
                "successful_repeats": (
                    sum(row["status"] == "success" for row in statuses)
                    if statuses
                    else len(records)
                ),
                "failed_repeats": sum(
                    row["status"] != "success" for row in statuses
                ),
                "oom_repeats": sum(row["status"] == "oom" for row in statuses),
                "validation_failed_repeats": sum(
                    row["status"] == "validation_failed" for row in statuses
                ),
                "capacity_overflow_repeats": sum(
                    row["status"] == "capacity_overflow" for row in statuses
                ),
                "missing_reference_repeats": sum(
                    row["status"] == "missing_reference" for row in statuses
                ),
                "error_repeats": sum(
                    row["status"]
                    not in {
                        "success",
                        "oom",
                        "validation_failed",
                        "capacity_overflow",
                        "missing_reference",
                    }
                    for row in statuses
                ),
                "unvalidated_successful_repeats": sum(
                    record.get("energy_validation_status") == "missing_reference"
                    for record in completed_records
                ),
                "seconds_per_step_median": (
                    f"{median(seconds_per_step_values):.9f}"
                    if seconds_per_step_values
                    else ""
                ),
                "md_wall_time_s_median": (
                    f"{median(md_wall_values):.6f}"
                    if md_wall_values
                    else ""
                ),
                "process_wall_time_s_median": (
                    f"{median(process_values):.6f}" if process_values else ""
                ),
                "peak_allocated_gib_max": (
                    f"{max(peak_allocated_values):.6f}"
                    if peak_allocated_values
                    else ""
                ),
                "peak_reserved_gib_max": (
                    f"{max(peak_reserved_values):.6f}"
                    if peak_reserved_values
                    else ""
                ),
                "cg_setup_wall_time_s_median": (
                    f"{median(cg_setup_values):.6f}" if cg_setup_values else ""
                ),
                "cg_hit_rate_min": (
                    f"{min(cg_hit_rates):.6f}" if cg_hit_rates else ""
                ),
                "cg_edge_capacity_max": (
                    max(
                        int(record["cuda_graph_edge_capacity"])
                        for record in completed_records
                        if record.get("cuda_graph_edge_capacity") is not None
                    )
                    if any(
                        record.get("cuda_graph_edge_capacity") is not None
                        for record in completed_records
                    )
                    else ""
                ),
                "energy_step_1_eV_median": (
                    f"{median(checkpoint_energies[1]):.12f}"
                    if checkpoint_energies[1]
                    else ""
                ),
                "energy_step_50_eV_median": (
                    f"{median(checkpoint_energies[50]):.12f}"
                    if checkpoint_energies[50]
                    else ""
                ),
                "energy_step_100_eV_median": (
                    f"{median(checkpoint_energies[100]):.12f}"
                    if checkpoint_energies[100]
                    else ""
                ),
                "energy_step_1000_eV_median": (
                    f"{median(checkpoint_energies[1000]):.12f}"
                    if checkpoint_energies[1000]
                    else ""
                ),
                "energy_validation_passed_repeats": sum(
                    record.get("energy_validation_pass") is True for record in records
                ),
                "energy_abs_error_step_1_eV_max": (
                    f"{max(energy_errors[1]):.12e}" if energy_errors[1] else ""
                ),
                "energy_abs_error_step_50_eV_max": (
                    f"{max(energy_errors[50]):.12e}" if energy_errors[50] else ""
                ),
                "energy_abs_error_step_100_eV_max": (
                    f"{max(energy_errors[100]):.12e}" if energy_errors[100] else ""
                ),
                "energy_abs_error_step_1000_eV_max": (
                    f"{max(energy_errors[1000]):.12e}"
                    if energy_errors[1000]
                    else ""
                ),
            }
        )

    if not rows:
        raise SystemExit(f"No baseline attempts found in {args.input_dir}")

    report_tsv = args.input_dir / f"{args.report_prefix}.tsv"
    with report_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    headers = list(rows[0])
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown.extend(
        "| " + " | ".join(str(row[name]) for name in headers) + " |" for row in rows
    )
    report_md = args.input_dir / f"{args.report_prefix}.md"
    report_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(report_md.read_text(encoding="utf-8"))
    print(f"TSV report: {report_tsv}")
    print(f"Markdown report: {report_md}")


if __name__ == "__main__":
    main()
