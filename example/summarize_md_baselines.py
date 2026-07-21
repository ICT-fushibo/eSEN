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
    process_times = {
        row["run_name"]: float(row["process_wall_time_s"])
        for row in status_rows
        if row["status"] == "success"
    }
    status_groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in status_rows:
        status_groups[(row["system"], float(row["temperature_K"]))].append(row)
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for path in args.input_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("backend") != args.backend:
            continue
        record["process_wall_time_s"] = process_times.get(path.stem)
        groups[(str(record["system"]), float(record["temperature_K"]))].append(record)

    rows: list[dict[str, object]] = []
    all_keys = set(groups) | set(status_groups)
    for system, temperature in sorted(
        all_keys, key=lambda key: (SYSTEM_ORDER.get(key[0], 999), key[1])
    ):
        records = groups[(system, temperature)]
        statuses = status_groups[(system, temperature)]
        process_values = [
            float(record["process_wall_time_s"])
            for record in records
            if record["process_wall_time_s"] is not None
        ]
        rows.append(
            {
                "system": system,
                "temperature_K": f"{temperature:g}",
                "atoms": records[0]["atoms"] if records else "",
                "attempted_repeats": len(statuses) if statuses else len(records),
                "successful_repeats": len(records),
                "failed_repeats": sum(
                    row["status"] != "success" for row in statuses
                ),
                "oom_repeats": sum(row["status"] == "oom" for row in statuses),
                "error_repeats": sum(
                    row["status"] not in {"success", "oom"} for row in statuses
                ),
                "seconds_per_step_median": (
                    f"{median(float(r['seconds_per_step']) for r in records):.9f}"
                    if records
                    else ""
                ),
                "md_wall_time_s_median": (
                    f"{median(float(r['md_wall_time_s']) for r in records):.6f}"
                    if records
                    else ""
                ),
                "process_wall_time_s_median": (
                    f"{median(process_values):.6f}" if process_values else ""
                ),
                "peak_allocated_gib_max": (
                    f"{max(float(r['peak_allocated_gib']) for r in records):.6f}"
                    if records
                    else ""
                ),
                "peak_reserved_gib_max": (
                    f"{max(float(r['peak_reserved_gib']) for r in records):.6f}"
                    if records
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
