#!/usr/bin/env python3
"""Compare corrected ASE and eager GPU-resident eSEN benchmark results."""

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


def load_records(
    directory: Path, backend: str
) -> dict[tuple[str, float], list[dict[str, object]]]:
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("backend") == backend:
            groups[(str(record["system"]), float(record["temperature_K"]))].append(
                record
            )
    return groups


def load_status(directory: Path) -> dict[tuple[str, float], list[str]]:
    path = directory / "run_status.tsv"
    groups: dict[tuple[str, float], list[str]] = defaultdict(list)
    if not path.is_file():
        return groups
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            groups[(row["system"], float(row["temperature_K"]))].append(
                row["status"]
            )
    return groups


def format_float(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_records(args.baseline_dir, "esen_ocpcalculator_eager")
    gpu = load_records(args.gpu_dir, "esen_gpu_resident_eager")
    baseline_status = load_status(args.baseline_dir)
    gpu_status = load_status(args.gpu_dir)
    keys = set(baseline) | set(gpu) | set(baseline_status) | set(gpu_status)
    if not keys:
        raise SystemExit("No benchmark records or statuses found")

    rows: list[dict[str, object]] = []
    for system, temperature in sorted(
        keys, key=lambda key: (SYSTEM_ORDER.get(key[0], 999), key[1])
    ):
        baseline_records = baseline[(system, temperature)]
        gpu_records = gpu[(system, temperature)]
        baseline_sps = (
            median(float(record["seconds_per_step"]) for record in baseline_records)
            if baseline_records
            else None
        )
        gpu_sps = (
            median(float(record["seconds_per_step"]) for record in gpu_records)
            if gpu_records
            else None
        )
        speedup = (
            baseline_sps / gpu_sps
            if baseline_sps is not None and gpu_sps is not None
            else None
        )
        baseline_peak = (
            max(float(record["peak_allocated_gib"]) for record in baseline_records)
            if baseline_records
            else None
        )
        gpu_peak = (
            max(float(record["peak_allocated_gib"]) for record in gpu_records)
            if gpu_records
            else None
        )
        rows.append(
            {
                "system": system,
                "temperature_K": f"{temperature:g}",
                "baseline_success": len(baseline_records),
                "baseline_oom": baseline_status[(system, temperature)].count("oom"),
                "gpu_success": len(gpu_records),
                "gpu_oom": gpu_status[(system, temperature)].count("oom"),
                "baseline_seconds_per_step": format_float(baseline_sps, 9),
                "gpu_seconds_per_step": format_float(gpu_sps, 9),
                "gpu_resident_speedup": format_float(speedup, 4),
                "baseline_peak_allocated_gib": format_float(baseline_peak, 6),
                "gpu_peak_allocated_gib": format_float(gpu_peak, 6),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "stage1_comparison.tsv"
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
    md_path = args.output_dir / "stage1_comparison.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"TSV comparison: {tsv_path}")
    print(f"Markdown comparison: {md_path}")


if __name__ == "__main__":
    main()
