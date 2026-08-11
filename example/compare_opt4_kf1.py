#!/usr/bin/env python3
"""Compare Opt4 KF1 against the matching Opt3/KF0 capture scopes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

from compare_md_backends import SYSTEM_ORDER, load_records, load_status


def _median(records: list[dict[str, object]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) not in (None, "")
    ]
    return median(values) if values else None


def _maximum(records: list[dict[str, object]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) not in (None, "")
    ]
    return max(values) if values else None


def _speedup(before: float | None, after: float | None) -> float | None:
    if before is None or after in (None, 0):
        return None
    return before / after


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def _status_counts(statuses: list[str]) -> str:
    names = ("success", "validation_failed", "oom", "capacity_overflow", "error")
    return ",".join(f"{name}:{statuses.count(name)}" for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kf0-fixed-dir", type=Path, required=True)
    parser.add_argument("--kf1-fixed-dir", type=Path, required=True)
    parser.add_argument("--kf0-whole-dir", type=Path, required=True)
    parser.add_argument("--kf1-whole-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    datasets = {
        "kf0_fixed": load_records(
            args.kf0_fixed_dir,
            "esen_gpu_resident_fixed_builder_model_cg",
        ),
        "kf1_fixed": load_records(
            args.kf1_fixed_dir,
            "esen_gpu_resident_fixed_builder_model_cg_kf1",
        ),
        "kf0_whole": load_records(
            args.kf0_whole_dir,
            "esen_gpu_resident_whole_step_cg",
        ),
        "kf1_whole": load_records(
            args.kf1_whole_dir,
            "esen_gpu_resident_whole_step_cg_kf1",
        ),
    }
    status_groups = {
        "kf0_fixed": load_status(args.kf0_fixed_dir),
        "kf1_fixed": load_status(args.kf1_fixed_dir),
        "kf0_whole": load_status(args.kf0_whole_dir),
        "kf1_whole": load_status(args.kf1_whole_dir),
    }
    baseline = (
        load_records(args.baseline_dir, "esen_ocpcalculator_eager")
        if args.baseline_dir is not None
        else {}
    )
    keys = set(baseline)
    for groups in (*datasets.values(), *status_groups.values()):
        keys.update(groups)
    if not keys:
        raise SystemExit("No Opt3/KF0 or Opt4/KF1 results found")

    rows: list[dict[str, object]] = []
    for system, temperature in sorted(
        keys, key=lambda item: (SYSTEM_ORDER.get(item[0], 999), item[1])
    ):
        key = (system, temperature)
        records = {name: groups.get(key, []) for name, groups in datasets.items()}
        seconds = {
            name: _median(group, "seconds_per_step")
            for name, group in records.items()
        }
        baseline_seconds = _median(baseline.get(key, []), "seconds_per_step")
        row: dict[str, object] = {
            "system": system,
            "temperature_K": temperature,
            "kf0_fixed_seconds_per_step": seconds["kf0_fixed"],
            "kf1_fixed_seconds_per_step": seconds["kf1_fixed"],
            "kf1_fixed_speedup_vs_kf0": _speedup(
                seconds["kf0_fixed"], seconds["kf1_fixed"]
            ),
            "kf0_whole_seconds_per_step": seconds["kf0_whole"],
            "kf1_whole_seconds_per_step": seconds["kf1_whole"],
            "kf1_whole_speedup_vs_kf0": _speedup(
                seconds["kf0_whole"], seconds["kf1_whole"]
            ),
            "kf1_whole_speedup_vs_baseline": _speedup(
                baseline_seconds, seconds["kf1_whole"]
            ),
            "kf1_whole_setup_s": _median(
                records["kf1_whole"], "setup_wall_time_s"
            ),
            "kf1_whole_peak_allocated_gib": _maximum(
                records["kf1_whole"], "peak_allocated_gib"
            ),
            "kf1_whole_peak_reserved_gib": _maximum(
                records["kf1_whole"], "peak_reserved_gib"
            ),
            "kf1_initial_force_error_max": _maximum(
                records["kf1_whole"],
                "initial_eager_force_max_abs_error_eV_per_A",
            ),
        }
        for step in (1, 50, 100, 1000):
            row[f"kf1_energy_error_step_{step}_eV_max"] = _maximum(
                records["kf1_whole"],
                f"energy_abs_error_step_{step}_eV",
            )
            row[f"kf1_energy_error_step_{step}_eV_per_atom_max"] = _maximum(
                records["kf1_whole"],
                f"energy_abs_error_step_{step}_eV_per_atom",
            )
        for name in datasets:
            row[f"{name}_status"] = _status_counts(
                status_groups[name].get(key, [])
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(args.output_dir / "opt4_kf1_ablation.tsv", rows)

    headers = list(rows[0])
    lines = [
        "# eSEN Opt4 KF1 消融",
        "",
        "KF1 仅融合固定邻居构图的 PBC 距离、cutoff 和 self-mask；",
        "top-k、padding、eSEN、autograd、NVT 与 Opt3/KF0 相同。",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row.get(field)) for field in headers) + " |"
        )
    fixed_gains = [
        float(row["kf1_fixed_speedup_vs_kf0"])
        for row in rows
        if row["kf1_fixed_speedup_vs_kf0"] is not None
    ]
    whole_gains = [
        float(row["kf1_whole_speedup_vs_kf0"])
        for row in rows
        if row["kf1_whole_speedup_vs_kf0"] is not None
    ]
    lines.extend(["", "## 暂定保留判据", ""])
    if fixed_gains:
        lines.append(f"- fixed/model-CG 中位增量：{median(fixed_gains):.4f}×。")
    if whole_gains:
        value = median(whole_gains)
        lines.append(f"- whole-step-CG 中位增量：{value:.4f}×。")
        lines.append(
            "- 当前性能门槛：达到约 1% 端到端提升后，再结合每次重复、MAD 与 profiling 决定是否保留。"
        )
    lines.append(
        "- 数值门槛保持为 1/50 步能量误差 <1e-5 eV/atom、初始最大力误差 <2e-4 eV/Å。"
    )
    (args.output_dir / "opt4_kf1_ablation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Opt4 KF1 report: {args.output_dir / 'opt4_kf1_ablation.md'}")


if __name__ == "__main__":
    main()

