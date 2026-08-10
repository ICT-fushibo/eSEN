#!/usr/bin/env python3
"""Summarize Opt3 profiling and classify the Whole-step regression."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
from statistics import median
from typing import Any, Iterable


BACKENDS = (
    "static-eager-breakdown",
    "fixed-builder-model-cg",
    "builder-cg-model-cg",
    "force-eval-cg",
    "whole-step-cg",
)
TRANSITIONS = (
    (
        "builder_capture",
        "fixed-builder-model-cg",
        "builder-cg-model-cg",
        "固定构图器进入独立 CUDA Graph",
    ),
    (
        "builder_model_merge",
        "builder-cg-model-cg",
        "force-eval-cg",
        "builder 与 model 合并为一张 CUDA Graph",
    ),
    (
        "integrator_capture",
        "force-eval-cg",
        "whole-step-cg",
        "NVT、固定地址状态和状态回写进入 CUDA Graph",
    ),
    (
        "fixed_to_whole_end_to_end",
        "fixed-builder-model-cg",
        "whole-step-cg",
        "Fixed-builder Model-CG 到 Whole-step CG 的端到端差异",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mad(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    centre = median(values)
    return median(abs(value - centre) for value in values)


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=keys,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_暂无数据。_\n"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(key, "")) for key in headers) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def load_json_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in root.rglob("*.json"):
        if path.name.endswith("torch_trace.json"):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(value, dict) and value.get("backend") in BACKENDS:
            value["_path"] = str(path)
            records.append(value)
    return records


def load_failure_records(root: Path) -> list[dict[str, Any]]:
    path = root / "failed_runs.tsv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def enrich_with_telemetry(root: Path, records: list[dict[str, Any]]) -> None:
    """Attach the first/last GPU state sample to each successful run."""

    telemetry = root / "telemetry"
    if not telemetry.is_dir():
        return

    def find_value(row: dict[str, str], prefix: str) -> str:
        key = next(
            (name for name in row if name.strip().startswith(prefix)), None
        )
        return "" if key is None else row.get(key, "").strip()

    for record in records:
        path = telemetry / f"{record.get('run_name', '')}.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            samples = list(csv.DictReader(handle))
        if not samples:
            continue
        first, last = samples[0], samples[-1]
        fields = {
            "pstate": "pstate",
            "sm_clock": "clocks.sm",
            "memory_clock": "clocks.mem",
            "power": "power.draw",
            "temperature": "temperature.gpu",
            "gpu_utilization": "utilization.gpu",
            "memory_used": "memory.used",
        }
        for output, prefix in fields.items():
            record[f"telemetry_before_{output}"] = find_value(first, prefix)
            record[f"telemetry_after_{output}"] = find_value(last, prefix)
        record["telemetry_samples"] = len(samples)


def timing_summary(
    records: list[dict[str, Any]], failures: list[dict[str, Any]]
):
    timing = [record for record in records if record.get("profile_kind") == "timing"]
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in timing:
        groups[
            (
                str(record["system"]),
                float(record["temperature_K"]),
                str(record["backend"]),
            )
        ].append(record)
    failure_groups: dict[
        tuple[str, float, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in failures:
        if record.get("profile_kind") != "timing":
            continue
        failure_groups[
            (
                str(record["system"]),
                float(record["temperature_K"]),
                str(record["backend"]),
            )
        ].append(record)

    summary = []
    components = []
    reruns = []
    for key in sorted(groups.keys() | failure_groups.keys()):
        rows = groups.get(key, [])
        failed_rows = failure_groups.get(key, [])
        system, temperature, backend = key
        times = [float(row["seconds_per_step"]) for row in rows]
        centre = median(times) if times else None
        dispersion = (mad(times) or 0.0) if times else None
        summary.append(
            {
                "system": system,
                "temperature_K": f"{temperature:g}",
                "backend": backend,
                "repeats": len(times),
                "failed_repeats": len(failed_rows),
                "seconds_per_step_median": fmt(centre, 9),
                "seconds_per_step_mad": fmt(dispersion, 9),
                "relative_mad": (
                    ""
                    if centre is None or dispersion is None
                    else f"{dispersion / centre:.6f}"
                ),
                "validation_failed": sum(
                    row.get("engineering_validation_status") == "failed"
                    for row in rows
                ),
                "oom": sum(row.get("status") == "oom" for row in failed_rows),
                "error": sum(
                    row.get("status") != "oom" for row in failed_rows
                ),
                "peak_reserved_gib_max": (
                    fmt(max(float(row["peak_reserved_gib"]) for row in rows), 6)
                    if rows
                    else ""
                ),
            }
        )
        if (
            centre is not None
            and dispersion is not None
            and dispersion / centre > 0.02
            and len(times) < 9
        ):
            used_repeats = {
                int(row["repeat"]) for row in rows
            } | {int(row["repeat"]) for row in failed_rows}
            first_new_repeat = max(used_repeats, default=0) + 1
            for repeat in range(first_new_repeat, 10):
                reruns.append(
                    {
                        "system": system,
                        "temperature_K": f"{temperature:g}",
                        "backend": backend,
                        "repeat": repeat,
                    }
                )

        component_names = sorted(
            {
                field.removeprefix("component_").removesuffix("_mean_ms")
                for row in rows
                for field in row
                if field.startswith("component_")
                and field.endswith("_mean_ms")
            }
        )
        for component in component_names:
            field = f"component_{component}_mean_ms"
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            if values:
                total_per_step = [
                    float(row[f"component_{component}_total_ms"])
                    / max(int(row["component_profile_steps"]), 1)
                    for row in rows
                    if row.get(f"component_{component}_total_ms") is not None
                ]
                calls_per_step = [
                    float(row[f"component_{component}_calls"])
                    / max(int(row["component_profile_steps"]), 1)
                    for row in rows
                    if row.get(f"component_{component}_calls") is not None
                ]
                components.append(
                    {
                        "system": system,
                        "temperature_K": f"{temperature:g}",
                        "backend": backend,
                        "component": component,
                        "mean_ms_median": f"{median(values):.6f}",
                        "mean_ms_mad": f"{(mad(values) or 0.0):.6f}",
                        "total_ms_per_profile_step_median": (
                            f"{median(total_per_step):.6f}"
                        ),
                        "calls_per_profile_step_median": (
                            f"{median(calls_per_step):.6f}"
                        ),
                        "repeats": len(values),
                    }
                )
    rerun_groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in reruns:
        rerun_groups[
            (int(row["repeat"]), str(row["system"]), str(row["temperature_K"]))
        ].append(row)
    interleaved_reruns = []
    for key, rows in sorted(rerun_groups.items()):
        repeat, system, _ = key
        rng = random.Random(42 + repeat + sum(ord(value) for value in system))
        rng.shuffle(rows)
        interleaved_reruns.extend(rows)
    return summary, components, interleaved_reruns, groups


def transition_summary(groups):
    rows = []
    for system, temperature in sorted({(key[0], key[1]) for key in groups}):
        for transition, before, after, description in TRANSITIONS:
            before_rows = groups.get((system, temperature, before), [])
            after_rows = groups.get((system, temperature, after), [])
            if not before_rows or not after_rows:
                continue
            before_values = [float(row["seconds_per_step"]) for row in before_rows]
            after_values = [float(row["seconds_per_step"]) for row in after_rows]
            before_median = median(before_values)
            after_median = median(after_values)
            common = {
                int(row["repeat"]): float(row["seconds_per_step"])
                for row in before_rows
            }
            after_by_repeat = {
                int(row["repeat"]): float(row["seconds_per_step"])
                for row in after_rows
            }
            paired = [
                after_by_repeat[repeat] - common[repeat]
                for repeat in sorted(common.keys() & after_by_repeat.keys())
            ]
            positive = sum(value > 0 for value in paired)
            needed = math.ceil(0.8 * len(paired)) if paired else 1
            delta = after_median - before_median
            significant = bool(
                len(paired) >= 5
                and after_median / before_median > 1.05
                and positive >= needed
                and delta > (mad(before_values) or 0.0) + (mad(after_values) or 0.0)
            )
            rows.append(
                {
                    "system": system,
                    "temperature_K": f"{temperature:g}",
                    "transition": transition,
                    "description": description,
                    "before_s_per_step": f"{before_median:.9f}",
                    "after_s_per_step": f"{after_median:.9f}",
                    "speedup_before_over_after": (
                        f"{before_median / after_median:.6f}"
                    ),
                    "delta_ms_per_step": f"{1000.0 * delta:.6f}",
                    "slowdown_percent": (
                        f"{100.0 * (after_median / before_median - 1.0):.3f}"
                    ),
                    "paired_positive": f"{positive}/{len(paired)}",
                    "significant": significant,
                }
            )
    return rows


def profiler_equivalence(records: list[dict[str, Any]]):
    """Check that Torch/NSYS instrumentation leaves the trajectory unchanged."""

    groups: dict[
        tuple[str, float, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in records:
        if record.get("profile_kind") == "timing":
            continue
        groups[
            (
                str(record["system"]),
                float(record["temperature_K"]),
                str(record["backend"]),
                int(record["steps"]),
            )
        ].append(record)
    rows = []
    for key, values in sorted(groups.items()):
        if len(values) < 2:
            continue
        positions = {row.get("final_positions_sha256") for row in values}
        momenta = {row.get("final_momenta_sha256") for row in values}
        forces = {row.get("final_forces_sha256") for row in values}
        energies = [float(row["final_energy_eV"]) for row in values]
        passed = bool(
            len(positions) == len(momenta) == len(forces) == 1
            and None not in positions | momenta | forces
        )
        rows.append(
            {
                "system": key[0],
                "temperature_K": f"{key[1]:g}",
                "backend": key[2],
                "steps": key[3],
                "instrumented_runs": len(values),
                "state_hashes_match": passed,
                "final_energy_range_eV": f"{max(energies) - min(energies):.12e}",
                "run_names": ";".join(str(row["run_name"]) for row in values),
            }
        )
    return rows


def identify_profile(path: Path) -> tuple[str, str, int] | None:
    name = path.name
    backend = next((value for value in BACKENDS if value in name), None)
    if backend is None:
        return None
    system = next(
        (
            value
            for value in ("Cu32", "Cu192", "H2O32", "H2O60")
            if name.startswith(value)
        ),
        None,
    )
    match = re.search(r"_(\d+)step_", name)
    if system is None or match is None:
        return None
    return system, backend, int(match.group(1))


def kernel_family(name: str) -> str:
    lowered = name.lower()
    rules = (
        ("topk_sort_select", r"topk|sort|radix|select"),
        ("gemm_bmm", r"gemm|sgemm|matmul|bmm|cutlass"),
        ("scatter_reduce", r"scatter|indexadd|index_add|reduce"),
        ("elementwise", r"elementwise|vectorized|pointwise"),
        ("copy_fill", r"copy|memcpy|fill|zero"),
    )
    for family, pattern in rules:
        if re.search(pattern, lowered):
            return family
    return "other"


def parse_nsys_kernel_files(root: Path):
    raw = []
    for path in root.rglob("*nsys_node.stats.csv"):
        identity = identify_profile(path)
        if identity is None:
            continue
        system, backend, steps = identity
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        reader = csv.reader(lines)
        header = None
        index = {}
        in_kernel_summary = False
        for line, values in zip(lines, reader):
            if line.lstrip().startswith("**"):
                in_kernel_summary = "CUDA GPU Kernel Summary" in line
                header = None
                index = {}
                continue
            if not in_kernel_summary:
                continue
            cleaned = [value.strip().strip('"') for value in values]
            if "Name" in cleaned and any("Total Time" in value for value in cleaned):
                header = cleaned
                index = {value: position for position, value in enumerate(header)}
                continue
            if header is None or len(cleaned) != len(header):
                continue
            total_key = next(
                (value for value in header if "Total Time" in value), None
            )
            instances_key = next(
                (value for value in header if value == "Instances"), None
            )
            if total_key is None:
                continue
            try:
                total_value = float(
                    cleaned[index[total_key]].replace(",", "")
                )
                instances = (
                    int(float(cleaned[index[instances_key]].replace(",", "")))
                    if instances_key is not None
                    else 0
                )
            except ValueError:
                continue
            unit_scale = 1.0
            if "(us)" in total_key or "(μs)" in total_key:
                unit_scale = 1e3
            elif "(ms)" in total_key:
                unit_scale = 1e6
            elif "(s)" in total_key:
                unit_scale = 1e9
            total_ns = total_value * unit_scale
            name = cleaned[index["Name"]]
            raw.append(
                {
                    "system": system,
                    "backend": backend,
                    "steps": steps,
                    "kernel": name,
                    "family": kernel_family(name),
                    "total_time_ns": total_ns,
                    "time_ms_per_step": total_ns / 1e6 / steps,
                    "instances": instances,
                    "instances_per_step": instances / steps,
                }
            )

    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"time": 0.0, "instances": 0.0}
    )
    for row in raw:
        key = (str(row["system"]), str(row["backend"]), str(row["family"]))
        grouped[key]["time"] += float(row["time_ms_per_step"])
        grouped[key]["instances"] += float(row["instances_per_step"])
    family_rows = [
        {
            "system": key[0],
            "backend": key[1],
            "family": key[2],
            "time_ms_per_step": f"{value['time']:.6f}",
            "instances_per_step": f"{value['instances']:.3f}",
        }
        for key, value in sorted(grouped.items())
    ]
    return raw, family_rows


def parse_nsys_graph_traces(root: Path):
    """Summarize low-overhead graph-mode GPU activity per traced MD step."""

    rows = []
    for path in root.rglob("*nsys_graph.gpu_trace.csv"):
        identity = identify_profile(path)
        if identity is None:
            continue
        system, backend, steps = identity
        starts = []
        durations = []
        header = None
        for values in csv.reader(
            path.read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            cleaned = [value.strip().strip('"') for value in values]
            if (
                any(value.startswith("Start") for value in cleaned)
                and any(value.startswith("Duration") for value in cleaned)
                and "Name" in cleaned
            ):
                header = cleaned
                continue
            if header is None or len(cleaned) != len(header):
                continue
            start_key = next(
                (value for value in header if value.startswith("Start")), None
            )
            duration_key = next(
                (value for value in header if value.startswith("Duration")),
                None,
            )
            if start_key is None or duration_key is None:
                continue
            item = dict(zip(header, cleaned))
            try:
                start = float(item[start_key].replace(",", ""))
                duration = float(item[duration_key].replace(",", ""))
            except ValueError:
                continue

            def to_ns(value: float, key: str) -> float:
                if "(us)" in key or "(μs)" in key:
                    return value * 1e3
                if "(ms)" in key:
                    return value * 1e6
                if "(s)" in key:
                    return value * 1e9
                return value

            starts.append(to_ns(start, start_key))
            durations.append(to_ns(duration, duration_key))
        if not starts:
            continue
        span_ns = max(
            start + duration for start, duration in zip(starts, durations)
        ) - min(starts)
        active_ns = sum(durations)
        rows.append(
            {
                "system": system,
                "backend": backend,
                "steps": steps,
                "gpu_trace_events": len(durations),
                "gpu_span_ms_per_step": f"{span_ns / 1e6 / steps:.6f}",
                "gpu_active_ms_per_step": f"{active_ns / 1e6 / steps:.6f}",
                "gpu_gap_ms_per_step": (
                    f"{max(span_ns - active_ns, 0.0) / 1e6 / steps:.6f}"
                ),
                "source": path.name,
            }
        )
    return rows


def kernel_delta_summary(raw_kernel_rows):
    rows = []
    for system in ("Cu192", "H2O60"):
        maps = {}
        for backend in ("force-eval-cg", "whole-step-cg"):
            maps[backend] = {
                str(row["kernel"]): float(row["time_ms_per_step"])
                for row in raw_kernel_rows
                if row["system"] == system and row["backend"] == backend
            }
        names = maps["force-eval-cg"].keys() | maps["whole-step-cg"].keys()
        deltas = sorted(
            (
                (
                    maps["whole-step-cg"].get(name, 0.0)
                    - maps["force-eval-cg"].get(name, 0.0),
                    name,
                )
                for name in names
            ),
            reverse=True,
        )
        positive_total = sum(max(delta, 0.0) for delta, _ in deltas)
        accumulated = 0.0
        for rank, (delta, name) in enumerate(
            ((delta, name) for delta, name in deltas if delta > 0), 1
        ):
            if rank > 3:
                break
            accumulated += delta
            rows.append(
                {
                    "system": system,
                    "rank": rank,
                    "kernel": name,
                    "delta_ms_per_step": f"{delta:.6f}",
                    "positive_delta_percent": (
                        f"{100.0 * delta / positive_total:.3f}"
                        if positive_total
                        else ""
                    ),
                    "cumulative_positive_delta_percent": (
                        f"{100.0 * accumulated / positive_total:.3f}"
                        if positive_total
                        else ""
                    ),
                }
            )
    return rows


def make_hot_filters(raw_kernel_rows):
    filters = []
    for row in kernel_delta_summary(raw_kernel_rows):
        for backend in ("force-eval-cg", "whole-step-cg"):
            filters.append(
                {
                    "system": row["system"],
                    "backend": backend,
                    "kernel_regex": re.escape(str(row["kernel"])),
                    "rank": row["rank"],
                    "delta_ms_per_step": row["delta_ms_per_step"],
                    "kernel": row["kernel"],
                }
            )
    return filters


def parse_ncu_csv(root: Path):
    rows = []
    for path in root.rglob("*.csv"):
        if "ncu" not in path.name:
            continue
        identity = identify_profile(path)
        if identity is None:
            continue
        system, backend, _ = identity
        graph_mode = "node" if "_ncu_node" in path.name else "graph"
        rank_match = re.search(r"_kernel_(\d+)", path.name)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        reader = csv.reader(lines)
        header = None
        for values in reader:
            cleaned = [value.strip().strip('"') for value in values]
            if "Metric Name" in cleaned and "Metric Value" in cleaned:
                header = cleaned
                continue
            if header is None or len(cleaned) != len(header):
                continue
            item = dict(zip(header, cleaned))
            rows.append(
                {
                    "source": path.name,
                    "system": system,
                    "backend": backend,
                    "graph_profiling": graph_mode,
                    "hot_kernel_rank": (
                        "" if rank_match is None else rank_match.group(1)
                    ),
                    "kernel": item.get("Kernel Name", item.get("Kernel", "")),
                    "metric": item.get("Metric Name", ""),
                    "unit": item.get("Metric Unit", ""),
                    "value": item.get("Metric Value", ""),
                }
            )
    return rows


def build_report(
    timing_rows,
    transition_rows,
    component_rows,
    kernel_rows,
    graph_rows,
    kernel_deltas,
    ncu_rows,
    failures,
    profiler_equivalence_rows,
) -> str:
    required_traces = {
        ("Cu192", "force-eval-cg"),
        ("Cu192", "whole-step-cg"),
        ("H2O60", "force-eval-cg"),
        ("H2O60", "whole-step-cg"),
    }
    available_traces = {
        (row["system"], row["backend"]) for row in graph_rows
    }
    root_cause_complete = bool(
        transition_rows
        and required_traces <= available_traces
        and any(row["system"] == "Cu192" for row in kernel_rows)
        and any(row["system"] == "H2O60" for row in kernel_rows)
        and profiler_equivalence_rows
        and all(row["state_hashes_match"] for row in profiler_equivalence_rows)
    )
    lines = [
        "# eSEN Opt3 Whole-step CUDA Graph 退化 Profiling",
        "",
        "## 完成状态",
        "",
        f"- 无 profiler timing 分组：{len(timing_rows)}。",
        f"- 捕获范围比较：{len(transition_rows)}。",
        f"- CUDA Event 分段记录：{len(component_rows)}。",
        f"- NSYS kernel-family 记录：{len(kernel_rows)}。",
        f"- NSYS graph timing 记录：{len(graph_rows)}。",
        f"- NCU metric 记录：{len(ncu_rows)}。",
        f"- OOM/错误记录：{len(failures)}。",
        f"- Profiler 无扰动检查：{len(profiler_equivalence_rows)}。",
        f"- 根因定位完成：{'是' if root_cause_complete else '否'}。",
        "",
    ]
    if not transition_rows:
        lines.extend(
            [
                "尚无服务器 timing 结果，当前不能给出退化根因。",
                "",
                "同步代码后先运行 `PHASE=timing`，再运行 `PHASE=nsys`。",
                "",
            ]
        )
        return "\n".join(lines)

    significant = [row for row in transition_rows if row["significant"]]
    cu_significant = [
        row
        for row in significant
        if row["system"] in {"Cu32", "Cu192"}
        and row["temperature_K"] == "300"
        and row["transition"] != "fixed_to_whole_end_to_end"
    ]
    primary = max(
        cu_significant,
        key=lambda row: float(row["delta_ms_per_step"]),
        default=None,
    )
    lines.extend(["## 根因判定", ""])
    if primary is None:
        lines.append(
            "交错重跑没有确认任何捕获范围存在稳定的 >5% 退化；原结果优先判为时段/GPU 状态漂移。"
        )
        recommendation = "先修正测试环境稳定性，不进入 kernel fusion。"
    else:
        lines.append(
            f"主要退化发生在 **{primary['description']}**："
            f"{primary['delta_ms_per_step']} ms/step，"
            f"慢 {primary['slowdown_percent']}%。"
        )
        transition = primary["transition"]
        if transition == "builder_capture":
            recommendation = (
                "优先检查 fixed builder 的 top-k、mask 与临时张量在 Graph 中的执行；KF1 候选为邻居距离/mask 融合。"
            )
        elif transition == "builder_model_merge":
            recommendation = (
                "先修正 builder 与 autograd/model 合图后的依赖或内存池结构，再决定 kernel fusion。"
            )
        else:
            recommendation = (
                "先修正 branchless NVT、固定地址状态或状态回写的 Whole-step Graph 结构，不应直接做模型 fusion。"
            )
    lines.extend(["", f"**下一步建议：** {recommendation}", ""])

    graph_map = {
        (row["system"], row["backend"]): float(
            row["gpu_span_ms_per_step"]
        )
        for row in graph_rows
    }
    device_classification = []
    for system in ("Cu192", "H2O60"):
        transition = next(
            (
                row
                for row in transition_rows
                if row["system"] == system
                and row["temperature_K"] == "300"
                and row["transition"] == "integrator_capture"
            ),
            None,
        )
        before = graph_map.get((system, "force-eval-cg"))
        after = graph_map.get((system, "whole-step-cg"))
        if transition is None or before is None or after is None:
            continue
        wall_delta = float(transition["delta_ms_per_step"])
        gpu_delta = after - before
        if wall_delta > 0 and gpu_delta >= 0.5 * wall_delta:
            classification = "真实 GPU kernel/依赖执行退化"
        elif wall_delta > 0 and abs(gpu_delta) <= 0.2 * wall_delta:
            classification = "主要是 host launch/同步/Python bookkeeping"
        elif abs(wall_delta) <= 0.05 * max(
            float(transition["before_s_per_step"]) * 1000.0, 1e-12
        ):
            classification = "Whole-step 差异很小，可能被 builder 成本掩盖"
        else:
            classification = "GPU 与 host 混合影响"
        device_classification.append(
            {
                "system": system,
                "wall_delta_ms/step": f"{wall_delta:.6f}",
                "NSYS_GPU_delta_ms/step": f"{gpu_delta:.6f}",
                "classification": classification,
            }
        )
    if device_classification:
        lines.extend(
            [
                "## GPU 端还是 Host 端",
                "",
                markdown_table(device_classification),
            ]
        )

    top_kernel_deltas = []
    if kernel_rows:
        grouped = defaultdict(dict)
        for row in kernel_rows:
            if row["system"] not in {"Cu192", "H2O60"}:
                continue
            grouped[(row["system"], row["family"])][row["backend"]] = float(
                row["time_ms_per_step"]
            )
        for (system, family), values in grouped.items():
            if "whole-step-cg" not in values or "force-eval-cg" not in values:
                continue
            delta = values["whole-step-cg"] - values["force-eval-cg"]
            top_kernel_deltas.append((delta, system, family))
        top_kernel_deltas.sort(reverse=True)
        lines.extend(["## Kernel family 增量", ""])
        for delta, system, family in top_kernel_deltas[:3]:
            lines.append(f"- {system} / {family}: {delta:.6f} ms/step。")
        lines.append("")

    if kernel_deltas:
        lines.extend(["## 退化贡献最大的 kernel", ""])
        for row in kernel_deltas:
            lines.append(
                f"- {row['system']} #{row['rank']}: "
                f"`{row['kernel']}`，+{row['delta_ms_per_step']} ms/step，"
                f"占正向 kernel 增量 {row['positive_delta_percent']}%。"
            )
        lines.append("")

    if profiler_equivalence_rows:
        lines.extend(
            [
                "## Profiler 无扰动检查",
                "",
                markdown_table(profiler_equivalence_rows),
            ]
        )

    compact_transitions = [
        {
            "system": row["system"],
            "T(K)": row["temperature_K"],
            "transition": row["transition"],
            "speedup": row["speedup_before_over_after"],
            "delta_ms": row["delta_ms_per_step"],
            "slowdown_%": row["slowdown_percent"],
            "significant": row["significant"],
        }
        for row in transition_rows
    ]
    lines.extend(["## 捕获范围消融", "", markdown_table(compact_transitions)])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_json_records(args.input_dir)
    enrich_with_telemetry(args.input_dir, records)
    failures = load_failure_records(args.input_dir)
    timing_rows, component_rows, reruns, groups = timing_summary(
        records, failures
    )
    transition_rows = transition_summary(groups)
    profiler_equivalence_rows = profiler_equivalence(records)
    raw_kernels, kernel_rows = parse_nsys_kernel_files(args.input_dir)
    graph_rows = parse_nsys_graph_traces(args.input_dir)
    kernel_deltas = kernel_delta_summary(raw_kernels)
    hot_filters = make_hot_filters(raw_kernels)
    ncu_rows = parse_ncu_csv(args.input_dir)

    outputs = {
        "profile_runs.tsv": [
            {key: value for key, value in row.items() if key != "_path"}
            for row in records
        ],
        "profile_summary.tsv": timing_rows,
        "component_timing.tsv": component_rows,
        "capture_scope_transitions.tsv": transition_rows,
        "kernel_summary.tsv": kernel_rows,
        "kernel_delta.tsv": kernel_deltas,
        "graph_timing.tsv": graph_rows,
        "hot_kernel_filters.tsv": hot_filters,
        "ncu_metrics.tsv": ncu_rows,
        "additional_runs.tsv": reruns,
        "failure_summary.tsv": failures,
        "profiler_equivalence.tsv": profiler_equivalence_rows,
    }
    for name, rows in outputs.items():
        write_tsv(args.output_dir / name, rows)
    report = build_report(
        timing_rows,
        transition_rows,
        component_rows,
        kernel_rows,
        graph_rows,
        kernel_deltas,
        ncu_rows,
        failures,
        profiler_equivalence_rows,
    )
    (args.output_dir / "opt3_whole_step_regression.md").write_text(
        report + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
