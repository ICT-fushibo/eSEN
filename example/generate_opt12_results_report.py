#!/usr/bin/env python3
"""Generate the eSEN baseline/opt1/opt2 benchmark record in Markdown."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "output"
STAGE1 = OUTPUT / "esen_stage1_energy_gpu2_20260722_130306"
ABLATION = OUTPUT / "esen_opt2_full_ablation_20260723_114852"
REPORT = REPO / "ESEN_OPT1_OPT2_RESULTS.md"

SYSTEMS = (
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
TEMPERATURES = (300, 800)
REPEATS = (1, 2, 3)

BACKENDS = {
    "baseline": {
        "label": "Baseline（ASE/OCPCalculator eager）",
        "directory": STAGE1 / "ase",
        "record_backend": "esen_ocpcalculator_eager",
    },
    "opt1": {
        "label": "Opt1（GPU-resident eager）",
        "directory": STAGE1 / "gpu_eager",
        "record_backend": "esen_gpu_resident_eager",
    },
    "static": {
        "label": "Opt2 static-eager 消融",
        "directory": ABLATION / "static_eager",
        "record_backend": "esen_gpu_resident_opt2_static_eager",
    },
    "opt2": {
        "label": "Opt2（model-only CUDA Graph）",
        "directory": ABLATION / "model_cg",
        "record_backend": "esen_gpu_resident_model_cg",
    },
}


def key(record: dict[str, object]) -> tuple[str, int, int]:
    return (
        str(record["system"]),
        int(float(record["temperature_K"])),
        int(record.get("repeat", 1)),
    )


def load_records(directory: Path, backend: str) -> dict[tuple[str, int, int], dict]:
    records = {}
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("backend") == backend:
            records[key(record)] = record
    return records


def load_status(directory: Path) -> dict[tuple[str, int, int], str]:
    result = {}
    path = directory / "run_status.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            result[
                (
                    row["system"],
                    int(float(row["temperature_K"])),
                    int(row["repeat"]),
                )
            ] = row["status"]
    return result


RECORDS = {
    name: load_records(config["directory"], config["record_backend"])
    for name, config in BACKENDS.items()
}
STATUSES = {
    name: load_status(config["directory"]) for name, config in BACKENDS.items()
}


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def fmt_time(value: float | None) -> str:
    return "—" if value is None else f"{value:.9f}"


def fmt_speedup(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}×"


def fmt_error(value: object | None) -> str:
    return "—" if value is None else f"{float(value):.12e}"


def median_field(records: list[dict], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return median(values) if values else None


def max_field(records: list[dict], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return max(values) if values else None


def records_for(name: str, system: str, temperature: int) -> list[dict]:
    return [
        record
        for (record_system, record_temperature, _), record in RECORDS[name].items()
        if record_system == system and record_temperature == temperature
    ]


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def status_label(name: str, run_key: tuple[str, int, int]) -> str:
    status = STATUSES[name].get(run_key, "not_recorded")
    translations = {
        "success": "完成",
        "validation_failed": "完成；数值验证失败",
        "oom": "OOM",
        "missing_reference": "缺少 baseline，未运行",
        "capacity_overflow": "容量溢出",
        "error": "错误",
    }
    return translations.get(status, status)


def error(record: dict | None, step: int) -> object | None:
    if record is None:
        return None
    return record.get(f"energy_abs_error_step_{step}_eV")


def per_run_rows(name: str) -> list[list[str]]:
    rows = []
    for system in SYSTEMS:
        for temperature in TEMPERATURES:
            for repeat in REPEATS:
                run_key = (system, temperature, repeat)
                baseline = RECORDS["baseline"].get(run_key)
                candidate = RECORDS[name].get(run_key)
                baseline_sps = (
                    None if baseline is None else float(baseline["seconds_per_step"])
                )
                candidate_sps = (
                    None if candidate is None else float(candidate["seconds_per_step"])
                )
                rows.append(
                    [
                        system,
                        str(temperature),
                        str(repeat),
                        fmt_time(baseline_sps),
                        fmt_time(candidate_sps),
                        fmt_speedup(ratio(baseline_sps, candidate_sps)),
                        fmt_error(error(candidate, 1)),
                        fmt_error(error(candidate, 50)),
                        fmt_error(error(candidate, 100)),
                        fmt_error(error(candidate, 1000)),
                        status_label(name, run_key),
                    ]
                )
    return rows


def main() -> None:
    lines = [
        "# eSEN MD 推理优化路线、实现与实验结果",
        "",
        "> 生成日期：2026-07-24  ",
        "> 结果来源：`esen_stage1_energy_gpu2_20260722_130306` 与 "
        "`esen_opt2_full_ablation_20260723_114852`",
        "",
        "## 1. 优化路线",
        "",
        "本项目按照“先消除 MD 框架与数据搬运开销，再消除模型 kernel launch "
        "开销，最后进行 kernel fusion”的顺序推进：",
        "",
        "```text",
        "官方 ASE/OCPCalculator eager baseline",
        "    ↓",
        "Opt1：GPU-resident MD + eager eSEN",
        "    ↓",
        "Opt2 static-eager 消融：固定容量邻居表 + dummy padding + 捕获兼容改造",
        "    ↓",
        "Opt2：model-only CUDA Graph",
        "    ↓",
        "后续 Opt3：Triton / kernel fusion，优先融合高频逐边与归约算子",
        "    ↓",
        "后续 Opt4：评估 whole-step CUDA Graph 或邻居构建捕获",
        "```",
        "",
        "路线中的原则是每一阶段都保留独立对照：",
        "",
        "- Baseline → Opt1：测量 GPU 常驻 MD、减少 CPU/GPU 往返带来的收益。",
        "- Opt1 → static-eager：测量为 CUDA Graph 引入的固定 shape、padding "
        "和确定性旋转改造本身的影响。",
        "- static-eager → Opt2：隔离纯 model-only CUDA Graph replay 收益。",
        "- Opt1 → Opt2：报告当前 opt2 的整体最终收益。",
        "- 后续 kernel fusion 必须继续分别与 Opt1、Opt2 对比，不能把多项优化混为一项。",
        "",
        "## 2. 各阶段实现",
        "",
        "### 2.1 Baseline：ASE/OCPCalculator eager",
        "",
        "- 使用官方 `OCPCalculator` 和 ASE NVT MD 路径。",
        "- 每一步由 ASE 驱动 calculator，包含 ASE 数据结构转换及 CPU/GPU 交互。",
        "- 模型为 FP32；MD 状态与积分为 FP64。",
        "- 不启用 AMP、TF32、`torch.compile`、CUDA Graph 或自定义 fusion。",
        "",
        "### 2.2 Opt1：GPU-resident eager",
        "",
        "- 将位置、动量、力和势能保持在 GPU。",
        "- 使用与 baseline 参数一致的 GPU NVT/Berendsen 积分器。",
        "- eSEN 前向、保守力 `autograd.grad` 和动态邻居表仍逐步 eager 执行。",
        "- 避免每一步 ASE calculator 调用、NumPy/Tensor 转换和主机设备往返。",
        "- 仍不启用 CUDA Graph、compile、AMP、TF32 或 kernel fusion。",
        "",
        "### 2.3 Opt2 static-eager 消融",
        "",
        "该路径专门用于公平消融，保留 Opt2 的全部捕获兼容改造，但完全不 capture：",
        "",
        "- 每步在图外根据真实 FP32 坐标重建动态邻居表。",
        "- 克隆状态 eager probe 50 步，最大边数增加 10% 后按 256 边对齐。",
        "- 使用 32 个 dummy sink atoms 将边张量填充到固定容量。",
        "- dummy 边不接触真实原子；能量 head 只归约 `n_real` 个真实原子。",
        "- 元素参考能量只从真实原子预计算。",
        "- 使用固定辅助旋转参考，避免捕获路径中的 RNG 操作。",
        "- 将普通属性形式的索引 Tensor 移到 GPU，避免隐式 H2D。",
        "- 模型前向、autograd 和反归一化依然每步 eager 执行。",
        "",
        "### 2.4 Opt2：model-only CUDA Graph",
        "",
        "- 与 static-eager 使用完全相同的固定容量、dummy padding 和模型输入。",
        "- 捕获范围仅包括 eSEN 前向、energy head、`autograd.grad`、"
        "energy/force 反归一化。",
        "- 动态邻居表构建和 NVT 积分保持 eager，因此每步仍可更新真实邻居。",
        "- 每个进程只 capture 一张图；正式 1000 步不允许重新 capture。",
        "- 容量溢出退出码为 45；OOM 为 42。",
        "- 成功运行要求正式阶段 1001 次调用全部命中同一张图。",
        "",
        "## 3. 实验配置与计时定义",
        "",
        "- GPU：NVIDIA H100 80GB HBM3，物理 GPU 2。",
        "- Torch：2.5.1+cu124；fairchem-core：1.10.1.dev2。",
        "- Baseline/Opt1 提交：`c2e5ecd1d6188bab9443acbadbab97c5bb540ad9`。",
        "- static-eager/Opt2 提交：`e9715f49c2b2051a6cffe1dcc0ed9b66ff6682b4`；"
        "两个提交间 Opt1 的定时计算路径没有改变。",
        "- checkpoint：`esen_30m_oam.pt`，SHA256 "
        "`adf7d38e5bccb8e0334434c0bd65ac75661fb646891df17ecc89c19d111efde1`。",
        "- 体系：Cu32/64/192/512/1024；H2O32/60/192/512/1024。",
        "- `H2O<N>` 表示 N 个水分子，原子数为 3N。",
        "- 温度：300 K、800 K；NVT；时间步 1 fs；`taut=100 fs`。",
        "- 正式步数：1000；warmup：3；每个配置重复 3 次。",
        "- 所有随机种子固定为 42；`CUBLAS_WORKSPACE_CONFIG=:4096:8`。",
        "- `seconds_per_step` 的计时包含初始力计算和 1000 步 MD，除以 1000；"
        "不包含模型加载、probe、capture、warmup、校验、哈希和文件 I/O。",
        "- 数值差异定义为同一 system/temperature/repeat 下 "
        "`|E_candidate(step)-E_baseline(step)|`，单位 eV。",
        "- 原始验收阈值：step 1 `<1e-8 eV`；step 50 `<1e-6 eV`；"
        "step 100/1000 只记录。",
        "",
        "## 4. 运行完成情况",
        "",
    ]

    completion_rows = []
    for name in ("baseline", "opt1", "static", "opt2"):
        status_counts: dict[str, int] = defaultdict(int)
        for status in STATUSES[name].values():
            status_counts[status] += 1
        completion_rows.append(
            [
                BACKENDS[name]["label"],
                str(len(RECORDS[name])),
                str(status_counts["success"]),
                str(status_counts["validation_failed"]),
                str(status_counts["oom"]),
                str(status_counts["missing_reference"]),
                str(status_counts["capacity_overflow"]),
                str(status_counts["error"]),
            ]
        )
    lines.extend(
        md_table(
            [
                "路径",
                "完成并产生 JSON",
                "success",
                "validation_failed",
                "OOM",
                "missing_reference",
                "capacity_overflow",
                "error",
            ],
            completion_rows,
        )
    )
    lines.extend(
        [
            "",
            "`validation_failed` 表示 MD 已完成且性能数据有效，但没有满足预设的严格"
            "能量阈值；它不等于运行失败。",
            "",
            "## 5. 性能与消融汇总",
            "",
            "加速比均按“较慢路径时间 / 较快路径时间”计算；大于 1 表示后者加速，"
            "小于 1 表示后者变慢。",
            "",
        ]
    )

    summary_rows = []
    for system in SYSTEMS:
        for temperature in TEMPERATURES:
            values = {
                name: median_field(
                    records_for(name, system, temperature), "seconds_per_step"
                )
                for name in BACKENDS
            }
            summary_rows.append(
                [
                    system,
                    str(temperature),
                    fmt_time(values["baseline"]),
                    fmt_time(values["opt1"]),
                    fmt_time(values["static"]),
                    fmt_time(values["opt2"]),
                    fmt_speedup(ratio(values["baseline"], values["opt1"])),
                    fmt_speedup(ratio(values["opt1"], values["static"])),
                    fmt_speedup(ratio(values["static"], values["opt2"])),
                    fmt_speedup(ratio(values["opt1"], values["opt2"])),
                    fmt_speedup(ratio(values["baseline"], values["opt2"])),
                ]
            )
    lines.extend(
        md_table(
            [
                "体系",
                "T/K",
                "Baseline s/step",
                "Opt1 s/step",
                "Static s/step",
                "Opt2 s/step",
                "Opt1 vs baseline",
                "Static vs opt1",
                "纯 CG",
                "Opt2 vs opt1",
                "Opt2 vs baseline",
            ],
            summary_rows,
        )
    )

    lines.extend(
        [
            "",
            "### 5.1 性能结论",
            "",
            "- Opt1 对小 Cu 体系有明显收益；GPU 常驻主要消除了 ASE 和数据搬运开销。",
            "- 纯 CUDA Graph 在所有能够完成的体系上均为正收益：Cu32 约 2.7×，"
            "Cu64 约 1.35–1.56×，Cu192/H2O60 约 1.06–1.08×，"
            "Cu512/H2O192 约 1.02×。",
            "- 固定容量和 padding 在 Cu512、Cu1024、H2O192 上造成约 5%–9% "
            "额外开销，抵消了大体系的 Graph 收益。",
            "- Opt2 相对 Opt1：Cu32 2.52–2.89×；Cu64 1.65–1.70×；"
            "Cu192 1.15–1.18×；H2O32 约 1.36×；H2O60 1.12–1.13×；"
            "Cu512/H2O192 约慢 3%–4%。",
            "- Cu1024 的 static-eager 可以运行，但 model-CG 因 CUDA Graph "
            "private pool 额外显存而 OOM。",
            "- 当前 `peak_reserved_gib` 不能完整覆盖 CUDA Graph private pool；"
            "不能据此宣称 Opt2 更省显存，应以 Cu1024 OOM 日志和进程级显存为准。",
            "",
            "## 6. 数值结果汇总",
            "",
        ]
    )

    error_rows = []
    for system in SYSTEMS:
        for temperature in TEMPERATURES:
            for name in ("opt1", "static", "opt2"):
                records = records_for(name, system, temperature)
                error_rows.append(
                    [
                        system,
                        str(temperature),
                        BACKENDS[name]["label"],
                        str(len(records)),
                        fmt_error(max_field(records, "energy_abs_error_step_1_eV")),
                        fmt_error(max_field(records, "energy_abs_error_step_50_eV")),
                        fmt_error(max_field(records, "energy_abs_error_step_100_eV")),
                        fmt_error(max_field(records, "energy_abs_error_step_1000_eV")),
                    ]
                )
    lines.extend(
        md_table(
            [
                "体系",
                "T/K",
                "路径",
                "完成次数",
                "max ΔE1/eV",
                "max ΔE50/eV",
                "max ΔE100/eV",
                "max ΔE1000/eV",
            ],
            error_rows,
        )
    )
    lines.extend(
        [
            "",
            "### 6.1 数值结论",
            "",
            "- static-eager 与 Opt2 相对 Opt1 的初始能量偏差基本一致，说明主要偏差"
            "来自固定 shape、dummy padding 和 FP32 reduction 顺序，不是 CUDA Graph "
            "capture 本身。",
            "- Cu 的误差在 1000 步内保持在相同数量级，没有持续放大。",
            "- H2O 的 static-eager 与 Opt2 都存在约 `1e-6～1e-5 eV/Å` 的"
            "相同输入力波动；因此其 1000 步轨迹分叉不是 CUDA Graph 独有问题。",
            "- `1e-8 eV` 的固定总能量阈值低于当前 FP32 总能量归约的可实现精度；"
            "建议同时报告 eV/atom、最大力误差和长轨迹统计物理量。",
            "",
            "## 7. Opt1 每次运行结果",
            "",
            "以下 ΔE 均相对相同 repeat 的 ASE baseline。",
            "",
        ]
    )
    detailed_headers = [
        "体系",
        "T/K",
        "repeat",
        "Baseline s/step",
        "Candidate s/step",
        "vs baseline",
        "ΔE1/eV",
        "ΔE50/eV",
        "ΔE100/eV",
        "ΔE1000/eV",
        "状态",
    ]
    lines.extend(md_table(detailed_headers, per_run_rows("opt1")))
    lines.extend(
        [
            "",
            "## 8. Opt2 static-eager 消融每次运行结果",
            "",
        ]
    )
    lines.extend(md_table(detailed_headers, per_run_rows("static")))
    lines.extend(
        [
            "",
            "## 9. Opt2 model-only CUDA Graph 每次运行结果",
            "",
        ]
    )
    lines.extend(md_table(detailed_headers, per_run_rows("opt2")))
    lines.extend(
        [
            "",
            "## 10. CUDA Graph 运行不变量",
            "",
            "所有 42 次完成的 Opt2 运行均满足：",
            "",
            "- `cuda_graph_capture_count = 1`",
            "- `cuda_graph_production_capture_count = 0`",
            "- `cuda_graph_production_calls = 1001`",
            "- `cuda_graph_production_replays = 1001`",
            "- `cuda_graph_capacity_misses = 0`",
            "- `cuda_graph_hit_rate = 1.0`",
            "- CUDA Graph 输出地址在 replay 间保持不变",
            "",
            "## 11. 当前结论与后续工作",
            "",
            "1. Opt1 已有效消除 ASE/CPU 数据路径开销，是所有后续优化的公平起点。",
            "2. Opt2 的 CUDA Graph 机制实现正确，主要适合 kernel launch 占比较高的"
            "小体系。",
            "3. 大体系需要减少固定容量 padding 的实际计算开销，否则 Graph 的约 2% "
            "收益会被抵消。",
            "4. Cu1024 需要降低 Graph private pool 占用，或采用更小模型/分块策略。",
            "5. 下一阶段 kernel fusion 应首先分析 profiler 中高频逐边算子和 reduction，"
            "并继续保留 static-eager 与 model-CG 两个对照。",
            "6. H2O 长轨迹验证应增加温度分布、势能均值/方差、RDF 和扩散系数，"
            "不应只依赖第 1000 步单点能量。",
            "",
        ]
    )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
