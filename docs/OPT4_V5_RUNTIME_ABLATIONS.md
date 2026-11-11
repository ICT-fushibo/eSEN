# Opt4 v5：三个独立运行时消融

冻结基线不变：Opt4 v5 FP32 / TF32 off、FP64 MD、dense 构图、CAP1 auto-safe、ROB1 窗口 10。
只做 whole-step；不组合三个候选，不启用 CAP2、CELL1 或额外 GEMM 改动，不启动正式矩阵。

| 单独开关 | 本轮变化 | 不变的部分 |
|---|---|---|
| `nhc` | Triton 单 CTA FP64 NHC half-step，支持 1–1024 atoms、tchain=3、tloop=1 | Suzuki–Yoshida 顺序、dt、质量、温度、kick/drift、事务快照 |
| `rob-compact` | 正常窗口只读 overflow scalar；仅失败时读取逐原子需求 | 每窗口同步、snapshot/restore、溢出判断、扩容、重捕获 |
| `shared-geometry` | 每次力评估内共享 10 个 Edgewise 的 Wigner 正/逆切片与 envelope | 所有 cuBLAS BMM/Linear、SO2、scatter、完整坐标梯度 |

共享几何不跨 step、不 detach，不共享 edge-degree embedding 的计算。
NHC 仅融合 thermostat half-step，不融合整个 MD step。该内核只做推理，拒绝需要梯度的输入。
默认 `--opt4-runtime-experiment none`；旧 CLI 默认值与冻结 tag 均未修改。

## 实验设计与判据

- 小体系：`anthracene_293K_Sharma_S`（48 atoms）。
- 大体系：`bulkLiMgAlZnSn_900K_J_Schmidt_VASP`（500 atoms）。
- smoke：30 steps，record_interval=10，每个体系 baseline/candidate 各一次。
- 消融：1,000 steps，3 repeats，逐对 AB/BA 交错、同对同 GPU；每个实验有自己的 v5 baseline。
- GPU 0：NHC；GPU 1：ROB 正常路径；GPU 2：共享几何。默认开跑前轮询等待 GPU 无计算进程且低利用率持续 10 秒。
- dt=0.25 fs、tdamp=25 fs、seed=0，温度读取各体系元数据。固定 probe=50、dummy=32、margin=0.10、slot rounding=8。
- 本轮不计算榜单统计量或离线 stress；这些任务不进入运行时间。

smoke 检查进程成功、实际开关、初始 E/F、Graph/ROB invariants、完整有限轨迹、同对容量布局一致和显存。
smoke 不因候选变慢而拒绝。消融必须有同代码/数据/参数/GPU 的成功 smoke，不能只提供一个 `accepted=true` 文件绕过检查。

消融初筛：两体系 median paired speedup 的几何平均 >=1.01x；任一体系不能 3/3 同方向慢超过 1%；
每体系 median peak-reserved 增量 <=1 GiB；所有运行正常窗口应为零 rollback、零 recovery capture。
一次初筛通过不等于正式接受/冻结。出现溢出不表示 ROB 恢复失败，但该运行不属于本轮正常路径计时样本。
两侧初始逐原子容量必须完全一致，否则拒绝把容量差异归因给本轮候选。

`seconds_per_step` 沿用原 runner 的 rollout wall-time 口径，包含记录与 ROB 窗口成本；setup/capture 单独记录。
HDF5 初始跨方案误差要求 energy/atom <1e-5 eV、force max <2e-4 eV/Å。
不要求长轨迹逐点一致；这不能替代后续物理统计验证。

## 服务器：gxn75

先将本次新增/修改文件同步到 `/home/fushibo/eSEN`。单纯 `git pull` 不会取得尚未提交的本地改动。

```bash
conda activate esen_opt
cd /home/fushibo/eSEN
export REFERENCE_H5=/home/fushibo/matbench-discovery-data/2026-06-29-dynamat-v1.0-reference-trajectories.h5
export MATBENCH_REPO=/home/fushibo/matbench-discovery
export CHECKPOINT=/home/fushibo/eSEN/esen_30m_oam.pt
export GPU_LIST="0 1 2"
export PYTHONPATH="$PWD/src:$PWD/example"

test -f "$REFERENCE_H5" && test -f "$CHECKPOINT"
bash -n example/run_opt4_runtime_experiments.sh
python example/run_opt4_runtime_ablation.py --help
```

### 单测：先运行，失败则不要启动 smoke

```bash
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 \
python -m pytest -q --noconftest \
    tests/core/applications/test_opt4_runtime_runner.py \
    tests/core/applications/test_esen_runtime_experiments.py \
    tests/core/applications/test_esen_matbench.py \
    tests/core/applications/test_esen_opt4_model_fusion.py
```

新增覆盖：NHC 20 次 half-step、100 harmonic steps、零/负 delta、非连续输入、
CUDA Graph 改变输入后重放、thermostat restore；ROB 第三步注入溢出并完整回滚；
共享几何的多消费者输出/梯度、空边、cutoff 隔离与 FusedEdgewise Graph；
错误配置、HDF5 重复/缺失帧、容量不一致、失败结果与 smoke 前置检查。

### Smoke：三个实验并行，各自独立

```bash
export SMOKE_ROOT="$PWD/example/md_out/opt4_v5_runtime_smoke_$(date '+%Y%m%d_%H%M%S')"
nohup env PHASE=smoke OUTPUT_ROOT="$SMOKE_ROOT" \
    bash example/run_opt4_runtime_experiments.sh \
    > "${SMOKE_ROOT}.log" 2>&1 &
export SMOKE_PID=$!
echo "PID=$SMOKE_PID ROOT=$SMOKE_ROOT"
```

```bash
tail -f "${SMOKE_ROOT}.log"
# Ctrl-C 只退出 tail。等运行结束：
wait "$SMOKE_PID"
echo "smoke exit=$?"
for EXP in nhc rob-compact shared-geometry; do
    python example/run_opt4_runtime_ablation.py \
        --output-dir "$SMOKE_ROOT/$EXP" --select-only
done
```

### 消融：成功 smoke 后运行，不自动组合候选

```bash
export ABL_ROOT="$PWD/example/md_out/opt4_v5_runtime_ablation_$(date '+%Y%m%d_%H%M%S')"
nohup env PHASE=ablation OUTPUT_ROOT="$ABL_ROOT" SMOKE_ROOT="$SMOKE_ROOT" \
    bash example/run_opt4_runtime_experiments.sh \
    > "${ABL_ROOT}.log" 2>&1 &
export ABL_PID=$!
echo "PID=$ABL_PID ROOT=$ABL_ROOT"
```

```bash
tail -f "${ABL_ROOT}.log"
wait "$ABL_PID"
echo "ablation exit=$?"
for EXP in nhc rob-compact shared-geometry; do
    python example/run_opt4_runtime_ablation.py \
        --output-dir "$ABL_ROOT/$EXP" --select-only
done
```

每个实验的 `selection.json` 为独立结论。顶层 shell exit=1 表示至少一个失败/拒绝，
不表示另两个也拒绝。每对完整命令、日志和 JSON/HDF5 保存在各实验 `runs/`，`pairs/` 索引当前完成对。

### 断点续跑

不要改变 `OUTPUT_ROOT`、GPU 分配或源代码。只跳过完整通过的 pair；半对/失败对重跑两侧，旧尝试保留。
源文件、实验配置、Python/依赖版本、GPU UUID/驱动、数据文件路径/大小/mtime 变化会拒绝复用目录。数据身份校验不是内容哈希；不要在保留大小/mtime 的情况下替换数据。
请勿同时启动两份同目录/同 GPU 的 runner；GPU 空闲检查不是跨进程预约锁。

```bash
# 消融断点续跑；smoke 同理使用 PHASE=smoke、OUTPUT_ROOT="$SMOKE_ROOT"
nohup env RESUME=1 PHASE=ablation OUTPUT_ROOT="$ABL_ROOT" SMOKE_ROOT="$SMOKE_ROOT" \
    bash example/run_opt4_runtime_experiments.sh \
    >> "${ABL_ROOT}.log" 2>&1 &
export ABL_PID=$!
```

修复代码后应使用新的 smoke/消融目录，不能跳过旧配置检查。

## 本地验证边界

开发机为 Windows CPU，无 CUDA/Triton 执行环境，且缺少完整 eSEN 依赖。
已运行隔离的 CPU 算法/选择器测试（绕开未使用的模型加载入口，若入口被调用立即报错）；
这些结果不代表完整模型集成或 GPU kernel 已通过。上面的服务器单测与 smoke 是必需关卡。
本轮不改变 Opt4 v5 冻结配置，也未运行服务器消融或正式实验。

本地已完成：26 项 CLI/HDF5/selector 测试通过；隔离 CPU 算法检查 22 通过、17 CUDA 跳过；
11 个 Python 文件通过 Python 3.9 语法解析，Bash 入口通过 `bash -n`。
FP64 exp 使用 Triton 3.1 的 `cuda.libdevice.exp` 双精度分派，接口核对见
[Triton 3.1 源码](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/extra/cuda/libdevice.py#L1053)。
