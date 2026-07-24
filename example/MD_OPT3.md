# eSEN Opt3：Whole-step CUDA Graph

Opt3 不修改 baseline、Opt1 或 Opt2。它新增两个使用同一固定形状动态邻居
构图器的后端：

- `fixed-builder-model-cg`：邻居构图在 CUDA Graph 外运行，模型和保守力
  autograd 使用 model-only CUDA Graph。
- `whole-step-cg`：Berendsen NVT 积分、邻居构图、模型、autograd 和状态更新
  位于同一张 CUDA Graph 中。

两者的差异用于隔离“扩大 CUDA Graph 捕获范围”的收益。Opt3 不启用 AMP、
TF32、`torch.compile`、Triton 或其他 kernel fusion。

## 数值与运行约定

- seed 固定为 42，`CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- MD 状态为 FP64，模型为 checkpoint 原有 FP32。
- 50 步 eager 探测每原子邻居容量，增加 10% 余量并按 8 向上取整。
- 每次 replay 都根据当前坐标重建邻居，未使用的固定槽位只连接 dummy atoms。
- 初始力和 1000 步 MD 共 replay 1001 次，只捕获一张图。
- 1、50 步工程阈值为 `1e-5 eV/atom`；初始最大力误差阈值为
  `2e-4 eV/Å`。
- 历史总能量阈值仍会报告，但不决定工程验证状态。
- 数值失败或容量溢出不会中断轨迹；结果写入后分别返回 43 或 45。

## 先运行烟测

```bash
cd /public-data/fushibo/eSEN

BASELINE_DIR=/path/to/ase_baseline \
GPU=6 \
bash example/test_md_opt3.sh
```

可以通过 `SYSTEMS`、`TEMPERATURES` 和 `STEPS` 调整烟测范围。正式测试前建议
依次运行 Cu32 和 H2O32 的 10 步、100 步测试。

## 完整 Opt3 与消融

```bash
cd /public-data/fushibo/eSEN

BASELINE_DIR=/path/to/ase_baseline \
GPU_EAGER_DIR=/path/to/opt1_gpu_eager \
MODEL_CG_DIR=/path/to/opt2_model_cg \
GPU=6 \
nohup bash example/run_opt3_full_ablation.sh \
  > example/opt3_full_ablation.log 2>&1 &
```

默认运行 10 个体系、300/800 K、1000 步和 3 次重复。输出目录包含：

- `fixed_builder_model_cg/`：固定构图 + model-only CG 结果。
- `whole_step_cg/`：whole-step CG 结果。
- `opt3_ablation.tsv/.md`：各体系中位性能、加速比、OOM 和验证状态。
- `opt3_runs.tsv/.md`：每次运行的 seconds/step，以及 1、50、100、1000
  步总能量和每原子能量误差。

runner 不会启动、关闭或配置 NVIDIA MPS，只设置 `CUDA_VISIBLE_DEVICES`。
