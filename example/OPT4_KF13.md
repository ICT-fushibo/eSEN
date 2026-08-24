# Opt4 KF13: Frozen SO3 Weight Cache

KF13 removes the 20 per-forward `SO3_Linear.weight.index_select` operations in
the ten SpectralAtomwise blocks.  The degree-expanded `[16,128,128]` FP32
weights are copied once while the frozen inference model is configured.  The
official `torch.einsum`/cuBLAS path, activation, model parameters, and input
gradient path are otherwise unchanged.

The cache is 1 MiB per SO3 linear and 20 MiB for the 30M model.  Result JSONs
record replacement count, expanded-weight count, cache bytes, and version
`opt4-model-fusion-v7-so3-weight-cache`.

## Experiment modes

`example/run_opt4_kf13_8gpu.py` polls idle GPUs and supports:

- `KF13_PRECISION=fp32`: Opt4 v3 FP32 versus Opt4 v3 FP32 + KF13.
- `KF13_PRECISION=tf32`: PREC1 TF32 versus PREC1 TF32 + KF13.
- `KF13_PHASE=smoke`: four representative systems, 300 K, 1 step, 1 repeat.
- `KF13_PHASE=ablation`: the same systems, 300 K, 100 steps, 3 repeats.
- `KF13_PHASE=formal`: ten systems, 300/800 K, 100 steps, 3 repeats.

Both sides of an A/B pair use the same precision and neighbor-capacity policy.
Whole-step holds CAP1-auto-safe fixed; model-only uses uniform capacity.
Energy and force validation remain telemetry and do not gate timing selection.

Run the matching selector after an ablation:

```bash
KF13_PRECISION=fp32 ROOT_OUTPUT_DIR=/path/to/ablation \
    bash example/select_opt4_kf13.sh
```

Formal mode requires `KF13_SELECTION_DIR` pointing to that selected ablation.
TF32 and FP32 selections are intentionally separate.

## PREC1 profiling

`example/run_opt4_prec1_profiling.sh` compares FP32 and TF32 for Cu512 and
H2O192 with NSYS graph/node traces.  It uses an initial-frame-only probe and
uniform whole-step capacity so base and candidate capture the same padded
shapes.  Setup, probing, capture, and warmup remain outside the profiler range.
