# Opt4 PREC1: isolated TF32 experiment

PREC1 compares the frozen Opt4 v3 configuration with TF32 disabled and
enabled.  Model fusions, neighbor capacity, MD dtype, seed, system order, and
CUDA Graph structure are held constant.  The default benchmark behavior is
unchanged because `--tf32-mode` defaults to `off`.

The candidate sets CUDA matmul and cuDNN TF32 flags, requests PyTorch
`float32_matmul_precision=high`, and verifies all three settings before model
construction.  Every JSON records the requested and effective policy.

Energy and force differences are telemetry only for this short performance
experiment.  A later Matbench trajectory evaluation is the correctness gate.

Phases for `run_opt4_prec1_8gpu.py`:

- `PREC1_PHASE=smoke`: four representative systems, 300 K, 1 step, 1 repeat.
- `PREC1_PHASE=ablation`: four representative systems, 300 K, 100 steps,
  3 interleaved repeats.
- `PREC1_PHASE=formal`: ten systems, 300/800 K, 100 steps, 3 interleaved
  repeats.  This phase requires `PREC1_SELECTION_DIR` containing accepted
  model-only and whole-step selector JSON files.

The polling scheduler is resumable by default.  Reuse the same
`ROOT_OUTPUT_DIR` to skip completed result JSON files.
