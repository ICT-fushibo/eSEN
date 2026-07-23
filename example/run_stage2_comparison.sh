#!/usr/bin/env bash
set -euo pipefail

# Run opt2 against existing ASE baseline and opt1 result directories, then
# produce the three-backend comparison.  Baseline OOM/missing references do
# not suppress opt2 performance attempts.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${BASELINE_DIR:?Set BASELINE_DIR to the ASE baseline output directory}"
: "${GPU_EAGER_DIR:?Set GPU_EAGER_DIR to the opt1 output directory}"

RUN_ID=${RUN_ID:-"esen_stage2_$(date '+%Y%m%d_%H%M%S')"}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
MODEL_CG_DIR=${MODEL_CG_DIR:-"$ROOT_OUTPUT_DIR/model_cg"}

mkdir -p "$ROOT_OUTPUT_DIR"

RUN_ID="${RUN_ID}_model_cg" \
OUTPUT_DIR="$MODEL_CG_DIR" \
BASELINE_DIR="$BASELINE_DIR" \
REQUIRE_BASELINE_REFERENCE=0 \
    bash "$REPO_ROOT/example/run_md_model_cuda_graph.sh"

python "$REPO_ROOT/example/compare_md_backends.py" \
    --baseline-dir "$BASELINE_DIR" \
    --gpu-dir "$GPU_EAGER_DIR" \
    --model-cg-dir "$MODEL_CG_DIR" \
    --output-dir "$ROOT_OUTPUT_DIR"

echo "Stage-2 comparison: $ROOT_OUTPUT_DIR"
