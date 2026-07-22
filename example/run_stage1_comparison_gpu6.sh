#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${RUN_ID:-"esen_stage1_$(date '+%Y%m%d_%H%M%S')"}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
GPU=${GPU:-6}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}

mkdir -p "$ROOT_OUTPUT_DIR"

common_env=(
    "GPU=$GPU"
    "STEPS=$STEPS"
    "WARMUP_STEPS=$WARMUP_STEPS"
    "REPEATS=$REPEATS"
    "SEED=42"
    "SYSTEMS=$SYSTEMS"
    "TEMPERATURES=$TEMPERATURES"
    "STRICT=0"
)

env "${common_env[@]}" \
    RUN_ID="${RUN_ID}_ase" \
    OUTPUT_DIR="$ROOT_OUTPUT_DIR/ase" \
    bash "$REPO_ROOT/example/run_md_baselines.sh"

env "${common_env[@]}" \
    RUN_ID="${RUN_ID}_gpu_eager" \
    OUTPUT_DIR="$ROOT_OUTPUT_DIR/gpu_eager" \
    BASELINE_DIR="$ROOT_OUTPUT_DIR/ase" \
    BASELINE_STEPS="$STEPS" \
    REQUIRE_BASELINE_REFERENCE=1 \
    bash "$REPO_ROOT/example/run_md_gpu_resident.sh"

python "$REPO_ROOT/example/compare_md_backends.py" \
    --baseline-dir "$ROOT_OUTPUT_DIR/ase" \
    --gpu-dir "$ROOT_OUTPUT_DIR/gpu_eager" \
    --output-dir "$ROOT_OUTPUT_DIR"

echo "Stage-1 comparison: $ROOT_OUTPUT_DIR"
