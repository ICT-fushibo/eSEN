#!/usr/bin/env bash
set -euo pipefail

# Pure FP32-vs-TF32 NSYS comparison for the frozen Opt4 v3 kernels.  Both
# variants use the same fusion mask, uniform neighbor policy, and an initial-
# frame-only probe so their fixed capacities are identical.  This deliberately
# isolates precision/kernel dispatch from CAP1 auto-safe trajectory divergence.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}

export PROFILE_LABEL=${PROFILE_LABEL:-opt4_v3_prec1_locked_capacity}
export FROZEN_CONFIG_TAG=${FROZEN_CONFIG_TAG:-opt4-v3-kf12-cap1-auto-safe}
export RUN_ID=${RUN_ID:-"opt4_prec1_nsys_gpu${GPU}_$(date '+%Y%m%d_%H%M%S')"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
export GPU
export SCOPES=${SCOPES:-"model-only whole-step"}
export SYSTEMS=${SYSTEMS:-"Cu512 H2O192"}
export TEMPERATURE=${TEMPERATURE:-300}
export TRACE_STEPS=${TRACE_STEPS:-20}

V3_MODEL_FUSIONS=so2-epilogue,so2-gate-bridge,so2-block-gemm
V3_WHOLE_FUSIONS=rmsnorm,$V3_MODEL_FUSIONS

export MODEL_BASE_STAGE=${MODEL_BASE_STAGE:-OPT4V3_FP32}
export MODEL_CANDIDATE_STAGE=${MODEL_CANDIDATE_STAGE:-PREC1_TF32}
export MODEL_BASE_FUSIONS=${MODEL_BASE_FUSIONS:-$V3_MODEL_FUSIONS}
export MODEL_CANDIDATE_FUSIONS=${MODEL_CANDIDATE_FUSIONS:-$V3_MODEL_FUSIONS}
export MODEL_BASE_TF32_MODE=off
export MODEL_CANDIDATE_TF32_MODE=on
export MODEL_PROBE_STEPS=${MODEL_PROBE_STEPS:-0}

export WHOLE_BASE_STAGE=${WHOLE_BASE_STAGE:-OPT4V3_FP32}
export WHOLE_CANDIDATE_STAGE=${WHOLE_CANDIDATE_STAGE:-PREC1_TF32}
export WHOLE_BASE_FUSIONS=${WHOLE_BASE_FUSIONS:-$V3_WHOLE_FUSIONS}
export WHOLE_CANDIDATE_FUSIONS=${WHOLE_CANDIDATE_FUSIONS:-$V3_WHOLE_FUSIONS}
export WHOLE_BASE_TF32_MODE=off
export WHOLE_CANDIDATE_TF32_MODE=on
export WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY=uniform
export WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY=uniform
export WHOLE_PROBE_STEPS=${WHOLE_PROBE_STEPS:-0}

bash "$REPO_ROOT/example/run_opt4_v1_profiling.sh"

"${PYTHON:-python}" -u "$REPO_ROOT/example/analyze_opt4_profiling.py" \
    --input-dir "$OUTPUT_DIR" \
    --base-stage "$MODEL_BASE_STAGE" \
    --candidate-stage "$MODEL_CANDIDATE_STAGE"

echo "PREC1 locked-capacity profiling report: $OUTPUT_DIR/opt4_v3_profiling.md"
