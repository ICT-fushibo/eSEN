#!/usr/bin/env bash
set -euo pipefail

# Confirm the frozen Opt4 v3 configuration against Opt4 v2.  Each scope is
# profiled on one physical GPU so graph-duration comparisons are not affected
# by cross-GPU clock differences.  The generic profiler excludes setup, Triton
# compilation, probing, capture, and warmup via cudaProfilerStart/Stop.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}

export PROFILE_LABEL=${PROFILE_LABEL:-opt4_v3_kf12_cap1_auto_safe}
export FROZEN_CONFIG_TAG=${FROZEN_CONFIG_TAG:-opt4-v3-kf12-cap1-auto-safe}
export RUN_ID=${RUN_ID:-"opt4_v3_nsys_gpu${GPU}_$(date '+%Y%m%d_%H%M%S')"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
export GPU
export SCOPES=${SCOPES:-"model-only whole-step"}
export SYSTEMS=${SYSTEMS:-"Cu512 H2O192"}
export TEMPERATURE=${TEMPERATURE:-300}
export TRACE_STEPS=${TRACE_STEPS:-20}

export MODEL_BASE_STAGE=${MODEL_BASE_STAGE:-OPT4V2}
export MODEL_BASE_FUSIONS=${MODEL_BASE_FUSIONS:-so2-epilogue,so2-gate-bridge}
export MODEL_CANDIDATE_STAGE=${MODEL_CANDIDATE_STAGE:-OPT4V3}
export MODEL_CANDIDATE_FUSIONS=${MODEL_CANDIDATE_FUSIONS:-so2-epilogue,so2-gate-bridge,so2-block-gemm}

export WHOLE_BASE_STAGE=${WHOLE_BASE_STAGE:-OPT4V2}
export WHOLE_BASE_FUSIONS=${WHOLE_BASE_FUSIONS:-rmsnorm,so2-epilogue,so2-gate-bridge}
export WHOLE_CANDIDATE_STAGE=${WHOLE_CANDIDATE_STAGE:-OPT4V3}
export WHOLE_CANDIDATE_FUSIONS=${WHOLE_CANDIDATE_FUSIONS:-rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm}
export WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY=${WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY:-uniform}
export WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY=${WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY:-auto-safe}
export NEIGHBOR_AUTO_MIN_REDUCTION=${NEIGHBOR_AUTO_MIN_REDUCTION:-0.05}
export NEIGHBOR_AUTO_GUARD_SLOTS=${NEIGHBOR_AUTO_GUARD_SLOTS:-1}
export WHOLE_PROBE_STEPS=${WHOLE_PROBE_STEPS:-100}

bash "$REPO_ROOT/example/run_opt4_v1_profiling.sh"

"${PYTHON:-python}" -u "$REPO_ROOT/example/analyze_opt4_profiling.py" \
    --input-dir "$OUTPUT_DIR" \
    --base-stage "$MODEL_BASE_STAGE" \
    --candidate-stage "$MODEL_CANDIDATE_STAGE"

echo "Opt4 v3 profiling report: $OUTPUT_DIR/opt4_v3_profiling.md"
