#!/usr/bin/env bash
set -euo pipefail

# Compare the frozen Opt4 v1/KF9 configuration with accepted Opt4 v2/KF10.
# The generic profiler keeps graph/node traces and all exported summaries
# separate from timing and Matbench outputs.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}

export PROFILE_LABEL=${PROFILE_LABEL:-opt4_v2_kf10}
export RUN_ID=${RUN_ID:-"opt4_v2_kf10_nsys_gpu${GPU}_$(date '+%Y%m%d_%H%M%S')"}
export GPU
export MODEL_BASE_STAGE=${MODEL_BASE_STAGE:-OPT4V1}
export MODEL_BASE_FUSIONS=${MODEL_BASE_FUSIONS:-so2-epilogue}
export MODEL_CANDIDATE_STAGE=${MODEL_CANDIDATE_STAGE:-KF10}
export MODEL_CANDIDATE_FUSIONS=${MODEL_CANDIDATE_FUSIONS:-so2-epilogue,so2-gate-bridge}
export WHOLE_BASE_STAGE=${WHOLE_BASE_STAGE:-OPT4V1}
export WHOLE_BASE_FUSIONS=${WHOLE_BASE_FUSIONS:-rmsnorm,so2-epilogue}
export WHOLE_CANDIDATE_STAGE=${WHOLE_CANDIDATE_STAGE:-KF10}
export WHOLE_CANDIDATE_FUSIONS=${WHOLE_CANDIDATE_FUSIONS:-rmsnorm,so2-epilogue,so2-gate-bridge}

exec bash "$REPO_ROOT/example/run_opt4_v1_profiling.sh"
