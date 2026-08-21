#!/usr/bin/env bash
set -euo pipefail

# NSYS graph/node confirmation for an accepted CAP1-auto allocation.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}

export PROFILE_LABEL=opt4_v2_cap1_auto
export RUN_ID=${RUN_ID:-"opt4_v2_cap1_auto_nsys_gpu${GPU}_$(date '+%Y%m%d_%H%M%S')"}
export GPU
export SCOPES=whole-step
export WHOLE_BASE_STAGE=OPT4V2
export WHOLE_BASE_FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge
export WHOLE_CANDIDATE_STAGE=CAP1AUTO
export WHOLE_CANDIDATE_FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge
export WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY=uniform
export WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY=auto
export NEIGHBOR_AUTO_MIN_REDUCTION=${NEIGHBOR_AUTO_MIN_REDUCTION:-0.05}

exec bash "$REPO_ROOT/example/run_opt4_v1_profiling.sh"
