#!/usr/bin/env bash
set -euo pipefail

# Large-system KF6-KF8 ablation on a polling GPU pool.  Stage selection stays
# serial (KF6 -> KF7 -> KF8); tasks within one stage are distributed across
# GPUs that have remained idle for two minutes.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_ROOT
export GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
export SCOPES=${SCOPES:-both}
export OPT4_STAGES=${OPT4_STAGES:-"KF6 KF7 KF8"}
export SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O512"}
export OPT4_FOCUS_SYSTEMS=${OPT4_FOCUS_SYSTEMS:-"Cu512 H2O512"}
export TEMPERATURES=${TEMPERATURES:-300}
export STEPS=${STEPS:-100}
export WARMUP_STEPS=${WARMUP_STEPS:-3}
export REPEATS=${REPEATS:-5}
export SELECT_MIN_PAIRED=${SELECT_MIN_PAIRED:-5}
export SELECT_MIN_FASTER=${SELECT_MIN_FASTER:-4}
export INITIAL_ACCEPTED_MODEL_ONLY=${INITIAL_ACCEPTED_MODEL_ONLY:-}
export INITIAL_ACCEPTED_WHOLE_STEP=${INITIAL_ACCEPTED_WHOLE_STEP:-rmsnorm}
export GPU_IDLE_SECONDS=${GPU_IDLE_SECONDS:-120}
export GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}
export GPU_IDLE_MEMORY_MIB=${GPU_IDLE_MEMORY_MIB:-1024}
export GPU_IDLE_UTILIZATION_PERCENT=${GPU_IDLE_UTILIZATION_PERCENT:-5}
: "${OPT4_SAVE_DIR:=${ROOT_OUTPUT_DIR:-$REPO_ROOT/example/md_out/esen_opt4_kf6_8_8gpu_$(date '+%Y%m%d_%H%M%S')}}"
export ROOT_OUTPUT_DIR=$OPT4_SAVE_DIR

exec "${PYTHON:-python}" "$REPO_ROOT/example/run_opt4_model_fusion_8gpu.py" "$@"
