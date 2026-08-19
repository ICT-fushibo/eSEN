#!/usr/bin/env bash
set -euo pipefail

# Poll GPU_LIST and launch one Matbench system after a GPU has remained idle
# for GPU_IDLE_SECONDS.  All four backends for that system stay on that GPU.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_ROOT
: "${MATBENCH_SAVE_DIR:=${SAVE_DIR:-$REPO_ROOT/example/md_out/esen_matbench_8gpu_$(date '+%Y%m%d_%H%M%S')}}"
export MATBENCH_SAVE_DIR
exec "${PYTHON:-python}" "$REPO_ROOT/example/run_esen_matbench_8gpu.py" "$@"
