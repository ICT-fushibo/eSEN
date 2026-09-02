#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OPT4_V5_PHASE=${OPT4_V5_PHASE:-formal}
export ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_v5_${OPT4_V5_PHASE}_8gpu_$(date '+%Y%m%d_%H%M%S')"}

python -u "$REPO_ROOT/example/run_opt4_v5_8gpu.py"

echo "Opt4 v5 ROB1 results: $ROOT_OUTPUT_DIR"
