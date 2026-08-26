#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${KF15_SELECTION_DIR:?Set KF15_SELECTION_DIR to the accepted ablation directory}"
export KF15_PHASE=formal
export ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf15_formal_8gpu_$(date '+%Y%m%d_%H%M%S')"}

python -u "$REPO_ROOT/example/run_opt4_kf15_8gpu.py"

echo "KF15 formal results: $ROOT_OUTPUT_DIR"
