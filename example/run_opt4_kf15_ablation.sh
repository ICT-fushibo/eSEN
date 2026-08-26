#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KF15_PHASE=ablation
export ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf15_ablation_8gpu_$(date '+%Y%m%d_%H%M%S')"}

python -u "$REPO_ROOT/example/run_opt4_kf15_8gpu.py"

echo "KF15 ablation results: $ROOT_OUTPUT_DIR"
echo "Run selector with: ROOT_OUTPUT_DIR=$ROOT_OUTPUT_DIR bash example/select_opt4_kf15.sh"
