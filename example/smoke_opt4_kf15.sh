#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KF15_PHASE=smoke
export ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf15_smoke_8gpu_$(date '+%Y%m%d_%H%M%S')"}

python -u "$REPO_ROOT/example/run_opt4_kf15_8gpu.py"
python -u "$REPO_ROOT/example/validate_opt4_kf15_smoke.py" \
    --input-dir "$ROOT_OUTPUT_DIR"

echo "KF15 smoke results: $ROOT_OUTPUT_DIR"
