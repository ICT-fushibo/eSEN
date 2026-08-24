#!/usr/bin/env bash
set -euo pipefail

# Polling one-step smoke for KF13.  Set KF13_PRECISION=fp32 (default) or tf32.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KF13_PRECISION=${KF13_PRECISION:-fp32}
export KF13_PRECISION
export KF13_PHASE=smoke
export ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf13_${KF13_PRECISION}_smoke_8gpu_$(date '+%Y%m%d_%H%M%S')"}

python -u "$REPO_ROOT/example/run_opt4_kf13_8gpu.py"
python -u "$REPO_ROOT/example/validate_opt4_kf13_smoke.py" \
    --input-dir "$ROOT_OUTPUT_DIR" --precision "$KF13_PRECISION"

echo "KF13 $KF13_PRECISION smoke results: $ROOT_OUTPUT_DIR"
