#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python -u "$REPO_ROOT/example/run_opt4_kf12_ablation_8gpu.py" "$@"
