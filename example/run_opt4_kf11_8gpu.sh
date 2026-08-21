#!/usr/bin/env bash
set -euo pipefail

# Poll idle GPUs and run the 10-system x 2-temperature Opt4 v2/KF11 matrix.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python -u "$REPO_ROOT/example/run_opt4_kf11_8gpu.py" "$@"
