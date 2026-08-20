#!/usr/bin/env bash
set -euo pipefail

# Poll idle GPUs and run the Opt4 KF9 v1 10-system x 2-temperature matrix.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python -u "$REPO_ROOT/example/run_opt4_v1_8gpu.py" "$@"
