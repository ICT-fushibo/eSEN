#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python -u "$REPO_ROOT/example/run_opt4_capacity_auto_safe_8gpu.py" "$@"
