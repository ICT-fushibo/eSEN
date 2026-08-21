#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_ROOT
exec python -u "$REPO_ROOT/example/run_opt4_capacity_8gpu.py" "$@"
