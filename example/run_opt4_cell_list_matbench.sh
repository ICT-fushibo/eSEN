#!/usr/bin/env bash
set -euo pipefail

# CELL1 is an experiment layered after the immutable opt4-v5-rob1 tag.
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export REPO_ROOT
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONPATH="$REPO_ROOT/src:${MATBENCH_REPO:-}:$(dirname "$REPO_ROOT")${PYTHONPATH:+:$PYTHONPATH}"

exec python -u "$REPO_ROOT/example/run_opt4_cell_list_matbench.py"
