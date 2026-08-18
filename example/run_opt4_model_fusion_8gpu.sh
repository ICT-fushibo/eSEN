#!/usr/bin/env bash
set -euo pipefail

# Convenience entry point for the idle-GPU queue scheduler.  All benchmark
# configuration is read from the environment; see the Python implementation
# for the scheduling and round-selection policy.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python}" "$REPO_ROOT/example/run_opt4_model_fusion_8gpu.py" "$@"
