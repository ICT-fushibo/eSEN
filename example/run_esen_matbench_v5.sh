#!/usr/bin/env bash
set -euo pipefail

# Explicit Matbench entry point for Opt4 v5.  The generic Matbench launcher
# remains Opt4 v4-compatible; this wrapper opts into v5 and ROB1.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OPT4_FUSION_STAGE=${OPT4_FUSION_STAGE:-OPT4V5_FP32_ROB1}
export OPT4_EXECUTION_SCOPE=${OPT4_EXECUTION_SCOPE:-whole-step}
export OPT4_NEIGHBOR_BUILDER=${OPT4_NEIGHBOR_BUILDER:-dense}
export OPT4_NEIGHBOR_CAPACITY_POLICY=${OPT4_NEIGHBOR_CAPACITY_POLICY:-auto-safe}
export ROB1=${ROB1:-1}
export ROB1_WINDOW_STEPS=${ROB1_WINDOW_STEPS:-0}
export ROB1_MAX_RETRIES=${ROB1_MAX_RETRIES:-2}
export OPT4_MODEL_FUSIONS=${OPT4_MODEL_FUSIONS:-rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm,so2-prepare-backward-reduce}

exec bash "$REPO_ROOT/example/run_esen_matbench.sh"
