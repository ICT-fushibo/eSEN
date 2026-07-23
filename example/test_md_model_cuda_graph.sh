#!/usr/bin/env bash
set -euo pipefail

# Small opt2 correctness/capture smoke test before the full benchmark matrix.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

GPU=${GPU:-6} \
STEPS=${STEPS:-10} \
WARMUP_STEPS=3 \
REPEATS=1 \
SYSTEMS=${SYSTEMS:-"Cu32 H2O32"} \
TEMPERATURES=${TEMPERATURES:-"300"} \
VALIDATE_OFFICIAL=1 \
REQUIRE_BASELINE_REFERENCE=0 \
STRICT=1 \
    bash "$REPO_ROOT/example/run_md_model_cuda_graph.sh"
