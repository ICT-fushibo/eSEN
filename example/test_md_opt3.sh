#!/usr/bin/env bash
set -euo pipefail

# Smoke-test both Opt3 backends before the full 60-run matrices.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${BASELINE_DIR:?Set BASELINE_DIR to the matching baseline directory}"
GPU=${GPU:-6}
SYSTEMS=${SYSTEMS:-"Cu32 H2O32"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-10}
REPEATS=${REPEATS:-1}

for backend in fixed-builder-model-cg whole-step-cg; do
    env \
        BACKEND="$backend" \
        GPU="$GPU" \
        SYSTEMS="$SYSTEMS" \
        TEMPERATURES="$TEMPERATURES" \
        STEPS="$STEPS" \
        REPEATS="$REPEATS" \
        WARMUP_STEPS=3 \
        BASELINE_DIR="$BASELINE_DIR" \
        BASELINE_STEPS=1000 \
        RUN_ID="esen_opt3_smoke_${backend}_$(date '+%Y%m%d_%H%M%S')" \
        bash "$REPO_ROOT/example/run_md_opt3.sh"
done
