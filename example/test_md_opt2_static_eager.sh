#!/usr/bin/env bash
set -euo pipefail

# One-configuration smoke test for the opt2 static-eager ablation control.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

GPU=${GPU:-2} \
STEPS=${STEPS:-10} \
WARMUP_STEPS=3 \
REPEATS=1 \
SYSTEMS=${SYSTEMS:-"Cu32"} \
TEMPERATURES=${TEMPERATURES:-"300"} \
VALIDATE_OFFICIAL=1 \
REQUIRE_BASELINE_REFERENCE=${REQUIRE_BASELINE_REFERENCE:-0} \
STRICT=${STRICT:-0} \
    bash "$REPO_ROOT/example/run_md_opt2_static_eager.sh"
