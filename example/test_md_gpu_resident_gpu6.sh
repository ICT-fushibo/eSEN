#!/usr/bin/env bash
set -euo pipefail

# Short correctness/performance smoke test on physical GPU 6.  The official
# OCPCalculator comparison is intentionally limited to the two small systems.
GPU=6 \
STEPS=${STEPS:-5} \
WARMUP_STEPS=3 \
REPEATS=1 \
SYSTEMS="Cu32 H2O32" \
TEMPERATURES="300" \
VALIDATE_OFFICIAL=1 \
STRICT=1 \
bash "$(dirname "$0")/run_md_gpu_resident.sh"
