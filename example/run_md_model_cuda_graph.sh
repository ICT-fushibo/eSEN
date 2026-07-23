#!/usr/bin/env bash
set -euo pipefail

# Opt2: GPU-resident MD with a model-only raw CUDA Graph.  Dynamic neighbor
# construction and the NVT integrator remain eager.  Missing/OOM ASE references
# do not suppress performance runs; their JSON records are marked unvalidated.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

BACKEND=model-cg \
REQUIRE_BASELINE_REFERENCE=${REQUIRE_BASELINE_REFERENCE:-0} \
    bash "$REPO_ROOT/example/run_md_gpu_resident.sh"
