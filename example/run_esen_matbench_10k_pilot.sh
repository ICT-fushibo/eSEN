#!/usr/bin/env bash
set -euo pipefail

# One-system, 10,000-step Matbench/DynaMat pilot.  All physical settings are
# pinned to the public protocol; only the rollout length differs from the
# 80,000-step leaderboard run.  Baseline and Opt2-Opt4 run serially on the
# same physical GPU so speedups and trajectory statistics are aligned.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}
REFERENCE_H5=${REFERENCE_H5:-/home/fushibo/matbench-discovery-data/2026-06-29-dynamat-v1.0-reference-trajectories.h5}
MATBENCH_REPO=${MATBENCH_REPO:-/home/fushibo/matbench-discovery}
CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/esen_30m_oam.pt}
SYSTEM=${SYSTEM:-anthracene_293K_Sharma_S}
SAVE_DIR=${SAVE_DIR:-$REPO_ROOT/example/md_out/matbench_10k_${SYSTEM}_gpu${GPU}_$(date '+%Y%m%d_%H%M%S')}

if [[ "${STEPS:-10000}" != "10000" ]]; then
    echo "This pilot pins STEPS=10000; got ${STEPS}" >&2
    exit 2
fi
if [[ "${RECORD_INTERVAL:-10}" != "10" ]]; then
    echo "Matbench record interval must remain 10; got ${RECORD_INTERVAL}" >&2
    exit 2
fi
if [[ ! -f "$REFERENCE_H5" ]]; then
    echo "Reference HDF5 not found: $REFERENCE_H5" >&2
    exit 2
fi
if [[ ! -d "$MATBENCH_REPO/matbench_discovery" ]]; then
    echo "Matbench source checkout not found: $MATBENCH_REPO" >&2
    echo "It is required for the official RDF/ADF/vDOS metric implementation." >&2
    exit 2
fi

export GPU REFERENCE_H5 MATBENCH_REPO CHECKPOINT SYSTEMS="$SYSTEM" SAVE_DIR
export BACKENDS="baseline opt2 opt3 opt4"
export STEPS=10000 RECORD_INTERVAL=10 SEED=0 STATISTICS=1
export OPT4_MODEL_FUSIONS="rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm,so2-prepare-backward-reduce"
export OPT4_FUSION_STAGE=OPT4V4_FP32
export OPT4_NEIGHBOR_CAPACITY_POLICY=auto-safe

exec bash "$REPO_ROOT/example/run_esen_matbench.sh"
