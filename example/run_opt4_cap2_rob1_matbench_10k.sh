#!/usr/bin/env bash
set -euo pipefail

# 10k-step bulkCu confirmation with official Matbench NHC/statistics protocol.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${REFERENCE_H5:?Set REFERENCE_H5}"
: "${MATBENCH_REPO:?Set MATBENCH_REPO}"
: "${SAVE_DIR:?Set SAVE_DIR}"

CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
SYSTEM=${SYSTEM:-bulkCu_1000K_Kapil}
BACKENDS=${BACKENDS:-opt4}
OFFLINE_STRESS=${OFFLINE_STRESS:-1}
STATISTICS=${STATISTICS:-1}

env \
    GPU="$GPU" \
    REFERENCE_H5="$REFERENCE_H5" \
    MATBENCH_REPO="$MATBENCH_REPO" \
    CHECKPOINT="$CHECKPOINT" \
    BACKENDS="$BACKENDS" \
    SYSTEMS="$SYSTEM" \
    STEPS=10000 \
    RECORD_INTERVAL=10 \
    SEED=0 \
    PROBE_STEPS="${PROBE_STEPS:-50}" \
    OPT4_NEIGHBOR_CAPACITY_POLICY=elastic \
    ROB1=1 \
    ROB1_WINDOW_STEPS=0 \
    ROB1_MAX_RETRIES=2 \
    CAP2_COMPACT_SLOT_STEP=4 \
    CAP2_COMPACT_MARGIN=0.0 \
    CAP2_MIN_REDUCTION=0.05 \
    OFFLINE_STRESS="$OFFLINE_STRESS" \
    STATISTICS="$STATISTICS" \
    SAVE_DIR="$SAVE_DIR" \
    bash "$REPO_ROOT/example/run_esen_matbench.sh"

if [[ -n "${BASE_RESULT_DIR:-}" ]]; then
    python "$REPO_ROOT/example/compare_opt4_cap2_matbench.py" \
        --base-dir "$BASE_RESULT_DIR" \
        --candidate-dir "$SAVE_DIR" \
        --system "$SYSTEM" \
        --output "$SAVE_DIR/CAP2_ROB1_10k_comparison.json"
fi
