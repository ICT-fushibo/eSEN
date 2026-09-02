#!/usr/bin/env bash
set -uo pipefail

# Official Matbench/DynaMat NHC runner for eSEN.  This is independent from the
# existing Cu/H2O, Opt4, and Berendsen benchmark scripts.  A failed/OOM system
# is recorded and does not stop the remaining systems or backends.
#
# Example smoke:
#   GPU=2 BACKEND=baseline SYSTEMS='bulkCuAu_500K-Artrith_VASP' \
#   STEPS=100 OUTPUT_DIR=/public-data/fushibo/eSEN/example/md_out/matbench_smoke \
#   bash example/run_esen_matbench.sh
#
# Example formal backend:
#   GPU=2 BACKEND=opt3 OUTPUT_DIR=/public-data/fushibo/eSEN/example/md_out/matbench_opt3 \
#   bash example/run_esen_matbench.sh

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT_DEFAULT=$(cd "$REPO_ROOT/.." && pwd)

GPU=${GPU:-0}
# BACKEND remains a compatibility alias for one backend.  BACKENDS can contain
# several whitespace-separated backends so one report can calculate speedups.
BACKENDS=${BACKENDS:-${BACKEND:-opt1}}
SYSTEMS=${SYSTEMS:-}
STEPS=${STEPS:-80000}
RECORD_INTERVAL=${RECORD_INTERVAL:-10}
SEED=${SEED:-0}
REFERENCE_NAME=2026-06-29-dynamat-v1.0-reference-trajectories.h5
if [[ -z "${REFERENCE_H5:-}" ]]; then
    if [[ -f "$ROOT_DEFAULT/matbench-discovery-data/$REFERENCE_NAME" ]]; then
        REFERENCE_H5="$ROOT_DEFAULT/matbench-discovery-data/$REFERENCE_NAME"
    else
        REFERENCE_H5="$ROOT_DEFAULT/matbench-discovery-data/md/$REFERENCE_NAME"
    fi
fi
MATBENCH_REPO=${MATBENCH_REPO:-"$ROOT_DEFAULT/matbench-discovery"}
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
# SAVE_DIR is the explicit persistence interface.  OUTPUT_DIR remains a
# backwards-compatible alias; SAVE_DIR wins when both are supplied.
SAVE_DIR=${SAVE_DIR:-${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/matbench_${BACKENDS// /_}_$(date '+%Y%m%d_%H%M%S')"}}
OUTPUT_DIR=$SAVE_DIR
PROBE_STEPS=${PROBE_STEPS:-50}
NEIGHBOR_MARGIN=${NEIGHBOR_MARGIN:-0.10}
NEIGHBOR_SLOT_STEP=${NEIGHBOR_SLOT_STEP:-8}
EDGE_STEP=${EDGE_STEP:-256}
DUMMY_ATOMS=${DUMMY_ATOMS:-32}
CAPTURE_WARMUP=${CAPTURE_WARMUP:-3}
MAX_NEIGHBORS=${MAX_NEIGHBORS:-300}
DEGENERACY_TOLERANCE=${DEGENERACY_TOLERANCE:-0.01}
OPT4_MODEL_FUSIONS=${OPT4_MODEL_FUSIONS:-rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm,so2-prepare-backward-reduce}
OPT4_FUSION_STAGE=${OPT4_FUSION_STAGE:-OPT4V4_FP32}
OPT4_NEIGHBOR_CAPACITY_POLICY=${OPT4_NEIGHBOR_CAPACITY_POLICY:-auto-safe}
NEIGHBOR_AUTO_MIN_REDUCTION=${NEIGHBOR_AUTO_MIN_REDUCTION:-0.05}
NEIGHBOR_AUTO_GUARD_SLOTS=${NEIGHBOR_AUTO_GUARD_SLOTS:-1}
ROB1=${ROB1:-0}
ROB1_WINDOW_STEPS=${ROB1_WINDOW_STEPS:-0}
ROB1_MAX_RETRIES=${ROB1_MAX_RETRIES:-2}
CAP2_COMPACT_SLOT_STEP=${CAP2_COMPACT_SLOT_STEP:-4}
CAP2_COMPACT_MARGIN=${CAP2_COMPACT_MARGIN:-0.0}
CAP2_MIN_REDUCTION=${CAP2_MIN_REDUCTION:-0.05}
CAP2_TEST_CAPACITY_LIMIT=${CAP2_TEST_CAPACITY_LIMIT:-0}
STATISTICS=${STATISTICS:-1}
OFFLINE_STRESS=${OFFLINE_STRESS:-0}
METRICS_ONLY=${METRICS_ONLY:-0}
STRICT=${STRICT:-0}
OVERWRITE=${OVERWRITE:-0}

if [[ "$SEED" != "0" ]]; then
    echo "Matbench seed is fixed at 0; got SEED=$SEED" >&2
    exit 2
fi
if [[ ! -f "$REFERENCE_H5" ]]; then
    echo "Reference HDF5 not found: $REFERENCE_H5" >&2
    exit 2
fi
if [[ "$METRICS_ONLY" != "1" && ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

# Do not create or write inside SAVE_DIR before the Python runner validates
# that a new output directory is empty.  In particular, opening the tee log
# there races with that validation and makes every fresh run fail unless
# --overwrite is used.
mkdir -p "$(dirname "$SAVE_DIR")"
read -r -a backends_array <<< "$BACKENDS"
args=(
    --backend "${backends_array[@]}"
    --reference-h5 "$REFERENCE_H5"
    --checkpoint "$CHECKPOINT"
    --matbench-repo "$MATBENCH_REPO"
    --save-dir "$SAVE_DIR"
    --gpu "$GPU"
    --steps "$STEPS"
    --record-interval "$RECORD_INTERVAL"
    --seed "$SEED"
    --probe-steps "$PROBE_STEPS"
    --neighbor-margin "$NEIGHBOR_MARGIN"
    --neighbor-slot-step "$NEIGHBOR_SLOT_STEP"
    --edge-step "$EDGE_STEP"
    --dummy-atoms "$DUMMY_ATOMS"
    --capture-warmup "$CAPTURE_WARMUP"
    --max-neighbors "$MAX_NEIGHBORS"
    --degeneracy-tolerance "$DEGENERACY_TOLERANCE"
    --opt4-model-fusions "$OPT4_MODEL_FUSIONS"
    --opt4-fusion-stage "$OPT4_FUSION_STAGE"
    --opt4-neighbor-capacity-policy "$OPT4_NEIGHBOR_CAPACITY_POLICY"
    --neighbor-auto-min-reduction "$NEIGHBOR_AUTO_MIN_REDUCTION"
    --neighbor-auto-guard-slots "$NEIGHBOR_AUTO_GUARD_SLOTS"
    --rob1-window-steps "$ROB1_WINDOW_STEPS"
    --rob1-max-retries "$ROB1_MAX_RETRIES"
    --cap2-compact-slot-step "$CAP2_COMPACT_SLOT_STEP"
    --cap2-compact-margin "$CAP2_COMPACT_MARGIN"
    --cap2-min-reduction "$CAP2_MIN_REDUCTION"
    --cap2-test-capacity-limit "$CAP2_TEST_CAPACITY_LIMIT"
)
if [[ -n "$SYSTEMS" ]]; then
    read -r -a systems_array <<< "$SYSTEMS"
    args+=(--systems "${systems_array[@]}")
fi
if [[ "$STATISTICS" != "1" ]]; then
    args+=(--no-statistics)
fi
if [[ "$OFFLINE_STRESS" == "1" ]]; then
    args+=(--offline-stress)
fi
if [[ "$ROB1" == "1" ]]; then
    args+=(--rob1)
else
    args+=(--no-rob1)
fi
if [[ "$METRICS_ONLY" == "1" ]]; then
    args+=(--metrics-only)
fi
if [[ "$STRICT" == "1" ]]; then
    args+=(--strict)
fi
if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
fi

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_ROOT/src:$MATBENCH_REPO:$ROOT_DEFAULT${PYTHONPATH:+:$PYTHONPATH}"
LOG_NAME=${BACKENDS// /_}
if [[ "$METRICS_ONLY" == "1" ]]; then
    LOG_NAME="${LOG_NAME}_metrics_only"
fi
LIVE_LOG="${SAVE_DIR%/}.${LOG_NAME}.log"

python -u "$REPO_ROOT/example/run_esen_matbench.py" "${args[@]}" \
    2>&1 | tee "$LIVE_LOG"
status=${PIPESTATUS[0]}

# Keep the live log beside SAVE_DIR while the runner owns its emptiness check,
# then place it with the other artifacts once the directory exists.
if [[ -d "$SAVE_DIR" ]]; then
    mv -f "$LIVE_LOG" "$SAVE_DIR/${LOG_NAME}.log"
fi
exit "$status"
