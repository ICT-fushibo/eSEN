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
REFERENCE_H5=${REFERENCE_H5:-"$ROOT_DEFAULT/matbench-discovery-data/md/2026-06-29-dynamat-v1.0-reference-trajectories.h5"}
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
STATISTICS=${STATISTICS:-1}
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
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

mkdir -p "$SAVE_DIR"
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
)
if [[ -n "$SYSTEMS" ]]; then
    read -r -a systems_array <<< "$SYSTEMS"
    args+=(--systems "${systems_array[@]}")
fi
if [[ "$STATISTICS" != "1" ]]; then
    args+=(--no-statistics)
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

python -u "$REPO_ROOT/example/run_esen_matbench.py" "${args[@]}" \
    2>&1 | tee "$SAVE_DIR/${LOG_NAME}.log"
exit "${PIPESTATUS[0]}"
