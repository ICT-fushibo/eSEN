#!/usr/bin/env bash
set -euo pipefail

# Opt4 KF1 ablation.  KF0 directories may point at existing Opt3 results, or
# RUN_KF0=1 can regenerate them under the same experiment root.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-6}
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
RUN_ID=${RUN_ID:-"esen_opt4_kf1_ablation_$(date '+%Y%m%d_%H%M%S')"}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}
BASELINE_DIR=${BASELINE_DIR:-}
RUN_KF0=${RUN_KF0:-0}
RUN_FIXED=${RUN_FIXED:-1}
RUN_WHOLE=${RUN_WHOLE:-1}
TRITON_BLOCK_SIZE=${TRITON_BLOCK_SIZE:-256}

KF0_FIXED_DIR=${KF0_FIXED_DIR:-"$ROOT_OUTPUT_DIR/kf0_fixed_builder_model_cg"}
KF0_WHOLE_DIR=${KF0_WHOLE_DIR:-"$ROOT_OUTPUT_DIR/kf0_whole_step_cg"}
KF1_FIXED_DIR=${KF1_FIXED_DIR:-"$ROOT_OUTPUT_DIR/kf1_fixed_builder_model_cg"}
KF1_WHOLE_DIR=${KF1_WHOLE_DIR:-"$ROOT_OUTPUT_DIR/kf1_whole_step_cg"}

if [[ "$RUN_KF0" != "1" ]]; then
    : "${KF0_FIXED_DIR:?Set KF0_FIXED_DIR to existing Opt3 fixed results}"
    : "${KF0_WHOLE_DIR:?Set KF0_WHOLE_DIR to existing Opt3 whole-step results}"
    if [[ "$RUN_FIXED" == "1" && ! -d "$KF0_FIXED_DIR" ]]; then
        echo "Existing KF0 fixed directory not found: $KF0_FIXED_DIR" >&2
        exit 2
    fi
    if [[ "$RUN_WHOLE" == "1" && ! -d "$KF0_WHOLE_DIR" ]]; then
        echo "Existing KF0 whole-step directory not found: $KF0_WHOLE_DIR" >&2
        exit 2
    fi
fi

mkdir -p "$ROOT_OUTPUT_DIR"
common_env=(
    GPU="$GPU"
    CHECKPOINT="$CHECKPOINT"
    STRUCTURE_DIR="$STRUCTURE_DIR"
    STEPS="$STEPS"
    WARMUP_STEPS="$WARMUP_STEPS"
    REPEATS="$REPEATS"
    SYSTEMS="$SYSTEMS"
    TEMPERATURES="$TEMPERATURES"
    SEED=42
    STRICT=0
    BASELINE_DIR="$BASELINE_DIR"
    BASELINE_STEPS=1000
    PROBE_STEPS=50
    NEIGHBOR_MARGIN=0.10
    NEIGHBOR_SLOT_STEP=8
    DUMMY_ATOMS=32
    CAPTURE_WARMUP=3
    MAX_NEIGHBORS=300
    DEGENERACY_TOLERANCE=0.01
    ENERGY_PER_ATOM_ATOL=1e-5
    FORCE_MAX_ATOL=2e-4
    TRITON_BLOCK_SIZE="$TRITON_BLOCK_SIZE"
)

run_backend() {
    local backend=$1
    local output_dir=$2
    env "${common_env[@]}" \
        BACKEND="$backend" \
        RUN_ID="${RUN_ID}_${backend}" \
        OUTPUT_DIR="$output_dir" \
        bash "$REPO_ROOT/example/run_md_opt3.sh"
}

if [[ "$RUN_FIXED" == "1" ]]; then
    if [[ "$RUN_KF0" == "1" ]]; then
        run_backend fixed-builder-model-cg "$KF0_FIXED_DIR"
    fi
    run_backend fixed-builder-model-cg-kf1 "$KF1_FIXED_DIR"
fi
if [[ "$RUN_WHOLE" == "1" ]]; then
    if [[ "$RUN_KF0" == "1" ]]; then
        run_backend whole-step-cg "$KF0_WHOLE_DIR"
    fi
    run_backend whole-step-cg-kf1 "$KF1_WHOLE_DIR"
fi

compare_args=(
    --kf0-fixed-dir "$KF0_FIXED_DIR"
    --kf1-fixed-dir "$KF1_FIXED_DIR"
    --kf0-whole-dir "$KF0_WHOLE_DIR"
    --kf1-whole-dir "$KF1_WHOLE_DIR"
    --output-dir "$ROOT_OUTPUT_DIR"
)
if [[ -n "$BASELINE_DIR" ]]; then
    compare_args+=(--baseline-dir "$BASELINE_DIR")
fi
python "$REPO_ROOT/example/compare_opt4_kf1.py" "${compare_args[@]}"

echo "Opt4 KF1 results: $ROOT_OUTPUT_DIR"
echo "KF1 report: $ROOT_OUTPUT_DIR/opt4_kf1_ablation.md"

