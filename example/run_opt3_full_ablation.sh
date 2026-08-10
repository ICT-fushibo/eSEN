#!/usr/bin/env bash
set -euo pipefail

# Run both new Opt3 backends.  Baseline, Opt1, and Opt2 directories are
# read-only report inputs and are never regenerated or modified.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${BASELINE_DIR:?Set BASELINE_DIR to the ASE baseline output directory}"
: "${GPU_EAGER_DIR:?Set GPU_EAGER_DIR to the existing Opt1 output directory}"
: "${MODEL_CG_DIR:?Set MODEL_CG_DIR to the existing Opt2 output directory}"

GPU=${GPU:-6}
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
RUN_ID=${RUN_ID:-"esen_opt3_full_ablation_$(date '+%Y%m%d_%H%M%S')"}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
FIXED_DIR=${FIXED_DIR:-"$ROOT_OUTPUT_DIR/fixed_builder_model_cg"}
WHOLE_DIR=${WHOLE_DIR:-"$ROOT_OUTPUT_DIR/whole_step_cg"}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}
RUN_FIXED=${RUN_FIXED:-1}
RUN_WHOLE=${RUN_WHOLE:-1}

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
)

if [[ "$RUN_FIXED" == "1" ]]; then
    env "${common_env[@]}" \
        BACKEND=fixed-builder-model-cg \
        RUN_ID="${RUN_ID}_fixed_builder_model_cg" \
        OUTPUT_DIR="$FIXED_DIR" \
        bash "$REPO_ROOT/example/run_md_opt3.sh"
fi
if [[ "$RUN_WHOLE" == "1" ]]; then
    env "${common_env[@]}" \
        BACKEND=whole-step-cg \
        RUN_ID="${RUN_ID}_whole_step_cg" \
        OUTPUT_DIR="$WHOLE_DIR" \
        bash "$REPO_ROOT/example/run_md_opt3.sh"
fi

python "$REPO_ROOT/example/compare_opt3_ablation.py" \
    --baseline-dir "$BASELINE_DIR" \
    --opt1-dir "$GPU_EAGER_DIR" \
    --opt2-dir "$MODEL_CG_DIR" \
    --fixed-dir "$FIXED_DIR" \
    --whole-dir "$WHOLE_DIR" \
    --output-dir "$ROOT_OUTPUT_DIR"

echo "Fixed-builder/model-CG: $FIXED_DIR"
echo "Whole-step CG: $WHOLE_DIR"
echo "Opt3 ablation report: $ROOT_OUTPUT_DIR"
echo "Phase-1 decision report: $ROOT_OUTPUT_DIR/opt3_phase1_decision.md"
