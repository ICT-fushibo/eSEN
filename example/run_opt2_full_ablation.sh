#!/usr/bin/env bash
set -euo pipefail

# Full opt2 benchmark plus its capture-compatible eager ablation control.
# Existing opt1 results are read-only input and are never regenerated or
# modified by this script.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${BASELINE_DIR:?Set BASELINE_DIR to the ASE baseline output directory}"
: "${GPU_EAGER_DIR:?Set GPU_EAGER_DIR to the existing opt1 output directory}"

GPU=${GPU:-2}
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
RUN_ID=${RUN_ID:-"esen_opt2_full_ablation_$(date '+%Y%m%d_%H%M%S')"}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
OPT2_DIR=${OPT2_DIR:-"$ROOT_OUTPUT_DIR/model_cg"}
STATIC_EAGER_DIR=${STATIC_EAGER_DIR:-"$ROOT_OUTPUT_DIR/static_eager"}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}
RUN_OPT2=${RUN_OPT2:-1}
RUN_STATIC_EAGER=${RUN_STATIC_EAGER:-1}

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
    MD_DTYPE=float64
    VALIDATE_OFFICIAL=0
    STRICT=0
    BASELINE_DIR="$BASELINE_DIR"
    BASELINE_STEPS=1000
    REQUIRE_BASELINE_REFERENCE=0
    CG_PROBE_STEPS=50
    CG_CAPACITY_MARGIN=0.10
    CG_EDGE_STEP=256
    CG_DUMMY_ATOMS=32
    CG_CAPTURE_WARMUP=3
    CG_REPLAY_ENERGY_ATOL=0.0
    CG_REPLAY_FORCE_ATOL=1e-6
)

if [[ "$RUN_OPT2" == "1" ]]; then
    env "${common_env[@]}" \
        RUN_ID="${RUN_ID}_model_cg" \
        OUTPUT_DIR="$OPT2_DIR" \
        bash "$REPO_ROOT/example/run_md_model_cuda_graph.sh"
fi

if [[ "$RUN_STATIC_EAGER" == "1" ]]; then
    env "${common_env[@]}" \
        RUN_ID="${RUN_ID}_static_eager" \
        OUTPUT_DIR="$STATIC_EAGER_DIR" \
        bash "$REPO_ROOT/example/run_md_opt2_static_eager.sh"
fi

python "$REPO_ROOT/example/compare_opt2_ablation.py" \
    --opt1-dir "$GPU_EAGER_DIR" \
    --static-eager-dir "$STATIC_EAGER_DIR" \
    --opt2-dir "$OPT2_DIR" \
    --output-dir "$ROOT_OUTPUT_DIR"

echo "Opt2 full benchmark: $OPT2_DIR"
echo "Opt2 static-eager control: $STATIC_EAGER_DIR"
echo "Opt2 ablation report: $ROOT_OUTPUT_DIR"
