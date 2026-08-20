#!/usr/bin/env bash
set -uo pipefail

# One-step compile/capture smoke for the KF10 SO2-Gate bridge.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf10_smoke_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-1}
REPEATS=${REPEATS:-1}
WARMUP_STEPS=${WARMUP_STEPS:-0}
BASELINE_STEPS=${BASELINE_STEPS:-100}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}

case "$SCOPE" in
    model-only)
        BASE_STAGE=OPT4V1
        BASE_FUSIONS=so2-epilogue
        ;;
    whole-step)
        BASE_STAGE=OPT4V1
        BASE_FUSIONS=rmsnorm,so2-epilogue
        ;;
    *) echo "SCOPE must be model-only or whole-step" >&2; exit 2 ;;
esac
CANDIDATE_FUSIONS="$BASE_FUSIONS,so2-gate-bridge"

mkdir -p "$OUTPUT_DIR"
set +e
env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
    BASELINE_STEPS="$BASELINE_STEPS" STEPS="$STEPS" REPEATS="$REPEATS" \
    WARMUP_STEPS="$WARMUP_STEPS" SYSTEMS="$SYSTEMS" \
    TEMPERATURES="$TEMPERATURES" OUTPUT_DIR="$OUTPUT_DIR" \
    BASE_STAGE="$BASE_STAGE" BASE_FUSIONS="$BASE_FUSIONS" \
    CANDIDATE_STAGE=KF10 CANDIDATE_FUSIONS="$CANDIDATE_FUSIONS" \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
code=$?
set -e

status=completed
if [[ $code -ne 0 ]]; then
    status=runner_failed
elif [[ ! -s "$OUTPUT_DIR/run_status.tsv" ]] || \
    ! awk -F '\t' 'NR > 1 { seen=1; if ($9 !~ /^(success|validation_failed)$/) bad=1 } END { exit !(seen && !bad) }' \
    "$OUTPUT_DIR/run_status.tsv"; then
    status=benchmark_failed
fi
printf 'stage\tstatus\texit_code\toutput_dir\nKF10\t%s\t%s\t%s\n' \
    "$status" "$code" "$OUTPUT_DIR" > "$OUTPUT_DIR/smoke_status.tsv"
echo "KF10 smoke: $status ($code)"
echo "KF10 smoke output: $OUTPUT_DIR"
[[ "$status" == completed ]]
