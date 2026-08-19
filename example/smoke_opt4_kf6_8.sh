#!/usr/bin/env bash
set -uo pipefail

# One-step compile/capture smoke for KF6-KF8.  This intentionally runs each
# candidate from the same accepted base so a failure is isolated to one stage.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf6_8_smoke_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 H2O32 H2O60 H2O192 H2O512"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-1}
REPEATS=${REPEATS:-1}
WARMUP_STEPS=${WARMUP_STEPS:-0}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
if [[ "$SCOPE" == whole-step ]]; then
    BASE_FUSIONS=${BASE_FUSIONS:-rmsnorm}
else
    BASE_FUSIONS=${BASE_FUSIONS:-}
fi
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}

case "$SCOPE" in model-only|whole-step) ;; *) echo "SCOPE must be model-only or whole-step" >&2; exit 2 ;; esac
mkdir -p "$OUTPUT_DIR"
SUMMARY="$OUTPUT_DIR/smoke_status.tsv"
printf 'stage\tstatus\texit_code\toutput_dir\n' > "$SUMMARY"
failures=0

for spec in "KF6:radial-mlp" "KF7:so3-mlp" "KF8:energy-head"; do
    stage=${spec%%:*}; fusion=${spec#*:}
    stage_dir="$OUTPUT_DIR/$stage"
    set +e
    env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
        STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
        BASELINE_STEPS="$BASELINE_STEPS" STEPS="$STEPS" REPEATS="$REPEATS" \
        WARMUP_STEPS="$WARMUP_STEPS" SYSTEMS="$SYSTEMS" \
        TEMPERATURES="$TEMPERATURES" OUTPUT_DIR="$stage_dir" \
        BASE_STAGE=KF0 BASE_FUSIONS="$BASE_FUSIONS" \
        CANDIDATE_STAGE="$stage" CANDIDATE_FUSIONS="${BASE_FUSIONS:+$BASE_FUSIONS,}$fusion" \
        SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
        bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
    code=$?
    set -e
    status=completed
    if [[ $code -ne 0 ]]; then
        status=runner_failed
    elif [[ ! -s "$stage_dir/run_status.tsv" ]] || \
         ! awk -F '\t' 'NR > 1 && $2 == "candidate" { seen=1; if ($9 !~ /^(success|validation_failed)$/) bad=1 } END { exit !(seen && !bad) }' \
             "$stage_dir/run_status.tsv"; then
        status=benchmark_failed
    fi
    if [[ "$status" != completed ]]; then failures=$((failures + 1)); fi
    printf '%s\t%s\t%s\t%s\n' "$stage" "$status" "$code" "$stage_dir" >> "$SUMMARY"
done

echo "KF6-KF8 smoke summary: $SUMMARY"
echo "Per-stage result/status files are under: $OUTPUT_DIR"
if [[ $failures -ne 0 ]]; then
    echo "KF6-KF8 smoke had $failures failed stage(s)" >&2
    exit 1
fi
