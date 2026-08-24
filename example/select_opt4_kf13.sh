#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ROOT_OUTPUT_DIR:?Set ROOT_OUTPUT_DIR to the completed KF13 ablation}"
KF13_PRECISION=${KF13_PRECISION:-fp32}
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"Cu512 H2O192"}
MINIMUM_GEOMEAN_SPEEDUP=${MINIMUM_GEOMEAN_SPEEDUP:-1.01}
MIN_PAIRED_REPEATS=${MIN_PAIRED_REPEATS:-3}
MIN_FASTER_DIRECTIONS=${MIN_FASTER_DIRECTIONS:-3}
MAXIMUM_PEAK_RESERVED_INCREASE_GIB=${MAXIMUM_PEAK_RESERVED_INCREASE_GIB:-1.0}

case "$KF13_PRECISION" in
    fp32)
        BASE_STAGE=OPT4V3_FP32
        CANDIDATE_STAGE=KF13_FP32
        EXPECTED_TF32_MODE=off
        ;;
    tf32)
        BASE_STAGE=PREC1_TF32
        CANDIDATE_STAGE=KF13_PREC1_TF32
        EXPECTED_TF32_MODE=on
        ;;
    *) echo "KF13_PRECISION must be fp32 or tf32" >&2; exit 2 ;;
esac

read -r -a focus <<< "$FOCUS_SYSTEMS"
overall=0
for scope in model-only whole-step; do
    scope_dir="$ROOT_OUTPUT_DIR/${scope//-/_}"
    output="$ROOT_OUTPUT_DIR/KF13_${KF13_PRECISION}_${scope//-/_}_selection.json"
    if [[ "$scope" == model-only ]]; then
        accepted_before=so2-epilogue,so2-gate-bridge,so2-block-gemm
    else
        accepted_before=rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm
    fi
    set +e
    python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
        --input-dir "$scope_dir" --scope "$scope" \
        --base-stage "$BASE_STAGE" --candidate-stage "$CANDIDATE_STAGE" \
        --candidate-fusion so3-weight-cache \
        --accepted-before "$accepted_before" \
        --expected-tf32-mode "$EXPECTED_TF32_MODE" \
        --focus-systems "${focus[@]}" \
        --minimum-geomean-speedup "$MINIMUM_GEOMEAN_SPEEDUP" \
        --min-paired-repeats "$MIN_PAIRED_REPEATS" \
        --min-faster-directions "$MIN_FASTER_DIRECTIONS" \
        --maximum-peak-reserved-increase-gib \
        "$MAXIMUM_PEAK_RESERVED_INCREASE_GIB" \
        --output "$output"
    code=$?
    set -e
    [[ $code -eq 0 ]] || overall=1
    echo "KF13 $KF13_PRECISION $scope selector exit=$code output=$output"
done
exit "$overall"
