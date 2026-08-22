#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ROOT_OUTPUT_DIR:?Set ROOT_OUTPUT_DIR to the completed PREC1 ablation}"
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"Cu512 H2O192"}
MINIMUM_GEOMEAN_SPEEDUP=${MINIMUM_GEOMEAN_SPEEDUP:-1.01}
MIN_PAIRED_REPEATS=${MIN_PAIRED_REPEATS:-3}
MIN_FASTER_DIRECTIONS=${MIN_FASTER_DIRECTIONS:-3}

read -r -a focus <<< "$FOCUS_SYSTEMS"
overall=0
for scope in model-only whole-step; do
    scope_dir="$ROOT_OUTPUT_DIR/${scope//-/_}"
    output="$ROOT_OUTPUT_DIR/PREC1_${scope//-/_}_selection.json"
    set +e
    python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
        --input-dir "$scope_dir" --scope "$scope" \
        --base-stage OPT4V3_FP32 --candidate-stage PREC1_TF32 \
        --candidate-fusion tf32 --require-tf32-pair \
        --focus-systems "${focus[@]}" \
        --minimum-geomean-speedup "$MINIMUM_GEOMEAN_SPEEDUP" \
        --min-paired-repeats "$MIN_PAIRED_REPEATS" \
        --min-faster-directions "$MIN_FASTER_DIRECTIONS" \
        --output "$output"
    code=$?
    set -e
    [[ $code -eq 0 ]] || overall=1
    echo "PREC1 $scope selector exit=$code output=$output"
done
exit "$overall"
