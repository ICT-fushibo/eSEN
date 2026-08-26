#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ROOT_OUTPUT_DIR:?Set ROOT_OUTPUT_DIR to the completed KF15 ablation}"
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"Cu512 H2O192"}
MINIMUM_GEOMEAN_SPEEDUP=${MINIMUM_GEOMEAN_SPEEDUP:-1.01}
MIN_PAIRED_REPEATS=${MIN_PAIRED_REPEATS:-3}
MIN_FASTER_DIRECTIONS=${MIN_FASTER_DIRECTIONS:-3}
MAXIMUM_PEAK_RESERVED_INCREASE_GIB=${MAXIMUM_PEAK_RESERVED_INCREASE_GIB:-1.0}

read -r -a focus <<< "$FOCUS_SYSTEMS"
overall=0
for scope in model-only whole-step; do
    scope_dir="$ROOT_OUTPUT_DIR/${scope//-/_}"
    output="$ROOT_OUTPUT_DIR/KF15_${scope//-/_}_selection.json"
    if [[ "$scope" == model-only ]]; then
        accepted_before=so2-epilogue,so2-gate-bridge,so2-block-gemm,so2-prepare-backward-reduce
    else
        accepted_before=rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm,so2-prepare-backward-reduce
    fi
    set +e
    python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
        --input-dir "$scope_dir" --scope "$scope" \
        --base-stage OPT4V4_FP32 --candidate-stage KF15_FP32 \
        --candidate-fusion wigner-so2-hybrid \
        --accepted-before "$accepted_before" \
        --expected-tf32-mode off \
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
    echo "KF15 $scope selector exit=$code output=$output"
done
exit "$overall"
