#!/usr/bin/env bash
set -uo pipefail

# Interleaved KF9 A/B timing and selector.  Numerical errors are telemetry;
# hard runtime failures and unhealthy CUDA Graphs still invalidate timing.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf9_ablation_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"Cu512 H2O192"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-5}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}
SELECT_MIN_PAIRED=${SELECT_MIN_PAIRED:-$REPEATS}
SELECT_MIN_FASTER=${SELECT_MIN_FASTER:-$((REPEATS - 1))}

case "$SCOPE" in
    model-only) BASE_STAGE=KF0; BASE_FUSIONS="" ;;
    whole-step) BASE_STAGE=KF4_base; BASE_FUSIONS=${BASE_FUSIONS:-rmsnorm} ;;
    *) echo "SCOPE must be model-only or whole-step" >&2; exit 2 ;;
esac

mkdir -p "$ROOT_OUTPUT_DIR"
ROUND_DIR="$ROOT_OUTPUT_DIR/round_KF9"
env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
    BASELINE_STEPS="$BASELINE_STEPS" STEPS="$STEPS" REPEATS="$REPEATS" \
    WARMUP_STEPS="$WARMUP_STEPS" SYSTEMS="$SYSTEMS" \
    TEMPERATURES="$TEMPERATURES" OUTPUT_DIR="$ROUND_DIR" \
    BASE_STAGE="$BASE_STAGE" BASE_FUSIONS="$BASE_FUSIONS" \
    CANDIDATE_STAGE=KF9 CANDIDATE_FUSIONS="${BASE_FUSIONS:+$BASE_FUSIONS,}so2-epilogue" \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
runner_code=$?

selection="$ROOT_OUTPUT_DIR/KF9_selection.json"
set +e
python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
    --input-dir "$ROUND_DIR" --scope "$SCOPE" \
    --base-stage "$BASE_STAGE" --candidate-stage KF9 \
    --candidate-fusion so2-epilogue --accepted-before "${BASE_FUSIONS}" \
    --focus-systems $FOCUS_SYSTEMS \
    --min-paired-repeats "$SELECT_MIN_PAIRED" \
    --min-faster-directions "$SELECT_MIN_FASTER" \
    --output "$selection"
selected=$?
set -e
if [[ $runner_code -ne 0 ]]; then
    selected=$runner_code
fi

python - "$ROOT_OUTPUT_DIR/accepted_fusions.json" "$SCOPE" "$FOCUS_SYSTEMS" "$selected" <<'PY'
import json
import pathlib
import sys

path, scope, focus, selected = sys.argv[1:]
accepted = ["so2-epilogue"] if int(selected) == 0 else []
if scope == "whole-step":
    accepted = ["rmsnorm"] + accepted
pathlib.Path(path).write_text(json.dumps({
    "scope": scope,
    "accepted_fusions": accepted,
    "focus_systems": focus.split(),
    "candidate_stage": "KF9",
    "policy": "energy/force errors are telemetry only",
}, indent=2) + "\n", encoding="utf-8")
PY

echo "KF9 selector exit: $selected"
echo "KF9 interleaved runner exit: $runner_code"
echo "KF9 selection: $selection"
echo "KF9 accepted fusions: $ROOT_OUTPUT_DIR/accepted_fusions.json"
echo "KF9 ablation output: $ROOT_OUTPUT_DIR"
exit "$selected"
