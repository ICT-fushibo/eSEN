#!/usr/bin/env bash
set -uo pipefail

# Three-repeat interleaved Opt4 v2 vs KF11 timing and acceptance selector.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-100}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf11_ablation_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"Cu512 H2O192"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}

case "$SCOPE" in
    model-only) BASE_FUSIONS=so2-epilogue,so2-gate-bridge ;;
    whole-step) BASE_FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge ;;
    *) echo "SCOPE must be model-only or whole-step" >&2; exit 2 ;;
esac
CANDIDATE_FUSIONS="$BASE_FUSIONS,wigner-so2-bridge"
ROUND_DIR="$ROOT_OUTPUT_DIR/round_KF11"
mkdir -p "$ROOT_OUTPUT_DIR"

env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
    BASELINE_STEPS="$BASELINE_STEPS" STEPS="$STEPS" REPEATS="$REPEATS" \
    WARMUP_STEPS="$WARMUP_STEPS" SYSTEMS="$SYSTEMS" \
    TEMPERATURES="$TEMPERATURES" OUTPUT_DIR="$ROUND_DIR" \
    BASE_STAGE=OPT4V2 BASE_FUSIONS="$BASE_FUSIONS" \
    CANDIDATE_STAGE=KF11 CANDIDATE_FUSIONS="$CANDIDATE_FUSIONS" \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
runner_code=$?

selection="$ROOT_OUTPUT_DIR/KF11_selection.json"
set +e
python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
    --input-dir "$ROUND_DIR" --scope "$SCOPE" \
    --base-stage OPT4V2 --candidate-stage KF11 \
    --candidate-fusion wigner-so2-bridge --accepted-before "$BASE_FUSIONS" \
    --focus-systems $FOCUS_SYSTEMS \
    --min-paired-repeats "$REPEATS" --min-faster-directions "$REPEATS" \
    --maximum-peak-reserved-increase-gib 1.0 \
    --output "$selection"
selected=$?
set -e
if [[ $runner_code -ne 0 ]]; then selected=$runner_code; fi

python - "$ROOT_OUTPUT_DIR/accepted_fusions.json" "$SCOPE" "$FOCUS_SYSTEMS" "$selected" "$BASE_FUSIONS" <<'PY'
import json
import pathlib
import sys

path, scope, focus, selected, base = sys.argv[1:]
accepted = [item for item in base.split(",") if item]
if int(selected) == 0:
    accepted.append("wigner-so2-bridge")
pathlib.Path(path).write_text(json.dumps({
    "scope": scope,
    "accepted_fusions": accepted,
    "focus_systems": focus.split(),
    "candidate_stage": "KF11",
    "policy": "energy/force errors are telemetry only",
}, indent=2) + "\n", encoding="utf-8")
PY

echo "KF11 selector exit: $selected"
echo "KF11 interleaved runner exit: $runner_code"
echo "KF11 selection: $selection"
echo "KF11 accepted fusions: $ROOT_OUTPUT_DIR/accepted_fusions.json"
echo "KF11 ablation output: $ROOT_OUTPUT_DIR"
exit "$selected"
