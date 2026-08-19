#!/usr/bin/env bash
set -uo pipefail

# Cumulative KF6-KF8 selector.  The default matrix deliberately contains one
# small and one large Cu/H2O system; use --focus-systems in the selector so
# Cu512/H2O512 determine acceptance while small systems are guardrails.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf6_8_ablation_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O512"}
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"Cu512 H2O512"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-5}
START_STAGE=${START_STAGE:-KF6}
if [[ "$SCOPE" == whole-step ]]; then
    ACCEPTED_FUSIONS=${ACCEPTED_FUSIONS:-rmsnorm}
else
    ACCEPTED_FUSIONS=${ACCEPTED_FUSIONS:-}
fi
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}
SELECT_MIN_PAIRED=${SELECT_MIN_PAIRED:-$REPEATS}
SELECT_MIN_FASTER=${SELECT_MIN_FASTER:-$((REPEATS - 1))}

case "$SCOPE" in model-only|whole-step) ;; *) echo "SCOPE must be model-only or whole-step" >&2; exit 2 ;; esac
mkdir -p "$ROOT_OUTPUT_DIR"
accepted="$ACCEPTED_FUSIONS"

for spec in "KF6:radial-mlp" "KF7:so3-mlp" "KF8:energy-head"; do
    stage=${spec%%:*}; candidate=${spec#*:}
    if [[ "$stage" < "$START_STAGE" ]]; then continue; fi
    candidate_fusions=${accepted:+$accepted,}$candidate
    round_dir="$ROOT_OUTPUT_DIR/round_$stage"
    if [[ -n "$accepted" ]]; then base_stage="${stage}_base"; else base_stage=KF0; fi
    env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
        STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
        BASELINE_STEPS="$BASELINE_STEPS" STEPS="$STEPS" REPEATS="$REPEATS" \
        WARMUP_STEPS="$WARMUP_STEPS" SYSTEMS="$SYSTEMS" \
        TEMPERATURES="$TEMPERATURES" OUTPUT_DIR="$round_dir" \
        BASE_STAGE="$base_stage" BASE_FUSIONS="$accepted" \
        CANDIDATE_STAGE="$stage" CANDIDATE_FUSIONS="$candidate_fusions" \
        SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
        bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"

    selection="$ROOT_OUTPUT_DIR/${stage}_selection.json"
    python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
        --input-dir "$round_dir" --scope "$SCOPE" \
        --base-stage "$base_stage" --candidate-stage "$stage" \
        --candidate-fusion "$candidate" --accepted-before "$accepted" \
        --focus-systems $FOCUS_SYSTEMS \
        --min-paired-repeats "$SELECT_MIN_PAIRED" \
        --min-faster-directions "$SELECT_MIN_FASTER" \
        --output "$selection"
    selected=$?
    if [[ $selected -eq 0 ]]; then accepted="$candidate_fusions"; fi
done

python - "$ROOT_OUTPUT_DIR/accepted_fusions.json" "$accepted" "$SCOPE" "$FOCUS_SYSTEMS" <<'PY'
import json, pathlib, sys
path, accepted, scope, focus = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "scope": scope,
    "accepted_fusions": [x for x in accepted.split(",") if x],
    "focus_systems": focus.split(),
    "stages": ["KF6", "KF7", "KF8"],
    "policy": "Cu512/H2O512 primary; non-focus systems are regression guardrails; numerical errors are telemetry",
}, indent=2) + "\n", encoding="utf-8")
PY
echo "Accepted KF6-KF8 fusions: ${accepted:-none}"
echo "Ablation output: $ROOT_OUTPUT_DIR"
