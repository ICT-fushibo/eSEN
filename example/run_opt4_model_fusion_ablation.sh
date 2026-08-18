#!/usr/bin/env bash
set -uo pipefail

# Sequential, cumulative Opt4 model-fusion selection for one CUDA Graph
# scope.  Run model-only and whole-step in separate output directories; the
# accepted fusion set is intentionally allowed to differ between scopes.
# Set START_STAGE=KF4 to resume after previously decided rounds, and
# ACCEPTED_FUSIONS="rmsnorm,gate" to seed the cumulative chain with an
# earlier decision (e.g. whole-step resuming at KF6).
#
# The default matrix is the full 10-system x {300,800}K grid.  Energy/force
# errors versus the baseline are telemetry only: acceptance is decided by
# structural CUDA Graph health plus stable paired timing (see
# select_opt4_model_fusions.py).

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
RUN_ID=${RUN_ID:-"esen_opt4_model_fusion_$(date '+%Y%m%d_%H%M%S')"}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}
# Selection thresholds derived from REPEATS unless overridden explicitly.
SELECT_MIN_PAIRED=${SELECT_MIN_PAIRED:-$REPEATS}
SELECT_MIN_FASTER=${SELECT_MIN_FASTER:-$((REPEATS - 1))}

case "$SCOPE" in
    model-only|whole-step) ;;
    *) echo "Unsupported Opt4 scope: $SCOPE" >&2; exit 2 ;;
esac

mkdir -p "$ROOT_OUTPUT_DIR"
START_STAGE=${START_STAGE:-KF2}
# Seed the cumulative chain with a previously accepted set (e.g. the
# small-matrix rmsnorm,gate decision for whole-step) when resuming at a
# later stage.
accepted=${ACCEPTED_FUSIONS:-}
base_stage=KF0

for spec in \
    "KF2:gather-wigner" \
    "KF3:reverse-scatter" \
    "KF4:rmsnorm" \
    "KF5:gate" \
    "KF6:radial-mlp" \
    "KF7:so3-mlp" \
    "KF8:energy-head"; do
    stage=${spec%%:*}
    if [[ "$stage" < "$START_STAGE" ]]; then
        echo "Skip $stage: START_STAGE=$START_STAGE"
        continue
    fi
    candidate=${spec#*:}
    candidate_fusions=${accepted:+$accepted,}$candidate
    round_dir="$ROOT_OUTPUT_DIR/round_$stage"
    if [[ -n "$accepted" ]]; then
        round_base_stage="${stage}_base"
    else
        round_base_stage=KF0
    fi
    env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
        SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
        STRUCTURE_DIR="$STRUCTURE_DIR" \
        BASELINE_DIR="$BASELINE_DIR" BASELINE_STEPS="$BASELINE_STEPS" \
        STEPS="$STEPS" WARMUP_STEPS="$WARMUP_STEPS" \
        REPEATS="$REPEATS" SYSTEMS="$SYSTEMS" TEMPERATURES="$TEMPERATURES" \
        OUTPUT_DIR="$round_dir" BASE_STAGE="$round_base_stage" \
        BASE_FUSIONS="$accepted" CANDIDATE_STAGE="$stage" \
        CANDIDATE_FUSIONS="$candidate_fusions" \
        bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
    round_status=$?
    if [[ $round_status -ne 0 ]]; then
        echo "Opt4 $stage A/B runner failed with exit code $round_status" >&2
        exit "$round_status"
    fi

    selection="$ROOT_OUTPUT_DIR/${stage}_selection.json"
    set +e
    python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
        --input-dir "$round_dir" \
        --scope "$SCOPE" \
        --base-stage "$round_base_stage" \
        --candidate-stage "$stage" \
        --candidate-fusion "$candidate" \
        --accepted-before "$accepted" \
        --min-paired-repeats "$SELECT_MIN_PAIRED" \
        --min-faster-directions "$SELECT_MIN_FASTER" \
        --output "$selection"
    selected=$?
    set -e
    if [[ $selected -eq 0 ]]; then
        accepted="$candidate_fusions"
        base_stage="$stage"
    fi
done

python - "$ROOT_OUTPUT_DIR/accepted_fusions.json" "$accepted" \
    "$base_stage" "$SCOPE" "$REPEATS" "$SELECT_MIN_PAIRED" \
    "$SELECT_MIN_FASTER" <<'PY'
import json, pathlib, sys
path, accepted, stage, scope = sys.argv[1:5]
repeats, min_paired, min_faster = sys.argv[5:8]
pathlib.Path(path).write_text(json.dumps({
    "scope": scope,
    "accepted_fusions": [item for item in accepted.split(",") if item],
    "final_stage": stage,
    "repeats": int(repeats),
    "selection_min_paired_repeats": int(min_paired),
    "selection_min_faster_directions": int(min_faster),
    "policy": "energy/force-vs-baseline errors are telemetry only",
}, indent=2) + "\n", encoding="utf-8")
PY

echo "Opt4 scope: $SCOPE"
echo "Accepted model fusions: ${accepted:-none}"
echo "Selection results: $ROOT_OUTPUT_DIR/accepted_fusions.json"
