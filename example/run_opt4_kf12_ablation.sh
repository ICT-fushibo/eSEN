#!/usr/bin/env bash
set -uo pipefail

# Three-repeat interleaved Opt4 v2 vs KF12 SO2 block-GEMM ablation.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-100}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf12_ablation_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"Cu512 H2O192"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}
PROBE_STEPS=${PROBE_STEPS:-100}
NEIGHBOR_AUTO_GUARD_SLOTS=${NEIGHBOR_AUTO_GUARD_SLOTS:-1}

case "$SCOPE" in
    model-only)
        BASE_FUSIONS=so2-epilogue,so2-gate-bridge
        BASE_POLICY=uniform
        BASE_STAGE_LABEL=OPT4V2
        CANDIDATE_STAGE_LABEL=KF12
        ;;
    whole-step)
        BASE_FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge
        # Hold guarded CAP1-auto constant so this round isolates KF12.
        BASE_POLICY=auto-safe
        BASE_STAGE_LABEL=OPT4V2CAP1SAFE
        CANDIDATE_STAGE_LABEL=KF12CAP1SAFE
        ;;
    *) echo "SCOPE must be model-only or whole-step" >&2; exit 2 ;;
esac
CANDIDATE_FUSIONS="$BASE_FUSIONS,so2-block-gemm"
ROUND_DIR="$ROOT_OUTPUT_DIR/round_KF12"
mkdir -p "$ROOT_OUTPUT_DIR"

env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
    BASELINE_STEPS="$BASELINE_STEPS" STEPS="$STEPS" REPEATS="$REPEATS" \
    WARMUP_STEPS="$WARMUP_STEPS" SYSTEMS="$SYSTEMS" \
    TEMPERATURES="$TEMPERATURES" OUTPUT_DIR="$ROUND_DIR" \
    BASE_STAGE="$BASE_STAGE_LABEL" BASE_FUSIONS="$BASE_FUSIONS" \
    CANDIDATE_STAGE="$CANDIDATE_STAGE_LABEL" \
    CANDIDATE_FUSIONS="$CANDIDATE_FUSIONS" \
    BASE_NEIGHBOR_CAPACITY_POLICY="$BASE_POLICY" \
    CANDIDATE_NEIGHBOR_CAPACITY_POLICY="$BASE_POLICY" \
    NEIGHBOR_AUTO_MIN_REDUCTION="${NEIGHBOR_AUTO_MIN_REDUCTION:-0.05}" \
    NEIGHBOR_AUTO_GUARD_SLOTS="$NEIGHBOR_AUTO_GUARD_SLOTS" \
    PROBE_STEPS="$PROBE_STEPS" \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
runner_code=$?

selection="$ROOT_OUTPUT_DIR/KF12_selection.json"
set +e
python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
    --input-dir "$ROUND_DIR" --scope "$SCOPE" \
    --base-stage "$BASE_STAGE_LABEL" \
    --candidate-stage "$CANDIDATE_STAGE_LABEL" \
    --candidate-fusion so2-block-gemm --accepted-before "$BASE_FUSIONS" \
    --focus-systems $FOCUS_SYSTEMS \
    --min-paired-repeats "$REPEATS" --min-faster-directions "$REPEATS" \
    --maximum-peak-reserved-increase-gib 1.0 \
    --output "$selection"
selected=$?
set -e
if [[ $runner_code -ne 0 ]]; then selected=$runner_code; fi

python - "$ROOT_OUTPUT_DIR/accepted_fusions.json" "$SCOPE" "$FOCUS_SYSTEMS" "$selected" "$BASE_FUSIONS" "$NEIGHBOR_AUTO_GUARD_SLOTS" "$CANDIDATE_STAGE_LABEL" <<'PY'
import json
import pathlib
import sys

path, scope, focus, selected, base, guard_slots, candidate_stage = sys.argv[1:]
accepted = [item for item in base.split(",") if item]
if int(selected) == 0:
    accepted.append("so2-block-gemm")
pathlib.Path(path).write_text(json.dumps({
    "scope": scope,
    "accepted_fusions": accepted,
    "focus_systems": focus.split(),
    "candidate_stage": candidate_stage,
    "neighbor_capacity_policy": "auto-safe" if scope == "whole-step" else "uniform",
    "neighbor_auto_guard_slots": int(guard_slots) if scope == "whole-step" else 0,
    "policy": "energy/force errors are telemetry only",
}, indent=2) + "\n", encoding="utf-8")
PY

echo "KF12 selector exit: $selected"
echo "KF12 interleaved runner exit: $runner_code"
echo "KF12 selection: $selection"
echo "KF12 accepted fusions: $ROOT_OUTPUT_DIR/accepted_fusions.json"
echo "KF12 ablation output: $ROOT_OUTPUT_DIR"
exit "$selected"
