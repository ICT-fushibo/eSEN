#!/usr/bin/env bash
set -uo pipefail

# Interleaved Opt4 v2 whole-step uniform-vs-per-atom capacity ablation.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-100}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_capacity_ablation_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
FOCUS_SYSTEMS=${FOCUS_SYSTEMS:-"H2O32 H2O192"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}
MINIMUM_GEOMEAN_SPEEDUP=${MINIMUM_GEOMEAN_SPEEDUP:-1.01}
MAXIMUM_SYSTEM_REGRESSION=${MAXIMUM_SYSTEM_REGRESSION:-0.01}
MIN_FASTER_DIRECTIONS=${MIN_FASTER_DIRECTIONS:-$REPEATS}

FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge
ROUND_DIR="$ROOT_OUTPUT_DIR/round_CAP1"
mkdir -p "$ROOT_OUTPUT_DIR"

env GPU="$GPU" SCOPE=whole-step CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
    BASELINE_STEPS="$BASELINE_STEPS" STEPS="$STEPS" REPEATS="$REPEATS" \
    WARMUP_STEPS="$WARMUP_STEPS" SYSTEMS="$SYSTEMS" \
    TEMPERATURES="$TEMPERATURES" OUTPUT_DIR="$ROUND_DIR" \
    BASE_STAGE=OPT4V2 BASE_FUSIONS="$FUSIONS" \
    CANDIDATE_STAGE=CAP1 CANDIDATE_FUSIONS="$FUSIONS" \
    BASE_NEIGHBOR_CAPACITY_POLICY=uniform \
    CANDIDATE_NEIGHBOR_CAPACITY_POLICY=atom \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
runner_code=$?

selection="$ROOT_OUTPUT_DIR/CAP1_selection.json"
set +e
python "$REPO_ROOT/example/select_opt4_model_fusions.py" \
    --input-dir "$ROUND_DIR" --scope whole-step \
    --base-stage OPT4V2 --candidate-stage CAP1 \
    --candidate-fusion atom-capacity --accepted-before "" \
    --focus-systems $FOCUS_SYSTEMS \
    --minimum-geomean-speedup "$MINIMUM_GEOMEAN_SPEEDUP" \
    --maximum-system-regression "$MAXIMUM_SYSTEM_REGRESSION" \
    --min-paired-repeats "$REPEATS" \
    --min-faster-directions "$MIN_FASTER_DIRECTIONS" \
    --maximum-peak-reserved-increase-gib 1.0 \
    --output "$selection"
selected=$?
set -e
if [[ $runner_code -ne 0 ]]; then selected=$runner_code; fi

python - "$ROOT_OUTPUT_DIR/accepted_capacity.json" "$selection" <<'PY'
import json
import pathlib
import sys

output, selection = map(pathlib.Path, sys.argv[1:])
record = json.loads(selection.read_text(encoding="utf-8"))
output.write_text(json.dumps({
    "accepted": bool(record.get("accepted")),
    "base_policy": "uniform",
    "candidate_policy": "atom",
    "model_fusions": "rmsnorm,so2-epilogue,so2-gate-bridge",
    "focus_systems": record.get("focus_systems", []),
    "policy": "energy/force errors are telemetry only",
}, indent=2) + "\n", encoding="utf-8")
PY

echo "CAP1 selector exit: $selected"
echo "CAP1 selection: $selection"
echo "CAP1 output: $ROOT_OUTPUT_DIR"
exit "$selected"
