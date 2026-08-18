#!/usr/bin/env bash
set -euo pipefail

# Full 10-system, 300/800 K interleaved control-versus-Final confirmation for
# one automatically selected scope.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
: "${ACCEPTED_FUSIONS_FILE:?Set accepted_fusions.json from the ablation runner}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
RUN_ID=${RUN_ID:-"esen_opt4_final_$(date '+%Y%m%d_%H%M%S')"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}
REPEATS=${REPEATS:-3}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}

case "$SCOPE" in
    model-only|whole-step) ;;
    *) echo "Unsupported Opt4 scope: $SCOPE" >&2; exit 2 ;;
esac

MODEL_FUSIONS=$(python - "$ACCEPTED_FUSIONS_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(data.get("accepted_fusions", [])))
PY
)
if [[ -z "$MODEL_FUSIONS" ]]; then
    echo "No fusion passed selection; the $SCOPE control remains final" >&2
    exit 3
fi
FILE_SCOPE=$(python - "$ACCEPTED_FUSIONS_FILE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("scope", ""))
PY
)
if [[ "$FILE_SCOPE" != "$SCOPE" ]]; then
    echo "Accepted-fusion scope mismatch: file=$FILE_SCOPE requested=$SCOPE" >&2
    exit 2
fi

env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    BASELINE_DIR="$BASELINE_DIR" BASELINE_STEPS="$BASELINE_STEPS" \
    OUTPUT_DIR="$OUTPUT_DIR" SYSTEMS="$SYSTEMS" \
    TEMPERATURES="$TEMPERATURES" REPEATS="$REPEATS" \
    STEPS="$STEPS" WARMUP_STEPS="$WARMUP_STEPS" \
    BASE_STAGE=KF0 BASE_FUSIONS="" CANDIDATE_STAGE=FINAL \
    CANDIDATE_FUSIONS="$MODEL_FUSIONS" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"

echo "Opt4 Final scope: $SCOPE"
echo "Opt4 Final fusions: $MODEL_FUSIONS"
echo "Opt4 Final results: $OUTPUT_DIR"
