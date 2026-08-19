#!/usr/bin/env bash
set -euo pipefail

# 1000-step confirmation for the fusion set accepted by the KF6-KF8
# large-system ablation.  Numerical validation remains telemetry.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
: "${ACCEPTED_FUSIONS_FILE:?Set accepted_fusions.json from ablation}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf6_8_formal_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 H2O32 H2O60 H2O192 H2O512"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-5}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}

MODEL_FUSIONS=$(python - "$ACCEPTED_FUSIONS_FILE" <<'PY'
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(data.get("accepted_fusions", [])))
PY
)
if [[ -z "$MODEL_FUSIONS" ]]; then
    echo "No KF6-KF8 fusion was accepted; refusing a misleading candidate run" >&2
    exit 3
fi
FILE_SCOPE=$(python - "$ACCEPTED_FUSIONS_FILE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("scope", ""))
PY
)
[[ "$FILE_SCOPE" == "$SCOPE" ]] || { echo "scope mismatch: $FILE_SCOPE vs $SCOPE" >&2; exit 2; }

# Whole-step's accepted chain starts from the previously accepted RMSNorm;
# model-only starts from Opt2's unmodified model graph.
if [[ "$SCOPE" == whole-step ]]; then BASE_FUSIONS=${BASE_FUSIONS:-rmsnorm}; else BASE_FUSIONS=${BASE_FUSIONS:-}; fi
env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
    BASELINE_STEPS="$BASELINE_STEPS" OUTPUT_DIR="$OUTPUT_DIR" \
    SYSTEMS="$SYSTEMS" TEMPERATURES="$TEMPERATURES" STEPS="$STEPS" \
    WARMUP_STEPS="$WARMUP_STEPS" REPEATS="$REPEATS" \
    BASE_STAGE=KF0 BASE_FUSIONS="$BASE_FUSIONS" \
    CANDIDATE_STAGE=FINAL CANDIDATE_FUSIONS="$MODEL_FUSIONS" \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
echo "KF6-KF8 formal output: $OUTPUT_DIR"
