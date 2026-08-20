#!/usr/bin/env bash
set -euo pipefail

# 1000-step confirmation for an accepted KF9 fusion set.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
: "${ACCEPTED_FUSIONS_FILE:?Set the KF9 accepted_fusions.json}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_kf9_formal_$(date '+%Y%m%d_%H%M%S')"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 H2O32 H2O60 H2O192 H2O512"}
TEMPERATURES=${TEMPERATURES:-"300"}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-5}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}

MODEL_FUSIONS=$(python - "$ACCEPTED_FUSIONS_FILE" <<'PY'
import json
import sys
print(",".join(json.load(open(sys.argv[1], encoding="utf-8")).get("accepted_fusions", [])))
PY
)
[[ "$MODEL_FUSIONS" == *so2-epilogue* ]] || {
    echo "KF9 was not accepted; refusing formal candidate run" >&2
    exit 3
}
FILE_SCOPE=$(python - "$ACCEPTED_FUSIONS_FILE" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("scope", ""))
PY
)
[[ "$FILE_SCOPE" == "$SCOPE" ]] || {
    echo "scope mismatch: $FILE_SCOPE vs $SCOPE" >&2
    exit 2
}

if [[ "$SCOPE" == whole-step ]]; then
    BASE_STAGE=KF4_base
    BASE_FUSIONS=${BASE_FUSIONS:-rmsnorm}
else
    BASE_STAGE=KF0
    BASE_FUSIONS=${BASE_FUSIONS:-}
fi

env GPU="$GPU" SCOPE="$SCOPE" CHECKPOINT="$CHECKPOINT" \
    STRUCTURE_DIR="$STRUCTURE_DIR" BASELINE_DIR="$BASELINE_DIR" \
    BASELINE_STEPS="$BASELINE_STEPS" OUTPUT_DIR="$OUTPUT_DIR" \
    SYSTEMS="$SYSTEMS" TEMPERATURES="$TEMPERATURES" STEPS="$STEPS" \
    WARMUP_STEPS="$WARMUP_STEPS" REPEATS="$REPEATS" \
    BASE_STAGE="$BASE_STAGE" BASE_FUSIONS="$BASE_FUSIONS" \
    CANDIDATE_STAGE=KF9 CANDIDATE_FUSIONS="$MODEL_FUSIONS" \
    SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"

unexpected_failures=$(awk -F '\t' '
    NR > 1 && $9 !~ /^(success|validation_failed)$/ &&
    !($5 == "H2O512" && $9 == "oom") { count++ }
    END { print count + 0 }
' "$OUTPUT_DIR/run_status.tsv")
h2o512_ooms=$(awk -F '\t' '
    NR > 1 && $5 == "H2O512" && $9 == "oom" { count++ }
    END { print count + 0 }
' "$OUTPUT_DIR/run_status.tsv")
formal_status=completed
formal_code=0
if [[ $unexpected_failures -gt 0 ]]; then
    formal_status=failed
    formal_code=4
elif [[ $h2o512_ooms -gt 0 ]]; then
    formal_status=partial_h2o512_oom
fi
printf 'scope\tstatus\tunexpected_failures\th2o512_ooms\toutput_dir\n%s\t%s\t%s\t%s\t%s\n' \
    "$SCOPE" "$formal_status" "$unexpected_failures" "$h2o512_ooms" \
    "$OUTPUT_DIR" > "$OUTPUT_DIR/formal_status.tsv"
echo "KF9 formal output: $OUTPUT_DIR"
echo "KF9 formal status: $formal_status"
exit "$formal_code"
