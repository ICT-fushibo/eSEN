#!/usr/bin/env bash
set -euo pipefail

# Combined one-step smoke: KF12 in both scopes and CAP1-auto-safe in whole-step.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
TEMPERATURES=${TEMPERATURES:-"300"}
PROBE_STEPS=${PROBE_STEPS:-100}
NEIGHBOR_AUTO_GUARD_SLOTS=${NEIGHBOR_AUTO_GUARD_SLOTS:-1}
NEIGHBOR_AUTO_MIN_REDUCTION=${NEIGHBOR_AUTO_MIN_REDUCTION:-0.05}
KF10=so2-epilogue,so2-gate-bridge
KF10_WHOLE=rmsnorm,$KF10
KF12=$KF10,so2-block-gemm
KF12_WHOLE=rmsnorm,$KF12

mkdir -p "$OUTPUT_DIR"

env GPU="$GPU" SCOPE=model-only STEPS=1 REPEATS=1 WARMUP_STEPS=0 \
    SYSTEMS="$SYSTEMS" TEMPERATURES="$TEMPERATURES" \
    CHECKPOINT="$CHECKPOINT" STRUCTURE_DIR="$STRUCTURE_DIR" \
    BASELINE_DIR="$BASELINE_DIR" BASELINE_STEPS=100 \
    OUTPUT_DIR="$OUTPUT_DIR/model_only" \
    BASE_STAGE=OPT4V2 BASE_FUSIONS="$KF10" \
    CANDIDATE_STAGE=KF12 CANDIDATE_FUSIONS="$KF12" \
    BASE_NEIGHBOR_CAPACITY_POLICY=uniform \
    CANDIDATE_NEIGHBOR_CAPACITY_POLICY=uniform \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"

env GPU="$GPU" SCOPE=whole-step STEPS=1 REPEATS=1 WARMUP_STEPS=0 \
    SYSTEMS="$SYSTEMS" TEMPERATURES="$TEMPERATURES" \
    CHECKPOINT="$CHECKPOINT" STRUCTURE_DIR="$STRUCTURE_DIR" \
    BASELINE_DIR="$BASELINE_DIR" BASELINE_STEPS=100 \
    OUTPUT_DIR="$OUTPUT_DIR/whole_step" \
    BASE_STAGE=OPT4V2 BASE_FUSIONS="$KF10_WHOLE" \
    CANDIDATE_STAGE=KF12CAP1SAFE CANDIDATE_FUSIONS="$KF12_WHOLE" \
    BASE_NEIGHBOR_CAPACITY_POLICY=uniform \
    CANDIDATE_NEIGHBOR_CAPACITY_POLICY=auto-safe \
    PROBE_STEPS="$PROBE_STEPS" \
    NEIGHBOR_AUTO_GUARD_SLOTS="$NEIGHBOR_AUTO_GUARD_SLOTS" \
    NEIGHBOR_AUTO_MIN_REDUCTION="$NEIGHBOR_AUTO_MIN_REDUCTION" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"

python - "$OUTPUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
failures = []
candidate_count = 0
for scope in ("model_only", "whole_step"):
    scope_root = root / scope
    with (scope_root / "run_status.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if row["status"] != "success":
            failures.append(f'{row["run_name"]}: status={row["status"]}')
    for path in sorted((scope_root / "results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("graph_invariants_pass"):
            failures.append(f"{path.name}: graph invariants failed")
        if data.get("cuda_graph_capture_count") != 1:
            failures.append(f"{path.name}: capture_count != 1")
        if data.get("cuda_graph_production_capture_count") != 0:
            failures.append(f"{path.name}: production capture occurred")
        if data.get("cuda_graph_production_replays") != 2:
            failures.append(f"{path.name}: production_replays != 2")
        if data.get("cuda_graph_capacity_misses") != 0:
            failures.append(f"{path.name}: capacity miss")
        if data.get("cuda_graph_hit_rate") != 1.0:
            failures.append(f"{path.name}: graph hit rate != 1")
        if "so2-block-gemm" in data.get("model_fusions", ""):
            candidate_count += 1
            if data.get("model_fusion_so2_block_gemm_convolution_replacements") != 20:
                failures.append(f"{path.name}: KF12 replacement count != 20")
        if data.get("neighbor_capacity_policy_requested") == "auto-safe":
            if data.get("neighbor_capacity_auto_guard_slots") != 1:
                failures.append(f"{path.name}: auto-safe guard_slots != 1")
            if data.get("neighbor_capacity_policy_effective") not in {
                "uniform",
                "atom-safe",
            }:
                failures.append(f"{path.name}: invalid auto-safe decision")

if candidate_count == 0:
    failures.append("no KF12 candidate result was found")
status = "failed" if failures else "passed"
(root / "smoke_status.tsv").write_text(
    "stage\tstatus\tcandidate_results\n"
    f"KF12+CAP1-auto-safe\t{status}\t{candidate_count}\n",
    encoding="utf-8",
)
if failures:
    raise SystemExit("combined smoke failed:\n" + "\n".join(failures))
print(f"combined smoke passed: {candidate_count} KF12 candidate results")
PY

echo "Combined smoke output: $OUTPUT_DIR"
