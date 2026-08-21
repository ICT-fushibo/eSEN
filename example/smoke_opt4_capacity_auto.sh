#!/usr/bin/env bash
set -euo pipefail

# One-step structural smoke for Opt4 v2 uniform vs CAP1-auto.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
TEMPERATURES=${TEMPERATURES:-"300"}
NEIGHBOR_AUTO_MIN_REDUCTION=${NEIGHBOR_AUTO_MIN_REDUCTION:-0.05}
FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge

env GPU="$GPU" SCOPE=whole-step STEPS=1 REPEATS=1 WARMUP_STEPS=1 \
    SYSTEMS="$SYSTEMS" TEMPERATURES="$TEMPERATURES" \
    CHECKPOINT="$CHECKPOINT" STRUCTURE_DIR="$STRUCTURE_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" BASE_STAGE=OPT4V2 BASE_FUSIONS="$FUSIONS" \
    CANDIDATE_STAGE=CAP1AUTO CANDIDATE_FUSIONS="$FUSIONS" \
    BASE_NEIGHBOR_CAPACITY_POLICY=uniform \
    CANDIDATE_NEIGHBOR_CAPACITY_POLICY=auto \
    NEIGHBOR_AUTO_MIN_REDUCTION="$NEIGHBOR_AUTO_MIN_REDUCTION" \
    bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"

python - "$OUTPUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
failures = []
with (root / "run_status.tsv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
for row in rows:
    if row["status"] != "success":
        failures.append(f'{row["run_name"]}: status={row["status"]}')

decisions = []
for path in sorted((root / "results").glob("*.json")):
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
    if data.get("neighbor_capacity_policy_requested") == "auto":
        decisions.append(
            (
                data["system"],
                data["neighbor_capacity_policy_effective"],
                data["neighbor_capacity_auto_candidate_reduction_vs_uniform"],
                data["neighbor_edge_capacity"],
                data["neighbor_uniform_edge_capacity"],
            )
        )

decision_path = root / "auto_decisions.tsv"
with decision_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(
        ("system", "effective_policy", "candidate_reduction", "edge_capacity", "uniform_edge_capacity")
    )
    writer.writerows(decisions)
print(decision_path.read_text(encoding="utf-8"), end="")
if failures:
    raise SystemExit("CAP1-auto smoke failed:\n" + "\n".join(failures))
print(f"CAP1-auto smoke passed: {len(rows)} runs")
PY
