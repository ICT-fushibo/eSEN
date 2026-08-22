#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${STRUCTURE_DIR:?Set STRUCTURE_DIR}"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/opt4_prec1_smoke_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"}
SYSTEMS=${SYSTEMS:-"Cu32 Cu512 H2O32 H2O192"}
MODEL_FUSIONS=so2-epilogue,so2-gate-bridge,so2-block-gemm
WHOLE_FUSIONS=rmsnorm,$MODEL_FUSIONS

for scope in model-only whole-step; do
    scope_dir="$OUTPUT_DIR/${scope//-/_}"
    if [[ "$scope" == model-only ]]; then
        fusions=$MODEL_FUSIONS
        capacity=uniform
    else
        fusions=$WHOLE_FUSIONS
        capacity=auto-safe
    fi
    env GPU="$GPU" SCOPE="$scope" OUTPUT_DIR="$scope_dir" \
        CHECKPOINT="$CHECKPOINT" STRUCTURE_DIR="$STRUCTURE_DIR" \
        BASELINE_DIR="${BASELINE_DIR:-}" BASELINE_STEPS="${BASELINE_STEPS:-100}" \
        SYSTEMS="$SYSTEMS" TEMPERATURES=300 STEPS=1 WARMUP_STEPS=1 \
        REPEATS=1 PROBE_STEPS=100 \
        BASE_STAGE=OPT4V3_FP32 CANDIDATE_STAGE=PREC1_TF32 \
        BASE_FUSIONS="$fusions" CANDIDATE_FUSIONS="$fusions" \
        BASE_NEIGHBOR_CAPACITY_POLICY="$capacity" \
        CANDIDATE_NEIGHBOR_CAPACITY_POLICY="$capacity" \
        BASE_TF32_MODE=off CANDIDATE_TF32_MODE=on \
        SOURCE_BUNDLE_SHA256="${SOURCE_BUNDLE_SHA256:-}" \
        bash "$REPO_ROOT/example/run_opt4_interleaved_stage.sh"
done

python - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
failures = []
candidate_count = 0
for path in root.glob("*/results/*PREC1_TF32*.json"):
    data = json.loads(path.read_text())
    candidate_count += 1
    expected_replays = int(data["steps"]) + 1
    checks = {
        "tf32": data.get("tf32") is True,
        "tf32_mode_requested": data.get("tf32_mode_requested") == "on",
        "tf32_matmul_allowed": data.get("tf32_matmul_allowed") is True,
        "tf32_cudnn_allowed": data.get("tf32_cudnn_allowed") is True,
        "float32_matmul_precision": data.get("float32_matmul_precision") == "high",
        "tf32_config_verified": data.get("tf32_config_verified") is True,
        "graph_invariants_pass": data.get("graph_invariants_pass") is True,
        "capture_count": data.get("cuda_graph_capture_count") == 1,
        "production_capture_count": data.get("cuda_graph_production_capture_count") == 0,
        "production_replays": data.get("cuda_graph_production_replays") == expected_replays,
        "capacity_misses": data.get("cuda_graph_capacity_misses") == 0,
        "graph_hit_rate": data.get("cuda_graph_hit_rate") == 1.0,
    }
    failures.extend(f"{path.name}: {name}" for name, ok in checks.items() if not ok)
for path in root.glob("*/results/*OPT4V3_FP32*.json"):
    data = json.loads(path.read_text())
    if not (
        data.get("tf32") is False
        and data.get("tf32_mode_requested") == "off"
        and data.get("tf32_matmul_allowed") is False
        and data.get("float32_matmul_precision") == "highest"
        and data.get("tf32_config_verified") is True
    ):
        failures.append(f"{path.name}: invalid FP32 control metadata")
if candidate_count == 0:
    failures.append("no PREC1 candidate JSON was found")
if failures:
    raise SystemExit("PREC1 smoke failed:\n" + "\n".join(failures))
print(f"PREC1 smoke passed: {candidate_count} candidate results")
PY

echo "PREC1 smoke results: $OUTPUT_DIR"
