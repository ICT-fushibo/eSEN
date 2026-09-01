#!/usr/bin/env bash
set -uo pipefail

# CAP2/ROB1 smoke: normal execution plus deliberately undersized recovery.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${REFERENCE_H5:?Set REFERENCE_H5 for bulkCu smoke}"

CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
MATBENCH_REPO=${MATBENCH_REPO:-"$REPO_ROOT/../matbench-discovery"}
# Use a smoke-specific name so that a SYSTEMS value exported by a previous
# formal run cannot accidentally expand this recovery test to the full matrix.
SMOKE_SYSTEMS=${SMOKE_SYSTEMS:-"Cu32 H2O32"}
V4_FUSIONS=${V4_FUSIONS:-rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm,so2-prepare-backward-reduce}

mkdir -p "$OUTPUT_DIR"
failures=0

run_standard() {
    local mode=$1 limit=$2 system=$3
    local root="$OUTPUT_DIR/standard_$mode"
    local run="${system}_300K_30step_CAP2_${mode}"
    mkdir -p "$root/results" "$root/logs"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/example${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$REPO_ROOT/example/benchmark_md_opt4.py" \
            --backend whole-step-cg-opt4 \
            --model-fusions "$V4_FUSIONS" \
            --fusion-stage "CAP2_${mode}" \
            --structure "$STRUCTURE_DIR/$system.cif" \
            --checkpoint "$CHECKPOINT" --system "$system" \
            --steps 30 --warmup-steps 0 --temperature 300 \
            --timestep 1.0 --taut 100.0 --seed 42 --repeat 1 \
            --probe-steps 50 --neighbor-margin 0.10 \
            --neighbor-slot-step 8 --neighbor-capacity-policy elastic \
            --neighbor-auto-min-reduction 0.05 \
            --neighbor-auto-guard-slots 1 --rob1 \
            --rob1-window-steps 10 --rob1-max-retries 2 \
            --cap2-compact-slot-step 4 --cap2-compact-margin 0.0 \
            --cap2-min-reduction 0.05 \
            --cap2-test-capacity-limit "$limit" \
            --dummy-atoms 32 --capture-warmup 3 --max-neighbors 300 \
            --degeneracy-tolerance 0.01 --tf32-mode off \
            --missing-baseline-reference --output-dir "$root/results" \
            --run-name "$run" > "$root/logs/$run.log" 2>&1
    local code=$?
    set +e
    if [[ $code -ne 0 && $code -ne 43 ]]; then
        echo "failed ($code): $run"
        failures=$((failures + 1))
    else
        echo "completed ($code): $run"
    fi
}

for mode in normal forced; do
    if [[ "$mode" == normal ]]; then limit=0; else limit=1; fi
    for system in $SMOKE_SYSTEMS; do
        run_standard "$mode" "$limit" "$system"
    done

    matbench_out="$OUTPUT_DIR/matbench_$mode"
    set +e
    env GPU="$GPU" REFERENCE_H5="$REFERENCE_H5" \
        MATBENCH_REPO="$MATBENCH_REPO" CHECKPOINT="$CHECKPOINT" \
        BACKENDS=opt4 SYSTEMS=bulkCu_1000K_Kapil STEPS=30 \
        RECORD_INTERVAL=10 SEED=0 PROBE_STEPS=50 CAPTURE_WARMUP=3 \
        OPT4_NEIGHBOR_CAPACITY_POLICY=elastic ROB1=1 \
        ROB1_WINDOW_STEPS=0 ROB1_MAX_RETRIES=2 \
        CAP2_COMPACT_SLOT_STEP=4 CAP2_COMPACT_MARGIN=0.0 \
        CAP2_MIN_REDUCTION=0.05 CAP2_TEST_CAPACITY_LIMIT="$limit" \
        OFFLINE_STRESS=0 STATISTICS=0 STRICT=1 SAVE_DIR="$matbench_out" \
        bash "$REPO_ROOT/example/run_esen_matbench.sh"
    code=$?
    set +e
    if [[ $code -ne 0 ]]; then
        echo "failed ($code): bulkCu_1000K_Kapil CAP2 $mode"
        failures=$((failures + 1))
    else
        echo "completed (0): bulkCu_1000K_Kapil CAP2 $mode"
    fi
done

python - "$OUTPUT_DIR" "$SMOKE_SYSTEMS" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np

root = pathlib.Path(sys.argv[1])
smoke_systems = sys.argv[2].split()
failures = []
checked = 0

paths = list(root.glob("standard_*/*/*.json"))
paths += list(root.glob("matbench_*/runs/opt4/*.json"))
for path in sorted(paths):
    row = json.loads(path.read_text(encoding="utf-8"))
    stats = row.get("graph_stats", row)
    # The directory components are named standard_forced/matbench_forced;
    # checking for an exact component named "forced" misclassifies them.
    mode = (
        "forced"
        if any(part.endswith("_forced") for part in path.parts)
        else "normal"
    )
    checked += 1
    expected_replays = 31
    checks = {
        "success": row.get("status", "success") == "success",
        "graph_invariants": row.get("graph_invariants_pass") is True,
        "setup_capture": stats.get("cuda_graph_capture_count") == 1,
        "committed_replays": stats.get("cuda_graph_committed_replays")
        == expected_replays,
        "committed_steps": stats.get("rob1_committed_physical_steps") == 30,
        "unrecovered": stats.get("rob1_unrecovered_overflows") == 0,
        "hit_rate": stats.get("cuda_graph_hit_rate") == 1.0,
        "snapshot_addresses": stats.get("rob1_snapshot_addresses_stable") is True,
        "sink_mode": stats.get("sink_padding_mode")
        == "distributed_dummy_self_edges",
        "sink_shift": stats.get("sink_nonzero_shift_verified") is True,
        "sink_cutoff": stats.get("sink_cutoff_zero_verified") is True,
    }
    if mode == "normal":
        checks["no_rollback"] = stats.get("rob1_rollback_count") == 0
        checks["no_recovery_capture"] = (
            stats.get("cuda_graph_recovery_capture_count") == 0
        )
    else:
        checks["rollback_exercised"] = stats.get("rob1_rollback_count", 0) >= 1
        checks["recovery_capture"] = (
            stats.get("cuda_graph_recovery_capture_count", 0) >= 1
        )
        checks["dummy_only_overflow"] = (
            stats.get("overflow_dummy_only_replays", 0) >= 1
        )
    for name, passed in checks.items():
        if not passed:
            failures.append(f"{path}: {name}")

expected = 2 * len(smoke_systems) + 2  # normal/forced standard + Matbench
if checked != expected:
    failures.append(f"expected {expected} smoke JSON files, found {checked}")

normal_h5 = root / "matbench_normal/trajectories/opt4/bulkCu_1000K_Kapil.h5"
forced_h5 = root / "matbench_forced/trajectories/opt4/bulkCu_1000K_Kapil.h5"
if normal_h5.is_file() and forced_h5.is_file():
    with h5py.File(normal_h5, "r") as normal, h5py.File(forced_h5, "r") as forced:
        np.testing.assert_array_equal(normal["md_step"][:], forced["md_step"][:])
        np.testing.assert_allclose(
            normal["positions"][:], forced["positions"][:], rtol=0, atol=2e-6
        )
        np.testing.assert_allclose(
            normal["momenta"][:], forced["momenta"][:], rtol=0, atol=2e-5
        )
        np.testing.assert_allclose(
            normal["forces"][:], forced["forces"][:], rtol=2e-4, atol=5e-4
        )
        np.testing.assert_allclose(
            normal["energy"][:], forced["energy"][:], rtol=2e-6, atol=1e-3
        )
else:
    failures.append("missing normal/forced bulkCu trajectory comparison")
if failures:
    raise SystemExit("CAP2/ROB1 smoke failed:\n" + "\n".join(failures))
print(f"CAP2/ROB1 smoke passed: {checked} runs")
PY
validator=$?

if [[ $failures -ne 0 || $validator -ne 0 ]]; then
    echo "CAP2/ROB1 smoke failed: command_failures=$failures validator=$validator" >&2
    exit 1
fi

echo "CAP2/ROB1 smoke results: $OUTPUT_DIR"
