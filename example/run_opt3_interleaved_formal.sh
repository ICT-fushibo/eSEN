#!/usr/bin/env bash
set -uo pipefail

# Fair Opt2 -> Opt3 formal benchmark.
#
# Each (system, temperature, repeat) is one block.  Backend order is shuffled
# deterministically inside every block, and system/temperature order is also
# shuffled per repeat.  Every backend runs in a fresh Python process.
#
# This script never starts, stops, or configures NVIDIA MPS.  GPU is mandatory:
#   GPU=2 BASELINE_DIR=/path/to/ase ... bash example/run_opt3_interleaved_formal.sh

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index, for example GPU=2}"

CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
RUN_ID=${RUN_ID:-"esen_opt3_interleaved_$(date '+%Y%m%d_%H%M%S')"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
# The existing ASE reference matrix normally contains r1-r3.  Use REPEATS=5
# only after matching r4/r5 baseline JSON files have been produced.
REPEATS=${REPEATS:-3}
SEED=42
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 H2O32 H2O60"}
TEMPERATURES=${TEMPERATURES:-"300 800"}
BACKENDS=${BACKENDS:-"model-cg fixed-builder-model-cg force-eval-cg whole-step-cg"}
# Missing/OOM ASE references do not suppress performance runs; they are
# recorded as missing_reference.  Set to 1 only for a strict reference matrix.
REQUIRE_BASELINE_REFERENCE=${REQUIRE_BASELINE_REFERENCE:-0}
REQUIRE_GPU_IDLE=${REQUIRE_GPU_IDLE:-1}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-200}
STRICT=${STRICT:-0}

PROBE_STEPS=${PROBE_STEPS:-50}
NEIGHBOR_MARGIN=${NEIGHBOR_MARGIN:-0.10}
NEIGHBOR_SLOT_STEP=${NEIGHBOR_SLOT_STEP:-8}
DUMMY_ATOMS=${DUMMY_ATOMS:-32}
CAPTURE_WARMUP=${CAPTURE_WARMUP:-3}
MAX_NEIGHBORS=${MAX_NEIGHBORS:-300}
DEGENERACY_TOLERANCE=${DEGENERACY_TOLERANCE:-0.01}
ENERGY_PER_ATOM_ATOL=${ENERGY_PER_ATOM_ATOL:-1e-5}
FORCE_MAX_ATOL=${FORCE_MAX_ATOL:-2e-4}

OPT2_EDGE_STEP=${OPT2_EDGE_STEP:-256}
OPT2_REPLAY_ENERGY_ATOL=${OPT2_REPLAY_ENERGY_ATOL:-0.0}
OPT2_REPLAY_FORCE_ATOL=${OPT2_REPLAY_FORCE_ATOL:-1e-6}

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ "$REQUIRE_BASELINE_REFERENCE" == "1" && -z "$BASELINE_DIR" ]]; then
    echo "BASELINE_DIR is required for formal energy validation" >&2
    exit 2
fi
if [[ -n "$BASELINE_DIR" && ! -d "$BASELINE_DIR" ]]; then
    echo "Baseline directory not found: $BASELINE_DIR" >&2
    exit 2
fi

read -r -a systems <<< "$SYSTEMS"
read -r -a temperatures <<< "$TEMPERATURES"
read -r -a backends <<< "$BACKENDS"
for backend in "${backends[@]}"; do
    case "$backend" in
        model-cg|fixed-builder-model-cg|force-eval-cg|whole-step-cg) ;;
        *) echo "Unsupported backend: $backend" >&2; exit 2 ;;
    esac
done
for system in "${systems[@]}"; do
    if [[ ! -f "$STRUCTURE_DIR/$system.cif" ]]; then
        echo "Structure not found: $STRUCTURE_DIR/$system.cif" >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR"/{results,logs,telemetry,snapshots}
STATUS_TSV="$OUTPUT_DIR/run_status.tsv"
printf 'backend\tsystem\ttemperature_K\trepeat\trun_name\tstatus\texit_code\tprocess_wall_time_s\tbaseline_reference_status\tbaseline_result\ttelemetry\n' \
    > "$STATUS_TSV"

GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
if [[ -z "$GPU_UUID" ]]; then
    echo "Unable to resolve physical GPU $GPU" >&2
    exit 2
fi

{
    echo "run_id=$RUN_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "gpu=$GPU"
    echo "gpu_uuid=$GPU_UUID"
    echo "checkpoint=$CHECKPOINT"
    echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    echo "structure_dir=$STRUCTURE_DIR"
    echo "baseline_dir=$BASELINE_DIR"
    echo "systems=$SYSTEMS"
    echo "temperatures=$TEMPERATURES"
    echo "backends=$BACKENDS"
    echo "steps=$STEPS"
    echo "warmup_steps=$WARMUP_STEPS"
    echo "repeats=$REPEATS"
    echo "seed=$SEED"
    echo "require_gpu_idle=$REQUIRE_GPU_IDLE"
    echo "telemetry_interval_ms=$TELEMETRY_INTERVAL_MS"
    echo "pythonhashseed=$SEED"
    echo "cublas_workspace_config=:4096:8"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        python -c 'import torch; print(f"python_torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
    nvidia-smi -i "$GPU" \
        --query-gpu=index,name,uuid,driver_version,memory.total,compute_mode \
        --format=csv,noheader
} > "$OUTPUT_DIR/run_metadata.txt"

active_compute_processes() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
        --format=csv,noheader 2>/dev/null | awk -F', ' -v uuid="$GPU_UUID" \
        '$1 == uuid {print $0}'
}

ensure_gpu_idle() {
    [[ "$REQUIRE_GPU_IDLE" == "1" ]] || return 0
    local active
    active=$(active_compute_processes)
    if [[ -n "$active" ]]; then
        echo "Requested GPU $GPU ($GPU_UUID) is not idle:" >&2
        echo "$active" >&2
        return 1
    fi
}

TELEMETRY_PID=
stop_telemetry() {
    if [[ -n "${TELEMETRY_PID:-}" ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
        TELEMETRY_PID=
    fi
}
trap stop_telemetry EXIT

start_telemetry() {
    local path=$1
    nvidia-smi -i "$GPU" \
        --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
        --format=csv -lms "$TELEMETRY_INTERVAL_MS" > "$path" 2>&1 &
    TELEMETRY_PID=$!
}

shuffle_values() {
    local salt=$1
    shift
    python -c 'import hashlib,random,sys; salt=sys.argv[1]; values=sys.argv[2:]; seed=int.from_bytes(hashlib.sha256(("42|"+salt).encode()).digest()[:8], "big"); random.Random(seed).shuffle(values); print(" ".join(values))' \
        "$salt" "$@"
}

classify_status() {
    local exit_code=$1 log_path=$2 result_path=$3
    if [[ $exit_code -eq 0 && -s "$result_path" ]]; then
        echo success
    elif [[ $exit_code -eq 42 ]] || grep -Eqi \
        'STATUS=oom|CUDA out of memory|OutOfMemoryError' "$log_path"; then
        echo oom
    elif [[ $exit_code -eq 45 ]] || grep -Eqi \
        'STATUS=capacity_overflow' "$log_path"; then
        echo capacity_overflow
    elif [[ $exit_code -eq 43 ]] || grep -Eqi \
        'STATUS=validation_failed' "$log_path"; then
        echo validation_failed
    else
        echo error
    fi
}

run_one() {
    local backend=$1 system=$2 temperature=$3 repeat=$4
    local suffix=${backend//-/_}
    local run_name="${system}_${temperature}K_${STEPS}step_esen_${suffix}_r${repeat}"
    local result_dir="$OUTPUT_DIR/results/$backend"
    local result_path="$result_dir/$run_name.json"
    local log_path="$OUTPUT_DIR/logs/$run_name.log"
    local telemetry_path="$OUTPUT_DIR/telemetry/$run_name.csv"
    local snapshot_before="$OUTPUT_DIR/snapshots/${run_name}_before.txt"
    local snapshot_after="$OUTPUT_DIR/snapshots/${run_name}_after.txt"
    local baseline_name="${system}_${temperature}K_${BASELINE_STEPS}step_esen_baseline_r${repeat}.json"
    local baseline_result=""
    local baseline_status=not_requested
    local reference_args=()

    mkdir -p "$result_dir"
    if [[ -n "$BASELINE_DIR" ]]; then
        baseline_result="$BASELINE_DIR/$baseline_name"
        if [[ -s "$baseline_result" ]]; then
            baseline_status=available
            reference_args+=(--baseline-result "$baseline_result")
        elif [[ "$REQUIRE_BASELINE_REFERENCE" == "1" ]]; then
            baseline_status=missing
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$backend" "$system" "$temperature" "$repeat" "$run_name" \
                missing_reference 44 0.000000 "$baseline_status" \
                "$baseline_result" "$telemetry_path" >> "$STATUS_TSV"
            echo "MISSING BASELINE: $baseline_result" >&2
            return 0
        else
            baseline_status=missing
            reference_args+=(--missing-baseline-reference)
        fi
    fi

    ensure_gpu_idle || return 1
    {
        date --iso-8601=seconds
        nvidia-smi -i "$GPU" \
            --query-gpu=index,uuid,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
            --format=csv
        active_compute_processes
    } > "$snapshot_before"
    start_telemetry "$telemetry_path"
    echo "Running $run_name on physical GPU $GPU ($GPU_UUID)"

    local start_ns end_ns process_wall_time exit_code
    start_ns=$(date +%s%N)
    set +e
    if [[ "$backend" == "model-cg" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONHASHSEED="$SEED" \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
            python -u "$REPO_ROOT/example/benchmark_md_gpu.py" \
                --backend model-cg \
                --structure "$STRUCTURE_DIR/$system.cif" \
                --checkpoint "$CHECKPOINT" \
                --system "$system" \
                --output-dir "$result_dir" \
                --run-name "$run_name" \
                --steps "$STEPS" \
                --warmup-steps "$WARMUP_STEPS" \
                --temperature "$temperature" \
                --timestep 1.0 --taut 100.0 \
                --seed "$SEED" --repeat "$repeat" --md-dtype float64 \
                --cg-probe-steps "$PROBE_STEPS" \
                --cg-capacity-margin "$NEIGHBOR_MARGIN" \
                --cg-edge-step "$OPT2_EDGE_STEP" \
                --cg-dummy-atoms "$DUMMY_ATOMS" \
                --cg-capture-warmup "$CAPTURE_WARMUP" \
                --cg-replay-energy-atol "$OPT2_REPLAY_ENERGY_ATOL" \
                --cg-replay-force-atol "$OPT2_REPLAY_FORCE_ATOL" \
                "${reference_args[@]}" > "$log_path" 2>&1
        exit_code=$?
    else
        CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONHASHSEED="$SEED" \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
            python -u "$REPO_ROOT/example/profile_opt3.py" \
                --backend "$backend" --profile-kind timing \
                --structure "$STRUCTURE_DIR/$system.cif" \
                --checkpoint "$CHECKPOINT" \
                --system "$system" \
                --temperature "$temperature" \
                --output-dir "$result_dir" \
                --run-name "$run_name" \
                --steps "$STEPS" --component-steps 0 \
                --warmup-steps "$WARMUP_STEPS" \
                --probe-steps "$PROBE_STEPS" \
                --repeat "$repeat" --seed "$SEED" \
                --timestep 1.0 --taut 100.0 \
                --neighbor-margin "$NEIGHBOR_MARGIN" \
                --neighbor-slot-step "$NEIGHBOR_SLOT_STEP" \
                --dummy-atoms "$DUMMY_ATOMS" \
                --capture-warmup "$CAPTURE_WARMUP" \
                --max-neighbors "$MAX_NEIGHBORS" \
                --degeneracy-tolerance "$DEGENERACY_TOLERANCE" \
                --energy-per-atom-atol "$ENERGY_PER_ATOM_ATOL" \
                --force-max-atol "$FORCE_MAX_ATOL" \
                "${reference_args[@]}" > "$log_path" 2>&1
        exit_code=$?
    fi
    set -e
    stop_telemetry
    end_ns=$(date +%s%N)
    process_wall_time=$(awk -v start="$start_ns" -v end="$end_ns" \
        'BEGIN { printf "%.6f", (end - start) / 1000000000 }')

    {
        date --iso-8601=seconds
        nvidia-smi -i "$GPU" \
            --query-gpu=index,uuid,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
            --format=csv
        active_compute_processes
    } > "$snapshot_after"

    local status
    status=$(classify_status "$exit_code" "$log_path" "$result_path")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$backend" "$system" "$temperature" "$repeat" "$run_name" \
        "$status" "$exit_code" "$process_wall_time" "$baseline_status" \
        "$baseline_result" "$telemetry_path" >> "$STATUS_TSV"
    echo "$status ($exit_code): $run_name"
}

failure=0
for repeat in $(seq 1 "$REPEATS"); do
    read -r -a repeat_systems <<< "$(shuffle_values "systems|$repeat" "${systems[@]}")"
    for system in "${repeat_systems[@]}"; do
        read -r -a repeat_temperatures <<< "$(shuffle_values "temperatures|$repeat|$system" "${temperatures[@]}")"
        for temperature in "${repeat_temperatures[@]}"; do
            read -r -a block_backends <<< "$(shuffle_values "backends|$repeat|$system|$temperature" "${backends[@]}")"
            for backend in "${block_backends[@]}"; do
                if ! run_one "$backend" "$system" "$temperature" "$repeat"; then
                    failure=1
                    echo "Runner stopped before $backend/$system/${temperature}K/r$repeat; GPU may not be idle" >&2
                    break 4
                fi
            done
        done
    done
done

python "$REPO_ROOT/example/summarize_opt3_interleaved.py" \
    --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR"

{
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "runner_failure=$failure"
} >> "$OUTPUT_DIR/run_metadata.txt"

echo "Opt3 interleaved results: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/opt3_interleaved_report.md"
if [[ "$STRICT" == "1" ]]; then
    if [[ "$failure" -ne 0 ]] || awk -F'\t' \
        'NR > 1 && $6 != "success" {found=1} END {exit !found}' \
        "$STATUS_TSV"; then
        exit 1
    fi
fi
