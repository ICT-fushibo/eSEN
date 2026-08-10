#!/usr/bin/env bash
set -euo pipefail

# Opt3 regression profiling.  This script never starts/stops MPS and uses
# absolute Nsight tool paths so no global PATH modification is required.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}
PHASE=${PHASE:-timing}
RUN_ID=${RUN_ID:-"esen_opt3_profile_$(date '+%Y%m%d_%H%M%S')"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
METADATA_FILE="$OUTPUT_DIR/run_metadata_${PHASE}.txt"
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
NCU=${NCU:-/usr/local/cuda/bin/ncu}
STEPS=${STEPS:-1000}
REPEATS=${REPEATS:-5}
COMPONENT_STEPS=${COMPONENT_STEPS:-$STEPS}
TRACE_STEPS=${TRACE_STEPS:-20}
SEED=42

if [[ "$PHASE" != "smoke" && "$PHASE" != "timing" && "$PHASE" != "torch" && \
      "$PHASE" != "nsys" && "$PHASE" != "ncu" && "$PHASE" != "all" ]]; then
    echo "PHASE must be smoke, timing, torch, nsys, ncu, or all" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"/{smoke,timing,torch,nsys,ncu,telemetry,logs}
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')

{
    echo "run_id=$RUN_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "gpu=$GPU"
    echo "gpu_uuid=$GPU_UUID"
    echo "checkpoint=$CHECKPOINT"
    echo "structure_dir=$STRUCTURE_DIR"
    echo "baseline_dir=$BASELINE_DIR"
    echo "seed=$SEED"
    echo "steps=$STEPS"
    echo "repeats=$REPEATS"
    echo "component_steps=$COMPONENT_STEPS"
    echo "trace_steps=$TRACE_STEPS"
    echo "nsys=$NSYS"
    echo "ncu=$NCU"
    [[ -x "$NSYS" ]] && "$NSYS" --version | tr '\n' ' '
    echo
    [[ -x "$NCU" ]] && "$NCU" --version | tr '\n' ' '
    echo
    nvidia-smi -i "$GPU" --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader
} > "$METADATA_FILE"

ensure_gpu_idle() {
    local active
    active=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
        --format=csv,noheader 2>/dev/null | awk -F', ' -v uuid="$GPU_UUID" \
        '$1 == uuid {print $0}')
    if [[ -n "$active" ]]; then
        echo "Requested GPU is not idle:" >&2
        echo "$active" >&2
        return 1
    fi
}

baseline_args() {
    local system=$1 temperature=$2 repeat=$3
    local path
    if [[ -z "$BASELINE_DIR" ]]; then
        return
    fi
    path="$BASELINE_DIR/${system}_${temperature}K_1000step_esen_baseline_r${repeat}.json"
    if [[ -f "$path" ]]; then
        printf '%s\n' "--baseline-result" "$path"
    else
        printf '%s\n' "--missing-baseline-reference"
    fi
}

start_telemetry() {
    local run_name=$1
    nvidia-smi -i "$GPU" \
        --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
        --format=csv -lms 200 \
        > "$OUTPUT_DIR/telemetry/${run_name}.csv" 2>&1 &
    TELEMETRY_PID=$!
}

stop_telemetry() {
    if [[ -n "${TELEMETRY_PID:-}" ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
        TELEMETRY_PID=
    fi
}
trap stop_telemetry EXIT

record_failure() {
    local kind=$1 backend=$2 system=$3 temperature=$4 repeat=$5 status=$6
    local path="$OUTPUT_DIR/failed_runs.tsv"
    if [[ ! -s "$path" ]]; then
        printf 'profile_kind\tbackend\tsystem\ttemperature_K\trepeat\texit_code\tstatus\n' \
            > "$path"
    fi
    local label=error
    [[ "$status" -eq 42 ]] && label=oom
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$kind" "$backend" "$system" "$temperature" "$repeat" \
        "$status" "$label" >> "$path"
}

run_python_profile() {
    local kind=$1 backend=$2 system=$3 temperature=$4 repeat=$5 steps=$6 out=$7
    local run_name="${system}_${temperature}K_${steps}step_${backend}_r${repeat}_${kind}"
    local reference=()
    mapfile -t reference < <(baseline_args "$system" "$temperature" "$repeat")
    ensure_gpu_idle
    start_telemetry "$run_name"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONHASHSEED="$SEED" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$REPO_ROOT/example/profile_opt3.py" \
            --backend "$backend" \
            --profile-kind "$kind" \
            --structure "$STRUCTURE_DIR/$system.cif" \
            --checkpoint "$CHECKPOINT" \
            --system "$system" \
            --temperature "$temperature" \
            --output-dir "$out" \
            --run-name "$run_name" \
            --steps "$steps" \
            --component-steps "$COMPONENT_STEPS" \
            --warmup-steps 3 \
            --probe-steps 50 \
            --repeat "$repeat" \
            --seed "$SEED" \
            "${reference[@]}" \
            > "$OUTPUT_DIR/logs/${run_name}.log" 2>&1
    local status=$?
    set -e
    stop_telemetry
    if [[ $status -eq 42 ]]; then
        record_failure "$kind" "$backend" "$system" "$temperature" \
            "$repeat" "$status"
        echo "OOM ($status): $run_name; continuing" >&2
        return 0
    fi
    if [[ $status -ne 0 && $status -ne 43 && $status -ne 45 ]]; then
        record_failure "$kind" "$backend" "$system" "$temperature" \
            "$repeat" "$status"
        echo "FAILED ($status): $run_name" >&2
        return "$status"
    fi
    echo "completed ($status): $run_name"
}

timing_order() {
    python -c 'import random,sys; values=["static-eager-breakdown","fixed-builder-model-cg","builder-cg-model-cg","force-eval-cg","whole-step-cg"]; random.Random(42+int(sys.argv[1])).shuffle(values); print(" ".join(values))' "$1"
}

run_timing_phase() {
    local systems=(Cu32 Cu192 H2O32 H2O60)
    local repeat system backend temperature
    for repeat in $(seq 1 "$REPEATS"); do
        read -r -a backends <<< "$(timing_order "$repeat")"
        for system in "${systems[@]}"; do
            for backend in "${backends[@]}"; do
                run_python_profile timing "$backend" "$system" 300 "$repeat" \
                    "$STEPS" "$OUTPUT_DIR/timing"
            done
        done
        # Cu512 is a drift control; only compare the two existing Opt3 paths.
        for temperature in 300 800; do
            if (( repeat % 2 )); then
                backends=(fixed-builder-model-cg whole-step-cg)
            else
                backends=(whole-step-cg fixed-builder-model-cg)
            fi
            for backend in "${backends[@]}"; do
                run_python_profile timing "$backend" Cu512 "$temperature" \
                    "$repeat" "$STEPS" "$OUTPUT_DIR/timing"
            done
        done
    done
}

run_smoke_phase() {
    local system backend smoke_steps
    for system in Cu32 H2O32; do
        for smoke_steps in 10 100; do
            for backend in static-eager-breakdown fixed-builder-model-cg \
                builder-cg-model-cg force-eval-cg whole-step-cg; do
                run_python_profile smoke "$backend" "$system" 300 1 \
                    "$smoke_steps" "$OUTPUT_DIR/smoke"
            done
        done
    done
}

run_adaptive_timing() {
    local requests="$OUTPUT_DIR/additional_runs.tsv"
    [[ -s "$requests" ]] || return 0
    local system temperature backend repeat
    while IFS=$'\t' read -r system temperature backend repeat; do
        [[ "$system" == "system" ]] && continue
        run_python_profile timing "$backend" "$system" "$temperature" \
            "$repeat" "$STEPS" "$OUTPUT_DIR/timing"
    done < "$requests"
}

run_torch_phase() {
    local system backend
    for system in Cu32 Cu192 H2O32 H2O60; do
        for backend in static-eager-breakdown fixed-builder-model-cg \
            builder-cg-model-cg force-eval-cg whole-step-cg; do
            run_python_profile torch-profiler "$backend" "$system" 300 1 \
                "$TRACE_STEPS" "$OUTPUT_DIR/torch"
        done
    done
}

run_nsys_one() {
    local mode=$1 backend=$2 system=$3
    local run_name="${system}_300K_${TRACE_STEPS}step_${backend}_nsys_${mode}"
    local prefix="$OUTPUT_DIR/nsys/$run_name"
    local reference=()
    mapfile -t reference < <(baseline_args "$system" 300 1)
    ensure_gpu_idle
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONHASHSEED="$SEED" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$NSYS" profile \
            --trace=cuda,nvtx,osrt \
            --sample=none \
            --cpuctxsw=none \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --cuda-graph-trace="$mode" \
            --force-overwrite=true \
            --output="$prefix" \
            python -u "$REPO_ROOT/example/profile_opt3.py" \
                --backend "$backend" \
                --profile-kind external-profiler \
                --structure "$STRUCTURE_DIR/$system.cif" \
                --checkpoint "$CHECKPOINT" \
                --system "$system" \
                --temperature 300 \
                --output-dir "$OUTPUT_DIR/nsys" \
                --run-name "$run_name" \
                --steps "$TRACE_STEPS" \
                --component-steps 0 \
                --repeat 1 \
                --seed "$SEED" \
                "${reference[@]}" \
            > "$OUTPUT_DIR/logs/${run_name}.log" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 42 ]]; then
        record_failure "nsys-$mode" "$backend" "$system" 300 1 "$status"
        echo "NSYS target OOM: $run_name; continuing" >&2
        return 0
    fi
    if [[ $status -ne 0 && $status -ne 43 && $status -ne 45 ]]; then
        record_failure "nsys-$mode" "$backend" "$system" 300 1 "$status"
        echo "NSYS failed ($status): $run_name" >&2
        return "$status"
    fi
    "$NSYS" stats \
        --report cuda_gpu_kern_sum,cuda_api_sum \
        --format csv \
        "$prefix.nsys-rep" \
        > "$prefix.stats.csv"
    "$NSYS" stats \
        --report cuda_gpu_trace \
        --format csv \
        "$prefix.nsys-rep" \
        > "$prefix.gpu_trace.csv"
}

run_nsys_phase() {
    if [[ ! -x "$NSYS" ]]; then
        echo "nsys not executable: $NSYS" >&2
        exit 2
    fi
    local system backend mode
    for system in Cu32 Cu192 H2O32 H2O60; do
        for backend in fixed-builder-model-cg builder-cg-model-cg \
            force-eval-cg whole-step-cg; do
            for mode in graph node; do
                run_nsys_one "$mode" "$backend" "$system"
            done
        done
    done
}

run_ncu_one() {
    local graph_mode=$1 backend=$2 system=$3 kernel_regex=${4:-} hot_rank=${5:-}
    local suffix=$graph_mode
    [[ -n "$kernel_regex" ]] && suffix="${graph_mode}_kernel_${hot_rank}"
    local run_name="${system}_300K_1step_${backend}_ncu_${suffix}"
    local output="$OUTPUT_DIR/ncu/$run_name"
    local kernel_args=()
    [[ -n "$kernel_regex" ]] && kernel_args=(--kernel-name "regex:${kernel_regex}")
    ensure_gpu_idle
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONHASHSEED="$SEED" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$NCU" \
            --target-processes all \
            --profile-from-start off \
            --graph-profiling "$graph_mode" \
            --replay-mode kernel \
            --clock-control none \
            --cache-control none \
            --section SpeedOfLight \
            --section LaunchStats \
            --section Occupancy \
            --section MemoryWorkloadAnalysis \
            --section SchedulerStats \
            --launch-count 1 \
            --force-overwrite \
            "${kernel_args[@]}" \
            --export "$output" \
            python -u "$REPO_ROOT/example/profile_opt3.py" \
                --backend "$backend" \
                --profile-kind external-profiler \
                --structure "$STRUCTURE_DIR/$system.cif" \
                --checkpoint "$CHECKPOINT" \
                --system "$system" \
                --temperature 300 \
                --output-dir "$OUTPUT_DIR/ncu" \
                --run-name "$run_name" \
                --steps 1 \
                --component-steps 0 \
                --repeat 1 \
                --seed "$SEED" \
            > "$OUTPUT_DIR/logs/${run_name}.log" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 42 ]]; then
        record_failure "ncu-$graph_mode" "$backend" "$system" 300 1 "$status"
        echo "NCU target OOM: $run_name; continuing" >&2
        return 0
    fi
    if [[ $status -ne 0 && $status -ne 43 && $status -ne 45 ]]; then
        record_failure "ncu-$graph_mode" "$backend" "$system" 300 1 "$status"
        echo "NCU failed ($status): $run_name" >&2
        return "$status"
    fi
    "$NCU" --import "$output.ncu-rep" --page raw --csv \
        > "$output.csv"
}

run_ncu_phase() {
    if [[ ! -x "$NCU" ]]; then
        echo "ncu not executable: $NCU" >&2
        exit 2
    fi
    local system backend
    for system in Cu192 H2O60; do
        for backend in force-eval-cg whole-step-cg; do
            run_ncu_one graph "$backend" "$system"
        done
    done
    local filters="$OUTPUT_DIR/hot_kernel_filters.tsv"
    if [[ -f "$filters" ]]; then
        while IFS=$'\t' read -r system backend regex rank _delta _kernel; do
            [[ "$system" == "system" || -z "$regex" ]] && continue
            run_ncu_one node "$backend" "$system" "$regex" "$rank"
        done < "$filters"
    else
        echo "No $filters; run the analyzer after NSYS before node-level NCU." >&2
    fi
}

run_analyzer() {
    python "$REPO_ROOT/example/analyze_opt3_profiling.py" \
        --input-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR"
}

case "$PHASE" in
    smoke) run_smoke_phase; run_analyzer ;;
    timing)
        run_timing_phase
        run_analyzer
        run_adaptive_timing
        run_analyzer
        ;;
    torch) run_torch_phase; run_analyzer ;;
    nsys) run_nsys_phase; run_analyzer ;;
    ncu) run_ncu_phase; run_analyzer ;;
    all)
        run_smoke_phase
        run_timing_phase
        run_analyzer
        run_adaptive_timing
        run_analyzer
        run_torch_phase
        run_nsys_phase
        run_analyzer
        run_ncu_phase
        run_analyzer
        ;;
esac

echo "finished_at=$(date --iso-8601=seconds)" >> "$METADATA_FILE"
echo "Profiling results: $OUTPUT_DIR"
