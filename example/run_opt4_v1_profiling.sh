#!/usr/bin/env bash
set -euo pipefail

# Nsight Systems confirmation profiling for Opt4 v1.  The benchmark entry
# points call cudaProfilerStart/Stop only around the production MD region, so
# model loading, Triton compilation, probing, graph capture, and warmup are
# excluded from every trace.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}
SCOPES=${SCOPES:-"whole-step"}
SYSTEMS=${SYSTEMS:-"Cu512 H2O192"}
TEMPERATURE=${TEMPERATURE:-300}
TRACE_STEPS=${TRACE_STEPS:-20}
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
RUN_ID=${RUN_ID:-"opt4_v1_nsys_gpu${GPU}_$(date '+%Y%m%d_%H%M%S')"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
PYTHON=${PYTHON:-python}
RESUME=${RESUME:-1}
PROFILE_LABEL=${PROFILE_LABEL:-opt4_v1}
FROZEN_CONFIG_TAG=${FROZEN_CONFIG_TAG:-}
MODEL_BASE_STAGE=${MODEL_BASE_STAGE:-OPT2}
MODEL_BASE_FUSIONS=${MODEL_BASE_FUSIONS:-}
MODEL_CANDIDATE_STAGE=${MODEL_CANDIDATE_STAGE:-OPT4V1}
MODEL_CANDIDATE_FUSIONS=${MODEL_CANDIDATE_FUSIONS:-so2-epilogue}
WHOLE_BASE_STAGE=${WHOLE_BASE_STAGE:-OPT3}
WHOLE_BASE_FUSIONS=${WHOLE_BASE_FUSIONS:-}
WHOLE_CANDIDATE_STAGE=${WHOLE_CANDIDATE_STAGE:-OPT4V1}
WHOLE_CANDIDATE_FUSIONS=${WHOLE_CANDIDATE_FUSIONS:-rmsnorm,so2-epilogue}
WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY=${WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY:-uniform}
WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY=${WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY:-uniform}
NEIGHBOR_AUTO_MIN_REDUCTION=${NEIGHBOR_AUTO_MIN_REDUCTION:-0.05}
NEIGHBOR_AUTO_GUARD_SLOTS=${NEIGHBOR_AUTO_GUARD_SLOTS:-1}
WHOLE_PROBE_STEPS=${WHOLE_PROBE_STEPS:-50}

if [[ ! -x "$NSYS" ]]; then
    echo "Nsight Systems is not executable: $NSYS" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ "$TRACE_STEPS" -lt 1 ]]; then
    echo "TRACE_STEPS must be positive" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"/{logs,reports,results,sqlite}
STATUS_FILE="$OUTPUT_DIR/profile_status.tsv"
if [[ ! -s "$STATUS_FILE" ]]; then
    printf 'scope\tvariant\tsystem\ttemperature_K\tmode\tstatus\texit_code\treport\n' \
        > "$STATUS_FILE"
fi

GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid \
    --format=csv,noheader | tr -d '[:space:]')

ensure_gpu_idle() {
    local active
    active=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
        --format=csv,noheader 2>/dev/null | awk -F', ' -v uuid="$GPU_UUID" \
        '$1 == uuid {print $0}')
    if [[ -n "$active" ]]; then
        echo "GPU $GPU ($GPU_UUID) has active compute processes:" >&2
        echo "$active" >&2
        return 1
    fi
}

record_status() {
    local scope=$1 variant=$2 system=$3 mode=$4 status=$5 code=$6 report=$7
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$scope" "$variant" "$system" "$TEMPERATURE" "$mode" \
        "$status" "$code" "$report" >> "$STATUS_FILE"
}

run_one() {
    local scope=$1 variant=$2 system=$3 mode=$4
    local stage backend script scope_label run_name prefix log_path fusion_value
    local capacity_policy=uniform
    local -a target_args fusion_args

    if [[ ! -f "$STRUCTURE_DIR/$system.cif" ]]; then
        echo "Structure not found: $STRUCTURE_DIR/$system.cif" >&2
        exit 2
    fi

    fusion_args=()
    if [[ "$scope" == "whole-step" ]]; then
        script="$REPO_ROOT/example/benchmark_md_opt4.py"
        scope_label=whole_step
        if [[ "$variant" == "base" ]]; then
            stage=$WHOLE_BASE_STAGE
            fusion_value=$WHOLE_BASE_FUSIONS
            capacity_policy=$WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY
        else
            stage=$WHOLE_CANDIDATE_STAGE
            fusion_value=$WHOLE_CANDIDATE_FUSIONS
            capacity_policy=$WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY
        fi
        if [[ -n "$fusion_value" ]]; then
            backend=whole-step-cg-opt4
            fusion_args=(
                --model-fusions "$fusion_value"
                --fusion-stage "$stage"
            )
        else
            backend=whole-step-cg
        fi
        target_args=(
            --backend "$backend"
            --structure "$STRUCTURE_DIR/$system.cif"
            --checkpoint "$CHECKPOINT"
            --system "$system"
            --steps "$TRACE_STEPS"
            --warmup-steps 3
            --temperature "$TEMPERATURE"
            --timestep 1.0
            --taut 100.0
            --seed 42
            --repeat 1
            --probe-steps "$WHOLE_PROBE_STEPS"
            --neighbor-margin 0.10
            --neighbor-slot-step 8
            --neighbor-capacity-policy "$capacity_policy"
            --neighbor-auto-min-reduction "$NEIGHBOR_AUTO_MIN_REDUCTION"
            --neighbor-auto-guard-slots "$NEIGHBOR_AUTO_GUARD_SLOTS"
            --dummy-atoms 32
            --capture-warmup 3
            --max-neighbors 300
            --degeneracy-tolerance 0.01
            --replay-energy-atol 0.0
            --replay-force-atol 2e-4
            --energy-per-atom-atol 1e-5
            --force-max-atol 2e-4
            --missing-baseline-reference
            --external-profiler
        )
    elif [[ "$scope" == "model-only" ]]; then
        script="$REPO_ROOT/example/benchmark_md_gpu.py"
        scope_label=model_only
        if [[ "$variant" == "base" ]]; then
            stage=$MODEL_BASE_STAGE
            fusion_value=$MODEL_BASE_FUSIONS
        else
            stage=$MODEL_CANDIDATE_STAGE
            fusion_value=$MODEL_CANDIDATE_FUSIONS
        fi
        if [[ -n "$fusion_value" ]]; then
            backend=model-cg-opt4
            fusion_args=(
                --model-fusions "$fusion_value"
                --fusion-stage "$stage"
            )
        else
            backend=model-cg
        fi
        target_args=(
            --backend "$backend"
            --structure "$STRUCTURE_DIR/$system.cif"
            --checkpoint "$CHECKPOINT"
            --system "$system"
            --steps "$TRACE_STEPS"
            --warmup-steps 3
            --temperature "$TEMPERATURE"
            --timestep 1.0
            --taut 100.0
            --seed 42
            --repeat 1
            --md-dtype float64
            --cg-probe-steps 50
            --cg-capacity-margin 0.10
            --cg-edge-step 256
            --cg-dummy-atoms 32
            --cg-capture-warmup 3
            --cg-replay-energy-atol 0.0
            --cg-replay-force-atol 2e-4
            --energy-per-atom-atol 1e-5
            --force-max-atol 2e-4
            --missing-baseline-reference
            --external-profiler
        )
    else
        echo "Unsupported scope: $scope (use model-only or whole-step)" >&2
        exit 2
    fi

    run_name="${system}_${TEMPERATURE}K_${TRACE_STEPS}step_${scope_label}_${stage}_nsys_${mode}"
    prefix="$OUTPUT_DIR/reports/$run_name"
    log_path="$OUTPUT_DIR/logs/$run_name.log"
    target_args+=(
        --output-dir "$OUTPUT_DIR/results"
        --run-name "$run_name"
    )
    target_args+=("${fusion_args[@]}")

    if [[ "$RESUME" == 1 && -s "$prefix.nsys-rep" && \
          -s "$prefix.stats.csv" && -s "$prefix.gpu_trace.csv" ]]; then
        echo "complete trace exists; skipping: $run_name"
        return 0
    fi

    ensure_gpu_idle
    echo "profiling: $run_name"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONHASHSEED=42 \
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
            "$PYTHON" -u "$script" "${target_args[@]}" \
            > "$log_path" 2>&1
    local status=$?
    set -e

    if [[ $status -ne 0 || ! -s "$prefix.nsys-rep" ]]; then
        record_status "$scope" "$variant" "$system" "$mode" failed \
            "$status" "$prefix.nsys-rep"
        echo "failed ($status): $run_name; see $log_path" >&2
        return 1
    fi

    "$NSYS" stats \
        --report cuda_gpu_kern_sum,cuda_api_sum \
        --format csv \
        "$prefix.nsys-rep" > "$prefix.stats.csv"
    "$NSYS" stats \
        --report cuda_gpu_trace \
        --format csv \
        "$prefix.nsys-rep" > "$prefix.gpu_trace.csv"
    "$NSYS" export \
        --type sqlite \
        --force-overwrite=true \
        --output "$OUTPUT_DIR/sqlite/$run_name.sqlite" \
        "$prefix.nsys-rep" > "$prefix.export.log" 2>&1

    record_status "$scope" "$variant" "$system" "$mode" success 0 \
        "$prefix.nsys-rep"
    echo "completed: $run_name"
}

{
    echo "run_id=$RUN_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "gpu=$GPU"
    echo "gpu_uuid=$GPU_UUID"
    echo "scopes=$SCOPES"
    echo "systems=$SYSTEMS"
    echo "temperature=$TEMPERATURE"
    echo "trace_steps=$TRACE_STEPS"
    echo "profile_label=$PROFILE_LABEL"
    echo "frozen_config_tag=$FROZEN_CONFIG_TAG"
    echo "model_base_stage=$MODEL_BASE_STAGE"
    echo "model_base_fusions=$MODEL_BASE_FUSIONS"
    echo "model_candidate_stage=$MODEL_CANDIDATE_STAGE"
    echo "model_candidate_fusions=$MODEL_CANDIDATE_FUSIONS"
    echo "whole_base_stage=$WHOLE_BASE_STAGE"
    echo "whole_base_fusions=$WHOLE_BASE_FUSIONS"
    echo "whole_candidate_stage=$WHOLE_CANDIDATE_STAGE"
    echo "whole_candidate_fusions=$WHOLE_CANDIDATE_FUSIONS"
    echo "whole_base_neighbor_capacity_policy=$WHOLE_BASE_NEIGHBOR_CAPACITY_POLICY"
    echo "whole_candidate_neighbor_capacity_policy=$WHOLE_CANDIDATE_NEIGHBOR_CAPACITY_POLICY"
    echo "neighbor_auto_min_reduction=$NEIGHBOR_AUTO_MIN_REDUCTION"
    echo "neighbor_auto_guard_slots=$NEIGHBOR_AUTO_GUARD_SLOTS"
    echo "whole_probe_steps=$WHOLE_PROBE_STEPS"
    echo "checkpoint=$CHECKPOINT"
    echo "structure_dir=$STRUCTURE_DIR"
    "$NSYS" --version | tr '\n' ' '
    echo
    nvidia-smi -i "$GPU" \
        --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader
} > "$OUTPUT_DIR/run_metadata.txt"

for scope in $SCOPES; do
    for system in $SYSTEMS; do
        for variant in base candidate; do
            for mode in graph node; do
                run_one "$scope" "$variant" "$system" "$mode"
            done
        done
    done
done

echo "$PROFILE_LABEL NSYS profiling complete: $OUTPUT_DIR"
echo "Status: $STATUS_FILE"
