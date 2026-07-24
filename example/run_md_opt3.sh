#!/usr/bin/env bash
set -uo pipefail

# Run one Opt3 backend over the formal eSEN MD matrix.  This script does not
# start, stop, or configure NVIDIA MPS; it only selects CUDA_VISIBLE_DEVICES.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BACKEND=${BACKEND:-whole-step-cg}
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
RUN_ID=${RUN_ID:-"esen_opt3_${BACKEND}_$(date '+%Y%m%d_%H%M%S')"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
GPU=${GPU:-6}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
SEED=${SEED:-42}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
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
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}

if [[ "$BACKEND" != "fixed-builder-model-cg" && "$BACKEND" != "whole-step-cg" ]]; then
    echo "BACKEND must be fixed-builder-model-cg or whole-step-cg" >&2
    exit 2
fi
if [[ "$SEED" != "42" ]]; then
    echo "All benchmark RNG seeds are fixed at 42" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

read -r -a systems <<< "$SYSTEMS"
read -r -a temperatures <<< "$TEMPERATURES"
for system in "${systems[@]}"; do
    if [[ ! -f "$STRUCTURE_DIR/$system.cif" ]]; then
        echo "Structure not found: $STRUCTURE_DIR/$system.cif" >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR"
STATUS_TSV="$OUTPUT_DIR/run_status.tsv"
printf 'system\ttemperature_K\trepeat\trun_name\tstatus\texit_code\tprocess_wall_time_s\tbaseline_result\n' > "$STATUS_TSV"

{
    echo "run_id=$RUN_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "checkpoint=$CHECKPOINT"
    echo "structure_dir=$STRUCTURE_DIR"
    echo "physical_gpu=$GPU"
    echo "backend=$BACKEND"
    echo "systems=$SYSTEMS"
    echo "temperatures=$TEMPERATURES"
    echo "steps=$STEPS"
    echo "warmup_steps=$WARMUP_STEPS"
    echo "repeats=$REPEATS"
    echo "seed=$SEED"
    echo "baseline_dir=$BASELINE_DIR"
    echo "probe_steps=$PROBE_STEPS"
    echo "neighbor_margin=$NEIGHBOR_MARGIN"
    echo "neighbor_slot_step=$NEIGHBOR_SLOT_STEP"
    echo "dummy_atoms=$DUMMY_ATOMS"
    echo "capture_warmup=$CAPTURE_WARMUP"
    echo "max_neighbors=$MAX_NEIGHBORS"
    echo "degeneracy_tolerance=$DEGENERACY_TOLERANCE"
    echo "energy_per_atom_atol=$ENERGY_PER_ATOM_ATOL"
    echo "force_max_atol=$FORCE_MAX_ATOL"
    echo "pythonhashseed=$SEED"
    echo "cublas_workspace_config=:4096:8"
    echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        python -c 'import torch; print(f"python_torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
    nvidia-smi -i "$GPU" --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader
} > "$OUTPUT_DIR/run_metadata.txt"

success_count=0
oom_count=0
validation_failed_count=0
capacity_overflow_count=0
missing_reference_count=0
error_count=0

for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    for temperature in "${temperatures[@]}"; do
        for repeat in $(seq 1 "$REPEATS"); do
            suffix=${BACKEND//-/_}
            run_name="${system}_${temperature}K_${STEPS}step_esen_${suffix}_r${repeat}"
            log_path="$OUTPUT_DIR/${run_name}.log"
            baseline_run_name="${system}_${temperature}K_${BASELINE_STEPS}step_esen_baseline_r${repeat}"
            baseline_result=""
            reference_args=()
            if [[ -n "$BASELINE_DIR" ]]; then
                baseline_result="$BASELINE_DIR/${baseline_run_name}.json"
                if [[ -f "$baseline_result" ]]; then
                    reference_args+=(--baseline-result "$baseline_result")
                else
                    reference_args+=(--missing-baseline-reference)
                    ((missing_reference_count += 1))
                fi
            fi

            echo "Running $run_name on physical GPU $GPU"
            start_ns=$(date +%s%N)
            set +e
            CUDA_VISIBLE_DEVICES="$GPU" \
            PYTHONHASHSEED="$SEED" \
            CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
                python -u "$REPO_ROOT/example/benchmark_md_opt3.py" \
                    --backend "$BACKEND" \
                    --structure "$structure" \
                    --checkpoint "$CHECKPOINT" \
                    --system "$system" \
                    --output-dir "$OUTPUT_DIR" \
                    --run-name "$run_name" \
                    --steps "$STEPS" \
                    --warmup-steps "$WARMUP_STEPS" \
                    --temperature "$temperature" \
                    --timestep 1.0 \
                    --taut 100.0 \
                    --seed "$SEED" \
                    --repeat "$repeat" \
                    --probe-steps "$PROBE_STEPS" \
                    --neighbor-margin "$NEIGHBOR_MARGIN" \
                    --neighbor-slot-step "$NEIGHBOR_SLOT_STEP" \
                    --dummy-atoms "$DUMMY_ATOMS" \
                    --capture-warmup "$CAPTURE_WARMUP" \
                    --max-neighbors "$MAX_NEIGHBORS" \
                    --degeneracy-tolerance "$DEGENERACY_TOLERANCE" \
                    --energy-per-atom-atol "$ENERGY_PER_ATOM_ATOL" \
                    --force-max-atol "$FORCE_MAX_ATOL" \
                    "${reference_args[@]}" \
                2>&1 | tee "$log_path"
            exit_code=${PIPESTATUS[0]}
            set -e
            end_ns=$(date +%s%N)
            process_wall_time=$(awk -v start="$start_ns" -v end="$end_ns" \
                'BEGIN { printf "%.6f", (end - start) / 1000000000 }')

            if [[ $exit_code -eq 0 && -f "$OUTPUT_DIR/${run_name}.json" ]]; then
                status=success
                ((success_count += 1))
            elif [[ $exit_code -eq 42 ]] || grep -Eqi \
                'BENCHMARK_STATUS=oom|CUDA out of memory|OutOfMemoryError' "$log_path"; then
                status=oom
                ((oom_count += 1))
            elif [[ $exit_code -eq 45 ]] || grep -Eqi \
                'BENCHMARK_STATUS=capacity_overflow' "$log_path"; then
                status=capacity_overflow
                ((capacity_overflow_count += 1))
            elif [[ $exit_code -eq 43 ]] || grep -Eqi \
                'BENCHMARK_STATUS=validation_failed' "$log_path"; then
                status=validation_failed
                ((validation_failed_count += 1))
            else
                status=error
                ((error_count += 1))
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$system" "$temperature" "$repeat" "$run_name" "$status" \
                "$exit_code" "$process_wall_time" "$baseline_result" \
                >> "$STATUS_TSV"
            echo "$status: $run_name"
        done
    done
done

if [[ "$BACKEND" == "whole-step-cg" ]]; then
    record_backend=esen_gpu_resident_whole_step_cg
    report_prefix=whole_step_cg_report
else
    record_backend=esen_gpu_resident_fixed_builder_model_cg
    report_prefix=fixed_builder_model_cg_report
fi
python "$REPO_ROOT/example/summarize_md_baselines.py" \
    --input-dir "$OUTPUT_DIR" \
    --backend "$record_backend" \
    --report-prefix "$report_prefix"

{
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "successful_runs=$success_count"
    echo "oom_runs=$oom_count"
    echo "validation_failed_runs=$validation_failed_count"
    echo "capacity_overflow_runs=$capacity_overflow_count"
    echo "missing_reference_runs=$missing_reference_count"
    echo "error_runs=$error_count"
} >> "$OUTPUT_DIR/run_metadata.txt"

echo "Results: $OUTPUT_DIR"
echo "success=$success_count oom=$oom_count validation_failed=$validation_failed_count capacity_overflow=$capacity_overflow_count missing_reference=$missing_reference_count error=$error_count"
if [[ "$STRICT" == "1" && $((oom_count + validation_failed_count + capacity_overflow_count + error_count)) -ne 0 ]]; then
    exit 1
fi
