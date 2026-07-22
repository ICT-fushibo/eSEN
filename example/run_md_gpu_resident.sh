#!/usr/bin/env bash
set -uo pipefail

# Eager GPU-resident eSEN benchmark.  There is intentionally no CUDA Graph,
# torch.compile, AMP, TF32, or custom fusion in this runner.
#
# Examples:
#   GPU=6 STEPS=10 REPEATS=1 SYSTEMS="Cu32 H2O32" TEMPERATURES="300" \
#     VALIDATE_OFFICIAL=1 REQUIRE_BASELINE_REFERENCE=0 \
#     bash example/run_md_gpu_resident.sh
#
#   GPU=6 STEPS=1000 WARMUP_STEPS=3 REPEATS=3 \
#     BASELINE_DIR=/path/to/esen_baseline_output \
#     bash example/run_md_gpu_resident.sh

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
RUN_ID=${RUN_ID:-"esen_gpu_eager_$(date '+%Y%m%d_%H%M%S')"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
GPU=${GPU:-6}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
MD_DTYPE=${MD_DTYPE:-float64}
VALIDATE_OFFICIAL=${VALIDATE_OFFICIAL:-0}
STRICT=${STRICT:-0}
SEED=${SEED:-42}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
REQUIRE_BASELINE_REFERENCE=${REQUIRE_BASELINE_REFERENCE:-1}
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}

read -r -a systems <<< "$SYSTEMS"
read -r -a temperatures <<< "$TEMPERATURES"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ "$SEED" != "42" ]]; then
    echo "All benchmark RNG seeds are fixed at 42; got SEED=$SEED" >&2
    exit 2
fi
if [[ "$REQUIRE_BASELINE_REFERENCE" == "1" && -z "$BASELINE_DIR" ]]; then
    echo "BASELINE_DIR is required for optimized MD energy validation" >&2
    exit 2
fi
for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    if [[ ! -f "$structure" ]]; then
        echo "Structure not found: $structure" >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR"
STATUS_TSV="$OUTPUT_DIR/run_status.tsv"
printf 'system\ttemperature_K\trepeat\trun_name\tstatus\texit_code\tprocess_wall_time_s\tbaseline_result\n' > "$STATUS_TSV"
STRUCTURE_HASH_TSV="$OUTPUT_DIR/structure_sha256.tsv"
printf 'system\tstructure\tsha256\n' > "$STRUCTURE_HASH_TSV"
for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    structure_sha=$(sha256sum "$structure" | awk '{print $1}')
    printf '%s\t%s\t%s\n' "$system" "$structure" "$structure_sha" \
        >> "$STRUCTURE_HASH_TSV"
done

{
    echo "run_id=$RUN_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "checkpoint=$CHECKPOINT"
    echo "structure_dir=$STRUCTURE_DIR"
    echo "physical_gpu=$GPU"
    echo "systems=$SYSTEMS"
    echo "temperatures=$TEMPERATURES"
    echo "steps=$STEPS"
    echo "warmup_steps=$WARMUP_STEPS"
    echo "repeats=$REPEATS"
    echo "seed=$SEED"
    echo "md_dtype=$MD_DTYPE"
    echo "validate_official=$VALIDATE_OFFICIAL"
    echo "baseline_dir=$BASELINE_DIR"
    echo "baseline_steps=$BASELINE_STEPS"
    echo "require_baseline_reference=$REQUIRE_BASELINE_REFERENCE"
    echo "pythonhashseed=$SEED"
    echo "cublas_workspace_config=:4096:8"
    echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        python -c 'import torch; print(f"python_torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
    nvidia-smi -i "$GPU" --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader
} > "$OUTPUT_DIR/run_metadata.txt"

validation_args=()
if [[ "$VALIDATE_OFFICIAL" == "1" ]]; then
    validation_args+=(--validate-official)
fi

success_count=0
oom_count=0
error_count=0
validation_failed_count=0
missing_reference_count=0

for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    for temperature in "${temperatures[@]}"; do
        for repeat in $(seq 1 "$REPEATS"); do
            run_name="${system}_${temperature}K_${STEPS}step_esen_gpu_eager_r${repeat}"
            log_path="$OUTPUT_DIR/${run_name}.log"
            baseline_run_name="${system}_${temperature}K_${BASELINE_STEPS}step_esen_baseline_r${repeat}"
            baseline_result=""
            reference_args=()
            if [[ -n "$BASELINE_DIR" ]]; then
                baseline_result="$BASELINE_DIR/${baseline_run_name}.json"
                if [[ -f "$baseline_result" ]]; then
                    reference_args+=(--baseline-result "$baseline_result")
                elif [[ "$REQUIRE_BASELINE_REFERENCE" == "1" ]]; then
                    echo "MISSING BASELINE: $baseline_result" >&2
                    ((missing_reference_count += 1))
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$system" "$temperature" "$repeat" "$run_name" \
                        "missing_reference" "44" "0.000000" "$baseline_result" \
                        >> "$STATUS_TSV"
                    continue
                fi
            fi
            echo "Running $run_name on physical GPU $GPU"
            start_ns=$(date +%s%N)

            set +e
            CUDA_VISIBLE_DEVICES="$GPU" \
            PYTHONHASHSEED="$SEED" \
            CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
                python -u "$REPO_ROOT/example/benchmark_md_gpu.py" \
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
                    --md-dtype "$MD_DTYPE" \
                    "${reference_args[@]}" \
                    "${validation_args[@]}" \
                2>&1 | tee "$log_path"
            exit_code=${PIPESTATUS[0]}
            set -e

            end_ns=$(date +%s%N)
            process_wall_time=$(awk -v start="$start_ns" -v end="$end_ns" \
                'BEGIN { printf "%.6f", (end - start) / 1000000000 }')

            result_json="$OUTPUT_DIR/${run_name}.json"
            if [[ $exit_code -eq 0 && -f "$result_json" ]]; then
                status=success
                ((success_count += 1))
            elif [[ $exit_code -eq 42 ]] || grep -Eqi \
                'BENCHMARK_STATUS=oom|CUDA out of memory|CUDA error: out of memory|OutOfMemoryError' \
                "$log_path"; then
                status=oom
                ((oom_count += 1))
                echo "OOM: $run_name; continuing" >&2
            elif [[ $exit_code -eq 43 ]] || grep -Eqi \
                'BENCHMARK_STATUS=validation_failed' "$log_path"; then
                status=validation_failed
                ((validation_failed_count += 1))
                echo "ENERGY VALIDATION FAILED: $run_name; continuing" >&2
            else
                status=error
                ((error_count += 1))
                echo "ERROR: $run_name (exit code $exit_code); continuing" >&2
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$system" "$temperature" "$repeat" "$run_name" "$status" \
                "$exit_code" "$process_wall_time" "$baseline_result" \
                >> "$STATUS_TSV"
        done
    done
done

python "$REPO_ROOT/example/summarize_md_baselines.py" \
    --input-dir "$OUTPUT_DIR" \
    --backend esen_gpu_resident_eager \
    --report-prefix gpu_resident_report

{
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "successful_runs=$success_count"
    echo "oom_runs=$oom_count"
    echo "validation_failed_runs=$validation_failed_count"
    echo "missing_reference_runs=$missing_reference_count"
    echo "error_runs=$error_count"
} >> "$OUTPUT_DIR/run_metadata.txt"

echo "Results: $OUTPUT_DIR"
echo "success=$success_count oom=$oom_count validation_failed=$validation_failed_count missing_reference=$missing_reference_count error=$error_count"

if [[ "$STRICT" == "1" && $((oom_count + validation_failed_count + missing_reference_count + error_count)) -ne 0 ]]; then
    exit 1
fi
