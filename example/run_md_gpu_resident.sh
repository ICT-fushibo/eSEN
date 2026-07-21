#!/usr/bin/env bash
set -uo pipefail

# Eager GPU-resident eSEN benchmark.  There is intentionally no CUDA Graph,
# torch.compile, AMP, TF32, or custom fusion in this runner.
#
# Examples:
#   GPU=6 STEPS=10 REPEATS=1 SYSTEMS="Cu32 H2O32" TEMPERATURES="300" \
#     VALIDATE_OFFICIAL=1 bash example/run_md_gpu_resident.sh
#
#   GPU=6 STEPS=1000 WARMUP_STEPS=3 REPEATS=3 \
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
SYSTEMS=${SYSTEMS:-"Cu32 Cu64 Cu192 Cu512 Cu1024 H2O32 H2O60 H2O192 H2O512 H2O1024"}
TEMPERATURES=${TEMPERATURES:-"300 800"}

read -r -a systems <<< "$SYSTEMS"
read -r -a temperatures <<< "$TEMPERATURES"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
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
printf 'system\ttemperature_K\trepeat\trun_name\tstatus\texit_code\tprocess_wall_time_s\n' > "$STATUS_TSV"

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
    echo "md_dtype=$MD_DTYPE"
    echo "validate_official=$VALIDATE_OFFICIAL"
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

for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    for temperature in "${temperatures[@]}"; do
        for repeat in $(seq 1 "$REPEATS"); do
            run_name="${system}_${temperature}K_${STEPS}step_esen_gpu_eager_r${repeat}"
            log_path="$OUTPUT_DIR/${run_name}.log"
            echo "Running $run_name on physical GPU $GPU"
            start_ns=$(date +%s%N)

            set +e
            CUDA_VISIBLE_DEVICES="$GPU" \
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
                    --seed 42 \
                    --md-dtype "$MD_DTYPE" \
                    "${validation_args[@]}" \
                2>&1 | tee "$log_path"
            exit_code=${PIPESTATUS[0]}
            set -e

            end_ns=$(date +%s%N)
            process_wall_time=$(awk -v start="$start_ns" -v end="$end_ns" \
                'BEGIN { printf "%.6f", (end - start) / 1000000000 }')

            if [[ $exit_code -eq 0 ]]; then
                status=success
                ((success_count += 1))
            elif [[ $exit_code -eq 42 ]] || grep -Eqi \
                'BENCHMARK_STATUS=oom|CUDA out of memory|CUDA error: out of memory|OutOfMemoryError' \
                "$log_path"; then
                status=oom
                ((oom_count += 1))
                echo "OOM: $run_name; continuing" >&2
            else
                status=error
                ((error_count += 1))
                echo "ERROR: $run_name (exit code $exit_code); continuing" >&2
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$system" "$temperature" "$repeat" "$run_name" "$status" \
                "$exit_code" "$process_wall_time" >> "$STATUS_TSV"
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
    echo "error_runs=$error_count"
} >> "$OUTPUT_DIR/run_metadata.txt"

echo "Results: $OUTPUT_DIR"
echo "success=$success_count oom=$oom_count error=$error_count"

if [[ "$STRICT" == "1" && $((oom_count + error_count)) -ne 0 ]]; then
    exit 1
fi
