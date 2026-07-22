#!/usr/bin/env bash
set -euo pipefail

# Optional:
#   CHECKPOINT=/path/to/esen_30m_oam.pt STRUCTURE_DIR=/path/to/cif_file
#   GPU=6 STEPS=1000 WARMUP_STEPS=3 REPEATS=3

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
RUN_ID=${RUN_ID:-"esen_baseline_$(date '+%Y%m%d_%H%M%S')"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out/$RUN_ID"}
GPU=${GPU:-6}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}
STRICT=${STRICT:-1}
SEED=${SEED:-42}
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

mkdir -p "$OUTPUT_DIR"

for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    if [[ ! -f "$structure" ]]; then
        echo "Structure not found: $structure" >&2
        echo "Generate all requested CIF files before running this benchmark." >&2
        exit 2
    fi
done

STATUS_TSV="$OUTPUT_DIR/run_status.tsv"
printf 'system\ttemperature_K\trepeat\trun_name\tstatus\texit_code\tprocess_wall_time_s\n' > "$STATUS_TSV"
STRUCTURE_HASH_TSV="$OUTPUT_DIR/structure_sha256.tsv"
printf 'system\tstructure\tsha256\n' > "$STRUCTURE_HASH_TSV"
for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    structure_sha=$(sha256sum "$structure" | awk '{print $1}')
    printf '%s\t%s\t%s\n' "$system" "$structure" "$structure_sha" \
        >> "$STRUCTURE_HASH_TSV"
done
failure_count=0
oom_count=0
error_count=0

{
    echo "run_id=$RUN_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "checkpoint=$CHECKPOINT"
    echo "structure_dir=$STRUCTURE_DIR"
    echo "physical_gpu=$GPU"
    echo "steps=$STEPS"
    echo "warmup_steps=$WARMUP_STEPS"
    echo "repeats=$REPEATS"
    echo "seed=$SEED"
    echo "systems=$SYSTEMS"
    echo "temperatures=$TEMPERATURES"
    echo "pythonhashseed=$SEED"
    echo "cublas_workspace_config=:4096:8"
    echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    python -c 'import torch; print(f"python_torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
    nvidia-smi -i "$GPU" --query-gpu=index,name,uuid,driver_version --format=csv,noheader
} > "$OUTPUT_DIR/run_metadata.txt"

for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    for temperature in "${temperatures[@]}"; do
        for repeat in $(seq 1 "$REPEATS"); do
            run_name="${system}_${temperature}K_${STEPS}step_esen_baseline_r${repeat}"
            echo "Running $run_name on physical GPU $GPU"
            start_ns=$(date +%s%N)
            set +e
            CUDA_VISIBLE_DEVICES="$GPU" \
            PYTHONHASHSEED="$SEED" \
            CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
                python -u "$REPO_ROOT/example/benchmark_md.py" \
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
                    --outputs energy forces \
                2>&1 | tee "$OUTPUT_DIR/${run_name}.log"
            exit_code=${PIPESTATUS[0]}
            set -e
            end_ns=$(date +%s%N)
            process_wall_time=$(awk -v start="$start_ns" -v end="$end_ns" \
                'BEGIN { printf "%.6f", (end - start) / 1000000000 }')

            result_json="$OUTPUT_DIR/${run_name}.json"
            if [[ $exit_code -eq 0 && -f "$result_json" ]]; then
                status=success
            elif [[ $exit_code -eq 42 ]] || grep -Eqi \
                'BENCHMARK_STATUS=oom|CUDA out of memory|CUDA error: out of memory|OutOfMemoryError' \
                "$OUTPUT_DIR/${run_name}.log"; then
                status=oom
                ((failure_count += 1))
                ((oom_count += 1))
                echo "OOM: $run_name; continuing" >&2
            else
                status=error
                ((failure_count += 1))
                ((error_count += 1))
                echo "FAILED: $run_name (exit code $exit_code); continuing" >&2
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$system" "$temperature" "$repeat" "$run_name" "$status" \
                "$exit_code" "$process_wall_time" >> "$STATUS_TSV"
        done
    done
done

python "$REPO_ROOT/example/summarize_md_baselines.py" --input-dir "$OUTPUT_DIR"
echo "finished_at=$(date --iso-8601=seconds)" >> "$OUTPUT_DIR/run_metadata.txt"
echo "failed_runs=$failure_count" >> "$OUTPUT_DIR/run_metadata.txt"
echo "oom_runs=$oom_count" >> "$OUTPUT_DIR/run_metadata.txt"
echo "error_runs=$error_count" >> "$OUTPUT_DIR/run_metadata.txt"
echo "Results: $OUTPUT_DIR"

if [[ "$STRICT" == "1" && $failure_count -ne 0 ]]; then
    echo "$failure_count run(s) failed; see run_status.tsv and per-run logs." >&2
    exit 1
fi
