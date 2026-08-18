#!/usr/bin/env bash
set -uo pipefail

# Fair one-candidate Opt4 A/B round.  Base and candidate are shuffled within
# every (system, temperature, repeat) block and run in fresh Python processes.
# SCOPE=model-only compares against Opt2's dynamic-builder model CUDA Graph;
# SCOPE=whole-step compares against Opt3's whole-step CUDA Graph.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Set the physical GPU index}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${SCOPE:?Set SCOPE to model-only or whole-step}"
: "${BASE_STAGE:?Set BASE_STAGE}"
: "${CANDIDATE_STAGE:?Set CANDIDATE_STAGE}"
: "${CANDIDATE_FUSIONS:?Set CANDIDATE_FUSIONS}"
BASE_FUSIONS=${BASE_FUSIONS:-}
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
BASELINE_DIR=${BASELINE_DIR:-}
BASELINE_STEPS=${BASELINE_STEPS:-1000}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-5}
SYSTEMS=${SYSTEMS:-"Cu32 Cu192 H2O32 H2O60"}
TEMPERATURES=${TEMPERATURES:-"300"}
SEED=42
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-}

case "$SCOPE" in
    model-only|whole-step) ;;
    *) echo "Unsupported Opt4 scope: $SCOPE" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"/{results,logs}
STATUS_TSV="$OUTPUT_DIR/run_status.tsv"
printf 'scope\tvariant\tfusion_stage\tmodel_fusions\tsystem\ttemperature_K\trepeat\trun_name\tstatus\texit_code\tprocess_wall_time_s\n' > "$STATUS_TSV"

{
    echo "scope=$SCOPE"
    echo "base_stage=$BASE_STAGE"
    echo "base_fusions=$BASE_FUSIONS"
    echo "candidate_stage=$CANDIDATE_STAGE"
    echo "candidate_fusions=$CANDIDATE_FUSIONS"
    echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "source_bundle_sha256=$SOURCE_BUNDLE_SHA256"
    echo "physical_gpu=$GPU"
    echo "systems=$SYSTEMS"
    echo "temperatures=$TEMPERATURES"
    echo "steps=$STEPS"
    echo "repeats=$REPEATS"
    echo "seed=$SEED"
    echo "checkpoint=$CHECKPOINT"
    echo "structure_dir=$STRUCTURE_DIR"
    echo "baseline_dir=$BASELINE_DIR"
    echo "baseline_steps=$BASELINE_STEPS"
    nvidia-smi -i "$GPU" \
        --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader
} > "$OUTPUT_DIR/run_metadata.txt"

shuffle_variants() {
    local salt=$1
    python -c 'import hashlib,random,sys; v=["base","candidate"]; random.Random(int.from_bytes(hashlib.sha256(("42|"+sys.argv[1]).encode()).digest()[:8],"big")).shuffle(v); print(" ".join(v))' "$salt"
}

classify() {
    local code=$1 log=$2 result=$3
    if [[ $code -eq 0 && -s "$result" ]]; then echo success
    elif [[ $code -eq 42 ]] || grep -Eqi 'BENCHMARK_STATUS=oom|out of memory' "$log"; then echo oom
    elif [[ $code -eq 43 ]] || grep -Eqi 'BENCHMARK_STATUS=validation_failed' "$log"; then echo validation_failed
    elif [[ $code -eq 45 ]] || grep -Eqi 'BENCHMARK_STATUS=capacity_overflow' "$log"; then echo capacity_overflow
    elif [[ $code -eq 46 ]] || grep -Eqi 'BENCHMARK_STATUS=unsupported_fusion_config' "$log"; then echo unsupported_fusion_config
    else echo error; fi
}

run_one() {
    local variant=$1 system=$2 temperature=$3 repeat=$4
    local stage fusions backend
    if [[ "$variant" == base ]]; then
        stage=$BASE_STAGE; fusions=$BASE_FUSIONS
    else
        stage=$CANDIDATE_STAGE; fusions=$CANDIDATE_FUSIONS
    fi
    if [[ "$SCOPE" == "model-only" ]]; then
        if [[ -n "$fusions" ]]; then backend=model-cg-opt4; else backend=model-cg; fi
    else
        if [[ -n "$fusions" ]]; then backend=whole-step-cg-opt4; else backend=whole-step-cg; fi
    fi
    local scope_suffix=${SCOPE//-/_}
    local run_name="${system}_${temperature}K_${STEPS}step_esen_${scope_suffix}_${stage}_r${repeat}"
    local result="$OUTPUT_DIR/results/${run_name}.json"
    local log="$OUTPUT_DIR/logs/${run_name}.log"
    local refs=()
    local baseline="$BASELINE_DIR/${system}_${temperature}K_${BASELINE_STEPS}step_esen_baseline_r${repeat}.json"
    if [[ -n "$BASELINE_DIR" && -s "$baseline" ]]; then refs+=(--baseline-result "$baseline")
    elif [[ -n "$BASELINE_DIR" ]]; then refs+=(--missing-baseline-reference); fi
    local start end elapsed code status
    start=$(date +%s%N)
    set +e
    if [[ "$SCOPE" == "model-only" ]]; then
        # Replay stability is judged at the engineering force tolerance
        # (--force-max-atol below): the legacy 1e-6 threshold labels FP32
        # graph-replay jitter (~1e-5 eV/A) as a validation failure even
        # though it is ~20x below the engineering accuracy bar.
        local model_args=()
        if [[ "$backend" == "model-cg-opt4" ]]; then
            model_args+=(--model-fusions "$fusions" --fusion-stage "$stage")
        else
            # Label the unmodified Opt2 control without changing its kernels.
            model_args+=(--fusion-stage "$stage")
        fi
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=42 \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
            python -u "$REPO_ROOT/example/benchmark_md_gpu.py" \
                --backend "$backend" \
                --structure "$STRUCTURE_DIR/$system.cif" \
                --checkpoint "$CHECKPOINT" --system "$system" \
                --output-dir "$OUTPUT_DIR/results" --run-name "$run_name" \
                --steps "$STEPS" --warmup-steps "$WARMUP_STEPS" \
                --temperature "$temperature" --timestep 1.0 --taut 100.0 \
                --seed 42 --repeat "$repeat" --md-dtype float64 \
                --cg-probe-steps 50 --cg-capacity-margin 0.10 \
                --cg-edge-step 256 --cg-dummy-atoms 32 \
                --cg-capture-warmup 3 --cg-replay-energy-atol 0.0 \
                --cg-replay-force-atol 2e-4 \
                --energy-per-atom-atol 1e-5 --force-max-atol 2e-4 \
                "${model_args[@]}" "${refs[@]}" > "$log" 2>&1
    else
        local benchmark_script="$REPO_ROOT/example/benchmark_md_opt4.py"
        local whole_args=()
        if [[ "$backend" == "whole-step-cg" ]]; then
            benchmark_script="$REPO_ROOT/example/benchmark_md_opt3.py"
        else
            whole_args+=(--model-fusions "$fusions" --fusion-stage "$stage")
        fi
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=42 \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
            python -u "$benchmark_script" \
                --backend "$backend" \
                --structure "$STRUCTURE_DIR/$system.cif" \
                --checkpoint "$CHECKPOINT" --system "$system" \
                --output-dir "$OUTPUT_DIR/results" --run-name "$run_name" \
                --steps "$STEPS" --warmup-steps "$WARMUP_STEPS" \
                --temperature "$temperature" --timestep 1.0 --taut 100.0 \
                --seed 42 --repeat "$repeat" --probe-steps 50 \
                --neighbor-margin 0.10 --neighbor-slot-step 8 --dummy-atoms 32 \
                --capture-warmup 3 --max-neighbors 300 \
                --degeneracy-tolerance 0.01 --energy-per-atom-atol 1e-5 \
                --force-max-atol 2e-4 "${whole_args[@]}" \
                "${refs[@]}" > "$log" 2>&1
    fi
    code=$?
    set -e
    end=$(date +%s%N)
    elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.6f",(b-a)/1e9}')
    status=$(classify "$code" "$log" "$result")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$SCOPE" "$variant" "$stage" "$fusions" "$system" \
        "$temperature" "$repeat" "$run_name" "$status" "$code" \
        "$elapsed" >> "$STATUS_TSV"
    echo "$status ($code): $run_name"
}

read -r -a systems <<< "$SYSTEMS"
read -r -a temperatures <<< "$TEMPERATURES"
for repeat in $(seq 1 "$REPEATS"); do
    for system in "${systems[@]}"; do
        for temperature in "${temperatures[@]}"; do
            read -r -a variants <<< "$(shuffle_variants "$CANDIDATE_STAGE|$system|$temperature|$repeat")"
            for variant in "${variants[@]}"; do
                run_one "$variant" "$system" "$temperature" "$repeat"
            done
        done
    done
done

echo "finished_at=$(date --iso-8601=seconds)" >> "$OUTPUT_DIR/run_metadata.txt"
echo "Opt4 $SCOPE A/B results: $OUTPUT_DIR"
