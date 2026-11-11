#!/usr/bin/env bash
# Launch only three independent comparisons; one experiment per GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${REFERENCE_H5:?Set REFERENCE_H5}"
: "${OUTPUT_ROOT:?Set a new OUTPUT_ROOT, or the existing root for RESUME=1}"
PHASE="${PHASE:-smoke}"
if [[ "$PHASE" != smoke && "$PHASE" != ablation ]]; then
    echo 'PHASE must be smoke or ablation' >&2
    exit 2
fi
if [[ "$PHASE" == ablation ]]; then
    : "${SMOKE_ROOT:?Set the successfully completed SMOKE_ROOT}"
fi
CHECKPOINT="${CHECKPOINT:-$PWD/esen_30m_oam.pt}"
MATBENCH_REPO="${MATBENCH_REPO:-$(dirname "$PWD")/matbench-discovery}"
read -r -a GPUS <<< "${GPU_LIST:-0 1 2}"
if [[ ${#GPUS[@]} != 3 || ${GPUS[0]} == "${GPUS[1]}" || ${GPUS[0]} == "${GPUS[2]}" || ${GPUS[1]} == "${GPUS[2]}" ]]; then
    echo 'GPU_LIST must contain three distinct GPU indices' >&2
    exit 2
fi
EXPERIMENTS=(nhc rob-compact shared-geometry)
mkdir -p "$OUTPUT_ROOT/logs"
PIDS=()
for INDEX in 0 1 2; do
    EXPERIMENT="${EXPERIMENTS[$INDEX]}"
    EXTRA=()
    [[ "${RESUME:-0}" != 1 ]] || EXTRA+=(--resume)
    [[ "$PHASE" != ablation ]] || EXTRA+=(--smoke-dir "$SMOKE_ROOT/$EXPERIMENT")
    PYTHONPATH="$PWD/src:$PWD/example" python -u example/run_opt4_runtime_ablation.py \
        --experiment "$EXPERIMENT" --phase "$PHASE" --gpu "${GPUS[$INDEX]}" \
        --reference-h5 "$REFERENCE_H5" --checkpoint "$CHECKPOINT" \
        --matbench-repo "$MATBENCH_REPO" --idle-seconds "${GPU_IDLE_SECONDS:-10}" \
        --output-dir "$OUTPUT_ROOT/$EXPERIMENT" "${EXTRA[@]}" \
        >> "$OUTPUT_ROOT/logs/$EXPERIMENT.log" 2>&1 &
    PIDS+=("$!")
    echo "$EXPERIMENT gpu=${GPUS[$INDEX]} pid=$! log=$OUTPUT_ROOT/logs/$EXPERIMENT.log"
done
FAILURES=0
for INDEX in 0 1 2; do
    STATUS=0
    wait "${PIDS[$INDEX]}" || STATUS=$?
    echo "${EXPERIMENTS[$INDEX]} exit=$STATUS"
    [[ "$STATUS" == 0 ]] || FAILURES=$((FAILURES + 1))
done
echo "Results: $OUTPUT_ROOT (failed/rejected=$FAILURES)"
[[ "$FAILURES" == 0 ]]
