#!/usr/bin/env bash
set -euo pipefail

# Launch the Opt4 ablation queue and the Matbench queue together without
# sharing a GPU between the two scheduler processes.  Each child still polls
# its assigned GPU subset and waits for GPU_IDLE_SECONDS before dispatching.
# Set RUN_OPT4=0 or RUN_MATBENCH=0 to run only one workload.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_OPT4=${RUN_OPT4:-1}
RUN_MATBENCH=${RUN_MATBENCH:-1}
OPT4_GPU_LIST=${OPT4_GPU_LIST:-"0 1 2 3"}
MATBENCH_GPU_LIST=${MATBENCH_GPU_LIST:-"4 5 6 7"}
ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR:-"$REPO_ROOT/example/md_out/queued_opt4_matbench_$(date '+%Y%m%d_%H%M%S')"}
OPT4_SAVE_DIR=${OPT4_SAVE_DIR:-"$ROOT_OUTPUT_DIR/opt4"}
MATBENCH_SAVE_DIR=${MATBENCH_SAVE_DIR:-"$ROOT_OUTPUT_DIR/matbench"}
OPT4_LOG=${OPT4_LOG:-"$ROOT_OUTPUT_DIR/opt4_queue.log"}
MATBENCH_LOG=${MATBENCH_LOG:-"$ROOT_OUTPUT_DIR/matbench_queue.log"}

mkdir -p "$OPT4_SAVE_DIR" "$MATBENCH_SAVE_DIR"
pids=()

if [[ "$RUN_OPT4" == 1 ]]; then
    GPU_LIST="$OPT4_GPU_LIST" \
    OPT4_SAVE_DIR="$OPT4_SAVE_DIR" ROOT_OUTPUT_DIR="$OPT4_SAVE_DIR" \
    bash "$REPO_ROOT/example/run_opt4_kf6_8_8gpu.sh" \
        > "$OPT4_LOG" 2>&1 &
    pids+=("$!")
fi

if [[ "$RUN_MATBENCH" == 1 ]]; then
    GPU_LIST="$MATBENCH_GPU_LIST" \
    MATBENCH_SAVE_DIR="$MATBENCH_SAVE_DIR" \
    bash "$REPO_ROOT/example/run_esen_matbench_8gpu.sh" \
        > "$MATBENCH_LOG" 2>&1 &
    pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
echo "Opt4 output: $OPT4_SAVE_DIR"
echo "Matbench output: $MATBENCH_SAVE_DIR"
exit "$status"
