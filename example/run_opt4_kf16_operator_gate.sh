#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:=0}"
: "${OUTPUT_DIR:=$REPO_ROOT/example/md_out/opt4_kf16_operator_gpu${GPU}_$(date '+%Y%m%d_%H%M%S')}"
: "${WARMUP:=3}"
: "${REPLAYS:=50}"
: "${BENCH_REPEATS:=7}"
: "${MINIMUM_SPEEDUP:=1.05}"

mkdir -p "$OUTPUT_DIR"/{results,logs}
failures=0
for system in Cu512 H2O192; do
    for mode in forward forward-backward; do
        for variant in base tiled; do
            run="${system}_${mode}_${variant}"
            set +e
            CUDA_VISIBLE_DEVICES="$GPU" \
            PYTHONHASHSEED=42 \
            CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/example" \
            python -u "$REPO_ROOT/example/benchmark_opt4_wigner_microbench.py" \
                --system "$system" \
                --mode "$mode" \
                --variant "$variant" \
                --warmup "$WARMUP" \
                --replays "$REPLAYS" \
                --repeats "$BENCH_REPEATS" \
                --output "$OUTPUT_DIR/results/$run.json" \
                > "$OUTPUT_DIR/logs/$run.log" 2>&1
            code=$?
            set -e
            if [[ $code -ne 0 ]]; then
                failures=$((failures + 1))
                echo "failed ($code): $run"
            else
                echo "success: $run"
            fi
        done
    done
done

set +e
python -u "$REPO_ROOT/example/select_opt4_kf16_operator_gate.py" \
    --input-dir "$OUTPUT_DIR" \
    --minimum-speedup "$MINIMUM_SPEEDUP" \
    --output "$OUTPUT_DIR/KF16_operator_gate.json"
selector_code=$?
set -e

echo "KF16 operator results: $OUTPUT_DIR"
if [[ $failures -ne 0 || $selector_code -ne 0 ]]; then
    echo "KF16 operator gate rejected (failures=$failures selector=$selector_code)"
    exit 1
fi
echo "KF16 operator gate accepted"
