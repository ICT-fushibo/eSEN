#!/usr/bin/env bash
set -euo pipefail

# Optional:
#   CHECKPOINT=/path/to/esen_30m_oam.pt STRUCTURE_DIR=/path/to/cif_file
#   GPU=1 STEPS=1000 WARMUP_STEPS=3 REPEATS=3

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT=${CHECKPOINT:-"$REPO_ROOT/esen_30m_oam.pt"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"$REPO_ROOT/../MatRIS-09bk/example/cif_file"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/example/md_out"}
GPU=${GPU:-1}
STEPS=${STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-3}
REPEATS=${REPEATS:-3}

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

systems=(Cu192 Cu512 Cu1024 H2O192 H2O512 H2O1024)

for system in "${systems[@]}"; do
    structure="$STRUCTURE_DIR/$system.cif"
    if [[ ! -f "$structure" ]]; then
        echo "Structure not found: $structure" >&2
        echo "Generate the six CIF files before running this benchmark." >&2
        exit 2
    fi

    if [[ "$system" == Cu* ]]; then
        temperature=800
    else
        temperature=300
    fi

    for repeat in $(seq 1 "$REPEATS"); do
        run_name="${system}_${temperature}K_${STEPS}step_esen_baseline_r${repeat}"
        echo "Running $run_name on physical GPU $GPU"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
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
                --seed 42 \
                --outputs energy forces \
            2>&1 | tee "$OUTPUT_DIR/${run_name}.log"
    done
done
