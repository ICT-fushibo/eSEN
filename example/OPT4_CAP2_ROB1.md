# Opt4 CAP2 / ROB1

CAP2 is an opt-in whole-step Opt4 capacity policy. The default remains the
frozen `auto-safe` policy with rollback disabled, so Opt3 and Opt4 v1-v4 runs
are unchanged.

The elastic policy starts from per-atom probe maxima rounded to four slots
when that saves at least 5% relative to CAP1-auto-safe. An overflowing replay
writes a dummy-only graph, and ROB1 discards its complete transaction. The
controller restores fixed-address MD/NHC state, promotes capacity, captures
one replacement graph, and replays the same physical steps. Capacity is
monotonic and at most two promotions are allowed.

## Unit tests

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src" \
python -m pytest -q --noconftest \
    tests/core/applications/test_esen_fixed_neighbor.py \
    tests/core/applications/test_esen_matbench.py \
    tests/core/applications/test_esen_opt4_model_fusion.py
```

## Smoke

The smoke includes normal Cu32/H2O32/bulkCu runs and forced-low-capacity runs
that must roll back, recapture, and reproduce the sufficient-capacity bulkCu
trajectory.
Standard smoke systems can be overridden with `SMOKE_SYSTEMS`; the generic
`SYSTEMS` variable is intentionally ignored so an exported formal-test matrix
cannot leak into this test.

```bash
GPU=0 \
OUTPUT_DIR="$PWD/example/md_out/cap2_rob1_smoke_$(date '+%Y%m%d_%H%M%S')" \
REFERENCE_H5=/home/fushibo/matbench-discovery-data/2026-06-29-dynamat-v1.0-reference-trajectories.h5 \
MATBENCH_REPO=/home/fushibo/matbench-discovery \
CHECKPOINT="$PWD/esen_30m_oam.pt" \
STRUCTURE_DIR=/home/fushibo/MatRIS-09bk/example/cif_file \
bash example/smoke_opt4_cap2_rob1.sh
```

## 100-step interleaved ablation

```bash
GPU=0 \
OUTPUT_DIR="$PWD/example/md_out/cap2_rob1_ablation_$(date '+%Y%m%d_%H%M%S')" \
CHECKPOINT="$PWD/esen_30m_oam.pt" \
STRUCTURE_DIR=/home/fushibo/MatRIS-09bk/example/cif_file \
SYSTEMS="Cu32 Cu512 H2O32 H2O192" TEMPERATURES=300 \
STEPS=100 REPEATS=3 \
bash example/run_opt4_cap2_rob1_ablation.sh
```

The runner writes `CAP2_ROB1_selection.json`. Energy and force differences are
telemetry and do not gate the timing decision.

## bulkCu 10k Matbench confirmation

`BASE_RESULT_DIR` may point to the completed frozen Opt4 v4 10k directory. If
provided, the runner also checks all physical-statistic delta thresholds.

```bash
GPU=0 \
REFERENCE_H5=/home/fushibo/matbench-discovery-data/2026-06-29-dynamat-v1.0-reference-trajectories.h5 \
MATBENCH_REPO=/home/fushibo/matbench-discovery \
CHECKPOINT="$PWD/esen_30m_oam.pt" \
SAVE_DIR="$PWD/example/md_out/cap2_rob1_bulkCu_10k_$(date '+%Y%m%d_%H%M%S')" \
BASE_RESULT_DIR=/path/to/frozen_opt4_v4_bulkCu_10k \
bash example/run_opt4_cap2_rob1_matbench_10k.sh
```

Completed trajectories can be re-evaluated without rerunning MD:

```bash
GPU=0 REFERENCE_H5=/path/to/reference.h5 \
MATBENCH_REPO=/path/to/matbench-discovery \
BACKENDS=opt4 SAVE_DIR=/path/to/completed/run \
METRICS_ONLY=1 bash example/run_esen_matbench.sh
```
