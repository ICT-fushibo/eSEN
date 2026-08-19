# Queued Opt4 and Matbench runs

Both queue scripts poll `nvidia-smi` and dispatch only after a GPU has had no
compute process, low utilization, and low memory use for
`GPU_IDLE_SECONDS` (default 120 seconds). They do not start, stop, or modify
MPS. Use separate output roots for Opt4 and Matbench.

## Opt4 KF6-KF8 ablation

`run_opt4_kf6_8_8gpu.sh` defaults to 8 GPUs, `Cu32 Cu512 H2O32 H2O512`,
`100` steps, `5` repeats, and `Cu512/H2O512` as the selector focus. Stage
selection is serial; tasks within a stage are polled and run concurrently.

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
SCOPES=both \
CHECKPOINT=/public-data/fushibo/eSEN/esen_30m_oam.pt \
STRUCTURE_DIR=/public-data/fushibo/MatRIS-09bk/example/cif_file \
BASELINE_DIR=/public-data/fushibo/eSEN/example/md_out/esen_stage1_energy_gpu2_20260722_130306/ase \
OPT4_SAVE_DIR=/public-data/fushibo/eSEN/example/md_out/opt4_kf6_8_8gpu_$(date +%Y%m%d_%H%M%S) \
bash example/run_opt4_kf6_8_8gpu.sh
```

Set `OPT4_STAGES="KF2 KF3 KF4 KF5 KF6 KF7 KF8"` to rerun the complete
Opt4 chain. Set `INITIAL_ACCEPTED_WHOLE_STEP=rmsnorm` to preserve the accepted
whole-step RMSNorm base when starting at KF6.

## Matbench baseline/Opt1/Opt2/Opt3

`run_esen_matbench_8gpu.sh` discovers the 17 HDF5 systems when `SYSTEMS` is
omitted. One queued process runs all four backends for one system on one GPU,
so its speedups are same-GPU comparisons. `SAVE_DIR` or `MATBENCH_SAVE_DIR`
is the persistence interface; each system is stored below
`<save-dir>/systems/<system>`.

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
REFERENCE_H5=/public-data/fushibo/matbench-discovery-data/md/2026-06-29-dynamat-v1.0-reference-trajectories.h5 \
CHECKPOINT=/public-data/fushibo/eSEN/esen_30m_oam.pt \
MATBENCH_REPO=/public-data/fushibo/matbench-discovery \
SAVE_DIR=/public-data/fushibo/eSEN/example/md_out/matbench_8gpu_$(date +%Y%m%d_%H%M%S) \
bash example/run_esen_matbench_8gpu.sh
```

For a smoke run, use `STEPS=100 SYSTEMS='bulkCuAu_500K-Artrith_VASP'`.
The queue writes `queue_status.tsv`, per-system reports and trajectories, and
the aggregate `matbench_esen_queue_report.{json,md}` plus
`matbench_esen_speedups.tsv`.

## Run both queues

The combined launcher uses disjoint GPU subsets so the two independent pollers
cannot race to assign the same GPU:

```bash
OPT4_GPU_LIST="0 1 2 3" \
MATBENCH_GPU_LIST="4 5 6 7" \
ROOT_OUTPUT_DIR=/public-data/fushibo/eSEN/example/md_out/queued_$(date +%Y%m%d_%H%M%S) \
CHECKPOINT=/public-data/fushibo/eSEN/esen_30m_oam.pt \
bash example/run_opt4_and_matbench_8gpu.sh
```

If another experiment is using some cards, remove them from the corresponding
GPU list. The poller will also refuse to launch on cards with active compute
processes.
