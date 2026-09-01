# eSEN Matbench MD

`run_esen_matbench.py` is a separate Matbench/DynaMat runner.  It does not
modify the existing Cu/H2O baseline, Opt1, Opt2, Opt3, or Opt4 scripts.

## Protocol

The formal protocol is the public Matbench DynaMat v1.0 protocol:

- 17 HDF5 systems, starting from the public frame 0;
- ASE-compatible `NoseHooverChainNVT`, `tchain=3`, `tloop=1`;
- `0.25 fs` timestep and `25 fs` thermostat time scale;
- Maxwell-Boltzmann velocities from `numpy.default_rng(seed=0)`;
- no warmup;
- 80,000 steps and one saved frame every 10 steps (`8,001` frames);
- predicted saved-frame interval `2.5 fs`.

The baseline uses a Matbench-only ASE driver that pins the ASE 3.28
Suzuki--Yoshida coefficient rule. This avoids the extra coefficient-count
division present in ASE 3.24 while leaving the installed ASE package and all
non-Matbench runners untouched.

The Matbench-specific GPU integrator and whole-step graph live in
`src/fairchem/core/applications/esen_matbench.py`.  The original
`GPUIntegrator` and existing CUDA Graph classes are not replaced.

The Matbench `opt4` backend uses the frozen Opt4 v4 FP32 configuration:
`rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm,so2-prepare-backward-reduce`
with the `auto-safe` fixed-neighbor capacity policy. It keeps the Matbench NHC
integrator rather than reusing the Cu/H2O Berendsen path.

## 10,000-step aligned pilot

This launcher pins every physical setting to Matbench and changes only the
step count. It runs baseline, Opt2, Opt3, and Opt4 serially on one GPU and uses
`bulkCu_1000K_Kapil` (108 atoms, with reference stress) by default. Its
reference trajectory covers the complete 2.5 ps prediction window:

```bash
GPU=0 \
REFERENCE_H5=/home/fushibo/matbench-discovery-data/2026-06-29-dynamat-v1.0-reference-trajectories.h5 \
MATBENCH_REPO=/home/fushibo/matbench-discovery \
SAVE_DIR=/home/fushibo/eSEN/example/md_out/matbench_10k_pilot \
bash example/run_esen_matbench_10k_pilot.sh
```

RDF/ADF/vDOS and pressure PMAE/PW1/error use the official implementation and
the common 0--2.5 ps time window. Predicted stress is evaluated after rollout
with the canonical checkpoint and is explicitly excluded from seconds/step.
A one-system 10,000-step result is a diagnostic pilot, not a published
17-system leaderboard score.

## Smoke

Run one backend and one small system first:

```bash
GPU=2 \
BACKEND=opt3 \
SYSTEMS='bulkCuAu_500K-Artrith_VASP' \
STEPS=100 \
OUTPUT_DIR=/public-data/fushibo/eSEN/example/md_out/matbench_opt3_smoke \
bash example/run_esen_matbench.sh
```

Repeat with `BACKEND=baseline`, `opt1`, `opt2`, and `opt4`, using separate output
directories.  The launcher never starts or stops MPS; it only sets
`CUDA_VISIBLE_DEVICES`.

## Formal backend runs

For a directly comparable speedup report, run all requested backends in one process
and one output directory.  Each backend is still initialized and recorded as an
independent run; the runner releases the previous system's model/cache before
continuing after OOM or capacity failure. `--save-dir` is an alias for
`--output-dir` and is the explicit persistence interface:

```bash
GPU=2 BACKENDS='baseline opt1 opt2 opt3 opt4' \
  SAVE_DIR=/public-data/fushibo/eSEN/example/md_out/matbench_all_gpu2 \
  bash example/run_esen_matbench.sh
```

For multiple GPUs, `run_esen_matbench_8gpu.sh` polls the GPU pool and assigns
one system per job. All requested backends for that system remain on the same
physical GPU, and each system is saved below `<SAVE_DIR>/systems/<system>`.
The queue writes `matbench_esen_queue_report.{json,md}` after pending jobs
finish.

If a backend must be isolated in a separate process, set `BACKEND=opt3` (or
another single value) and use a unique output directory.  Combine those reports
only after confirming that the runs used the same GPU and software environment.
The output directory must be unique unless `OVERWRITE=1` is explicitly set.

## Timing and metrics

`rollout_wall_time_s` includes initial force evaluation, all NHC steps, and the
same frame-sampling device copies for every backend.  Probe, graph capture,
model loading, HDF5 serialization, and metric calculation are reported
separately.

The runner computes public RDF, ADF, and vDOS metrics using the source checkout's
`matbench_discovery.metrics.md`.  It has a compatibility loader for checkouts whose
top-level Matbench rollout helper imports optional UI dependencies or is not usable in
the eSEN Python environment.  If the metric implementation itself is unavailable,
rollout results are still retained and the report marks metrics as unavailable.  The
public HDF5 has no energy/force labels, and the current eSEN energy-force path does
not calculate stress, so private RMSE and pressure metrics are marked unavailable.
Prediction trajectories are retained in HDF5 for a later offline stress pass.

The report reads `models/esen/esen-30m-oam.yml` and displays the published
eSEN-30M-OAM values separately.  H200 published runtime is not used as a speed
denominator for a local H100 run.

## Output layout

The run directory contains `run_metadata.json`, `run_status.tsv`, one JSON result
and HDF5 trajectory per backend/system, `matbench_esen_report.{json,md}`,
`matbench_esen_speedups.tsv`, and `matbench_esen_metrics.tsv`.  Per-backend metric
CSV/aggregate JSON files are written below `metrics/`.  Incomplete trajectories are
kept with `complete=false` and are excluded from public metric aggregation.
