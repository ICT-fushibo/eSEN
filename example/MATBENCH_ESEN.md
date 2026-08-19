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

The Matbench-specific GPU integrator and whole-step graph live in
`src/fairchem/core/applications/esen_matbench.py`.  The original
`GPUIntegrator` and existing CUDA Graph classes are not replaced.

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

Repeat with `BACKEND=baseline`, `opt1`, and `opt2`, using separate output
directories.  The launcher never starts or stops MPS; it only sets
`CUDA_VISIBLE_DEVICES`.

## Formal backend runs

For a directly comparable speedup report, run all four backends in one process
and one output directory.  Each backend is still initialized and recorded as an
independent run; the runner releases the previous system's model/cache before
continuing after OOM or capacity failure:

```bash
GPU=2 BACKENDS='baseline opt1 opt2 opt3' \
  OUTPUT_DIR=/public-data/fushibo/eSEN/example/md_out/matbench_all_gpu2 \
  bash example/run_esen_matbench.sh
```

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
