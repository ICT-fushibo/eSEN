# eSEN eager GPU-resident MD

This is the stage-1 optimization baseline.  Positions, momenta, masses,
forces, the PyG batch, and model inputs remain on the GPU.  It intentionally
does not enable CUDA Graphs, `torch.compile`, AMP, TF32, or custom kernel
fusion.  The official ASE/`OCPCalculator` benchmark remains in
`benchmark_md.py`.

The NVT path uses the same unconstrained Velocity-Verlet/Berendsen equations
and parameters as the ASE baseline.  MD state defaults to FP64, while the
checkpoint model keeps its FP32 input dtype; conversion happens on the GPU.
Stress regression and activation checkpointing are disabled because this MD
benchmark consumes only energy and forces.

## GPU 6 smoke test

On the server:

```bash
cd /public-data/fushibo/eSEN
conda activate esen_opt
bash example/test_md_gpu_resident_gpu6.sh
```

This runs five steps for Cu32 and H2O32 at 300 K and compares the initial
energy and forces with `OCPCalculator` before timing.

## All systems at 300 K and 800 K

To rerun the corrected ASE baseline and the GPU-resident path back-to-back and
generate a speedup table:

```bash
cd /public-data/fushibo/eSEN
conda activate esen_opt

CHECKPOINT=/public-data/fushibo/eSEN/esen_30m_oam.pt \
STRUCTURE_DIR=/public-data/fushibo/MatRIS-09bk/example/cif_file \
GPU=6 STEPS=1000 WARMUP_STEPS=3 REPEATS=3 \
bash example/run_stage1_comparison_gpu6.sh
```

The combined directory contains `stage1_comparison.tsv` and
`stage1_comparison.md`.  A speedup is reported only when both paths complete;
OOM counts are retained separately.

To run only the GPU-resident path:

```bash
cd /public-data/fushibo/eSEN
conda activate esen_opt

CHECKPOINT=/public-data/fushibo/eSEN/esen_30m_oam.pt \
STRUCTURE_DIR=/public-data/fushibo/MatRIS-09bk/example/cif_file \
GPU=6 STEPS=1000 WARMUP_STEPS=3 REPEATS=3 \
bash example/run_md_gpu_resident.sh
```

Each attempt uses a fresh Python process.  CUDA OOM is recorded as `oom` in
`run_status.tsv` and does not stop later systems.  Successful JSON results are
summarized in `gpu_resident_report.tsv` and `gpu_resident_report.md`.

Useful short-run overrides:

```bash
GPU=6 STEPS=10 REPEATS=1 \
SYSTEMS="Cu32 Cu64 H2O32 H2O60" TEMPERATURES="300" \
bash example/run_md_gpu_resident.sh
```

Set `STRICT=1` if the batch shell should return a failure status when any run
OOMs or fails.  The default is to finish the full matrix and report all
statuses.
