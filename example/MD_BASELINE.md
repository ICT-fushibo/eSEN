# eSEN MD baseline

This benchmark uses the official `OCPCalculator` with the
`eSEN-30M-OAM` checkpoint and ASE `NVTBerendsen` dynamics.

Baseline contract:

- eager fairchem inference;
- no `torch.compile`;
- no explicit CUDA Graph;
- AMP disabled;
- TF32 disabled;
- on-the-fly neighbor graph construction retained;
- energy and force outputs only;
- three untimed MD warm-up steps, followed by restoration of the initial state;
- model loading, warm-up, validation, hashing, and result I/O excluded from timing;
- no trajectory or per-step logfile in the timed region.

## Checkpoint download

The checkpoint is gated. First open `https://huggingface.co/facebook/OMAT24`
in a browser, accept its terms, and create a read token. On the server, avoid
putting the token in shell history:

```bash
read -s HF_TOKEN
export HF_TOKEN
bash example/download_esen_checkpoint.sh
unset HF_TOKEN
```

If the server cannot reach Hugging Face, run the download on an accessible
machine and transfer `esen_30m_oam.pt` to the server. Access still requires an
account that accepted the model terms.

## Structures

The expected structures and atom counts are:

| system | atoms |
| --- | ---: |
| Cu32 | 32 |
| Cu64 | 64 |
| Cu192 | 192 |
| Cu512 | 512 |
| Cu1024 | 1024 |
| H2O32 | 96 |
| H2O60 | 180 |
| H2O192 | 576 |
| H2O512 | 1536 |
| H2O1024 | 3072 |

Generate the CIF files with the existing MatRIS helper if needed:

```bash
python ../MatRIS-09bk/example/generate_structures.py \
    --systems Cu32 Cu64 Cu192 Cu512 Cu1024 \
              H2O32 H2O60 H2O192 H2O512 H2O1024
```

## One smoke test

```bash
CUDA_VISIBLE_DEVICES=1 python -u example/benchmark_md.py \
    --structure ../MatRIS-09bk/example/cif_file/Cu192.cif \
    --checkpoint esen_30m_oam.pt \
    --system Cu192 \
    --temperature 800 \
    --steps 10 \
    --warmup-steps 3
```

## Formal ten-system, two-temperature benchmark

```bash
CHECKPOINT="$PWD/esen_30m_oam.pt" \
STRUCTURE_DIR="$PWD/../MatRIS-09bk/example/cif_file" \
GPU=6 STEPS=1000 WARMUP_STEPS=3 REPEATS=3 \
bash example/run_md_baselines.sh
```

Each system is tested at both 300 K and 800 K. Each repeat runs in a fresh
Python process. JSON results, process status, logs, and median reports are
written under a timestamped directory in `example/md_out` by default.
