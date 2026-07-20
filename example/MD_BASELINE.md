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
| Cu192 | 192 |
| Cu512 | 512 |
| Cu1024 | 1024 |
| H2O192 | 576 |
| H2O512 | 1536 |
| H2O1024 | 3072 |

Generate the CIF files with the existing MatRIS helper if needed:

```bash
python ../MatRIS-09bk/example/generate_structures.py
```

## One smoke test

```bash
CUDA_VISIBLE_DEVICES=1 python -u example/benchmark_md.py \
    --structure ../MatRIS-09bk/example/cif_file/Cu192.cif \
    --checkpoint checkpoints/esen_30m_oam.pt \
    --system Cu192 \
    --temperature 800 \
    --steps 10 \
    --warmup-steps 3
```

## Six-system benchmark

```bash
CHECKPOINT="$PWD/checkpoints/esen_30m_oam.pt" \
STRUCTURE_DIR="$PWD/../MatRIS-09bk/example/cif_file" \
GPU=1 STEPS=1000 WARMUP_STEPS=3 REPEATS=3 \
bash example/run_md_baselines.sh
```

Each repeat runs in a fresh Python process. JSON results and an append-only
`summary.tsv` are written under `example/md_out` by default.
