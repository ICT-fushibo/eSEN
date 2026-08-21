# Opt4 CAP1: heterogeneous fixed neighbor capacity

CAP1 is an opt-in whole-step CUDA Graph ablation on top of the frozen Opt4 v2
fusion mask (`rmsnorm,so2-epilogue,so2-gate-bridge`).  It does not change the
default `uniform` capacity policy used by Opt3 or Opt4 v1/v2.

The `atom` policy records the largest official neighbor degree observed for
each atom during the setup probe, adds the existing margin, and rounds each
capacity by `neighbor-slot-step`.  The resulting per-atom capacities are fixed
for capture and replay.  This benefits any structure with heterogeneous local
coordination; it is not keyed to water or to particular atomic numbers.

Because a long diffusive trajectory may leave the probed local environment,
`fixed_builder_capacity_misses=0` is mandatory.  Do not use CAP1 for long
Matbench trajectories until that protocol has an appropriately long probe or
an independent capacity-safety study.

## CAP1-auto

The unconditional CAP1 experiment showed a stable Cu32 regression even though
H2O32/H2O192 improved by about 1.15x.  `auto` therefore evaluates the rounded
per-atom allocation after the existing probe and selects it only when its
fixed edge capacity is at least 5% smaller than uniform capacity.  Otherwise
it captures the original uniform graph.  The decision is chemistry-agnostic,
occurs before capture, and adds no replay-time branch.

Every result records:

- `neighbor_capacity_policy_requested=auto`;
- `neighbor_capacity_policy_effective=uniform|atom`;
- `neighbor_capacity_auto_candidate_reduction_vs_uniform`;
- `neighbor_capacity_auto_selected`.

The threshold is configurable with `--neighbor-auto-min-reduction` or the
runner environment variable `NEIGHBOR_AUTO_MIN_REDUCTION`.

## CAP1-auto-safe

The full 300/800 K matrix found deterministic local-slot overflows in
H2O60/H2O192 at 800 K even though roughly 15% of the global edge buffer was
unused.  `auto-safe` preserves the original `auto` implementation for result
reproducibility and adds one protected slot bucket after margin/rounding.  It
then reapplies the minimum-reduction gate, so systems whose protected layout
no longer saves at least 5% fall back to uniform before capture.

Configuration and telemetry:

- `--neighbor-capacity-policy auto-safe`;
- `--neighbor-auto-guard-slots 1` (or `NEIGHBOR_AUTO_GUARD_SLOTS=1`);
- `neighbor_capacity_policy_effective=atom-safe|uniform`;
- unprotected and protected capacity reductions are both recorded;
- any capacity miss makes `graph_invariants_pass=false` and the performance
  sample ineligible.

Use `PROBE_STEPS=100`/`WHOLE_PROBE_STEPS=100` for the current 100-step
acceptance matrix.  The guard is a conservative fixed-shape policy, not a
mathematical guarantee for an arbitrarily long diffusive trajectory.

KF12 and CAP1-auto-safe can be compiled/captured in one smoke:

```bash
GPU=0 \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
OUTPUT_DIR=/path/to/kf12_cap1_safe_smoke \
bash example/smoke_opt4_kf12_cap1_safe.sh
```

The isolated CAP1-auto-safe polling ablation holds KF12 fixed on both sides:

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
ROOT_OUTPUT_DIR=/path/to/cap1_auto_safe_ablation \
bash example/run_opt4_capacity_auto_safe_8gpu.sh
```

The combined end-to-end matrix compares Opt4 v2 with KF12+CAP1-auto-safe:

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
ROOT_OUTPUT_DIR=/path/to/kf12_cap1_safe_formal \
bash example/run_opt4_kf12_cap1_safe_8gpu.sh
```

### Unit tests

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/src" \
python -m pytest -q --noconftest \
  tests/core/applications/test_esen_fixed_neighbor.py
```

### CAP1-auto one-step smoke

```bash
GPU=0 \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
OUTPUT_DIR=/path/to/cap1_auto_smoke \
bash example/smoke_opt4_capacity_auto.sh
```

The smoke writes `auto_decisions.tsv` in addition to the normal result and
status files.

### CAP1-auto 100-step polling ablation

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
GPU_IDLE_SECONDS=120 \
GPU_POLL_SECONDS=10 \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
ROOT_OUTPUT_DIR=/path/to/cap1_auto_8gpu \
nohup bash example/run_opt4_capacity_auto_8gpu.sh \
  > /path/to/cap1_auto_8gpu.log 2>&1 &
```

After the queue completes, select it with:

```bash
python example/select_opt4_model_fusions.py \
  --input-dir /path/to/cap1_auto_8gpu/whole_step \
  --scope whole-step \
  --base-stage OPT4V2 \
  --candidate-stage CAP1AUTO \
  --candidate-fusion auto-capacity \
  --accepted-before "" \
  --focus-systems H2O32 H2O192 \
  --min-paired-repeats 3 \
  --min-faster-directions 3 \
  --maximum-peak-reserved-increase-gib 1.0 \
  --output /path/to/cap1_auto_8gpu/CAP1AUTO_selection.json
```

The single-GPU equivalent is `run_opt4_capacity_auto_ablation.sh`.

### Profiling after acceptance

```bash
GPU=0 \
SYSTEMS="H2O32 H2O192" \
TRACE_STEPS=20 \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
OUTPUT_DIR=/path/to/cap1_auto_nsys \
bash example/run_opt4_capacity_auto_profiling.sh
```

## One-step smoke

```bash
GPU=0 \
SCOPE=whole-step \
STEPS=1 \
REPEATS=1 \
SYSTEMS="Cu32 Cu512 H2O32 H2O192" \
TEMPERATURES=300 \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
OUTPUT_DIR=/path/to/cap1_smoke \
BASE_STAGE=OPT4V2 \
BASE_FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge \
CANDIDATE_STAGE=CAP1 \
CANDIDATE_FUSIONS=rmsnorm,so2-epilogue,so2-gate-bridge \
BASE_NEIGHBOR_CAPACITY_POLICY=uniform \
CANDIDATE_NEIGHBOR_CAPACITY_POLICY=atom \
bash example/run_opt4_interleaved_stage.sh
```

Smoke requires complete JSON output, one capture, two production replays, no
capacity miss, and stable output addresses.  Energy and force are telemetry.

## 100-step single-GPU ablation

```bash
GPU=0 \
STEPS=100 \
REPEATS=3 \
SYSTEMS="Cu32 Cu512 H2O32 H2O192" \
FOCUS_SYSTEMS="H2O32 H2O192" \
TEMPERATURES=300 \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
ROOT_OUTPUT_DIR=/path/to/cap1_ablation \
bash example/run_opt4_capacity_ablation.sh
```

## Idle-GPU polling queue

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
GPU_IDLE_SECONDS=120 \
CHECKPOINT=/path/to/esen_30m_oam.pt \
STRUCTURE_DIR=/path/to/cif_file \
ROOT_OUTPUT_DIR=/path/to/cap1_8gpu \
nohup bash example/run_opt4_capacity_8gpu.sh \
  > /path/to/cap1_8gpu.log 2>&1 &
```

After the queue completes:

```bash
python example/select_opt4_model_fusions.py \
  --input-dir /path/to/cap1_8gpu/whole_step \
  --scope whole-step \
  --base-stage OPT4V2 \
  --candidate-stage CAP1 \
  --candidate-fusion atom-capacity \
  --accepted-before "" \
  --focus-systems H2O32 H2O192 \
  --min-paired-repeats 3 \
  --min-faster-directions 3 \
  --maximum-peak-reserved-increase-gib 1.0 \
  --output /path/to/cap1_8gpu/CAP1_selection.json
```

The candidate is accepted only with zero overflow/miss, healthy Graph
invariants, at least `1.01x` focus geomean speedup, and no stable regression
over 1% on Cu guardrails.  NSYS is collected only after timing acceptance.
