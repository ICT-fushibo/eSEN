# Opt4 v5: ROB1 safety

Opt4 v5 freezes the accepted Opt4 v4 model fusion mask and adds transactional
rollback/replay to the whole-step CUDA Graph path.  The initial capacity is
the CAP1-auto-safe allocation, not CAP2's compact allocation.  Therefore a
normal trajectory has exactly the v4 graph and timing path.  If a fixed
neighbor capacity is exceeded, ROB1 snapshots the complete MD/NHC state,
discards the transaction, promotes capacity, recaptures one graph, and replays
the same physical steps.

The old Opt4 v1-v4 and CAP2 runners remain available.  Use the dedicated v5
entry point for new experiments:

```bash
OPT4_V5_PHASE=smoke GPU_LIST="0 1" \
  bash example/run_opt4_v5_8gpu.sh
```

The defaults are:

- model-only: v4 fusion mask, ROB1 not applicable;
- whole-step base: v4 fusion mask, CAP1-auto-safe, ROB1 disabled;
- whole-step candidate: same mask and capacity, ROB1 enabled;
- FP32 model, TF32 off, 3 repeats for the formal 10-system matrix.

ROB1 telemetry is recorded in each result under `graph_stats`, including
attempted/committed/discarded replays, rollback and recovery-capture counts,
promotion history, and the initial/final capacity.  A recovered overflow is a
successful run; `unrecovered_overflows` or a failed graph invariant is not.

For the Matbench protocol, use `run_esen_matbench_v5.sh`.  The generic
`run_esen_matbench.sh` remains v4-compatible so historical commands are
reproducible.
