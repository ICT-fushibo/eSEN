# Opt4 KF14: SO2 prepare backward reduction

KF14 targets the `_so2_prepare_backward_kernel` identified in the Opt4 v3
NSYS traces.  Its radial-gradient addresses are private to each edge and
channel, but the original kernel uses an atomic add for every one of the 14
coefficient contributions.  KF14 assigns one Triton program to each edge,
reduces the nine distinct radial gradients in registers, and writes each
result once without an atomic operation.

The frozen Opt4 v3 FP32 masks are unchanged:

```text
model-only: so2-epilogue,so2-gate-bridge,so2-block-gemm
whole-step: rmsnorm,so2-epilogue,so2-gate-bridge,so2-block-gemm
```

The KF14 candidate adds `so2-prepare-backward-reduce`.  The fusion explicitly
requires both `so2-epilogue` and `so2-gate-bridge`; it replaces only the ten
conv1 prepare backward calls.  Forward, cuBLAS Linear operations, CAP1
auto-safe, and all frozen configurations retain their existing behavior.

## Test phases

- Smoke: Cu32, Cu512, H2O32, H2O192; 300 K; 1 step; 1 repeat.
- Ablation: the same systems; 300 K; 100 steps; 3 interleaved repeats.
- Formal: ten Cu/H2O systems; 300/800 K; 100 steps; 3 repeats.

The polling runner resumes from existing result JSON files and dispatches only
to GPUs that satisfy the configured idle window.  Energy and force validation
remain telemetry and do not determine timing-task completion.

The selector focuses on Cu512 and H2O192.  It requires a focus geomean speedup
of at least 1.01x, 3/3 faster paired repeats per focus system, no stable small
system regression above 1%, no graph/capacity failure, and no stable peak
reserved increase above 1 GiB.

After acceptance, profile Cu512 and H2O192 in NSYS graph/node modes.  Confirm
that `_so2_prepare_backward_kernel` is replaced by
`_so2_prepare_backward_reduce_kernel`, graph duration falls in the same
direction as the unprofiled benchmark, and no new kernel becomes dominant.
`example/run_opt4_kf14_profiling.sh` provides this post-acceptance comparison;
profiler timings never enter the selector or seconds-per-step result.
