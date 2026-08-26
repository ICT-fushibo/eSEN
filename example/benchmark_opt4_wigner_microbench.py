#!/usr/bin/env python3
"""Counter-free CUDA Graph microbenchmark for the KF11 Wigner/SO2 bridge.

This diagnostic compares the Opt4-v4 Edgewise producer sequence

    gather/cat -> Wigner BMM -> KF14 SO2 prepare

with the existing fused ``wigner_so2_prepare`` implementation.  It uses CUDA
Events around batches of CUDA Graph replays, so it does not require access to
GPU performance counters and its timings are not NCU replay timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from fairchem.core.applications.esen_opt4_model_fusion import (
    _SO2PrepareBackwardReduce,
    model_fusion_available,
    wigner_so2_prepare,
)
from fairchem.core.models.esen.common.so3 import CoefficientMapping


SYSTEM_SHAPES = {
    "Cu512": (512, 44032),
    "H2O192": (576, 55552),
}
OUT_MASK = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=tuple(SYSTEM_SHAPES), required=True)
    parser.add_argument("--variant", choices=("base", "bridge"), required=True)
    parser.add_argument(
        "--mode", choices=("forward", "forward-backward"), required=True
    )
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--edges", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.replays < 1 or args.repeats < 1:
        parser.error("warmup, replays, and repeats must be positive")
    return args


def make_reference(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    out_mask: torch.Tensor,
    radial: torch.Tensor,
    to_m: torch.Tensor,
):
    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gathered = torch.cat(
            (x.index_select(0, edge_index[0]), x.index_select(0, edge_index[1])),
            dim=2,
        )
        rotated = torch.bmm(wigner.index_select(1, out_mask), gathered)
        # This is the exact SO2-prepare autograd implementation enabled by
        # Opt4-v4/KF14.  Keeping it here isolates only the KF11 producer bridge
        # instead of comparing against a different eager permutation path.
        return _SO2PrepareBackwardReduce.apply(
            rotated, radial, to_m, True, 2304
        )

    return forward


def make_bridge(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    out_mask: torch.Tensor,
    radial: torch.Tensor,
    to_m: torch.Tensor,
):
    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return wigner_so2_prepare(
            x, edge_index, wigner, out_mask, radial, to_m
        )

    return forward


def capture(
    forward,
    inputs: tuple[torch.Tensor, ...],
    grad_outputs: tuple[torch.Tensor, ...] | None,
    warmup: int,
):
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(warmup):
            outputs = forward()
            if grad_outputs is not None:
                torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        outputs = forward()
        input_gradients = (
            torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs)
            if grad_outputs is not None
            else ()
        )
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    return graph, tuple(outputs), tuple(input_gradients)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.variant == "bridge" and not model_fusion_available():
        raise RuntimeError("The Triton model-fusion runtime is unavailable")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    default_nodes, default_edges = SYSTEM_SHAPES[args.system]
    nodes = args.nodes or default_nodes
    edges = args.edges or default_edges
    requires_grad = args.mode == "forward-backward"
    device = torch.device("cuda")

    mapping = CoefficientMapping(3, 2).to(device)
    to_m = mapping.to_m.argmax(dim=1).to(dtype=torch.long).contiguous()
    out_mask = torch.tensor(OUT_MASK, device=device, dtype=torch.long)
    edge_index = torch.randint(
        0, nodes, (2, edges), device=device, dtype=torch.long
    )
    x = torch.randn(
        nodes, 16, 128, device=device, requires_grad=requires_grad
    )
    wigner = torch.randn(
        edges, 16, 16, device=device, requires_grad=requires_grad
    )
    radial = torch.randn(
        edges, 2304, device=device, requires_grad=requires_grad
    )
    inputs = (x, wigner, radial)

    factory = make_reference if args.variant == "base" else make_bridge
    forward = factory(x, edge_index, wigner, out_mask, radial, to_m)
    sample_outputs = forward()
    grad_outputs = (
        tuple(torch.randn_like(output) for output in sample_outputs)
        if requires_grad
        else None
    )
    del sample_outputs
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    graph, outputs, input_gradients = capture(
        forward, inputs, grad_outputs, args.warmup
    )
    allocated_after_capture = torch.cuda.memory_allocated()
    reserved_after_capture = torch.cuda.memory_reserved()

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    addresses_before = tuple(t.data_ptr() for t in outputs + input_gradients)
    samples_ms: list[float] = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / args.replays)
    addresses_after = tuple(t.data_ptr() for t in outputs + input_gradients)
    output_checksums = [float(t.double().sum()) for t in outputs]
    gradient_checksums = [float(t.double().sum()) for t in input_gradients]

    median_ms = statistics.median(samples_ms)
    mad_ms = statistics.median(abs(value - median_ms) for value in samples_ms)
    wigner_flops = 2 * edges * 14 * 16 * 256
    wigner_minimum_bytes = 4 * edges * (
        14 * 16 + 16 * 256 + 14 * 256
    )
    record = {
        "system": args.system,
        "variant": args.variant,
        "mode": args.mode,
        "nodes": nodes,
        "edges": edges,
        "warmup": args.warmup,
        "replays_per_repeat": args.replays,
        "repeats": args.repeats,
        "samples_ms_per_replay": samples_ms,
        "median_ms_per_replay": median_ms,
        "mad_ms_per_replay": mad_ms,
        "addresses_stable": addresses_before == addresses_after,
        "output_checksums": output_checksums,
        "gradient_checksums": gradient_checksums,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "capture_allocated_delta_gib": (
            allocated_after_capture - allocated_before
        ) / 1024**3,
        "capture_reserved_delta_gib": (
            reserved_after_capture - reserved_before
        ) / 1024**3,
        "wigner_bmm_theoretical_flops": wigner_flops,
        "wigner_bmm_minimum_bytes": wigner_minimum_bytes,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tf32": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"Result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
