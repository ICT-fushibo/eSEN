#!/usr/bin/env python3
"""Profile Opt3 CUDA Graph capture-scope ablations.

This entrypoint is diagnostic only.  It never changes the production Opt3
benchmark and keeps setup, probe, capture, and profiler warmup outside the
reported production timing window.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

from benchmark_md_gpu import (
    EXPECTED_ATOMS,
    capture_rng_state,
    git_commit,
    package_version,
    restore_rng_state,
    sha256,
)
from benchmark_md_opt3 import _engineering_energy_validation
from md_energy_reference import (
    REQUIRED_SEED,
    checkpoint_energy_fields,
    compare_checkpoint_energies,
    load_baseline_reference,
    reached_energy_checkpoints,
    seed_everything,
    validate_reference_metadata,
)


REPO = Path(__file__).resolve().parent.parent
BACKENDS = (
    "static-eager-breakdown",
    "fixed-builder-model-cg",
    "builder-cg-model-cg",
    "force-eval-cg",
    "whole-step-cg",
)


def append_profile_tsv(path: Path, record: dict[str, Any]) -> None:
    """Append heterogeneous backend records with one stable union schema."""

    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    if path.is_file() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields.extend(reader.fieldnames or [])
            rows.extend(reader)
    for field in record:
        if field not in fields:
            fields.append(field)
    rows.append(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def tensor_sha256(tensor) -> str:
    value = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument(
        "--profile-kind",
        choices=("timing", "smoke", "torch-profiler", "external-profiler"),
        default="timing",
    )
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO / "esen_30m_oam.pt"
    )
    parser.add_argument("--system", required=True, choices=EXPECTED_ATOMS)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--component-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--probe-steps", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=REQUIRED_SEED)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--taut", type=float, default=100.0)
    parser.add_argument("--neighbor-margin", type=float, default=0.10)
    parser.add_argument("--neighbor-slot-step", type=int, default=8)
    parser.add_argument("--dummy-atoms", type=int, default=32)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--max-neighbors", type=int, default=300)
    parser.add_argument("--degeneracy-tolerance", type=float, default=0.01)
    parser.add_argument("--energy-per-atom-atol", type=float, default=1e-5)
    parser.add_argument("--force-max-atol", type=float, default=2e-4)
    parser.add_argument("--baseline-result", type=Path, default=None)
    parser.add_argument("--missing-baseline-reference", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.component_steps < 0:
        parser.error("steps must be positive and component-steps non-negative")
    if args.warmup_steps < 0 or args.probe_steps < 0:
        parser.error("warmup/probe steps must be non-negative")
    if args.seed != REQUIRED_SEED:
        parser.error(f"--seed must be {REQUIRED_SEED}")
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.baseline_result is not None and args.missing_baseline_reference:
        parser.error("baseline path and missing-reference flag are exclusive")
    return args


class SegmentRecorder:
    """Collect asynchronous CUDA Event timings without per-step sync."""

    def __init__(self, torch_module, *, events: bool, ranges: bool) -> None:
        self.torch = torch_module
        self.events = events
        self.ranges = ranges
        self.records: dict[str, list[tuple[Any, Any]]] = {}

    def call(self, name: str, function: Callable[[], Any]) -> Any:
        start = end = None
        if self.events:
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
        context = (
            self.torch.autograd.profiler.record_function(f"opt3::{name}")
            if self.ranges
            else nullcontext()
        )
        if self.ranges:
            self.torch.cuda.nvtx.range_push(f"opt3::{name}")
        try:
            with context:
                return function()
        finally:
            if self.ranges:
                self.torch.cuda.nvtx.range_pop()
            if self.events:
                assert start is not None and end is not None
                end.record()
                self.records.setdefault(name, []).append((start, end))

    def summarize(self) -> dict[str, float]:
        if not self.events:
            return {}
        self.torch.cuda.synchronize()
        result = {}
        for name, pairs in self.records.items():
            values = [start.elapsed_time(end) for start, end in pairs]
            result[f"component_{name}_total_ms"] = sum(values)
            result[f"component_{name}_mean_ms"] = sum(values) / len(values)
            result[f"component_{name}_calls"] = len(values)
        return result


class ForceMD:
    """Exact eager NVT loop with a profiling force-evaluation adapter."""

    def __init__(self, state, integrator, adapter) -> None:
        self.state = state
        self.integrator = integrator
        self.adapter = adapter

    def evaluate(self, recorder: SegmentRecorder) -> tuple[Any, Any]:
        forces, energy = self.adapter.evaluate(self.state.positions, recorder)
        self.state.forces = forces.to(dtype=self.state.positions.dtype)
        self.state.potential_energy = energy
        return self.state.forces, energy

    def step(self, recorder: SegmentRecorder) -> tuple[Any, Any]:
        from fairchem.core.applications.esen_opt3_profiling import (
            eager_nvt_post,
            eager_nvt_pre,
        )

        assert self.state.forces is not None
        half_momenta, positions = recorder.call(
            "nvt_pre",
            lambda: eager_nvt_pre(
                self.state.positions,
                self.state.momenta,
                self.state.forces,
                self.integrator,
            ),
        )
        model_forces, energy = self.adapter.evaluate(positions, recorder)
        forces = model_forces.to(dtype=positions.dtype)
        momenta = recorder.call(
            "nvt_post",
            lambda: eager_nvt_post(half_momenta, forces, self.integrator),
        )
        self.state.positions = positions
        self.state.momenta = momenta
        self.state.forces = forces
        self.state.potential_energy = energy
        return forces, energy


class ForceAdapter:
    def __init__(self, backend: str, evaluator) -> None:
        self.backend = backend
        self.evaluator = evaluator

    def evaluate(self, positions, recorder: SegmentRecorder):
        if self.backend == "static-eager-breakdown":
            recorder.call(
                "position_handoff",
                lambda: self.evaluator.copy_positions(positions),
            )
            recorder.call("fixed_builder", self.evaluator.build)
            return recorder.call("model_eager", self.evaluator.model_forward)
        if self.backend == "fixed-builder-model-cg":
            def copy_positions():
                import torch

                with torch.no_grad():
                    self.evaluator.static_positions[
                        : self.evaluator.num_atoms
                    ].copy_(positions)

            recorder.call(
                "position_handoff",
                copy_positions,
            )
            recorder.call(
                "fixed_builder",
                lambda: self.evaluator.fixed_builder.build(
                    self.evaluator.static_positions[
                        : self.evaluator.num_atoms
                    ]
                ),
            )

            def replay_model():
                assert self.evaluator.graph is not None
                self.evaluator.graph.replay()
                self.evaluator.total_replays += 1
                self.evaluator.production_replays += 1
                self.evaluator.production_calls += 1
                return (
                    self.evaluator.static_forces,
                    self.evaluator.static_energy,
                )

            forces, energy = recorder.call("model_graph", replay_model)
            assert forces is not None and energy is not None
            return forces, energy
        if self.backend == "builder-cg-model-cg":
            recorder.call(
                "position_handoff",
                lambda: self.evaluator.copy_positions(positions),
            )
            recorder.call("builder_graph", self.evaluator.replay_builder)
            return recorder.call("model_graph", self.evaluator.replay_model)
        if self.backend == "force-eval-cg":
            recorder.call(
                "position_handoff",
                lambda: self.evaluator.copy_positions(positions),
            )
            return recorder.call(
                "force_eval_graph", self.evaluator.replay_force_eval
            )
        raise ValueError(f"Unsupported force adapter backend: {self.backend}")

    def reset(self) -> None:
        if self.backend == "fixed-builder-model-cg":
            self.evaluator.reset_production_stats()
        else:
            self.evaluator.reset_production_stats()

    def stats(self) -> dict[str, Any]:
        return self.evaluator.stats()


def _make_backend(torch, args, evaluator, capacity, state, integrator):
    from fairchem.core.applications.esen_opt3_profiling import (
        ESENBuilderGraphModelGraphEvaluator,
        ESENForceEvalCUDAGraphEvaluator,
        ESENStaticEagerProfilingEvaluator,
    )
    from fairchem.core.applications.esen_whole_step_cuda_graph import (
        ESENFixedBuilderModelCUDAGraphEvaluator,
        ESENWholeStepCUDAGraphMD,
    )

    common = dict(
        neighbors_per_atom=capacity,
        dummy_atoms=args.dummy_atoms,
        max_neighbors=args.max_neighbors,
        degeneracy_tolerance=args.degeneracy_tolerance,
    )
    if args.backend == "static-eager-breakdown":
        profile_evaluator = ESENStaticEagerProfilingEvaluator(
            evaluator, **common
        )
        return ForceMD(
            state, integrator, ForceAdapter(args.backend, profile_evaluator)
        )
    if args.backend == "fixed-builder-model-cg":
        profile_evaluator = ESENFixedBuilderModelCUDAGraphEvaluator(
            evaluator, capture_warmup=args.capture_warmup, **common
        )
        profile_evaluator.capture(state.positions)
        return ForceMD(
            state, integrator, ForceAdapter(args.backend, profile_evaluator)
        )
    if args.backend == "builder-cg-model-cg":
        profile_evaluator = ESENBuilderGraphModelGraphEvaluator(
            evaluator, capture_warmup=args.capture_warmup, **common
        )
        profile_evaluator.capture(state.positions)
        return ForceMD(
            state, integrator, ForceAdapter(args.backend, profile_evaluator)
        )
    if args.backend == "force-eval-cg":
        profile_evaluator = ESENForceEvalCUDAGraphEvaluator(
            evaluator, capture_warmup=args.capture_warmup, **common
        )
        profile_evaluator.capture(state.positions)
        return ForceMD(
            state, integrator, ForceAdapter(args.backend, profile_evaluator)
        )
    return ESENWholeStepCUDAGraphMD(
        state,
        evaluator,
        integrator,
        capture_warmup=args.capture_warmup,
        **common,
    )


def _reset_backend(backend, initial_state, backend_name: str) -> None:
    if backend_name == "whole-step-cg":
        if not backend.captured:
            backend.capture(initial_state)
        backend.reset_production(initial_state)
    else:
        backend.state.restore_(initial_state)
        backend.adapter.reset()


def _run_backend(
    backend,
    backend_name: str,
    steps: int,
    recorder: SegmentRecorder,
) -> tuple[dict[int, Any], Any]:
    checkpoints = set(reached_energy_checkpoints(steps))
    energies = {}
    if backend_name == "whole-step-cg":
        backend.evaluate_initial()
        for step in range(1, steps + 1):
            _, energy = recorder.call(
                "total_step",
                lambda: recorder.call("whole_step_graph", backend.step),
            )
            if step in checkpoints:
                energies[step] = energy.detach().clone()
        return energies, backend.state_view()

    backend.evaluate(recorder)
    for step in range(1, steps + 1):
        _, energy = recorder.call(
            "total_step", lambda: backend.step(recorder)
        )
        if step in checkpoints:
            energies[step] = energy.detach().clone()
    return energies, backend.state


def _validate_fixed_graph(backend, backend_name: str, eager_evaluator, positions):
    """Compare the fixed-builder active edge sequence with official eSEN."""

    import torch

    if backend_name == "whole-step-cg":
        fixed_builder = backend.fixed_builder
        edge_index = backend.core.static_edge_index
        cell_offsets = backend.core.static_cell_offsets
    else:
        profile_evaluator = backend.adapter.evaluator
        core = getattr(profile_evaluator, "core", profile_evaluator)
        fixed_builder = core.fixed_builder
        edge_index = core.static_edge_index
        cell_offsets = core.static_cell_offsets
    if edge_index is None or cell_offsets is None:
        raise RuntimeError("Fixed builder output buffers are uninitialized")

    official = eager_evaluator.build_neighbor_graph(positions)
    real = edge_index[1] < fixed_builder.num_atoms
    actual_edges = edge_index[:, real]
    actual_offsets = cell_offsets[real]
    official_edges = official["edge_index"]
    official_offsets = official["cell_offsets"].to(
        device=actual_offsets.device, dtype=actual_offsets.dtype
    )
    edge_match = bool(torch.equal(actual_edges, official_edges))
    offset_match = bool(torch.equal(actual_offsets, official_offsets))
    return {
        "fixed_builder_edge_sequence_matches_official": edge_match,
        "fixed_builder_cell_offsets_match_official": offset_match,
        "fixed_builder_graph_matches_official": edge_match and offset_match,
        "fixed_builder_validation_real_edges": int(actual_edges.shape[1]),
        "official_validation_real_edges": int(official_edges.shape[1]),
    }


def _graph_invariants(backend_name: str, stats: dict[str, Any], steps: int):
    """Validate the capture/replay contract for one timed trajectory."""

    force_evaluations = steps + 1
    expected = {
        "static-eager-breakdown": (0, 0),
        "fixed-builder-model-cg": (1, force_evaluations),
        "builder-cg-model-cg": (2, force_evaluations),
        "force-eval-cg": (1, force_evaluations),
        "whole-step-cg": (1, force_evaluations),
    }
    expected_captures, expected_replays = expected[backend_name]
    captures = int(stats.get("cuda_graph_capture_count", -1))
    replays = int(stats.get("cuda_graph_production_replays", -1))
    production_captures = int(
        stats.get("cuda_graph_production_capture_count", 0)
    )
    capacity_misses = int(stats.get("cuda_graph_capacity_misses", 0))
    calls = int(stats.get("cuda_graph_production_calls", -1))
    pass_value = bool(
        captures == expected_captures
        and replays == expected_replays
        and calls == force_evaluations
        and production_captures == 0
        and capacity_misses == 0
    )
    if backend_name == "builder-cg-model-cg":
        pass_value = pass_value and bool(
            int(stats.get("cuda_graph_builder_production_replays", -1))
            == force_evaluations
            and int(stats.get("cuda_graph_model_production_replays", -1))
            == force_evaluations
        )
    if backend_name != "static-eager-breakdown":
        pass_value = pass_value and bool(
            stats.get("cuda_graph_replay_output_addresses_stable", False)
        )
    return {
        "graph_invariants_pass": pass_value,
        "graph_expected_capture_count": expected_captures,
        "graph_expected_production_replays": expected_replays,
        "graph_expected_production_calls": force_evaluations,
    }


def _start_external_profiler(torch) -> None:
    result = torch.cuda.cudart().cudaProfilerStart()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStart failed: {result}")


def _stop_external_profiler(torch) -> None:
    result = torch.cuda.cudart().cudaProfilerStop()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStop failed: {result}")


def _device_memory_used(torch, device) -> int | None:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError):
        return None
    return int(total_bytes - free_bytes)


def main() -> int:
    process_start = time.perf_counter()
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(REQUIRED_SEED):
        raise RuntimeError(f"Launch with PYTHONHASHSEED={REQUIRED_SEED}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(REPO / "src"))

    import torch
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from fairchem.core.applications.esen_fixed_neighbor import (
        maximum_neighbors_in_graph,
        neighbor_capacity_from_probe,
    )
    from fairchem.core.applications.esen_gpu_md import (
        ESENEnergyForceEvaluator,
        GPUIntegrator,
        GPUMDState,
        GPUResidentMD,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.structure.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("structure or checkpoint file is missing")
    seed_everything(torch, args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    atoms = read(args.structure)
    if len(atoms) != EXPECTED_ATOMS[args.system]:
        raise ValueError("structure atom count does not match system")
    checkpoint_hash = sha256(args.checkpoint)
    structure_hash = sha256(args.structure)
    baseline = None
    if args.baseline_result is not None:
        baseline = load_baseline_reference(args.baseline_result)
        validate_reference_metadata(
            baseline,
            {
                "system": args.system,
                "atoms": len(atoms),
                "temperature_K": args.temperature,
                "timestep_fs": args.timestep,
                "taut_fs": args.taut,
                "seed": args.seed,
                "repeat": args.repeat,
                "checkpoint_sha256": checkpoint_hash,
                "structure_sha256": structure_hash,
            },
        )

    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=args.temperature, force_temp=True, rng=rng
    )
    Stationary(atoms)
    device = torch.device("cuda:0")
    state = GPUMDState(
        positions=torch.as_tensor(
            atoms.get_positions(), dtype=torch.float64, device=device
        ).clone(),
        momenta=torch.as_tensor(
            atoms.get_momenta(), dtype=torch.float64, device=device
        ).clone(),
    )
    masses = torch.as_tensor(
        atoms.get_masses(), dtype=torch.float64, device=device
    )
    evaluator = ESENEnergyForceEvaluator(
        atoms,
        args.checkpoint,
        device=device,
        seed=args.seed,
        disable_amp=True,
    )
    integrator = GPUIntegrator(
        masses,
        timestep_fs=args.timestep,
        temperature_K=args.temperature,
        taut_fs=args.taut,
        fix_com=True,
        degrees_of_freedom=atoms.get_number_of_degrees_of_freedom(),
    )
    initial_state = state.clone()

    probe_rng = capture_rng_state(torch)
    eager_probe = GPUResidentMD(state, evaluator, integrator)
    probe_degrees = [
        maximum_neighbors_in_graph(
            evaluator.build_neighbor_graph(state.positions)["edge_index"],
            len(atoms),
        )
    ]
    for _ in range(args.probe_steps):
        eager_probe.run(1)
        graph = evaluator.build_neighbor_graph(state.positions)
        probe_degrees.append(
            maximum_neighbors_in_graph(graph["edge_index"], len(atoms))
        )
    torch.cuda.synchronize()
    capacity = neighbor_capacity_from_probe(
        max(probe_degrees),
        margin=args.neighbor_margin,
        slot_step=args.neighbor_slot_step,
    )
    state.restore_(initial_state)
    restore_rng_state(torch, probe_rng)
    eager_forces, eager_energy = evaluator(state.positions)
    eager_forces = eager_forces.detach().clone()
    eager_energy = eager_energy.detach().clone()
    torch.cuda.synchronize()
    setup_rng = capture_rng_state(torch)

    torch.cuda.synchronize()
    setup_allocated_before = torch.cuda.memory_allocated(device)
    setup_reserved_before = torch.cuda.memory_reserved(device)
    setup_device_used_before = _device_memory_used(torch, device)
    setup_start = time.perf_counter()
    backend = _make_backend(torch, args, evaluator, capacity, state, integrator)
    if args.backend == "whole-step-cg":
        backend.capture(initial_state)
    torch.cuda.synchronize()
    setup_wall_time = time.perf_counter() - setup_start
    setup_allocated_delta = (
        torch.cuda.memory_allocated(device) - setup_allocated_before
    )
    setup_reserved_delta = (
        torch.cuda.memory_reserved(device) - setup_reserved_before
    )
    setup_device_used_after = _device_memory_used(torch, device)

    # Trajectory-neutral warmup.
    _reset_backend(backend, initial_state, args.backend)
    warmup_recorder = SegmentRecorder(torch, events=False, ranges=False)
    _run_backend(backend, args.backend, args.warmup_steps, warmup_recorder)
    torch.cuda.synchronize()
    restore_rng_state(torch, setup_rng)

    # Initial-force validation after restoring the original state.
    _reset_backend(backend, initial_state, args.backend)
    validation_recorder = SegmentRecorder(torch, events=False, ranges=False)
    if args.backend == "whole-step-cg":
        initial_forces, initial_energy = backend.evaluate_initial()
    else:
        initial_forces, initial_energy = backend.evaluate(validation_recorder)
    torch.cuda.synchronize()
    edge_validation = _validate_fixed_graph(
        backend, args.backend, evaluator, initial_state.positions
    )
    initial_force_error = float(
        (initial_forces - eager_forces).abs().max().item()
    )
    initial_energy_error = abs(
        float(initial_energy.item()) - float(eager_energy.item())
    )
    restore_rng_state(torch, setup_rng)

    _reset_backend(backend, initial_state, args.backend)
    recorder = SegmentRecorder(
        torch,
        events=False,
        ranges=args.profile_kind in ("torch-profiler", "external-profiler"),
    )
    profiler = None
    if args.profile_kind == "torch-profiler":
        if (
            torch.profiler.ProfilerActivity.CUDA
            not in torch.profiler.supported_activities()
        ):
            raise RuntimeError("PyTorch profiler CUDA activity is unavailable")
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profiler.__enter__()
    if args.profile_kind == "external-profiler":
        _start_external_profiler(torch)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    timed_start = time.perf_counter()
    checkpoint_tensors, final_state = _run_backend(
        backend, args.backend, args.steps, recorder
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - timed_start

    if args.profile_kind == "external-profiler":
        _stop_external_profiler(torch)
    if profiler is not None:
        profiler.__exit__(None, None, None)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(
            str(args.output_dir / f"{args.run_name}.torch_trace.json")
        )
        (args.output_dir / f"{args.run_name}.torch_profile.txt").write_text(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=200
            )
            + "\n",
            encoding="utf-8",
        )

    checkpoint_energies = {
        step: float(value.item()) for step, value in checkpoint_tensors.items()
    }
    if final_state.forces is None or final_state.potential_energy is None:
        raise RuntimeError("profiling MD completed without energy or forces")
    timed_final_positions = final_state.positions.detach().clone()
    timed_final_momenta = final_state.momenta.detach().clone()
    timed_final_forces = final_state.forces.detach().clone()
    timed_final_energy = final_state.potential_energy.detach().clone()
    stats = (
        backend.stats()
        if args.backend == "whole-step-cg"
        else backend.adapter.stats()
    )
    graph_fields = _graph_invariants(args.backend, stats, args.steps)
    component_fields: dict[str, Any] = {}
    if args.profile_kind in ("timing", "smoke") and args.component_steps:
        _reset_backend(backend, initial_state, args.backend)
        component_recorder = SegmentRecorder(torch, events=True, ranges=False)
        _run_backend(
            backend,
            args.backend,
            min(args.component_steps, args.steps),
            component_recorder,
        )
        component_fields = component_recorder.summarize()

    finite = (
        torch.isfinite(timed_final_positions).all()
        & torch.isfinite(timed_final_momenta).all()
        & torch.isfinite(timed_final_forces).all()
        & torch.isfinite(timed_final_energy).all()
    )
    if not bool(finite.item()):
        raise FloatingPointError("profiling MD produced NaN or Inf")

    legacy_fields: dict[str, Any] = {}
    legacy_pass = None
    if baseline is not None:
        legacy_fields, legacy_pass = compare_checkpoint_energies(
            checkpoint_energies, baseline
        )
    engineering_fields, energy_pass = _engineering_energy_validation(
        checkpoint_energies,
        baseline,
        atoms=len(atoms),
        atol_per_atom=args.energy_per_atom_atol,
    )
    force_pass = initial_force_error < args.force_max_atol
    capacity_misses = int(stats.get("cuda_graph_capacity_misses", 0))
    graph_pass = bool(graph_fields["graph_invariants_pass"])
    edge_pass = bool(edge_validation["fixed_builder_graph_matches_official"])
    engineering_pass = (
        None
        if energy_pass is None
        and force_pass
        and graph_pass
        and edge_pass
        and capacity_misses == 0
        else bool(
            force_pass
            and energy_pass is not False
            and graph_pass
            and edge_pass
            and capacity_misses == 0
        )
    )

    record: dict[str, Any] = {
        "backend": args.backend,
        "run_name": args.run_name,
        "profile_kind": args.profile_kind,
        "system": args.system,
        "atoms": len(atoms),
        "temperature_K": args.temperature,
        "steps": args.steps,
        "component_profile_steps": min(args.component_steps, args.steps),
        "repeat": args.repeat,
        "seed": args.seed,
        "seconds_per_step": elapsed / args.steps,
        "md_wall_time_s": elapsed,
        "process_wall_time_s": time.perf_counter() - process_start,
        "setup_wall_time_s": setup_wall_time,
        "profiling_setup_allocated_delta_gib": setup_allocated_delta / 1024**3,
        "profiling_setup_reserved_delta_gib": setup_reserved_delta / 1024**3,
        "profiling_setup_device_used_delta_gib": (
            None
            if setup_device_used_before is None or setup_device_used_after is None
            else (setup_device_used_after - setup_device_used_before) / 1024**3
        ),
        "probe_max_neighbors_per_atom": max(probe_degrees),
        "neighbor_capacity_per_atom": capacity,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "initial_eager_energy_abs_error_eV": initial_energy_error,
        "initial_eager_force_max_abs_error_eV_per_A": initial_force_error,
        "engineering_force_validation_pass": force_pass,
        "engineering_validation_pass": engineering_pass,
        "engineering_validation_status": (
            "missing_reference"
            if engineering_pass is None
            else ("passed" if engineering_pass else "failed")
        ),
        "legacy_energy_validation_pass": legacy_pass,
        "capacity_overflow": capacity_misses > 0,
        "final_energy_eV": float(timed_final_energy.item()),
        "final_max_force_eV_per_A": float(timed_final_forces.abs().max().item()),
        "final_positions_sha256": tensor_sha256(timed_final_positions),
        "final_momenta_sha256": tensor_sha256(timed_final_momenta),
        "final_forces_sha256": tensor_sha256(timed_final_forces),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "fairchem_core_version": package_version("fairchem-core"),
        "repo_commit": git_commit(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "structure": str(args.structure.resolve()),
        "structure_sha256": structure_hash,
        "baseline_result": (
            "" if args.baseline_result is None else str(args.baseline_result.resolve())
        ),
    }
    record.update(stats)
    record.update(graph_fields)
    record.update(edge_validation)
    record.update(component_fields)
    record.update(checkpoint_energy_fields(checkpoint_energies))
    record.update(legacy_fields)
    record.update(engineering_fields)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.run_name}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_profile_tsv(args.output_dir / "profile_runs.tsv", record)
    print(json.dumps(record, indent=2))
    print(f"Result: {json_path}")

    if capacity_misses:
        print("PROFILE_STATUS=capacity_overflow", file=sys.stderr)
        return 45
    if engineering_pass is False:
        print("PROFILE_STATUS=validation_failed", file=sys.stderr)
        return 43
    return 0


def entrypoint() -> int:
    try:
        return main()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"PROFILE_STATUS=oom: {exc}", file=sys.stderr)
            return 42
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())
