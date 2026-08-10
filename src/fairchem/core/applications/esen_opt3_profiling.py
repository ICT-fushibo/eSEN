"""Profiling-only Opt3 capture-scope ablations.

These helpers deliberately live outside the production Opt3 implementation.
They reuse the same fixed-shape neighbor builder and padded eSEN model while
exposing the position handoff, builder replay, and model replay separately so
that profiling can attribute a regression to one capture-scope transition.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from fairchem.core.applications.esen_gpu_md import ESENEnergyForceEvaluator
from fairchem.core.applications.esen_whole_step_cuda_graph import (
    ESENFixedBuilderModelCUDAGraphEvaluator,
)


def eager_nvt_pre(
    positions: Tensor,
    momenta: Tensor,
    forces: Tensor,
    integrator,
) -> tuple[Tensor, Tensor]:
    """Exact pre-force half of :meth:`GPUIntegrator.step`."""

    half_momenta = integrator.scale_velocities(momenta)
    half_momenta = half_momenta + 0.5 * integrator.dt * forces
    if integrator.fix_com:
        half_momenta = half_momenta - half_momenta.sum(
            dim=0, keepdim=True
        ) / float(half_momenta.shape[0])
    next_positions = (
        positions + integrator.dt * half_momenta / integrator.masses
    )
    return half_momenta, next_positions


def eager_nvt_post(
    half_momenta: Tensor,
    new_forces: Tensor,
    integrator,
) -> Tensor:
    """Exact post-force half of :meth:`GPUIntegrator.step`."""

    return half_momenta + 0.5 * integrator.dt * new_forces


def _capture_on_stream(
    device: torch.device,
    warmup,
    body,
    *,
    warmup_steps: int,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream, float, int, int]:
    """Warm and capture a profiling graph on one stable side stream."""

    current = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(warmup_steps):
            warmup()
    current.wait_stream(stream)
    torch.cuda.synchronize(device)

    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    capture_start = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        body()
    torch.cuda.synchronize(device)
    return (
        graph,
        stream,
        time.perf_counter() - capture_start,
        torch.cuda.memory_allocated(device) - allocated_before,
        torch.cuda.memory_reserved(device) - reserved_before,
    )


class ESENStaticEagerProfilingEvaluator:
    """Fixed builder and padded model without CUDA Graph capture."""

    def __init__(
        self,
        eager_evaluator: ESENEnergyForceEvaluator,
        *,
        neighbors_per_atom: int,
        dummy_atoms: int = 32,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
    ) -> None:
        self.core = ESENFixedBuilderModelCUDAGraphEvaluator(
            eager_evaluator,
            neighbors_per_atom=neighbors_per_atom,
            dummy_atoms=dummy_atoms,
            capture_warmup=0,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        self.core.model.backbone.otf_graph = False
        self.num_atoms = self.core.num_atoms
        self.model_dtype = self.core.model_dtype
        self.force_evaluations = 0

    @torch.no_grad()
    def copy_positions(self, positions: Tensor) -> None:
        self.core.static_positions[: self.num_atoms].copy_(positions)

    @torch.no_grad()
    def build(self) -> None:
        self.core.fixed_builder.build(
            self.core.static_positions[: self.num_atoms]
        )

    def model_forward(self) -> tuple[Tensor, Tensor]:
        forces, energy = self.core._static_forward()
        self.force_evaluations += 1
        return forces, energy

    def reset_production_stats(self) -> None:
        self.force_evaluations = 0
        self.core.fixed_builder.reset_stats()

    def stats(self) -> dict[str, Any]:
        return {
            **self.core.fixed_builder.stats(),
            "profiling_capture_scope": "static_eager",
            "cuda_graph_capture_count": 0,
            "cuda_graph_production_replays": 0,
            "cuda_graph_production_calls": self.force_evaluations,
            "cuda_graph_capacity_misses": int(
                self.core.fixed_builder.capacity_misses.item()
            ),
            "cuda_graph_capture_allocated_delta_gib": 0.0,
            "cuda_graph_capture_reserved_delta_gib": 0.0,
        }


class ESENBuilderGraphModelGraphEvaluator:
    """Fixed builder graph followed by the existing model-only graph."""

    def __init__(
        self,
        eager_evaluator: ESENEnergyForceEvaluator,
        *,
        neighbors_per_atom: int,
        dummy_atoms: int = 32,
        capture_warmup: int = 3,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
    ) -> None:
        self.core = ESENFixedBuilderModelCUDAGraphEvaluator(
            eager_evaluator,
            neighbors_per_atom=neighbors_per_atom,
            dummy_atoms=dummy_atoms,
            capture_warmup=capture_warmup,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        self.device = self.core.device
        self.num_atoms = self.core.num_atoms
        self.model_dtype = self.core.model_dtype
        self.capture_warmup = int(capture_warmup)
        self.builder_graph: torch.cuda.CUDAGraph | None = None
        self.builder_stream: torch.cuda.Stream | None = None
        self.builder_capture_wall_time_s = 0.0
        self.builder_capture_allocated_delta_bytes = 0
        self.builder_capture_reserved_delta_bytes = 0
        self.builder_replays = 0
        self.model_replays = 0

    def capture(self, positions: Tensor) -> None:
        self.core.capture(positions)

        def build() -> None:
            self.core.fixed_builder.build(
                self.core.static_positions[: self.num_atoms]
            )

        (
            self.builder_graph,
            self.builder_stream,
            self.builder_capture_wall_time_s,
            self.builder_capture_allocated_delta_bytes,
            self.builder_capture_reserved_delta_bytes,
        ) = _capture_on_stream(
            self.device,
            build,
            build,
            warmup_steps=self.capture_warmup,
        )
        self.core.fixed_builder.reset_stats()

    @torch.no_grad()
    def copy_positions(self, positions: Tensor) -> None:
        self.core.static_positions[: self.num_atoms].copy_(positions)

    def replay_builder(self) -> None:
        if self.builder_graph is None:
            raise RuntimeError("Builder graph has not been captured")
        self.builder_graph.replay()
        self.builder_replays += 1

    def replay_model(self) -> tuple[Tensor, Tensor]:
        if self.core.graph is None:
            raise RuntimeError("Model graph has not been captured")
        self.core.graph.replay()
        self.model_replays += 1
        assert self.core.static_forces is not None
        assert self.core.static_energy is not None
        return self.core.static_forces, self.core.static_energy

    def reset_production_stats(self) -> None:
        self.builder_replays = 0
        self.model_replays = 0
        self.core.fixed_builder.reset_stats()

    def stats(self) -> dict[str, Any]:
        builder_stats = self.core.fixed_builder.stats()
        misses = int(builder_stats["fixed_builder_capacity_misses"])
        calls = self.builder_replays
        return {
            **builder_stats,
            "profiling_capture_scope": "builder_graph_plus_model_graph",
            "cuda_graph_capture_count": 2,
            "cuda_graph_builder_production_replays": self.builder_replays,
            "cuda_graph_model_production_replays": self.model_replays,
            "cuda_graph_production_replays": self.model_replays,
            "cuda_graph_production_calls": calls,
            "cuda_graph_capacity_misses": misses,
            "cuda_graph_hit_rate": (
                self.model_replays / calls if calls else 0.0
            ),
            "cuda_graph_builder_capture_wall_time_s": (
                self.builder_capture_wall_time_s
            ),
            "cuda_graph_builder_capture_allocated_delta_gib": (
                self.builder_capture_allocated_delta_bytes / 1024**3
            ),
            "cuda_graph_builder_capture_reserved_delta_gib": (
                self.builder_capture_reserved_delta_bytes / 1024**3
            ),
            "cuda_graph_model_capture_wall_time_s": (
                self.core.capture_wall_time_s
            ),
            "cuda_graph_capture_allocated_delta_gib": (
                self.builder_capture_allocated_delta_bytes
                + self.core.capture_allocated_delta_bytes
            )
            / 1024**3,
            "cuda_graph_capture_reserved_delta_gib": (
                self.builder_capture_reserved_delta_bytes
                + self.core.capture_reserved_delta_bytes
            )
            / 1024**3,
            "cuda_graph_replay_output_addresses_stable": (
                self.core.replay_output_addresses_stable
            ),
        }


class ESENForceEvalCUDAGraphEvaluator:
    """One graph containing fixed builder plus padded eSEN force evaluation."""

    def __init__(
        self,
        eager_evaluator: ESENEnergyForceEvaluator,
        *,
        neighbors_per_atom: int,
        dummy_atoms: int = 32,
        capture_warmup: int = 3,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
    ) -> None:
        self.core = ESENFixedBuilderModelCUDAGraphEvaluator(
            eager_evaluator,
            neighbors_per_atom=neighbors_per_atom,
            dummy_atoms=dummy_atoms,
            capture_warmup=0,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        self.device = self.core.device
        self.num_atoms = self.core.num_atoms
        self.model_dtype = self.core.model_dtype
        self.capture_warmup = int(capture_warmup)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.static_forces: Tensor | None = None
        self.static_energy: Tensor | None = None
        self.capture_wall_time_s = 0.0
        self.capture_allocated_delta_bytes = 0
        self.capture_reserved_delta_bytes = 0
        self.production_replays = 0
        self.output_addresses: tuple[int, int] | None = None

    def _body(self) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            self.core.fixed_builder.build(
                self.core.static_positions[: self.num_atoms]
            )
        return self.core._static_forward()

    def capture(self, positions: Tensor) -> None:
        self.core.model.backbone.otf_graph = False
        with torch.no_grad():
            self.core.static_positions[: self.num_atoms].copy_(positions)
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = stream
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            capture_positions = self.core.static_positions.detach().clone()
            capture_positions.requires_grad_(True)
            self.core.static_positions = capture_positions
            self.core.real_batch.pos = capture_positions[: self.num_atoms]
            self.core.static_batch.pos = capture_positions
            for _ in range(self.capture_warmup):
                self._body()
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)

        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            static_forces, static_energy = self._body()
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - start
        self.capture_allocated_delta_bytes = (
            torch.cuda.memory_allocated(self.device) - allocated_before
        )
        self.capture_reserved_delta_bytes = (
            torch.cuda.memory_reserved(self.device) - reserved_before
        )
        self.graph = graph
        self.static_forces = static_forces
        self.static_energy = static_energy
        self.output_addresses = (
            static_forces.data_ptr(),
            static_energy.data_ptr(),
        )
        self.core.fixed_builder.reset_stats()

    @torch.no_grad()
    def copy_positions(self, positions: Tensor) -> None:
        self.core.static_positions[: self.num_atoms].copy_(positions)

    def replay_force_eval(self) -> tuple[Tensor, Tensor]:
        if self.graph is None:
            raise RuntimeError("Force-evaluation graph has not been captured")
        self.graph.replay()
        self.production_replays += 1
        assert self.static_forces is not None
        assert self.static_energy is not None
        return self.static_forces, self.static_energy

    def reset_production_stats(self) -> None:
        self.production_replays = 0
        self.core.fixed_builder.reset_stats()

    def stats(self) -> dict[str, Any]:
        builder_stats = self.core.fixed_builder.stats()
        misses = int(builder_stats["fixed_builder_capacity_misses"])
        addresses_stable = bool(
            self.output_addresses is not None
            and self.static_forces is not None
            and self.static_energy is not None
            and self.output_addresses
            == (
                self.static_forces.data_ptr(),
                self.static_energy.data_ptr(),
            )
        )
        return {
            **builder_stats,
            "profiling_capture_scope": "builder_plus_model_graph",
            "cuda_graph_capture_count": 1,
            "cuda_graph_production_replays": self.production_replays,
            "cuda_graph_production_calls": self.production_replays,
            "cuda_graph_capacity_misses": misses,
            "cuda_graph_hit_rate": 1.0 if self.production_replays else 0.0,
            "cuda_graph_capture_wall_time_s": self.capture_wall_time_s,
            "cuda_graph_capture_allocated_delta_gib": (
                self.capture_allocated_delta_bytes / 1024**3
            ),
            "cuda_graph_capture_reserved_delta_gib": (
                self.capture_reserved_delta_bytes / 1024**3
            ),
            "cuda_graph_replay_output_addresses_stable": addresses_stable,
        }
