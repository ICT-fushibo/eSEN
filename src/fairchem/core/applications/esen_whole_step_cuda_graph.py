"""Opt3 fixed-builder and whole-step CUDA Graph execution for eSEN MD.

This module is intentionally separate from the opt1 and opt2 implementations.
It provides both the fixed-builder/model-only control and the full NVT-step
capture used to isolate the incremental benefit of widening CUDA Graph scope.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from ase import units
from torch import Tensor

from fairchem.core.applications.esen_cuda_graph import (
    CUDAGraphCapacityError,
    ESENModelCUDAGraphEvaluator,
)
from fairchem.core.applications.esen_fixed_neighbor import (
    FixedShapePBCNeighborBuilder,
)
from fairchem.core.applications.esen_gpu_md import (
    ESENEnergyForceEvaluator,
    GPUIntegrator,
    GPUMDState,
)


def _pbc_vector(batch, device: torch.device) -> Tensor:
    pbc = getattr(batch, "pbc", None)
    if pbc is None:
        return torch.ones(3, device=device, dtype=torch.bool)
    return torch.as_tensor(pbc, device=device, dtype=torch.bool).reshape(-1, 3)[0]


def _device_memory_used(device: torch.device) -> int | None:
    """Best-effort device memory including CUDA Graph private pools."""

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError):
        return None
    return int(total_bytes - free_bytes)


def _branchless_nvt_position_proposal(
    positions: Tensor,
    momenta: Tensor,
    forces: Tensor,
    integrator: GPUIntegrator,
    advance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return half-step momenta and the position used for force evaluation."""

    kinetic_energy = (
        0.5 * momenta.square() / integrator.masses
    ).sum()
    old_temperature = (
        2.0
        * kinetic_energy
        / (integrator.degrees_of_freedom * units.kB)
    ).clamp_min(1e-12)
    scale = torch.sqrt(
        1.0
        + (integrator.temperature_target / old_temperature - 1.0)
        * (integrator.dt / integrator.taut)
    ).clamp(min=0.9, max=1.1)
    half_momenta = momenta * scale + 0.5 * integrator.dt * forces
    if integrator.fix_com:
        half_momenta = half_momenta - half_momenta.sum(
            dim=0, keepdim=True
        ) / float(positions.shape[0])
    advanced_positions = (
        positions
        + integrator.dt * half_momenta / integrator.masses
    )
    evaluation_positions = positions + advance * (
        advanced_positions - positions
    )
    return half_momenta, evaluation_positions


def _branchless_nvt_momentum_finish(
    old_momenta: Tensor,
    half_momenta: Tensor,
    new_forces: Tensor,
    integrator: GPUIntegrator,
    advance: Tensor,
) -> Tensor:
    """Complete the second half-step or preserve momentum for advance=0."""

    advanced_momenta = (
        half_momenta + 0.5 * integrator.dt * new_forces
    )
    return old_momenta + advance * (advanced_momenta - old_momenta)


class ESENFixedBuilderModelCUDAGraphEvaluator(ESENModelCUDAGraphEvaluator):
    """Opt3 ablation control: fixed builder eager, model-only graph captured."""

    def __init__(
        self,
        eager_evaluator: ESENEnergyForceEvaluator,
        *,
        neighbors_per_atom: int,
        dummy_atoms: int = 32,
        capture_warmup: int = 3,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
        replay_energy_atol: float = 0.0,
        replay_force_atol: float = 1e-6,
    ) -> None:
        self.neighbors_per_atom = int(neighbors_per_atom)
        edge_capacity = eager_evaluator.num_atoms * self.neighbors_per_atom
        super().__init__(
            eager_evaluator,
            edge_capacity=edge_capacity,
            dummy_atoms=dummy_atoms,
            capture_warmup=capture_warmup,
            replay_energy_atol=replay_energy_atol,
            replay_force_atol=replay_force_atol,
        )

        sample = ESENModelCUDAGraphEvaluator._build_real_graph(
            self, eager_evaluator.model_positions
        )
        self._initialize_static_edges(sample)
        assert self.static_edge_index is not None
        assert self.static_cell_offsets is not None
        self.fixed_builder = FixedShapePBCNeighborBuilder(
            num_atoms=self.num_atoms,
            cell=self.static_batch.cell.reshape(-1, 3, 3)[0],
            pbc=_pbc_vector(self.static_batch, self.device),
            cutoff=float(self.model.backbone.cutoff),
            neighbors_per_atom=self.neighbors_per_atom,
            dummy_atoms=self.dummy_atoms,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
            output_edge_index=self.static_edge_index,
            output_cell_offsets=self.static_cell_offsets,
        )

    @torch.no_grad()
    def _build_real_graph(self, positions: Tensor) -> dict[str, Any]:
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Expected positions {(self.num_atoms, 3)}, got {positions.shape}"
            )
        self.static_positions[: self.num_atoms].copy_(positions)
        edge_index, cell_offsets = self.fixed_builder.build(
            self.static_positions[: self.num_atoms]
        )
        return {"edge_index": edge_index, "cell_offsets": cell_offsets}

    def _staticize(self, graph: dict[str, Any]) -> int:
        # The builder already wrote directly into the model's static buffers.
        return self.edge_capacity

    def reset_production_stats(self) -> None:
        super().reset_production_stats()
        self.fixed_builder.reset_stats()

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        if not self.captured or self.graph is None:
            raise RuntimeError("CUDA Graph must be captured before replay")
        self._build_real_graph(positions)
        self.graph.replay()
        self.total_replays += 1
        self.production_replays += 1
        self.production_calls += 1
        assert self.static_forces is not None
        assert self.static_energy is not None
        return self.static_forces, self.static_energy

    def raise_for_overflow(self) -> None:
        stats = self.fixed_builder.stats()
        misses = int(stats["fixed_builder_capacity_misses"])
        if misses:
            required = int(stats["fixed_builder_max_included_neighbors"])
            raise CUDAGraphCapacityError(required, self.neighbors_per_atom)

    def stats(self) -> dict[str, Any]:
        record = super().stats()
        builder_stats = self.fixed_builder.stats()
        calls = self.production_calls
        misses = int(builder_stats["fixed_builder_capacity_misses"])
        record.update(builder_stats)
        record.update(
            {
                "cuda_graph_production_calls": calls,
                "cuda_graph_capacity_misses": misses,
                "cuda_graph_hit_rate": (
                    self.production_replays / calls if calls else 0.0
                ),
                "cuda_graph_min_real_edges": builder_stats[
                    "fixed_builder_min_real_edges"
                ],
                "cuda_graph_max_real_edges": builder_stats[
                    "fixed_builder_max_real_edges"
                ],
                "cuda_graph_max_padding_fraction": builder_stats[
                    "fixed_builder_max_padding_fraction"
                ],
            }
        )
        return record


class ESENWholeStepCUDAGraphMD:
    """One captured graph containing neighbor build, eSEN, and one NVT step."""

    def __init__(
        self,
        state: GPUMDState,
        eager_evaluator: ESENEnergyForceEvaluator,
        integrator: GPUIntegrator,
        *,
        neighbors_per_atom: int,
        dummy_atoms: int = 32,
        capture_warmup: int = 3,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
    ) -> None:
        if eager_evaluator.device.type != "cuda":
            raise ValueError("Whole-step CUDA Graph requires CUDA")
        if state.positions.device != eager_evaluator.device:
            raise ValueError("MD state and evaluator must use the same device")
        if state.positions.shape != (eager_evaluator.num_atoms, 3):
            raise ValueError("MD state has the wrong position shape")
        if capture_warmup < 0:
            raise ValueError("capture_warmup must be non-negative")

        self.device = eager_evaluator.device
        self.num_atoms = eager_evaluator.num_atoms
        self.neighbors_per_atom = int(neighbors_per_atom)
        self.edge_capacity = self.num_atoms * self.neighbors_per_atom
        self.capture_warmup = int(capture_warmup)
        self.integrator = integrator
        self.core = ESENModelCUDAGraphEvaluator(
            eager_evaluator,
            edge_capacity=self.edge_capacity,
            dummy_atoms=dummy_atoms,
            capture_warmup=capture_warmup,
        )
        sample = self.core._build_real_graph(state.positions)
        self.core._initialize_static_edges(sample)
        assert self.core.static_edge_index is not None
        assert self.core.static_cell_offsets is not None
        self.fixed_builder = FixedShapePBCNeighborBuilder(
            num_atoms=self.num_atoms,
            cell=self.core.static_batch.cell.reshape(-1, 3, 3)[0],
            pbc=_pbc_vector(self.core.static_batch, self.device),
            cutoff=float(self.core.model.backbone.cutoff),
            neighbors_per_atom=self.neighbors_per_atom,
            dummy_atoms=dummy_atoms,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
            output_edge_index=self.core.static_edge_index,
            output_cell_offsets=self.core.static_cell_offsets,
        )
        self.core.model.backbone.otf_graph = False

        self.positions = state.positions.detach().clone()
        self.momenta = state.momenta.detach().clone()
        self.forces = torch.zeros_like(self.positions)
        self.potential_energy = torch.zeros(
            (), device=self.device, dtype=self.core.model_dtype
        )
        self.advance = torch.zeros(
            (), device=self.device, dtype=self.positions.dtype
        )
        self.step_counter = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.captured = False
        self.capture_count = 0
        self.production_capture_count = 0
        self.production_replays = 0
        self.total_replays = 0
        self.capture_wall_time_s = 0.0
        self.capture_allocated_delta_bytes = 0
        self.capture_reserved_delta_bytes = 0
        self.capture_device_used_delta_bytes: int | None = None
        self.output_addresses_stable = False

    @torch.no_grad()
    def restore_state_(self, state: GPUMDState) -> None:
        """Restore fixed-address state tensors for warmup-neutral execution."""

        self.positions.copy_(state.positions)
        self.momenta.copy_(state.momenta)
        if state.forces is None:
            self.forces.zero_()
        else:
            self.forces.copy_(state.forces)
        if state.potential_energy is None:
            self.potential_energy.zero_()
        else:
            self.potential_energy.copy_(state.potential_energy)
        self.advance.zero_()
        self.step_counter.zero_()

    def state_view(self) -> GPUMDState:
        return GPUMDState(
            positions=self.positions,
            momenta=self.momenta,
            forces=self.forces,
            potential_energy=self.potential_energy,
        )

    def _graph_body(self) -> None:
        """Execute the branchless initial-evaluation/NVT-step transaction."""

        with torch.no_grad():
            old_momenta = self.momenta
            half_momenta, evaluation_positions = (
                _branchless_nvt_position_proposal(
                    self.positions,
                    old_momenta,
                    self.forces,
                    self.integrator,
                    self.advance,
                )
            )
            self.core.static_positions[: self.num_atoms].copy_(
                evaluation_positions
            )
            graph_step = self.step_counter + self.advance.to(torch.long)
            self.fixed_builder.build(
                self.core.static_positions[: self.num_atoms],
                step=graph_step,
            )

        model_forces, model_energy = self.core._static_forward()

        with torch.no_grad():
            forces = model_forces.to(dtype=self.positions.dtype)
            final_momenta = _branchless_nvt_momentum_finish(
                old_momenta,
                half_momenta,
                forces,
                self.integrator,
                self.advance,
            )
            self.positions.copy_(evaluation_positions)
            self.momenta.copy_(final_momenta)
            self.forces.copy_(forces)
            self.potential_energy.copy_(model_energy)
            self.step_counter.add_(self.advance.to(torch.long))

    def capture(self, initial_state: GPUMDState) -> None:
        """Warm and capture exactly one graph, then restore the initial state."""

        if self.captured:
            raise RuntimeError("Whole-step CUDA Graph has already been captured")
        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = side_stream
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            capture_positions = self.core.static_positions.detach().clone()
            capture_positions.requires_grad_(True)
            self.core.static_positions = capture_positions
            self.core.real_batch.pos = capture_positions[: self.num_atoms]
            self.core.static_batch.pos = capture_positions
            self.restore_state_(initial_state)
            self.advance.fill_(1.0)
            for _ in range(self.capture_warmup):
                self._graph_body()
            self.restore_state_(initial_state)
            self.advance.fill_(1.0)
            self.fixed_builder.reset_stats()
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)

        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        device_used_before = _device_memory_used(self.device)
        capture_start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side_stream):
            self._graph_body()
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - capture_start
        self.capture_allocated_delta_bytes = (
            torch.cuda.memory_allocated(self.device) - allocated_before
        )
        self.capture_reserved_delta_bytes = (
            torch.cuda.memory_reserved(self.device) - reserved_before
        )
        device_used_after = _device_memory_used(self.device)
        if device_used_before is not None and device_used_after is not None:
            self.capture_device_used_delta_bytes = (
                device_used_after - device_used_before
            )

        self.graph = graph
        self.capture_count = 1
        self.captured = True
        addresses = (
            self.positions.data_ptr(),
            self.momenta.data_ptr(),
            self.forces.data_ptr(),
            self.potential_energy.data_ptr(),
        )
        self.restore_state_(initial_state)
        self.fixed_builder.reset_stats()
        torch.cuda.synchronize(self.device)
        self.output_addresses_stable = addresses == (
            self.positions.data_ptr(),
            self.momenta.data_ptr(),
            self.forces.data_ptr(),
            self.potential_energy.data_ptr(),
        )
        if not self.output_addresses_stable:
            raise RuntimeError("Whole-step CUDA Graph state addresses changed")

    def reset_production(self, initial_state: GPUMDState) -> None:
        if not self.captured:
            raise RuntimeError("Capture must complete before production")
        self.restore_state_(initial_state)
        self.fixed_builder.reset_stats()
        self.production_replays = 0

    def evaluate_initial(self) -> tuple[Tensor, Tensor]:
        """Replay once with advance=0, then arm subsequent replays for MD."""

        if self.graph is None:
            raise RuntimeError("Capture must complete before replay")
        self.graph.replay()
        self.production_replays += 1
        self.total_replays += 1
        self.advance.fill_(1.0)
        return self.forces, self.potential_energy

    def step(self) -> tuple[Tensor, Tensor]:
        if self.graph is None:
            raise RuntimeError("Capture must complete before replay")
        self.graph.replay()
        self.production_replays += 1
        self.total_replays += 1
        return self.forces, self.potential_energy

    def raise_for_overflow(self) -> None:
        stats = self.fixed_builder.stats()
        misses = int(stats["fixed_builder_capacity_misses"])
        if misses:
            required = int(stats["fixed_builder_max_included_neighbors"])
            raise CUDAGraphCapacityError(required, self.neighbors_per_atom)

    def stats(self) -> dict[str, Any]:
        builder_stats = self.fixed_builder.stats()
        calls = int(builder_stats["fixed_builder_build_calls"])
        misses = int(builder_stats["fixed_builder_capacity_misses"])
        return {
            **builder_stats,
            "cuda_graph_capture_count": self.capture_count,
            "cuda_graph_production_capture_count": self.production_capture_count,
            "cuda_graph_total_replays": self.total_replays,
            "cuda_graph_production_replays": self.production_replays,
            "cuda_graph_production_calls": calls,
            "cuda_graph_capacity_misses": misses,
            "cuda_graph_hit_rate": (
                self.production_replays / calls if calls else 0.0
            ),
            "cuda_graph_edge_capacity": self.edge_capacity,
            "cuda_graph_min_real_edges": builder_stats[
                "fixed_builder_min_real_edges"
            ],
            "cuda_graph_max_real_edges": builder_stats[
                "fixed_builder_max_real_edges"
            ],
            "cuda_graph_max_padding_fraction": builder_stats[
                "fixed_builder_max_padding_fraction"
            ],
            "cuda_graph_dummy_atoms": self.core.dummy_atoms,
            "cuda_graph_capture_warmup": self.capture_warmup,
            "cuda_graph_replay_output_addresses_stable": (
                self.output_addresses_stable
            ),
            "cuda_graph_capture_wall_time_s": self.capture_wall_time_s,
            "cuda_graph_capture_allocated_delta_gib": (
                self.capture_allocated_delta_bytes / 1024**3
            ),
            "cuda_graph_capture_reserved_delta_gib": (
                self.capture_reserved_delta_bytes / 1024**3
            ),
            "cuda_graph_capture_device_used_delta_gib": (
                None
                if self.capture_device_used_delta_bytes is None
                else self.capture_device_used_delta_bytes / 1024**3
            ),
        }
