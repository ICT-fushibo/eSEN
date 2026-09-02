"""Opt3 fixed-builder and whole-step CUDA Graph execution for eSEN MD.

This module is intentionally separate from the opt1 and opt2 implementations.
It provides both the fixed-builder/model-only control and the full NVT-step
capture used to isolate the incremental benefit of widening CUDA Graph scope.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
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
    promote_elastic_neighbor_capacities,
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


def _normalize_neighbor_capacities(
    num_atoms: int,
    neighbors_per_atom: int,
    neighbor_capacities: Tensor | Sequence[int] | None,
) -> tuple[int, tuple[int, ...] | None]:
    """Validate optional heterogeneous slots and return total edge capacity."""

    if neighbor_capacities is None:
        return num_atoms * int(neighbors_per_atom), None
    values = torch.as_tensor(
        neighbor_capacities, device="cpu", dtype=torch.long
    ).reshape(-1)
    if values.shape != (num_atoms,):
        raise ValueError(
            "neighbor_capacities must contain one value per real atom"
        )
    if bool((values < 1).any()):
        raise ValueError("neighbor capacities must be positive")
    normalized = tuple(int(value) for value in values.tolist())
    return sum(normalized), normalized


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
        neighbor_capacities: Tensor | Sequence[int] | None = None,
        neighbor_capacity_policy: str = "uniform",
        dummy_atoms: int = 32,
        capture_warmup: int = 3,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
        replay_energy_atol: float = 0.0,
        replay_force_atol: float = 1e-6,
    ) -> None:
        self.neighbors_per_atom = int(neighbors_per_atom)
        edge_capacity, self.neighbor_capacities = _normalize_neighbor_capacities(
            eager_evaluator.num_atoms,
            self.neighbors_per_atom,
            neighbor_capacities,
        )
        self.neighbor_capacity_policy = str(neighbor_capacity_policy)
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
            neighbor_capacities=self.neighbor_capacities,
            capacity_policy=self.neighbor_capacity_policy,
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
            required = int(stats["fixed_builder_max_overflow_required"])
            capacity = int(stats["fixed_builder_max_overflow_capacity"])
            raise CUDAGraphCapacityError(required, capacity)

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
        neighbor_capacities: Tensor | Sequence[int] | None = None,
        neighbor_capacity_policy: str = "uniform",
        dummy_atoms: int = 32,
        capture_warmup: int = 3,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
        overflow_to_dummy_only: bool = False,
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
        self.edge_capacity, self.neighbor_capacities = (
            _normalize_neighbor_capacities(
                self.num_atoms,
                self.neighbors_per_atom,
                neighbor_capacities,
            )
        )
        self.neighbor_capacity_policy = str(neighbor_capacity_policy)
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
            neighbor_capacities=self.neighbor_capacities,
            capacity_policy=self.neighbor_capacity_policy,
            dummy_atoms=dummy_atoms,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
            overflow_to_dummy_only=overflow_to_dummy_only,
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
            required = int(stats["fixed_builder_max_overflow_required"])
            capacity = int(stats["fixed_builder_max_overflow_capacity"])
            raise CUDAGraphCapacityError(required, capacity)

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


class UnrecoveredCapacityOverflow(CUDAGraphCapacityError):
    """Raised after CAP2 exhausts its monotonic promotion budget."""

    def __init__(
        self,
        required_edges: int,
        edge_capacity: int,
        graph_stats: dict[str, Any],
    ) -> None:
        super().__init__(required_edges, edge_capacity)
        self.graph_stats = graph_stats


class WholeStepTransactionSnapshot:
    """Fixed-address GPU backup for one rollback transaction."""

    def __init__(self, whole: ESENWholeStepCUDAGraphMD) -> None:
        self.positions = torch.empty_like(whole.positions)
        self.momenta = torch.empty_like(whole.momenta)
        self.forces = torch.empty_like(whole.forces)
        self.potential_energy = torch.empty_like(whole.potential_energy)
        self.step_counter = torch.empty_like(whole.step_counter)
        self.advance = torch.empty_like(whole.advance)
        eta = getattr(whole.integrator, "eta", None)
        p_eta = getattr(whole.integrator, "p_eta", None)
        self.eta = torch.empty_like(eta) if isinstance(eta, Tensor) else None
        self.p_eta = (
            torch.empty_like(p_eta) if isinstance(p_eta, Tensor) else None
        )
        self._addresses = self.addresses()

    def addresses(self) -> tuple[int, ...]:
        tensors = [
            self.positions,
            self.momenta,
            self.forces,
            self.potential_energy,
            self.step_counter,
            self.advance,
        ]
        if self.eta is not None:
            tensors.append(self.eta)
        if self.p_eta is not None:
            tensors.append(self.p_eta)
        return tuple(tensor.data_ptr() for tensor in tensors)

    @property
    def addresses_stable(self) -> bool:
        return self.addresses() == self._addresses

    @torch.no_grad()
    def save_from_(self, whole: ESENWholeStepCUDAGraphMD) -> None:
        self.positions.copy_(whole.positions)
        self.momenta.copy_(whole.momenta)
        self.forces.copy_(whole.forces)
        self.potential_energy.copy_(whole.potential_energy)
        self.step_counter.copy_(whole.step_counter)
        self.advance.copy_(whole.advance)
        if self.eta is not None:
            self.eta.copy_(whole.integrator.eta)
        if self.p_eta is not None:
            self.p_eta.copy_(whole.integrator.p_eta)

    def state_view(self) -> GPUMDState:
        return GPUMDState(
            positions=self.positions,
            momenta=self.momenta,
            forces=self.forces,
            potential_energy=self.potential_energy,
        )

    @torch.no_grad()
    def restore_integrator_(self, integrator: GPUIntegrator) -> None:
        if self.eta is not None:
            integrator.eta.copy_(self.eta)
        if self.p_eta is not None:
            integrator.p_eta.copy_(self.p_eta)

    @torch.no_grad()
    def restore_into_(self, whole: ESENWholeStepCUDAGraphMD) -> None:
        whole.positions.copy_(self.positions)
        whole.momenta.copy_(self.momenta)
        whole.forces.copy_(self.forces)
        whole.potential_energy.copy_(self.potential_energy)
        whole.step_counter.copy_(self.step_counter)
        whole.advance.copy_(self.advance)
        self.restore_integrator_(whole.integrator)


class TransactionalWholeStepCUDAGraphController:
    """ROB1 wrapper around a single active whole-step CUDA Graph.

    A failed transaction is never committed.  Its complete MD state is
    restored, the capacity is promoted from device-observed per-atom demand,
    and the same physical steps are replayed with a newly captured graph.

    The initial allocation may be CAP1-auto-safe or CAP2 elastic.  Keeping
    that choice independent from transactional recovery lets frozen Opt4
    configurations add rollback safety without inheriting CAP2's aggressive
    compact allocation.
    """

    def __init__(
        self,
        state: GPUMDState,
        eager_evaluator: ESENEnergyForceEvaluator,
        integrator: GPUIntegrator,
        *,
        whole_class: type[ESENWholeStepCUDAGraphMD],
        atomic_numbers: Tensor | Sequence[int],
        neighbor_capacities: Tensor | Sequence[int],
        initial_capacity_policy: str = "elastic",
        max_promotions: int = 2,
        whole_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if max_promotions < 0 or max_promotions > 2:
            raise ValueError("CAP2 max_promotions must be between zero and two")
        capacities = tuple(
            int(value)
            for value in torch.as_tensor(
                neighbor_capacities, device="cpu", dtype=torch.long
            )
            .reshape(-1)
            .tolist()
        )
        numbers = tuple(
            int(value)
            for value in torch.as_tensor(
                atomic_numbers, device="cpu", dtype=torch.long
            )
            .reshape(-1)
            .tolist()
        )
        if len(capacities) != eager_evaluator.num_atoms:
            raise ValueError("CAP2 requires one capacity per real atom")
        if len(numbers) != len(capacities):
            raise ValueError("atomic_numbers must match CAP2 capacities")
        if initial_capacity_policy not in {"auto-safe", "elastic"}:
            raise ValueError(
                "ROB1 initial capacity policy must be auto-safe or elastic"
            )
        self.eager_evaluator = eager_evaluator
        self.integrator = integrator
        self.whole_class = whole_class
        self.atomic_numbers = numbers
        self.current_capacities = capacities
        self.initial_capacities = capacities
        self.initial_capacity_policy = str(initial_capacity_policy)
        self.max_promotions = int(max_promotions)
        self.whole_kwargs = dict(whole_kwargs or {})
        self.whole: ESENWholeStepCUDAGraphMD | None = self._new_whole(state)
        self.snapshot: WholeStepTransactionSnapshot | None = None
        self.setup_capture_count = 0
        self.recovery_capture_count = 0
        self.capture_wall_time_s = 0.0
        self.recovery_capture_wall_time_s = 0.0
        self.attempted_replays = 0
        self.committed_replays = 0
        self.discarded_replays = 0
        self.committed_physical_steps = 0
        self.rollback_count = 0
        self.retried_physical_steps = 0
        self.detected_overflow_replays = 0
        self.unrecovered_overflows = 0
        self.promotion_count = 0
        self.promotion_history: list[dict[str, Any]] = []
        self._retired_stats: list[dict[str, Any]] = []

    def _new_whole(self, state: GPUMDState) -> ESENWholeStepCUDAGraphMD:
        kwargs = dict(self.whole_kwargs)
        kwargs.update(
            neighbors_per_atom=max(self.current_capacities),
            neighbor_capacities=self.current_capacities,
            neighbor_capacity_policy=(
                f"{self.initial_capacity_policy}-rob1"
            ),
            overflow_to_dummy_only=True,
        )
        return self.whole_class(
            state,
            self.eager_evaluator,
            self.integrator,
            **kwargs,
        )

    def capture(self, initial_state: GPUMDState) -> None:
        if self.whole is None:
            raise RuntimeError("CAP2 controller has no active graph")
        self.whole.capture(initial_state)
        self.setup_capture_count = 1
        self.capture_wall_time_s = self.whole.capture_wall_time_s
        self.snapshot = WholeStepTransactionSnapshot(self.whole)

    def _active(self) -> ESENWholeStepCUDAGraphMD:
        if self.whole is None or self.snapshot is None:
            raise RuntimeError("CAP2 graph must be captured before replay")
        return self.whole

    @staticmethod
    def _synchronize(whole: ESENWholeStepCUDAGraphMD) -> None:
        if torch.device(whole.device).type == "cuda":
            torch.cuda.synchronize(whole.device)

    def reset_production(self, initial_state: GPUMDState) -> None:
        whole = self._active()
        whole.reset_production(initial_state)
        self.attempted_replays = 0
        self.committed_replays = 0
        self.discarded_replays = 0
        self.committed_physical_steps = 0
        self.rollback_count = 0
        self.retried_physical_steps = 0
        self.detected_overflow_replays = 0
        self.unrecovered_overflows = 0
        # Capacity is monotonic for the lifetime of the controller.  In
        # particular, a setup/warmup promotion must still consume one of the
        # two allowed promotions; resetting the physical MD state may not
        # silently reset the CAP2 safety state or resurrect an old graph.

    def _promote_and_recapture(
        self,
        demand: Sequence[int],
        *,
        transaction_steps: int,
    ) -> None:
        whole = self._active()
        assert self.snapshot is not None
        if self.promotion_count >= self.max_promotions:
            self.snapshot.restore_into_(whole)
            self._synchronize(whole)
            self.unrecovered_overflows += 1
            stats = self.stats()
            required = max(int(value) for value in demand)
            raise UnrecoveredCapacityOverflow(
                required, max(self.current_capacities), stats
            )
        previous = self.current_capacities
        promoted, policy = promote_elastic_neighbor_capacities(
            previous,
            demand,
            self.atomic_numbers,
            promotion_index=self.promotion_count,
        )
        if sum(promoted) <= sum(previous):
            self.snapshot.restore_into_(whole)
            self._synchronize(whole)
            self.unrecovered_overflows += 1
            stats = self.stats()
            raise UnrecoveredCapacityOverflow(
                max(int(value) for value in demand),
                max(previous),
                stats,
            )
        recovery_started = time.perf_counter()
        self.snapshot.restore_into_(whole)
        self._synchronize(whole)
        retired = whole.stats()
        self._retired_stats.append(retired)
        old_edge_capacity = sum(previous)
        whole.graph = None
        whole.capture_stream = None
        self.whole = None
        del whole
        gc.collect()
        torch.cuda.empty_cache()

        self.current_capacities = promoted
        self.snapshot.restore_integrator_(self.integrator)
        state = self.snapshot.state_view()
        replacement = self._new_whole(state)
        replacement.capture(state)
        self._synchronize(replacement)
        capture_elapsed = time.perf_counter() - recovery_started
        self.snapshot.restore_into_(replacement)
        replacement.fixed_builder.reset_stats()
        self.whole = replacement
        self.promotion_history.append(
            {
                "promotion_index": self.promotion_count + 1,
                "policy": policy,
                "transaction_steps": transaction_steps,
                "required_max_neighbors": max(int(value) for value in demand),
                "old_edge_capacity": old_edge_capacity,
                "new_edge_capacity": sum(promoted),
                "old_capacity_min": min(previous),
                "old_capacity_max": max(previous),
                "new_capacity_min": min(promoted),
                "new_capacity_max": max(promoted),
                "capture_wall_time_s": capture_elapsed,
            }
        )
        self.promotion_count += 1
        self.recovery_capture_count += 1
        self.recovery_capture_wall_time_s += capture_elapsed

    def _run_transaction(
        self,
        steps: int,
        *,
        initial: bool,
        checkpoint_offsets: Sequence[int] = (),
    ) -> tuple[Tensor, Tensor, dict[int, Tensor]]:
        if steps < 0 or (initial and steps != 0) or (not initial and steps < 1):
            raise ValueError("invalid ROB1 transaction length")
        requested_offsets = {int(value) for value in checkpoint_offsets}
        if initial and requested_offsets:
            raise ValueError("initial transactions do not have step offsets")
        if any(value < 1 or value > steps for value in requested_offsets):
            raise ValueError("checkpoint offsets must lie inside the transaction")
        while True:
            whole = self._active()
            assert self.snapshot is not None
            self.snapshot.save_from_(whole)
            whole.fixed_builder.reset_window_stats()
            if initial:
                forces, energy = whole.evaluate_initial()
                replay_count = 1
            else:
                forces = whole.forces
                energy = whole.potential_energy
                pending_checkpoints: dict[int, Tensor] = {}
                for offset in range(1, steps + 1):
                    forces, energy = whole.step()
                    if offset in requested_offsets:
                        pending_checkpoints[offset] = energy.detach().clone()
                replay_count = steps
            if initial:
                pending_checkpoints = {}
            self._synchronize(whole)
            window = whole.fixed_builder.window_stats()
            self.attempted_replays += replay_count
            misses = int(window["fixed_builder_window_capacity_misses"])
            if misses == 0:
                self.committed_replays += replay_count
                if not initial:
                    self.committed_physical_steps += steps
                return forces, energy, pending_checkpoints

            self.detected_overflow_replays += int(
                window["fixed_builder_window_overflow_dummy_only_replays"]
            )
            self.discarded_replays += replay_count
            self.rollback_count += 1
            if not initial:
                self.retried_physical_steps += steps
            demand = window[
                "fixed_builder_window_maximum_included_neighbors_by_atom"
            ]
            self._promote_and_recapture(
                demand, transaction_steps=(0 if initial else steps)
            )

    def evaluate_initial(self) -> tuple[Tensor, Tensor]:
        forces, energy, _ = self._run_transaction(0, initial=True)
        return forces, energy

    def run_steps(self, steps: int) -> tuple[Tensor, Tensor]:
        forces, energy, _ = self._run_transaction(steps, initial=False)
        return forces, energy

    def run_steps_with_checkpoints(
        self,
        steps: int,
        checkpoint_offsets: Sequence[int],
    ) -> tuple[Tensor, Tensor, dict[int, Tensor]]:
        """Run one transaction and publish only checkpoints from its commit."""

        return self._run_transaction(
            steps,
            initial=False,
            checkpoint_offsets=checkpoint_offsets,
        )

    def step(self) -> tuple[Tensor, Tensor]:
        return self.run_steps(1)

    def state_view(self) -> GPUMDState:
        return self._active().state_view()

    def stats(self) -> dict[str, Any]:
        whole = self._active()
        active = whole.stats()
        histories = [*self._retired_stats, active]
        addresses_stable = bool(
            self.snapshot is not None
            and self.snapshot.addresses_stable
            and all(
                row.get("cuda_graph_replay_output_addresses_stable", False)
                for row in histories
            )
        )
        min_real_values = [
            row.get("fixed_builder_min_real_edges")
            for row in histories
            if row.get("fixed_builder_min_real_edges") is not None
        ]
        max_real_values = [
            row.get("fixed_builder_max_real_edges")
            for row in histories
            if row.get("fixed_builder_max_real_edges") is not None
        ]
        sink_min_values = [
            row.get("sink_padding_edges_min")
            for row in histories
            if row.get("sink_padding_edges_min") is not None
        ]
        sink_max_values = [
            row.get("sink_padding_edges_max")
            for row in histories
            if row.get("sink_padding_edges_max") is not None
        ]
        total_calls = sum(
            int(row.get("fixed_builder_build_calls", 0)) for row in histories
        )
        total_dummy_only = sum(
            int(row.get("overflow_dummy_only_replays", 0))
            for row in histories
        )
        active.update(
            {
                "cuda_graph_capture_count": self.setup_capture_count,
                "cuda_graph_recovery_capture_count": self.recovery_capture_count,
                "cuda_graph_total_capture_count": (
                    self.setup_capture_count + self.recovery_capture_count
                ),
                "cuda_graph_production_capture_count": self.recovery_capture_count,
                "cuda_graph_capture_wall_time_s": self.capture_wall_time_s,
                "cuda_graph_recovery_capture_wall_time_s": (
                    self.recovery_capture_wall_time_s
                ),
                "cuda_graph_total_capture_wall_time_s": (
                    self.capture_wall_time_s + self.recovery_capture_wall_time_s
                ),
                "setup_capture_count": self.setup_capture_count,
                "setup_capture_wall_time_s": self.capture_wall_time_s,
                "recovery_capture_count": self.recovery_capture_count,
                "recovery_capture_wall_time_s": (
                    self.recovery_capture_wall_time_s
                ),
                "cuda_graph_total_replays": self.attempted_replays,
                "cuda_graph_attempted_replays": self.attempted_replays,
                "cuda_graph_production_replays": self.committed_replays,
                "cuda_graph_committed_replays": self.committed_replays,
                "cuda_graph_discarded_replays": self.discarded_replays,
                "cuda_graph_production_calls": self.committed_replays,
                "cuda_graph_attempted_calls": total_calls,
                "cuda_graph_capacity_misses": self.unrecovered_overflows,
                "cuda_graph_recovered_capacity_misses": self.rollback_count,
                "cuda_graph_hit_rate": (
                    1.0 if self.committed_replays else 0.0
                ),
                "cuda_graph_attempted_hit_rate": (
                    self.committed_replays / self.attempted_replays
                    if self.attempted_replays
                    else 0.0
                ),
                "cuda_graph_edge_capacity": sum(self.current_capacities),
                "cuda_graph_initial_edge_capacity": sum(self.initial_capacities),
                "cuda_graph_final_edge_capacity": sum(self.current_capacities),
                "cuda_graph_min_real_edges": (
                    min(min_real_values) if min_real_values else None
                ),
                "cuda_graph_max_real_edges": (
                    max(max_real_values) if max_real_values else None
                ),
                "cuda_graph_replay_output_addresses_stable": addresses_stable,
                "rob1_enabled": True,
                "rob1_attempted_replays": self.attempted_replays,
                "rob1_committed_replays": self.committed_replays,
                "rob1_discarded_replays": self.discarded_replays,
                "rob1_committed_physical_steps": self.committed_physical_steps,
                "rob1_rollback_count": self.rollback_count,
                "rob1_retried_physical_steps": self.retried_physical_steps,
                "rob1_unrecovered_overflows": self.unrecovered_overflows,
                "capacity_misses_after_final_retry": (
                    self.unrecovered_overflows
                ),
                "unrecovered_overflow_count": self.unrecovered_overflows,
                "rob1_snapshot_addresses_stable": (
                    self.snapshot.addresses_stable
                    if self.snapshot is not None
                    else False
                ),
                "rob1_initial_capacity_policy": self.initial_capacity_policy,
                "rob1_promotion_count": self.promotion_count,
                "rob1_promotion_history": self.promotion_history,
                "rob1_initial_capacities": list(self.initial_capacities),
                "rob1_final_capacities": list(self.current_capacities),
                # Preserve CAP2 telemetry for existing result consumers.  A
                # CAP1-auto-safe ROB1 run is explicitly not a CAP2 run.
                "cap2_promotion_count": (
                    self.promotion_count
                    if self.initial_capacity_policy == "elastic"
                    else 0
                ),
                "cap2_promotion_history": (
                    self.promotion_history
                    if self.initial_capacity_policy == "elastic"
                    else []
                ),
                "cap2_initial_capacities": (
                    list(self.initial_capacities)
                    if self.initial_capacity_policy == "elastic"
                    else []
                ),
                "cap2_final_capacities": (
                    list(self.current_capacities)
                    if self.initial_capacity_policy == "elastic"
                    else []
                ),
                "sink_padding_edges_min": (
                    min(sink_min_values) if sink_min_values else None
                ),
                "sink_padding_edges_max": (
                    max(sink_max_values) if sink_max_values else None
                ),
                "overflow_dummy_only_replays": total_dummy_only,
            }
        )
        return active


# Backwards-compatible name retained for CAP2 scripts and imports.
ElasticWholeStepCUDAGraphController = TransactionalWholeStepCUDAGraphController
