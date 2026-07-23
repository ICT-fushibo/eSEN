"""Model-only CUDA Graph execution for GPU-resident eSEN molecular dynamics.

The neighbor graph remains dynamic and is rebuilt eagerly for every force
evaluation.  Its ragged edge tensors are copied into one fixed-capacity batch;
padding edges are routed exclusively through dummy sink atoms.  Only the eSEN
forward, conservative-force autograd, and prediction denormalization are
captured.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import Tensor

from fairchem.core.applications.esen_gpu_md import (
    ESENEnergyForceEvaluator,
    _resolve_model_output,
)


class CUDAGraphCapacityError(RuntimeError):
    """Raised when a production graph has more edges than the captured graph."""

    def __init__(self, required_edges: int, edge_capacity: int) -> None:
        self.required_edges = int(required_edges)
        self.edge_capacity = int(edge_capacity)
        super().__init__(
            f"CUDA Graph edge capacity exceeded: required={required_edges}, "
            f"capacity={edge_capacity}"
        )


class CUDAGraphValidationError(RuntimeError):
    """Raised when a captured graph fails an untimed correctness check."""


def edge_capacity_from_probe(
    maximum_edges: int,
    *,
    margin: float = 0.10,
    edge_step: int = 256,
) -> int:
    """Return one conservative fixed edge capacity for a probed trajectory."""

    if maximum_edges < 1:
        raise ValueError("maximum_edges must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if edge_step < 1:
        raise ValueError("edge_step must be positive")
    required = max(maximum_edges + 1, math.ceil(maximum_edges * (1.0 + margin)))
    return int(math.ceil(required / edge_step) * edge_step)


@torch.no_grad()
def staticize_neighbor_graph_(
    static_edge_index: Tensor,
    static_cell_offsets: Tensor,
    real_edge_index: Tensor,
    real_cell_offsets: Tensor,
    *,
    n_real: int,
    dummy_sink_template: Tensor,
    padding_offset_template: Tensor,
) -> int:
    """Copy a ragged real graph into fixed-capacity CUDA Graph input tensors."""

    if static_edge_index.ndim != 2 or static_edge_index.shape[0] != 2:
        raise ValueError("static_edge_index must have shape [2, edge_capacity]")
    if real_edge_index.ndim != 2 or real_edge_index.shape[0] != 2:
        raise ValueError("real_edge_index must have shape [2, num_edges]")
    if n_real < 1:
        raise ValueError("n_real must be positive")
    edge_capacity = int(static_edge_index.shape[1])
    num_edges = int(real_edge_index.shape[1])
    if num_edges > edge_capacity:
        raise CUDAGraphCapacityError(num_edges, edge_capacity)
    if static_cell_offsets.shape != (edge_capacity, 3):
        raise ValueError("static_cell_offsets has the wrong shape")
    if real_cell_offsets.shape != (num_edges, 3):
        raise ValueError("real_cell_offsets has the wrong shape")
    if dummy_sink_template.numel() < edge_capacity:
        raise ValueError("dummy_sink_template is smaller than edge_capacity")
    if padding_offset_template.shape != (edge_capacity, 3):
        raise ValueError("padding_offset_template has the wrong shape")
    if num_edges:
        static_edge_index[:, :num_edges].copy_(real_edge_index)
        static_cell_offsets[:num_edges].copy_(real_cell_offsets)
    padding = edge_capacity - num_edges
    if padding:
        sinks = dummy_sink_template[:padding]
        static_edge_index[0, num_edges:].copy_(sinks)
        static_edge_index[1, num_edges:].copy_(sinks)
        static_cell_offsets[num_edges:].copy_(
            padding_offset_template[:padding]
        )
    return num_edges


class ESENModelCUDAGraphEvaluator:
    """Replay a fixed-capacity eSEN model graph over dynamic neighbor graphs."""

    def __init__(
        self,
        eager_evaluator: ESENEnergyForceEvaluator,
        *,
        edge_capacity: int,
        dummy_atoms: int = 32,
        capture_warmup: int = 3,
    ) -> None:
        if eager_evaluator.device.type != "cuda":
            raise ValueError("Model-only CUDA Graph requires a CUDA evaluator")
        if edge_capacity < 1:
            raise ValueError("edge_capacity must be positive")
        if dummy_atoms < 1:
            raise ValueError("dummy_atoms must be positive")
        if capture_warmup < 0:
            raise ValueError("capture_warmup must be non-negative")

        self.eager_evaluator = eager_evaluator
        self.device = eager_evaluator.device
        self.calculator = eager_evaluator.calculator
        self.trainer = eager_evaluator.trainer
        self.model = eager_evaluator.model
        self.model_dtype = eager_evaluator.model_dtype
        self.num_atoms = eager_evaluator.num_atoms
        self.num_graphs = eager_evaluator.num_graphs
        if self.num_graphs != 1:
            raise ValueError("Model-only CUDA Graph MD supports one structure")

        self.edge_capacity = int(edge_capacity)
        self.dummy_atoms = int(dummy_atoms)
        self.capture_warmup = int(capture_warmup)
        self.total_atoms = self.num_atoms + self.dummy_atoms

        # A real-only batch drives the dynamic OTF graph builder and the
        # denormalizers.  A padded batch with static tensor shapes is consumed
        # by the captured model.  Their real positions share storage.
        self.real_batch = eager_evaluator.batch.clone()
        initial_real_positions = eager_evaluator.model_positions.detach().clone()
        dummy_positions = initial_real_positions.new_zeros(self.dummy_atoms, 3)
        self.static_positions = torch.cat(
            (initial_real_positions, dummy_positions), dim=0
        ).detach()
        self.static_positions.requires_grad_(True)
        self.real_batch.pos = self.static_positions[: self.num_atoms]

        self.static_batch = eager_evaluator.batch.clone()
        self.static_batch.pos = self.static_positions
        dummy_numbers = self.static_batch.atomic_numbers.new_zeros(self.dummy_atoms)
        self.static_batch.atomic_numbers = torch.cat(
            (self.static_batch.atomic_numbers, dummy_numbers), dim=0
        )
        dummy_batch = self.static_batch.batch.new_zeros(self.dummy_atoms)
        self.static_batch.batch = torch.cat(
            (self.static_batch.batch, dummy_batch), dim=0
        )
        self.static_batch.natoms = self.static_batch.natoms.new_tensor(
            [self.total_atoms]
        )
        self.static_batch.atomic_numbers_full = self.static_batch.atomic_numbers
        self.static_batch.batch_full = self.static_batch.batch
        self.static_batch.nedges = self.static_batch.natoms.new_tensor(
            [self.edge_capacity]
        )
        self.static_batch.n_real = self.num_atoms

        self.energy_element_reference: Tensor | None = None
        if "energy" in self.trainer.elementrefs:
            zero_energy = initial_real_positions.new_zeros(self.num_graphs, 1)
            with torch.no_grad():
                self.energy_element_reference = self.trainer.elementrefs[
                    "energy"
                ](zero_energy, self.real_batch).detach()
        if "forces" in self.trainer.elementrefs:
            raise ValueError(
                "Force element references are not supported by model CUDA Graph"
            )

        self.static_edge_index: Tensor | None = None
        self.static_cell_offsets: Tensor | None = None
        self.dummy_sink_template: Tensor | None = None
        self.padding_offset_template: Tensor | None = None
        self.fixed_rotation_reference: Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_forces: Tensor | None = None
        self.static_energy: Tensor | None = None
        self.captured = False

        self.capture_count = 0
        self.capture_wall_time_s = 0.0
        self.capture_allocated_delta_bytes = 0
        self.capture_reserved_delta_bytes = 0
        self.total_replays = 0
        self.production_replays = 0
        self.production_calls = 0
        self.production_capacity_misses = 0
        self.production_min_edges: int | None = None
        self.production_max_edges: int | None = None
        self.replay_stability_energy_abs_error = 0.0
        self.replay_stability_force_max_abs_error = 0.0
        self.replay_output_addresses_stable = False

    @torch.no_grad()
    def _build_real_graph(self, positions: Tensor) -> dict[str, Any]:
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Expected positions with shape {(self.num_atoms, 3)}, "
                f"got {tuple(positions.shape)}"
            )
        if positions.device != self.device:
            raise ValueError(
                f"Positions must be on {self.device}, got {positions.device}"
            )
        self.static_positions[: self.num_atoms].copy_(positions)
        return self.model.backbone.generate_graph(
            self.real_batch, otf_graph=True
        )

    def _initialize_static_edges(self, sample_graph: dict[str, Any]) -> None:
        real_edge_index = sample_graph["edge_index"]
        real_cell_offsets = sample_graph["cell_offsets"]
        self.static_edge_index = real_edge_index.new_empty(2, self.edge_capacity)
        self.static_cell_offsets = real_cell_offsets.new_empty(
            self.edge_capacity, 3
        )
        self.static_batch.edge_index = self.static_edge_index
        self.static_batch.cell_offsets = self.static_cell_offsets

        self.dummy_sink_template = torch.arange(
            self.edge_capacity,
            device=self.static_edge_index.device,
            dtype=self.static_edge_index.dtype,
        )
        self.dummy_sink_template.remainder_(self.dummy_atoms)
        self.dummy_sink_template.add_(self.num_atoms)

        # Dummy self-edges use a nonzero periodic image that is safely beyond
        # the model cutoff.  They never connect to real atoms, but a nonzero
        # vector also keeps rotation construction finite.
        cell = self.static_batch.cell.detach().reshape(-1, 3, 3)[0].cpu()
        vector_norms = torch.linalg.vector_norm(cell, dim=1)
        axis = int(torch.argmax(vector_norms).item())
        axis_norm = float(vector_norms[axis].item())
        if not math.isfinite(axis_norm) or axis_norm <= 0:
            raise ValueError("Cannot construct dummy edge offset from invalid cell")
        cutoff = float(self.model.backbone.cutoff)
        shift = max(2, math.ceil((cutoff + 1.0) / axis_norm) + 1)
        self.padding_offset_template = self.static_cell_offsets.new_zeros(
            self.edge_capacity, 3
        )
        self.padding_offset_template[:, axis] = shift

        # eSEN normally samples a random auxiliary vector for each edge when
        # constructing its local rotation frame.  A captured RNG operation
        # advances on replay and would make identical graph replays differ.
        # These three non-collinear component values are safely separated from
        # every edge direction by the fallback rotations in init_edge_rot_mat.
        reference = self.static_positions.new_tensor([0.37, -0.61, 0.71])
        self.fixed_rotation_reference = reference.expand(
            self.edge_capacity, 3
        ).contiguous()
        self.model.backbone.cuda_graph_fixed_rotation_reference = (
            self.fixed_rotation_reference
        )

    @torch.no_grad()
    def _staticize(self, graph: dict[str, Any]) -> int:
        if self.static_edge_index is None:
            self._initialize_static_edges(graph)
        assert self.static_edge_index is not None
        assert self.static_cell_offsets is not None
        assert self.dummy_sink_template is not None
        assert self.padding_offset_template is not None
        return staticize_neighbor_graph_(
            self.static_edge_index,
            self.static_cell_offsets,
            graph["edge_index"],
            graph["cell_offsets"],
            n_real=self.num_atoms,
            dummy_sink_template=self.dummy_sink_template,
            padding_offset_template=self.padding_offset_template,
        )

    def _static_forward(self) -> tuple[Tensor, Tensor]:
        # The head computes conservative forces through autograd.grad, so
        # inference_mode is intentionally not used.
        with torch.enable_grad():
            raw_outputs = self.model(self.static_batch)
            raw_energy = _resolve_model_output(raw_outputs, "energy")
            raw_forces = _resolve_model_output(raw_outputs, "forces")
            raw_energy = raw_energy.reshape(self.num_graphs, -1)
            raw_forces = raw_forces.reshape(self.total_atoms, -1).narrow(
                0, 0, self.num_atoms
            )
            # Normalizer modules are capture-safe.  Elemental energy references
            # are composition constants and were precomputed from the real-only
            # batch because the generic helper performs a GPU-to-host shape
            # read that is unsafe during capture.
            energy = raw_energy
            if "energy" in self.trainer.normalizers:
                energy = self.trainer.normalizers["energy"](energy)
            if self.energy_element_reference is not None:
                energy = energy + self.energy_element_reference
            forces = raw_forces
            if "forces" in self.trainer.normalizers:
                forces = self.trainer.normalizers["forces"](forces)
        return (
            forces.reshape(self.num_atoms, 3).detach(),
            energy.reshape(-1)[0].detach(),
        )

    def capture(self, positions: Tensor) -> None:
        """Warm and capture the single fixed-capacity model graph."""

        if self.captured:
            raise RuntimeError("CUDA Graph has already been captured")
        sample_graph = self._build_real_graph(positions)
        self._staticize(sample_graph)

        # The captured path must consume the supplied fixed graph instead of
        # invoking the ragged OTF builder internally.
        self.model.backbone.otf_graph = False
        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            for _ in range(self.capture_warmup):
                self._static_forward()
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)

        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        capture_start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_forces, static_energy = self._static_forward()
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - capture_start
        self.capture_allocated_delta_bytes = (
            torch.cuda.memory_allocated(self.device) - allocated_before
        )
        self.capture_reserved_delta_bytes = (
            torch.cuda.memory_reserved(self.device) - reserved_before
        )
        self.graph = graph
        self.static_forces = static_forces
        self.static_energy = static_energy
        self.capture_count = 1
        self.captured = True

        # Validate two consecutive replays of the exact same static inputs.
        # This is setup work and is deliberately excluded from production
        # replay counts and MD timing.
        force_address = self.static_forces.data_ptr()
        energy_address = self.static_energy.data_ptr()
        graph.replay()
        first_forces = self.static_forces.clone()
        first_energy = self.static_energy.clone()
        graph.replay()
        second_forces = self.static_forces.clone()
        second_energy = self.static_energy.clone()
        torch.cuda.synchronize(self.device)
        self.total_replays += 2
        self.replay_output_addresses_stable = (
            self.static_forces.data_ptr() == force_address
            and self.static_energy.data_ptr() == energy_address
        )
        self.replay_stability_force_max_abs_error = float(
            (second_forces - first_forces).abs().max().item()
        )
        self.replay_stability_energy_abs_error = float(
            (second_energy - first_energy).abs().max().item()
        )
        if not self.replay_output_addresses_stable:
            raise CUDAGraphValidationError(
                "CUDA Graph output addresses changed between replays"
            )
        if (
            self.replay_stability_force_max_abs_error != 0.0
            or self.replay_stability_energy_abs_error != 0.0
        ):
            raise CUDAGraphValidationError(
                "Identical CUDA Graph replays were not bitwise stable: "
                f"energy_error={self.replay_stability_energy_abs_error}, "
                f"force_error={self.replay_stability_force_max_abs_error}"
            )

    def reset_production_stats(self) -> None:
        self.production_replays = 0
        self.production_calls = 0
        self.production_capacity_misses = 0
        self.production_min_edges = None
        self.production_max_edges = None

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        if not self.captured or self.graph is None:
            raise RuntimeError("CUDA Graph must be captured before replay")
        graph = self._build_real_graph(positions)
        num_edges = int(graph["edge_index"].shape[1])
        self.production_calls += 1
        if num_edges > self.edge_capacity:
            self.production_capacity_misses += 1
            raise CUDAGraphCapacityError(num_edges, self.edge_capacity)
        self._staticize(graph)
        self.graph.replay()
        self.total_replays += 1
        self.production_replays += 1
        self.production_min_edges = (
            num_edges
            if self.production_min_edges is None
            else min(self.production_min_edges, num_edges)
        )
        self.production_max_edges = (
            num_edges
            if self.production_max_edges is None
            else max(self.production_max_edges, num_edges)
        )
        assert self.static_forces is not None
        assert self.static_energy is not None
        return self.static_forces, self.static_energy

    def stats(self) -> dict[str, int | float | None]:
        hit_rate = (
            self.production_replays / self.production_calls
            if self.production_calls
            else 0.0
        )
        max_padding_fraction = (
            (self.edge_capacity - self.production_min_edges) / self.edge_capacity
            if self.production_min_edges is not None
            else None
        )
        return {
            "cuda_graph_capture_count": self.capture_count,
            "cuda_graph_production_capture_count": 0,
            "cuda_graph_total_replays": self.total_replays,
            "cuda_graph_production_replays": self.production_replays,
            "cuda_graph_production_calls": self.production_calls,
            "cuda_graph_capacity_misses": self.production_capacity_misses,
            "cuda_graph_hit_rate": hit_rate,
            "cuda_graph_edge_capacity": self.edge_capacity,
            "cuda_graph_min_real_edges": self.production_min_edges,
            "cuda_graph_max_real_edges": self.production_max_edges,
            "cuda_graph_max_padding_fraction": max_padding_fraction,
            "cuda_graph_dummy_atoms": self.dummy_atoms,
            "cuda_graph_capture_warmup": self.capture_warmup,
            "cuda_graph_replay_output_addresses_stable": (
                self.replay_output_addresses_stable
            ),
            "cuda_graph_replay_stability_energy_abs_error_eV": (
                self.replay_stability_energy_abs_error
            ),
            "cuda_graph_replay_stability_force_max_abs_error_eV_per_A": (
                self.replay_stability_force_max_abs_error
            ),
            "cuda_graph_capture_wall_time_s": self.capture_wall_time_s,
            "cuda_graph_capture_allocated_delta_gib": (
                self.capture_allocated_delta_bytes / 1024**3
            ),
            "cuda_graph_capture_reserved_delta_gib": (
                self.capture_reserved_delta_bytes / 1024**3
            ),
        }
