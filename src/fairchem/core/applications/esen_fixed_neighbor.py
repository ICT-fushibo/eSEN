"""Capture-safe fixed-shape PBC neighbor construction for eSEN MD.

The official :func:`radius_graph_pbc` implementation produces ragged tensors
and reads several CUDA values on the host.  This module enumerates the same
single-structure PBC candidate universe once, then selects a fixed number of
slots per centre atom using only fixed-shape tensor operations.  Unused slots
are routed exclusively between dummy atoms.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def neighbor_capacity_from_probe(
    maximum_neighbors: int,
    *,
    margin: float = 0.10,
    slot_step: int = 8,
) -> int:
    """Add probe headroom and round a per-atom neighbor capacity upward."""

    if maximum_neighbors < 1:
        raise ValueError("maximum_neighbors must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if slot_step < 1:
        raise ValueError("slot_step must be positive")
    required = max(
        maximum_neighbors + 1,
        math.ceil(maximum_neighbors * (1.0 + margin)),
    )
    return int(math.ceil(required / slot_step) * slot_step)


def maximum_neighbors_in_graph(edge_index: Tensor, num_atoms: int) -> int:
    """Return the largest centre-atom degree in an official eSEN graph."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_atoms < 1:
        raise ValueError("num_atoms must be positive")
    if edge_index.shape[1] == 0:
        return 0
    counts = torch.bincount(edge_index[1], minlength=num_atoms)
    return int(counts.max().item())


def _pbc_repetitions(cell: Tensor, cutoff: float, pbc: Tensor) -> tuple[int, int, int]:
    """Match the plane-distance repetition calculation in radius_graph_pbc."""

    cell64 = cell.detach().to(device="cpu", dtype=torch.float64).reshape(3, 3)
    pbc_cpu = pbc.detach().to(device="cpu", dtype=torch.bool).reshape(3)
    cross_a2a3 = torch.cross(cell64[1], cell64[2], dim=0)
    volume = torch.dot(cell64[0], cross_a2a3)
    if not bool(torch.isfinite(volume)) or float(volume.abs()) == 0.0:
        raise ValueError("Cannot enumerate PBC images for a singular cell")
    reciprocal = (
        cross_a2a3,
        torch.cross(cell64[2], cell64[0], dim=0),
        torch.cross(cell64[0], cell64[1], dim=0),
    )
    repetitions = []
    for axis in range(3):
        if bool(pbc_cpu[axis]):
            inv_plane_distance = torch.linalg.vector_norm(
                reciprocal[axis] / volume
            )
            repetitions.append(int(torch.ceil(cutoff * inv_plane_distance).item()))
        else:
            repetitions.append(0)
    return tuple(repetitions)  # type: ignore[return-value]


class FixedShapePBCNeighborBuilder:
    """Build one fixed-capacity graph for a single periodic structure.

    The output tensors always have ``num_atoms * neighbors_per_atom`` edges.
    Active edges preserve the official candidate-enumeration order.  Inactive
    slots are dummy self-edges with a far periodic offset.
    """

    def __init__(
        self,
        *,
        num_atoms: int,
        cell: Tensor,
        pbc: Tensor,
        cutoff: float,
        neighbors_per_atom: int,
        dummy_atoms: int,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
        output_edge_index: Tensor | None = None,
        output_cell_offsets: Tensor | None = None,
    ) -> None:
        if num_atoms < 1:
            raise ValueError("num_atoms must be positive")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if neighbors_per_atom < 1:
            raise ValueError("neighbors_per_atom must be positive")
        if dummy_atoms < 1:
            raise ValueError("dummy_atoms must be positive")
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be positive")
        if degeneracy_tolerance < 0:
            raise ValueError("degeneracy_tolerance must be non-negative")

        self.num_atoms = int(num_atoms)
        self.cutoff = float(cutoff)
        self.neighbors_per_atom = int(neighbors_per_atom)
        self.dummy_atoms = int(dummy_atoms)
        self.max_neighbors = int(max_neighbors)
        self.degeneracy_tolerance = float(degeneracy_tolerance)
        self.device = cell.device
        self.position_dtype = cell.dtype
        self.edge_capacity = self.num_atoms * self.neighbors_per_atom
        self.repetitions = _pbc_repetitions(cell, cutoff, pbc)

        axes = [
            torch.arange(
                -rep,
                rep + 1,
                device=self.device,
                dtype=self.position_dtype,
            )
            for rep in self.repetitions
        ]
        unit_cell = torch.cartesian_prod(*axes).reshape(-1, 3)
        self.unit_cell_offsets = unit_cell.contiguous()
        self.num_cells = int(unit_cell.shape[0])
        self.candidates_per_atom = self.num_atoms * self.num_cells
        if self.neighbors_per_atom > self.candidates_per_atom:
            raise ValueError(
                "neighbors_per_atom cannot exceed the fixed candidate count: "
                f"{self.neighbors_per_atom} > {self.candidates_per_atom}"
            )

        self.cell = cell.detach().reshape(3, 3)
        self.candidate_sources = torch.arange(
            self.num_atoms, device=self.device, dtype=torch.long
        ).repeat_interleave(self.num_cells)
        self.candidate_cell_offsets = self.unit_cell_offsets.repeat(
            self.num_atoms, 1
        )
        self.candidate_ids = torch.arange(
            self.candidates_per_atom, device=self.device, dtype=torch.long
        ).view(1, -1)

        if output_edge_index is None:
            output_edge_index = torch.empty(
                2, self.edge_capacity, device=self.device, dtype=torch.long
            )
        if output_cell_offsets is None:
            output_cell_offsets = torch.empty(
                self.edge_capacity,
                3,
                device=self.device,
                dtype=self.position_dtype,
            )
        if output_edge_index.shape != (2, self.edge_capacity):
            raise ValueError("output_edge_index has the wrong shape")
        if output_cell_offsets.shape != (self.edge_capacity, 3):
            raise ValueError("output_cell_offsets has the wrong shape")
        self.edge_index = output_edge_index
        self.cell_offsets = output_cell_offsets

        slot_ids = torch.arange(
            self.edge_capacity, device=self.device, dtype=torch.long
        )
        self.dummy_sinks = (
            slot_ids.remainder(self.dummy_atoms) + self.num_atoms
        )
        vector_norms = torch.linalg.vector_norm(self.cell, dim=1)
        axis = int(torch.argmax(vector_norms).item())
        axis_norm = float(vector_norms[axis].item())
        if not math.isfinite(axis_norm) or axis_norm <= 0:
            raise ValueError("Cannot construct dummy offset from an invalid cell")
        far_shift = max(2, math.ceil((self.cutoff + 1.0) / axis_norm) + 1)
        self.padding_cell_offsets = self.cell_offsets.new_zeros(
            self.edge_capacity, 3
        )
        self.padding_cell_offsets[:, axis] = far_shift

        # Device-resident production diagnostics.  Updating these tensors is
        # capture-safe and does not introduce a host synchronization.
        self.build_calls = torch.zeros((), device=self.device, dtype=torch.long)
        self.capacity_misses = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.first_overflow_step = torch.full(
            (), -1, device=self.device, dtype=torch.long
        )
        self.current_real_edges = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.minimum_real_edges = torch.full(
            (), self.edge_capacity, device=self.device, dtype=torch.long
        )
        self.maximum_real_edges = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_raw_neighbors = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_included_neighbors = torch.zeros(
            (), device=self.device, dtype=torch.long
        )

    def reset_stats(self) -> None:
        """Reset production counters without changing their addresses."""

        self.build_calls.zero_()
        self.capacity_misses.zero_()
        self.first_overflow_step.fill_(-1)
        self.current_real_edges.zero_()
        self.minimum_real_edges.fill_(self.edge_capacity)
        self.maximum_real_edges.zero_()
        self.maximum_raw_neighbors.zero_()
        self.maximum_included_neighbors.zero_()

    def build(
        self,
        positions: Tensor,
        *,
        step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Update and return the fixed-address edge and cell-offset tensors."""

        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Expected positions {(self.num_atoms, 3)}, got {positions.shape}"
            )
        if positions.device != self.device:
            raise ValueError(
                f"Positions must be on {self.device}, got {positions.device}"
            )

        with torch.no_grad():
            shifted_sources = (
                positions.index_select(0, self.candidate_sources)
                + torch.mm(
                    self.candidate_cell_offsets.to(dtype=positions.dtype),
                    self.cell.to(dtype=positions.dtype),
                )
            )
            delta = shifted_sources.unsqueeze(0) - positions.unsqueeze(1)
            distance_sqr = delta.square().sum(dim=-1)
            cutoff_sqr = self.cutoff * self.cutoff
            valid = (distance_sqr <= cutoff_sqr) & (distance_sqr > 0.0001)
            raw_counts = valid.sum(dim=1)

            masked_distance = torch.where(
                valid,
                distance_sqr,
                torch.full_like(distance_sqr, torch.inf),
            )
            selection_k = min(
                self.max_neighbors + 1, self.candidates_per_atom
            )
            nearest = torch.topk(
                masked_distance,
                k=selection_k,
                dim=1,
                largest=False,
                sorted=True,
            ).values
            if self.candidates_per_atom > self.max_neighbors:
                effective_cutoff = (
                    nearest[:, self.max_neighbors] + self.degeneracy_tolerance
                )
            else:
                effective_cutoff = torch.full_like(
                    raw_counts, cutoff_sqr, dtype=distance_sqr.dtype
                )
            effective_cutoff = torch.where(
                raw_counts > self.max_neighbors,
                effective_cutoff,
                torch.full_like(effective_cutoff, cutoff_sqr),
            )
            included = valid & (
                distance_sqr <= effective_cutoff.unsqueeze(1)
            )
            included_counts = included.sum(dim=1)

            # Selecting the smallest original candidate indices restores the
            # enumeration order used by masked_select in radius_graph_pbc.
            candidate_order = torch.where(
                included,
                self.candidate_ids.expand(self.num_atoms, -1),
                torch.full_like(
                    self.candidate_ids.expand(self.num_atoms, -1),
                    self.candidates_per_atom,
                ),
            )
            selected = torch.topk(
                candidate_order,
                k=self.neighbors_per_atom,
                dim=1,
                largest=False,
                sorted=True,
            ).values
            slot_valid = selected < self.candidates_per_atom
            safe_selected = selected.clamp_max(self.candidates_per_atom - 1)

            sources = self.candidate_sources.index_select(
                0, safe_selected.reshape(-1)
            )
            centres = torch.arange(
                self.num_atoms, device=self.device, dtype=torch.long
            ).view(-1, 1).expand(-1, self.neighbors_per_atom).reshape(-1)
            selected_offsets = self.candidate_cell_offsets.index_select(
                0, safe_selected.reshape(-1)
            )
            flat_valid = slot_valid.reshape(-1)
            self.edge_index[0].copy_(
                torch.where(flat_valid, sources, self.dummy_sinks)
            )
            self.edge_index[1].copy_(
                torch.where(flat_valid, centres, self.dummy_sinks)
            )
            self.cell_offsets.copy_(
                torch.where(
                    flat_valid.unsqueeze(1),
                    selected_offsets.to(dtype=self.cell_offsets.dtype),
                    self.padding_cell_offsets,
                )
            )

            real_edges = flat_valid.sum()
            overflow = (included_counts > self.neighbors_per_atom).any()
            call_step = self.build_calls if step is None else step
            self.current_real_edges.copy_(real_edges)
            self.minimum_real_edges.copy_(
                torch.minimum(self.minimum_real_edges, real_edges)
            )
            self.maximum_real_edges.copy_(
                torch.maximum(self.maximum_real_edges, real_edges)
            )
            self.maximum_raw_neighbors.copy_(
                torch.maximum(self.maximum_raw_neighbors, raw_counts.max())
            )
            self.maximum_included_neighbors.copy_(
                torch.maximum(
                    self.maximum_included_neighbors, included_counts.max()
                )
            )
            self.capacity_misses.add_(overflow.to(dtype=torch.long))
            first = (self.first_overflow_step < 0) & overflow
            self.first_overflow_step.copy_(
                torch.where(first, call_step, self.first_overflow_step)
            )
            self.build_calls.add_(1)
        return self.edge_index, self.cell_offsets

    def stats(self) -> dict[str, Any]:
        """Synchronize once and return host-side builder diagnostics."""

        calls = int(self.build_calls.item())
        minimum = int(self.minimum_real_edges.item()) if calls else None
        maximum = int(self.maximum_real_edges.item()) if calls else None
        misses = int(self.capacity_misses.item())
        first_overflow = int(self.first_overflow_step.item())
        return {
            "fixed_builder_build_calls": calls,
            "fixed_builder_capacity_misses": misses,
            "fixed_builder_first_overflow_step": (
                first_overflow if first_overflow >= 0 else None
            ),
            "fixed_builder_edge_capacity": self.edge_capacity,
            "fixed_builder_neighbors_per_atom": self.neighbors_per_atom,
            "fixed_builder_min_real_edges": minimum,
            "fixed_builder_max_real_edges": maximum,
            "fixed_builder_max_padding_fraction": (
                None
                if minimum is None
                else (self.edge_capacity - minimum) / self.edge_capacity
            ),
            "fixed_builder_max_raw_neighbors": int(
                self.maximum_raw_neighbors.item()
            ),
            "fixed_builder_max_included_neighbors": int(
                self.maximum_included_neighbors.item()
            ),
            "fixed_builder_candidate_universe_size": (
                self.num_atoms * self.candidates_per_atom
            ),
            "fixed_builder_candidates_per_atom": self.candidates_per_atom,
            "fixed_builder_num_pbc_cells": self.num_cells,
            "fixed_builder_pbc_repetitions": list(self.repetitions),
            "fixed_builder_degeneracy_tolerance": self.degeneracy_tolerance,
            "fixed_builder_max_neighbors": self.max_neighbors,
        }
