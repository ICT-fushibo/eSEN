"""Capture-safe fixed-shape PBC neighbor construction for eSEN MD.

The official :func:`radius_graph_pbc` implementation produces ragged tensors
and reads several CUDA values on the host.  This module enumerates the same
single-structure PBC candidate universe once, then selects a fixed number of
slots per centre atom using only fixed-shape tensor operations.  Unused slots
are routed exclusively between dummy atoms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    counts = neighbor_counts_in_graph(edge_index, num_atoms)
    return int(counts.max().item())


def neighbor_counts_in_graph(edge_index: Tensor, num_atoms: int) -> Tensor:
    """Return the centre-atom degree vector for an official eSEN graph."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_atoms < 1:
        raise ValueError("num_atoms must be positive")
    if edge_index.shape[1] == 0:
        return torch.zeros(
            num_atoms, device=edge_index.device, dtype=torch.long
        )
    return torch.bincount(edge_index[1], minlength=num_atoms)[:num_atoms]


def species_neighbor_capacities_from_probe(
    maximum_neighbors_by_atom: Tensor | Sequence[int],
    atomic_numbers: Tensor | Sequence[int],
    *,
    margin: float = 0.10,
    slot_step: int = 8,
) -> tuple[int, ...]:
    """Derive one conservative static slot capacity per chemical species.

    Every atom of a species receives the largest capacity required by any atom
    of that species during the probe.  This remains safe when equivalent atoms
    exchange local environments while avoiding the global worst-case padding
    imposed by a single uniform capacity.
    """

    maxima = torch.as_tensor(
        maximum_neighbors_by_atom, device="cpu", dtype=torch.long
    ).reshape(-1)
    numbers = torch.as_tensor(
        atomic_numbers, device="cpu", dtype=torch.long
    ).reshape(-1)
    if maxima.numel() < 1 or maxima.shape != numbers.shape:
        raise ValueError(
            "maximum_neighbors_by_atom and atomic_numbers must be non-empty "
            "vectors with the same shape"
        )
    if bool((maxima < 0).any()):
        raise ValueError("maximum neighbor counts must be non-negative")
    if bool((numbers < 1).any()):
        raise ValueError("atomic numbers must be positive")

    capacities = torch.empty_like(maxima)
    for atomic_number in torch.unique(numbers, sorted=True):
        species_mask = numbers == atomic_number
        species_maximum = int(maxima[species_mask].max().item())
        # An isolated species still needs at least one static padding slot.
        capacity = neighbor_capacity_from_probe(
            max(1, species_maximum), margin=margin, slot_step=slot_step
        )
        capacities[species_mask] = capacity
    return tuple(int(value) for value in capacities.tolist())


def atom_neighbor_capacities_from_probe(
    maximum_neighbors_by_atom: Tensor | Sequence[int],
    *,
    margin: float = 0.10,
    slot_step: int = 8,
) -> tuple[int, ...]:
    """Derive a rounded static slot capacity for every atom.

    This policy targets systems with heterogeneous local coordination even
    when atoms share the same element.  It is intentionally opt-in because a
    long diffusive trajectory can outgrow a probe-derived per-atom bound; the
    fixed builder's device-side capacity-miss telemetry remains authoritative.
    """

    maxima = torch.as_tensor(
        maximum_neighbors_by_atom, device="cpu", dtype=torch.long
    ).reshape(-1)
    if maxima.numel() < 1:
        raise ValueError("maximum_neighbors_by_atom must be non-empty")
    if bool((maxima < 0).any()):
        raise ValueError("maximum neighbor counts must be non-negative")
    return tuple(
        neighbor_capacity_from_probe(
            max(1, int(maximum)), margin=margin, slot_step=slot_step
        )
        for maximum in maxima.tolist()
    )


def auto_neighbor_capacities_from_probe(
    maximum_neighbors_by_atom: Tensor | Sequence[int],
    *,
    margin: float = 0.10,
    slot_step: int = 8,
    minimum_reduction: float = 0.05,
    guard_slots: int = 0,
) -> tuple[tuple[int, ...] | None, float]:
    """Select guarded per-atom slots only when they remove enough padding.

    The decision is made once after the eager probe and before CUDA Graph
    capture.  Returning ``None`` selects the original uniform-capacity path,
    so replay contains no policy branch.  The returned reduction is the
    fractional edge-capacity reduction of the per-atom candidate relative to
    the uniform allocation, regardless of which path is selected.

    ``guard_slots`` promotes every heterogeneous capacity by that many slot
    buckets, capped at the uniform capacity.  A one-slot guard is the safe
    CAP1-auto policy: it covers one additional rounded neighbor bucket while
    automatically falling back to uniform when the protected allocation no
    longer clears ``minimum_reduction``.  The default remains zero so existing
    CAP1-auto experiments retain their exact allocation semantics.
    """

    if (
        not math.isfinite(minimum_reduction)
        or not 0.0 <= minimum_reduction <= 1.0
    ):
        raise ValueError("minimum_reduction must be finite and between 0 and 1")
    if guard_slots < 0:
        raise ValueError("guard_slots must be non-negative")
    maxima = torch.as_tensor(
        maximum_neighbors_by_atom, device="cpu", dtype=torch.long
    ).reshape(-1)
    if maxima.numel() < 1:
        raise ValueError("maximum_neighbors_by_atom must be non-empty")
    if bool((maxima < 0).any()):
        raise ValueError("maximum neighbor counts must be non-negative")

    atom_capacities = atom_neighbor_capacities_from_probe(
        maxima,
        margin=margin,
        slot_step=slot_step,
    )
    uniform_capacity = neighbor_capacity_from_probe(
        max(1, int(maxima.max().item())),
        margin=margin,
        slot_step=slot_step,
    )
    if guard_slots:
        guard = guard_slots * slot_step
        atom_capacities = tuple(
            min(uniform_capacity, capacity + guard)
            for capacity in atom_capacities
        )
    uniform_edge_capacity = uniform_capacity * int(maxima.numel())
    reduction = (
        uniform_edge_capacity - sum(atom_capacities)
    ) / uniform_edge_capacity
    selected = atom_capacities if reduction >= minimum_reduction else None
    return selected, float(reduction)


def elastic_neighbor_capacities_from_probe(
    maximum_neighbors_by_atom: Tensor | Sequence[int],
    safe_capacities: Tensor | Sequence[int],
    *,
    margin: float = 0.0,
    slot_step: int = 4,
    minimum_reduction: float = 0.05,
) -> tuple[tuple[int, ...], bool, float, tuple[int, ...]]:
    """Choose a compact CAP2 start only when it beats CAP1-auto-safe.

    The compact candidate keeps at least one neighbor beyond every probed
    per-atom maximum and rounds to ``slot_step``.  ``safe_capacities`` is the
    effective CAP1-auto-safe allocation (including a uniform fallback).  The
    selected allocation is therefore never larger than the frozen baseline.
    """

    if (
        not math.isfinite(minimum_reduction)
        or not 0.0 <= minimum_reduction <= 1.0
    ):
        raise ValueError("minimum_reduction must be finite and between 0 and 1")
    compact = atom_neighbor_capacities_from_probe(
        maximum_neighbors_by_atom,
        margin=margin,
        slot_step=slot_step,
    )
    safe = tuple(
        int(value)
        for value in torch.as_tensor(
            safe_capacities, device="cpu", dtype=torch.long
        )
        .reshape(-1)
        .tolist()
    )
    if len(compact) != len(safe) or not safe:
        raise ValueError("safe_capacities must match the probe atom count")
    if any(value < 1 for value in safe):
        raise ValueError("safe capacities must be positive")
    safe_edges = sum(safe)
    reduction = (safe_edges - sum(compact)) / safe_edges
    selected = compact if reduction >= minimum_reduction else safe
    return selected, selected == compact, float(reduction), compact


def promote_elastic_neighbor_capacities(
    current_capacities: Tensor | Sequence[int],
    maximum_required_by_atom: Tensor | Sequence[int],
    atomic_numbers: Tensor | Sequence[int],
    *,
    promotion_index: int,
    species_slot_step: int = 4,
    uniform_margin: float = 0.10,
    uniform_slot_step: int = 8,
) -> tuple[tuple[int, ...], str]:
    """Return the next monotonic CAP2 allocation after a failed transaction.

    Promotion zero groups the actual window demand by chemical species and
    keeps one spare neighbor before four-slot rounding.  Promotion one uses a
    conservative uniform allocation.  Higher promotion indices are rejected;
    the controller treats them as an unrecoverable capacity overflow.
    """

    current = torch.as_tensor(
        current_capacities, device="cpu", dtype=torch.long
    ).reshape(-1)
    required = torch.as_tensor(
        maximum_required_by_atom, device="cpu", dtype=torch.long
    ).reshape(-1)
    numbers = torch.as_tensor(
        atomic_numbers, device="cpu", dtype=torch.long
    ).reshape(-1)
    if current.numel() < 1 or current.shape != required.shape:
        raise ValueError("current capacities and actual demand must match")
    if numbers.shape != current.shape:
        raise ValueError("atomic_numbers must match the capacity vector")
    if bool((current < 1).any()) or bool((required < 0).any()):
        raise ValueError("capacities must be positive and demand non-negative")
    if promotion_index == 0:
        promoted = current.clone()
        for atomic_number in torch.unique(numbers, sorted=True):
            mask = numbers == atomic_number
            species_required = int(required[mask].max().item())
            species_capacity = neighbor_capacity_from_probe(
                max(1, species_required),
                margin=0.0,
                slot_step=species_slot_step,
            )
            promoted[mask] = torch.maximum(
                promoted[mask],
                torch.full_like(promoted[mask], species_capacity),
            )
        return tuple(int(value) for value in promoted.tolist()), "species"
    if promotion_index == 1:
        uniform_capacity = neighbor_capacity_from_probe(
            max(1, int(required.max().item())),
            margin=uniform_margin,
            slot_step=uniform_slot_step,
        )
        promoted = torch.maximum(
            current, torch.full_like(current, uniform_capacity)
        )
        return tuple(int(value) for value in promoted.tolist()), "uniform"
    raise ValueError("CAP2 supports at most two capacity promotions")


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


def _cell_plane_distances(cell: Tensor) -> tuple[float, float, float]:
    """Return the three real-space distances between opposite cell planes."""

    cell64 = cell.detach().to(device="cpu", dtype=torch.float64).reshape(3, 3)
    cross_a2a3 = torch.cross(cell64[1], cell64[2], dim=0)
    volume = torch.dot(cell64[0], cross_a2a3)
    if not bool(torch.isfinite(volume)) or float(volume.abs()) == 0.0:
        raise ValueError("Cannot construct a cell list for a singular cell")
    reciprocal = (
        cross_a2a3,
        torch.cross(cell64[2], cell64[0], dim=0),
        torch.cross(cell64[0], cell64[1], dim=0),
    )
    distances = tuple(
        float(1.0 / torch.linalg.vector_norm(vector / volume))
        for vector in reciprocal
    )
    if any(not math.isfinite(value) or value <= 0 for value in distances):
        raise ValueError("Cannot construct a cell list for an invalid cell")
    return distances  # type: ignore[return-value]


def cell_list_grid_shape(
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
) -> tuple[int, int, int]:
    """Choose a conservative fixed cell-list grid for a periodic structure.

    Each bin is at least ``cutoff`` wide in the corresponding reciprocal-plane
    direction whenever the periodic cell is large enough.  Small cells retain
    one bin and use a wider neighboring-bin stencil so multiple periodic images
    remain visible.
    """

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    pbc_cpu = pbc.detach().to(device="cpu", dtype=torch.bool).reshape(3)
    if not bool(pbc_cpu.all()):
        raise ValueError("GPU cell-list construction currently requires 3D PBC")
    return tuple(
        max(1, int(math.floor(distance / cutoff)))
        for distance in _cell_plane_distances(cell)
    )  # type: ignore[return-value]


def _cell_list_bin_ids(
    positions: Tensor,
    cell: Tensor,
    grid_shape: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return wrapped bin ids, integer image indices, and bin coordinates."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_atoms, 3]")
    grid = torch.as_tensor(
        tuple(int(value) for value in grid_shape),
        device=positions.device,
        dtype=torch.long,
    )
    if grid.shape != (3,) or bool((grid < 1).any()):
        raise ValueError("grid_shape must contain three positive values")
    inverse_cell = torch.linalg.inv(cell.to(dtype=positions.dtype).reshape(3, 3))
    fractional = torch.mm(positions, inverse_cell)
    image_indices = torch.floor(fractional).to(dtype=torch.long)
    wrapped = fractional - image_indices.to(dtype=fractional.dtype)
    bin_coordinates = torch.floor(
        wrapped * grid.to(dtype=wrapped.dtype)
    ).to(dtype=torch.long)
    bin_coordinates = torch.minimum(bin_coordinates, grid - 1)
    bin_ids = (
        (bin_coordinates[:, 0] * grid[1] + bin_coordinates[:, 1])
        * grid[2]
        + bin_coordinates[:, 2]
    )
    return bin_ids, image_indices, bin_coordinates


def cell_list_max_occupancy(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    *,
    grid_shape: Sequence[int] | None = None,
) -> Tensor:
    """Return the maximum GPU bin occupancy without a host synchronization."""

    grid_shape = (
        cell_list_grid_shape(cell, pbc, cutoff)
        if grid_shape is None
        else tuple(int(value) for value in grid_shape)
    )
    bin_ids, _, _ = _cell_list_bin_ids(positions, cell, grid_shape)
    num_bins = math.prod(grid_shape)
    counts = torch.zeros(
        num_bins, device=positions.device, dtype=torch.long
    )
    counts.scatter_add_(0, bin_ids, torch.ones_like(bin_ids))
    return counts.max()


def cell_list_bin_capacity_from_probe(
    maximum_occupancy: int,
    *,
    margin: float = 0.25,
    slot_step: int = 8,
) -> int:
    """Add headroom to a probed bin occupancy and round it upward."""

    if maximum_occupancy < 1:
        raise ValueError("maximum_occupancy must be positive")
    return neighbor_capacity_from_probe(
        maximum_occupancy, margin=margin, slot_step=slot_step
    )


class FixedShapePBCNeighborBuilder:
    """Build one fixed-capacity graph for a single periodic structure.

    With uniform slots the output tensors have
    ``num_atoms * neighbors_per_atom`` edges; optional heterogeneous capacities
    use their fixed sum instead.  Active edges preserve the official candidate
    enumeration order.  Inactive slots are dummy self-edges with a far
    periodic offset.
    """

    def __init__(
        self,
        *,
        num_atoms: int,
        cell: Tensor,
        pbc: Tensor,
        cutoff: float,
        neighbors_per_atom: int,
        neighbor_capacities: Tensor | Sequence[int] | None = None,
        capacity_policy: str = "uniform",
        dummy_atoms: int,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
        overflow_to_dummy_only: bool = False,
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
        if neighbor_capacities is None:
            capacity_values = torch.full(
                (self.num_atoms,), int(neighbors_per_atom), dtype=torch.long
            )
        else:
            capacity_values = torch.as_tensor(
                neighbor_capacities, device="cpu", dtype=torch.long
            ).reshape(-1)
            if capacity_values.shape != (self.num_atoms,):
                raise ValueError(
                    "neighbor_capacities must contain one value per real atom"
                )
            if bool((capacity_values < 1).any()):
                raise ValueError("neighbor capacities must be positive")
        self.capacity_policy = str(capacity_policy)
        self.neighbors_per_atom = int(capacity_values.max().item())
        self._neighbor_capacities_cpu = tuple(
            int(value) for value in capacity_values.tolist()
        )
        self.dummy_atoms = int(dummy_atoms)
        self.max_neighbors = int(max_neighbors)
        self.degeneracy_tolerance = float(degeneracy_tolerance)
        self.overflow_to_dummy_only = bool(overflow_to_dummy_only)
        self.device = cell.device
        self.position_dtype = cell.dtype
        self.edge_capacity = int(capacity_values.sum().item())
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

        self.neighbor_capacities = capacity_values.to(device=self.device)
        self.slot_centres = torch.repeat_interleave(
            torch.arange(self.num_atoms, device=self.device, dtype=torch.long),
            self.neighbor_capacities,
        )
        capacity_starts = (
            torch.cumsum(self.neighbor_capacities, dim=0)
            - self.neighbor_capacities
        )
        self.slot_ranks = torch.arange(
            self.edge_capacity, device=self.device, dtype=torch.long
        ) - capacity_starts.index_select(0, self.slot_centres)
        self.slot_selection_indices = (
            self.slot_centres * self.neighbors_per_atom + self.slot_ranks
        )
        unique_capacities, unique_counts = torch.unique(
            capacity_values, sorted=True, return_counts=True
        )
        self.capacity_histogram = {
            int(capacity): int(count)
            for capacity, count in zip(
                unique_capacities.tolist(), unique_counts.tolist()
            )
        }

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
        padding_distance = axis_norm * far_shift
        self.sink_nonzero_shift_verified = bool(far_shift != 0)
        self.sink_cutoff_zero_verified = bool(padding_distance > self.cutoff)

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
        self.maximum_capacity_excess = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_overflow_required = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_overflow_capacity = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_included_neighbors_by_atom = torch.zeros(
            self.num_atoms, device=self.device, dtype=torch.long
        )
        self.minimum_padding_edges = torch.full(
            (), self.edge_capacity, device=self.device, dtype=torch.long
        )
        self.maximum_padding_edges = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.overflow_dummy_only_replays = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.window_capacity_misses = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.window_overflow_dummy_only_replays = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.window_maximum_included_neighbors_by_atom = torch.zeros(
            self.num_atoms, device=self.device, dtype=torch.long
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
        self.maximum_capacity_excess.zero_()
        self.maximum_overflow_required.zero_()
        self.maximum_overflow_capacity.zero_()
        self.maximum_included_neighbors_by_atom.zero_()
        self.minimum_padding_edges.fill_(self.edge_capacity)
        self.maximum_padding_edges.zero_()
        self.overflow_dummy_only_replays.zero_()
        self.reset_window_stats()

    def reset_window_stats(self) -> None:
        """Reset transaction-local telemetry without changing addresses."""

        self.window_capacity_misses.zero_()
        self.window_overflow_dummy_only_replays.zero_()
        self.window_maximum_included_neighbors_by_atom.zero_()

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

            self._select_write_and_update_stats(
                included,
                raw_counts,
                included_counts,
                step=step,
            )
        return self.edge_index, self.cell_offsets

    def _select_write_and_update_stats(
        self,
        included: Tensor,
        raw_counts: Tensor,
        included_counts: Tensor,
        *,
        step: Tensor | None,
    ) -> None:
        """Select configured slots, write static outputs, and update counters."""

        # Selecting the smallest original candidate indices restores the
        # enumeration order used by masked_select in radius_graph_pbc.
        candidate_ids = self.candidate_ids.expand(self.num_atoms, -1)
        candidate_order = torch.where(
            included,
            candidate_ids,
            torch.full_like(candidate_ids, self.candidates_per_atom),
        )
        selected = torch.topk(
            candidate_order,
            k=self.neighbors_per_atom,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        flat_selected = selected.reshape(-1).index_select(
            0, self.slot_selection_indices
        )
        flat_valid = flat_selected < self.candidates_per_atom
        safe_selected = flat_selected.clamp_max(self.candidates_per_atom - 1)

        sources = self.candidate_sources.index_select(0, safe_selected)
        selected_offsets = self.candidate_cell_offsets.index_select(
            0, safe_selected
        )
        self._write_and_update_stats(
            sources,
            selected_offsets,
            flat_valid,
            raw_counts,
            included_counts,
            step=step,
        )

    def _write_and_update_stats(
        self,
        sources: Tensor,
        selected_offsets: Tensor,
        flat_valid: Tensor,
        raw_counts: Tensor,
        included_counts: Tensor,
        *,
        step: Tensor | None,
        extra_overflow: Tensor | None = None,
        extra_required: Tensor | None = None,
        extra_capacity: Tensor | None = None,
    ) -> None:
        """Write fixed slots and update capture-safe capacity telemetry."""

        capacity_excess = torch.clamp_min(
            included_counts - self.neighbor_capacities, 0
        )
        current_excess, overflow_atom = capacity_excess.max(dim=0)
        current_required = included_counts.index_select(
            0, overflow_atom.reshape(1)
        ).reshape(())
        current_capacity = self.neighbor_capacities.index_select(
            0, overflow_atom.reshape(1)
        ).reshape(())
        overflow = current_excess > 0
        if extra_overflow is not None:
            if extra_required is None or extra_capacity is None:
                raise ValueError(
                    "extra overflow requires its required and capacity values"
                )
            extra_excess = torch.clamp_min(
                extra_required - extra_capacity, 0
            )
            replace_with_extra = extra_excess > current_excess
            current_excess = torch.maximum(current_excess, extra_excess)
            current_required = torch.where(
                replace_with_extra, extra_required, current_required
            )
            current_capacity = torch.where(
                replace_with_extra, extra_capacity, current_capacity
            )
            overflow = overflow | extra_overflow

        if self.overflow_to_dummy_only:
            output_valid = flat_valid & ~overflow
            # Slot ids already map round-robin to the fixed dummy sinks.  On
            # overflow every slot becomes padding, so the distribution is
            # exactly balanced without adding a replay-time prefix scan.
            padding_sinks = self.dummy_sinks
        else:
            output_valid = flat_valid
            padding_sinks = self.dummy_sinks
        self.edge_index[0].copy_(
            torch.where(output_valid, sources, padding_sinks)
        )
        self.edge_index[1].copy_(
            torch.where(output_valid, self.slot_centres, padding_sinks)
        )
        self.cell_offsets.copy_(
            torch.where(
                output_valid.unsqueeze(1),
                selected_offsets.to(dtype=self.cell_offsets.dtype),
                self.padding_cell_offsets,
            )
        )

        real_edges = output_valid.sum()
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
        if self.overflow_to_dummy_only:
            padding_edges = self.edge_capacity - real_edges
            self.maximum_included_neighbors_by_atom.copy_(
                torch.maximum(
                    self.maximum_included_neighbors_by_atom, included_counts
                )
            )
            self.window_maximum_included_neighbors_by_atom.copy_(
                torch.maximum(
                    self.window_maximum_included_neighbors_by_atom,
                    included_counts,
                )
            )
            self.minimum_padding_edges.copy_(
                torch.minimum(self.minimum_padding_edges, padding_edges)
            )
            self.maximum_padding_edges.copy_(
                torch.maximum(self.maximum_padding_edges, padding_edges)
            )
        replace_overflow = current_excess > self.maximum_capacity_excess
        self.maximum_capacity_excess.copy_(
            torch.where(
                replace_overflow,
                current_excess,
                self.maximum_capacity_excess,
            )
        )
        self.maximum_overflow_required.copy_(
            torch.where(
                replace_overflow,
                current_required,
                self.maximum_overflow_required,
            )
        )
        self.maximum_overflow_capacity.copy_(
            torch.where(
                replace_overflow,
                current_capacity,
                self.maximum_overflow_capacity,
            )
        )
        self.capacity_misses.add_(overflow.to(dtype=torch.long))
        if self.overflow_to_dummy_only:
            self.window_capacity_misses.add_(overflow.to(dtype=torch.long))
            self.overflow_dummy_only_replays.add_(overflow.to(dtype=torch.long))
            self.window_overflow_dummy_only_replays.add_(
                overflow.to(dtype=torch.long)
            )
        first = (self.first_overflow_step < 0) & overflow
        self.first_overflow_step.copy_(
            torch.where(first, call_step, self.first_overflow_step)
        )
        self.build_calls.add_(1)

    def window_stats(self) -> dict[str, Any]:
        """Synchronize once and return transaction-local demand telemetry."""

        return {
            "fixed_builder_window_capacity_misses": int(
                self.window_capacity_misses.item()
            ),
            "fixed_builder_window_overflow_dummy_only_replays": int(
                self.window_overflow_dummy_only_replays.item()
            ),
            "fixed_builder_window_maximum_included_neighbors_by_atom": [
                int(value)
                for value in self.window_maximum_included_neighbors_by_atom.tolist()
            ],
        }

    def stats(self) -> dict[str, Any]:
        """Synchronize once and return host-side builder diagnostics."""

        calls = int(self.build_calls.item())
        minimum = int(self.minimum_real_edges.item()) if calls else None
        maximum = int(self.maximum_real_edges.item()) if calls else None
        misses = int(self.capacity_misses.item())
        first_overflow = int(self.first_overflow_step.item())
        if calls and self.overflow_to_dummy_only:
            min_padding = int(self.minimum_padding_edges.item())
            max_padding = int(self.maximum_padding_edges.item())
        elif calls:
            min_padding = self.edge_capacity - maximum
            max_padding = self.edge_capacity - minimum
        else:
            min_padding = None
            max_padding = None
        return {
            "fixed_builder_backend": "dense",
            "fixed_builder_build_calls": calls,
            "fixed_builder_capacity_misses": misses,
            "fixed_builder_first_overflow_step": (
                first_overflow if first_overflow >= 0 else None
            ),
            "fixed_builder_edge_capacity": self.edge_capacity,
            "fixed_builder_neighbors_per_atom": self.neighbors_per_atom,
            "fixed_builder_capacity_policy": self.capacity_policy,
            "fixed_builder_neighbor_capacity_min": min(
                self._neighbor_capacities_cpu
            ),
            "fixed_builder_neighbor_capacity_max": max(
                self._neighbor_capacities_cpu
            ),
            "fixed_builder_neighbor_capacity_mean": (
                self.edge_capacity / self.num_atoms
            ),
            "fixed_builder_neighbor_capacity_histogram": {
                str(capacity): count
                for capacity, count in self.capacity_histogram.items()
            },
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
            "fixed_builder_max_capacity_excess": int(
                self.maximum_capacity_excess.item()
            ),
            "fixed_builder_max_overflow_required": int(
                self.maximum_overflow_required.item()
            ),
            "fixed_builder_max_overflow_capacity": int(
                self.maximum_overflow_capacity.item()
            ),
            "fixed_builder_maximum_included_neighbors_by_atom": [
                int(value)
                for value in self.maximum_included_neighbors_by_atom.tolist()
            ],
            "sink_padding_mode": "distributed_dummy_self_edges",
            "sink_dummy_atoms": self.dummy_atoms,
            "sink_padding_edges_min": min_padding,
            "sink_padding_edges_max": max_padding,
            "sink_distribution_min": (
                None if max_padding is None else max_padding // self.dummy_atoms
            ),
            "sink_distribution_max": (
                None
                if max_padding is None
                else math.ceil(max_padding / self.dummy_atoms)
            ),
            "sink_nonzero_shift_verified": self.sink_nonzero_shift_verified,
            "sink_cutoff_zero_verified": self.sink_cutoff_zero_verified,
            "overflow_to_dummy_only": self.overflow_to_dummy_only,
            "overflow_dummy_only_replays": int(
                self.overflow_dummy_only_replays.item()
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


class CellListFixedShapePBCNeighborBuilder(FixedShapePBCNeighborBuilder):
    """Capture-safe GPU cell list feeding the existing fixed edge slots.

    At every build, atoms are wrapped into a fixed fractional grid, sorted by
    bin, and gathered only from neighboring bins.  A fixed bin capacity keeps
    every intermediate shape CUDA-Graph safe.  Bin-capacity overflow shares
    the normal builder overflow path, including dummy-only output for ROB1.
    """

    def __init__(
        self,
        *,
        reference_positions: Tensor,
        cell_list_bin_capacity: int = 0,
        cell_list_bin_margin: float = 0.25,
        cell_list_bin_step: int = 8,
        **kwargs: Any,
    ) -> None:
        cell = kwargs["cell"]
        pbc = kwargs["pbc"]
        cutoff = float(kwargs["cutoff"])
        super().__init__(**kwargs)

        if reference_positions.shape != (self.num_atoms, 3):
            raise ValueError(
                "reference_positions must contain one row per real atom"
            )
        if reference_positions.device != self.device:
            raise ValueError("reference_positions must use the builder device")
        if cell_list_bin_capacity < 0:
            raise ValueError("cell-list bin capacity must be non-negative")
        if cell_list_bin_margin < 0 or cell_list_bin_step < 1:
            raise ValueError("invalid cell-list bin-capacity parameters")

        self.dense_candidates_per_atom = self.candidates_per_atom
        self.cell_list_grid_shape = cell_list_grid_shape(cell, pbc, cutoff)
        plane_distances = _cell_plane_distances(cell)
        search_radii = tuple(
            max(
                1,
                int(
                    math.ceil(
                        cutoff * bins / plane_distance
                    )
                ),
            )
            for bins, plane_distance in zip(
                self.cell_list_grid_shape, plane_distances
            )
        )
        self.cell_list_search_radii = search_radii
        offset_axes = [
            torch.arange(
                -radius,
                radius + 1,
                device=self.device,
                dtype=torch.long,
            )
            for radius in search_radii
        ]
        self.cell_list_neighbor_bin_offsets = torch.cartesian_prod(
            *offset_axes
        ).reshape(-1, 3).contiguous()
        self.cell_list_neighbor_bin_count = int(
            self.cell_list_neighbor_bin_offsets.shape[0]
        )
        self.cell_list_num_bins = math.prod(self.cell_list_grid_shape)
        self.cell_list_grid = torch.as_tensor(
            self.cell_list_grid_shape,
            device=self.device,
            dtype=torch.long,
        )
        self.cell_list_inverse_cell = torch.linalg.inv(
            self.cell.to(dtype=reference_positions.dtype)
        )
        self.cell_list_repetitions = torch.as_tensor(
            self.repetitions,
            device=self.device,
            dtype=torch.long,
        )
        self.cell_list_offset_sizes = 2 * self.cell_list_repetitions + 1
        self.cell_list_atom_ids = torch.arange(
            self.num_atoms, device=self.device, dtype=torch.long
        )

        probed_occupancy = int(
            cell_list_max_occupancy(
                reference_positions,
                cell,
                pbc,
                cutoff,
                grid_shape=self.cell_list_grid_shape,
            ).item()
        )
        if cell_list_bin_capacity:
            bin_capacity = int(cell_list_bin_capacity)
        else:
            bin_capacity = cell_list_bin_capacity_from_probe(
                probed_occupancy,
                margin=cell_list_bin_margin,
                slot_step=cell_list_bin_step,
            )
        minimum_for_output = math.ceil(
            self.neighbors_per_atom / self.cell_list_neighbor_bin_count
        )
        bin_capacity = max(bin_capacity, minimum_for_output)
        if bin_capacity > self.num_atoms:
            bin_capacity = self.num_atoms
        if bin_capacity < 1:
            raise ValueError("cell-list bin capacity must be positive")
        self.cell_list_bin_capacity = int(bin_capacity)
        self.cell_list_bin_capacity_tensor = torch.as_tensor(
            self.cell_list_bin_capacity,
            device=self.device,
            dtype=torch.long,
        )
        self.cell_list_bin_ranks = torch.arange(
            self.cell_list_bin_capacity,
            device=self.device,
            dtype=torch.long,
        )
        self.candidates_per_atom = (
            self.cell_list_neighbor_bin_count * self.cell_list_bin_capacity
        )
        if self.candidates_per_atom < self.neighbors_per_atom:
            raise ValueError(
                "cell-list candidate slots cannot cover the edge capacity"
            )

        # Dense candidate tensors were allocated by the compatibility base
        # initializer.  They are not retained by the cell-list production path.
        del self.candidate_sources
        del self.candidate_cell_offsets
        del self.candidate_ids

        self.cell_list_maximum_bin_occupancy = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.cell_list_bin_overflow_replays = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.window_cell_list_maximum_bin_occupancy = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.window_cell_list_bin_overflow_replays = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.cell_list_probed_maximum_bin_occupancy = probed_occupancy

    def reset_stats(self) -> None:
        super().reset_stats()
        self.cell_list_maximum_bin_occupancy.zero_()
        self.cell_list_bin_overflow_replays.zero_()
        self.window_cell_list_maximum_bin_occupancy.zero_()
        self.window_cell_list_bin_overflow_replays.zero_()

    def reset_window_stats(self) -> None:
        super().reset_window_stats()
        if hasattr(self, "window_cell_list_maximum_bin_occupancy"):
            self.window_cell_list_maximum_bin_occupancy.zero_()
            self.window_cell_list_bin_overflow_replays.zero_()

    def build(
        self,
        positions: Tensor,
        *,
        step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Expected positions {(self.num_atoms, 3)}, got {positions.shape}"
            )
        if positions.device != self.device:
            raise ValueError(
                f"Positions must be on {self.device}, got {positions.device}"
            )

        with torch.no_grad():
            fractional = torch.mm(positions, self.cell_list_inverse_cell)
            image_indices = torch.floor(fractional).to(dtype=torch.long)
            wrapped = fractional - image_indices.to(dtype=fractional.dtype)
            bin_coordinates = torch.floor(
                wrapped * self.cell_list_grid.to(dtype=wrapped.dtype)
            ).to(dtype=torch.long)
            bin_coordinates = torch.minimum(
                bin_coordinates, self.cell_list_grid - 1
            )
            bin_ids = (
                (
                    bin_coordinates[:, 0] * self.cell_list_grid[1]
                    + bin_coordinates[:, 1]
                )
                * self.cell_list_grid[2]
                + bin_coordinates[:, 2]
            )
            sort_keys = bin_ids * (self.num_atoms + 1) + self.cell_list_atom_ids
            sorted_atoms = self.cell_list_atom_ids.index_select(
                0, torch.argsort(sort_keys)
            )
            bin_counts = torch.zeros(
                self.cell_list_num_bins,
                device=self.device,
                dtype=torch.long,
            )
            bin_counts.scatter_add_(
                0, bin_ids, torch.ones_like(bin_ids)
            )
            bin_starts = torch.cumsum(bin_counts, dim=0) - bin_counts
            maximum_bin_occupancy = bin_counts.max()
            bin_overflow = (
                maximum_bin_occupancy > self.cell_list_bin_capacity_tensor
            )
            self.cell_list_maximum_bin_occupancy.copy_(
                torch.maximum(
                    self.cell_list_maximum_bin_occupancy,
                    maximum_bin_occupancy,
                )
            )
            self.window_cell_list_maximum_bin_occupancy.copy_(
                torch.maximum(
                    self.window_cell_list_maximum_bin_occupancy,
                    maximum_bin_occupancy,
                )
            )
            self.cell_list_bin_overflow_replays.add_(
                bin_overflow.to(dtype=torch.long)
            )
            self.window_cell_list_bin_overflow_replays.add_(
                bin_overflow.to(dtype=torch.long)
            )

            raw_neighbor_coordinates = (
                bin_coordinates.unsqueeze(1)
                + self.cell_list_neighbor_bin_offsets.unsqueeze(0)
            )
            neighbor_coordinates = torch.remainder(
                raw_neighbor_coordinates,
                self.cell_list_grid.reshape(1, 1, 3),
            )
            periodic_shifts = torch.div(
                raw_neighbor_coordinates,
                self.cell_list_grid.reshape(1, 1, 3),
                rounding_mode="floor",
            )
            neighbor_bin_ids = (
                (
                    neighbor_coordinates[:, :, 0] * self.cell_list_grid[1]
                    + neighbor_coordinates[:, :, 1]
                )
                * self.cell_list_grid[2]
                + neighbor_coordinates[:, :, 2]
            )
            neighbor_counts = bin_counts.index_select(
                0, neighbor_bin_ids.reshape(-1)
            ).reshape(self.num_atoms, self.cell_list_neighbor_bin_count)
            neighbor_starts = bin_starts.index_select(
                0, neighbor_bin_ids.reshape(-1)
            ).reshape(self.num_atoms, self.cell_list_neighbor_bin_count)
            lookup = (
                neighbor_starts.unsqueeze(2)
                + self.cell_list_bin_ranks.reshape(1, 1, -1)
            )
            occupied = self.cell_list_bin_ranks.reshape(1, 1, -1) < (
                neighbor_counts.unsqueeze(2)
            )
            safe_lookup = lookup.clamp_max(self.num_atoms - 1)
            candidate_sources = sorted_atoms.index_select(
                0, safe_lookup.reshape(-1)
            ).reshape(self.num_atoms, -1)
            candidate_shifts = periodic_shifts.unsqueeze(2).expand(
                -1, -1, self.cell_list_bin_capacity, -1
            ).reshape(self.num_atoms, -1, 3)
            source_images = image_indices.index_select(
                0, candidate_sources.reshape(-1)
            ).reshape(self.num_atoms, -1, 3)
            candidate_offsets = (
                candidate_shifts
                - source_images
                + image_indices.unsqueeze(1)
            )
            within_official_images = (
                candidate_offsets.abs()
                <= self.cell_list_repetitions.reshape(1, 1, 3)
            ).all(dim=2)
            occupied = occupied.reshape(self.num_atoms, -1)

            shifted_sources = positions.index_select(
                0, candidate_sources.reshape(-1)
            ).reshape(self.num_atoms, self.candidates_per_atom, 3)
            shifted_sources = shifted_sources + torch.matmul(
                candidate_offsets.to(dtype=positions.dtype),
                self.cell.to(dtype=positions.dtype),
            )
            delta = shifted_sources - positions.unsqueeze(1)
            distance_sqr = delta.square().sum(dim=-1)
            cutoff_sqr = self.cutoff * self.cutoff
            valid = (
                occupied
                & within_official_images
                & (distance_sqr <= cutoff_sqr)
                & (distance_sqr > 0.0001)
            )
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
                    nearest[:, self.max_neighbors]
                    + self.degeneracy_tolerance
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

            offset_indices = (
                candidate_offsets + self.cell_list_repetitions.reshape(1, 1, 3)
            )
            cell_offset_ids = (
                (
                    offset_indices[:, :, 0] * self.cell_list_offset_sizes[1]
                    + offset_indices[:, :, 1]
                )
                * self.cell_list_offset_sizes[2]
                + offset_indices[:, :, 2]
            )
            official_ids = (
                candidate_sources * self.num_cells + cell_offset_ids
            )
            invalid_id = self.dense_candidates_per_atom
            candidate_order = torch.where(
                included,
                official_ids,
                torch.full_like(official_ids, invalid_id),
            )
            selected = torch.topk(
                candidate_order,
                k=self.neighbors_per_atom,
                dim=1,
                largest=False,
                sorted=True,
            )
            selected_sources = torch.gather(
                candidate_sources, 1, selected.indices
            )
            selected_offsets = torch.gather(
                candidate_offsets,
                1,
                selected.indices.unsqueeze(2).expand(-1, -1, 3),
            )
            selected_valid = selected.values < invalid_id
            flat_sources = selected_sources.reshape(-1).index_select(
                0, self.slot_selection_indices
            )
            flat_offsets = selected_offsets.reshape(-1, 3).index_select(
                0, self.slot_selection_indices
            )
            flat_valid = selected_valid.reshape(-1).index_select(
                0, self.slot_selection_indices
            )
            self._write_and_update_stats(
                flat_sources,
                flat_offsets,
                flat_valid,
                raw_counts,
                included_counts,
                step=step,
                extra_overflow=bin_overflow,
                extra_required=maximum_bin_occupancy,
                extra_capacity=self.cell_list_bin_capacity_tensor,
            )
        return self.edge_index, self.cell_offsets

    def window_stats(self) -> dict[str, Any]:
        record = super().window_stats()
        record.update(
            {
                "cell_list_window_maximum_bin_occupancy": int(
                    self.window_cell_list_maximum_bin_occupancy.item()
                ),
                "cell_list_window_bin_overflow_replays": int(
                    self.window_cell_list_bin_overflow_replays.item()
                ),
            }
        )
        return record

    def stats(self) -> dict[str, Any]:
        record = super().stats()
        record.update(
            {
                "fixed_builder_backend": "cell-list",
                "cell_list_grid_shape": list(self.cell_list_grid_shape),
                "cell_list_num_bins": self.cell_list_num_bins,
                "cell_list_search_radii": list(self.cell_list_search_radii),
                "cell_list_neighbor_bin_count": self.cell_list_neighbor_bin_count,
                "cell_list_bin_capacity": self.cell_list_bin_capacity,
                "cell_list_probed_maximum_bin_occupancy": (
                    self.cell_list_probed_maximum_bin_occupancy
                ),
                "cell_list_maximum_bin_occupancy": int(
                    self.cell_list_maximum_bin_occupancy.item()
                ),
                "cell_list_bin_overflow_replays": int(
                    self.cell_list_bin_overflow_replays.item()
                ),
                "cell_list_dense_candidates_per_atom": (
                    self.dense_candidates_per_atom
                ),
                "cell_list_candidate_reduction": (
                    1.0
                    - self.candidates_per_atom / self.dense_candidates_per_atom
                ),
            }
        )
        return record


def make_fixed_shape_pbc_neighbor_builder(
    neighbor_builder: str,
    **kwargs: Any,
) -> FixedShapePBCNeighborBuilder:
    """Construct the frozen dense builder or the experimental cell list."""

    if neighbor_builder == "dense":
        kwargs.pop("reference_positions", None)
        kwargs.pop("cell_list_bin_capacity", None)
        kwargs.pop("cell_list_bin_margin", None)
        kwargs.pop("cell_list_bin_step", None)
        return FixedShapePBCNeighborBuilder(**kwargs)
    if neighbor_builder == "cell-list":
        return CellListFixedShapePBCNeighborBuilder(**kwargs)
    raise ValueError("neighbor_builder must be dense or cell-list")
