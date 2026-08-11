"""Opt4 kernel-fusion experiments for eSEN molecular dynamics.

KF1 fuses the fixed neighbor builder's PBC shift, squared-distance, cutoff,
and self-edge mask into one Triton kernel.  Neighbor selection, non-strict
degeneracy handling, padding, the eSEN model, autograd, and NVT integration
remain identical to Opt3.  The Opt3 classes are not modified or monkey-patched.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from fairchem.core.applications.esen_fixed_neighbor import (
    FixedShapePBCNeighborBuilder,
)
from fairchem.core.applications.esen_gpu_md import (
    ESENEnergyForceEvaluator,
    GPUIntegrator,
    GPUMDState,
)
from fairchem.core.applications.esen_whole_step_cuda_graph import (
    ESENFixedBuilderModelCUDAGraphEvaluator,
    ESENWholeStepCUDAGraphMD,
    _pbc_vector,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in CPU-only environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _pbc_distance_cutoff_kernel(
        positions_ptr,
        unit_offsets_ptr,
        cell_ptr,
        distance_ptr,
        valid_ptr,
        num_atoms: tl.constexpr,
        num_cells: tl.constexpr,
        candidates_per_atom: tl.constexpr,
        total_values: tl.constexpr,
        cutoff_sqr: tl.constexpr,
        block_size: tl.constexpr,
    ):
        linear = tl.program_id(axis=0) * block_size + tl.arange(0, block_size)
        active = linear < total_values
        centre = linear // candidates_per_atom
        candidate = linear - centre * candidates_per_atom
        source = candidate // num_cells
        image = candidate - source * num_cells

        source_base = source * 3
        centre_base = centre * 3
        image_base = image * 3

        sx = tl.load(positions_ptr + source_base, mask=active, other=0.0)
        sy = tl.load(positions_ptr + source_base + 1, mask=active, other=0.0)
        sz = tl.load(positions_ptr + source_base + 2, mask=active, other=0.0)
        cx = tl.load(positions_ptr + centre_base, mask=active, other=0.0)
        cy = tl.load(positions_ptr + centre_base + 1, mask=active, other=0.0)
        cz = tl.load(positions_ptr + centre_base + 2, mask=active, other=0.0)

        ox = tl.load(unit_offsets_ptr + image_base, mask=active, other=0.0)
        oy = tl.load(unit_offsets_ptr + image_base + 1, mask=active, other=0.0)
        oz = tl.load(unit_offsets_ptr + image_base + 2, mask=active, other=0.0)

        cell_00 = tl.load(cell_ptr)
        cell_01 = tl.load(cell_ptr + 1)
        cell_02 = tl.load(cell_ptr + 2)
        cell_10 = tl.load(cell_ptr + 3)
        cell_11 = tl.load(cell_ptr + 4)
        cell_12 = tl.load(cell_ptr + 5)
        cell_20 = tl.load(cell_ptr + 6)
        cell_21 = tl.load(cell_ptr + 7)
        cell_22 = tl.load(cell_ptr + 8)
        dx = sx + ox * cell_00 + oy * cell_10 + oz * cell_20 - cx
        dy = sy + ox * cell_01 + oy * cell_11 + oz * cell_21 - cy
        dz = sz + ox * cell_02 + oy * cell_12 + oz * cell_22 - cz
        distance_sqr = dx * dx + dy * dy + dz * dz
        valid = (distance_sqr <= cutoff_sqr) & (distance_sqr > 0.0001)
        tl.store(distance_ptr + linear, distance_sqr, mask=active)
        tl.store(valid_ptr + linear, valid, mask=active)


def triton_neighbor_fusion_available() -> bool:
    """Return whether the optional Triton dependency can be imported."""

    return triton is not None


class TritonDistanceFixedShapePBCNeighborBuilder(
    FixedShapePBCNeighborBuilder
):
    """Opt3 fixed builder with only distance/mask calculation fused."""

    fusion_name = "triton_pbc_distance_cutoff_mask"

    def __init__(self, *args, triton_block_size: int = 256, **kwargs) -> None:
        if triton is None:
            raise RuntimeError(
                "Opt4 KF1 requires Triton; install the Triton version bundled "
                "with the active PyTorch CUDA environment"
            )
        super().__init__(*args, **kwargs)
        if self.device.type != "cuda":
            raise ValueError("Opt4 KF1 Triton builder requires a CUDA device")
        if self.position_dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "Opt4 KF1 supports float32 or float64 positions, got "
                f"{self.position_dtype}"
            )
        if triton_block_size < 32 or triton_block_size > 1024:
            raise ValueError("triton_block_size must be between 32 and 1024")
        if triton_block_size & (triton_block_size - 1):
            raise ValueError("triton_block_size must be a power of two")
        self.triton_block_size = int(triton_block_size)
        self._distance_sqr = torch.empty(
            self.num_atoms,
            self.candidates_per_atom,
            device=self.device,
            dtype=self.position_dtype,
        )
        self._valid_candidates = torch.empty(
            self.num_atoms,
            self.candidates_per_atom,
            device=self.device,
            dtype=torch.bool,
        )

    def _compute_distance_sqr_and_valid(
        self, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        if not positions.is_contiguous():
            raise ValueError("Opt4 KF1 positions must be contiguous")
        total_values = self.num_atoms * self.candidates_per_atom
        grid = (triton.cdiv(total_values, self.triton_block_size),)
        _pbc_distance_cutoff_kernel[grid](
            positions,
            self.unit_cell_offsets,
            self.cell,
            self._distance_sqr,
            self._valid_candidates,
            num_atoms=self.num_atoms,
            num_cells=self.num_cells,
            candidates_per_atom=self.candidates_per_atom,
            total_values=total_values,
            cutoff_sqr=self.cutoff * self.cutoff,
            block_size=self.triton_block_size,
        )
        return self._distance_sqr, self._valid_candidates

    def build(
        self,
        positions: Tensor,
        *,
        step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build the Opt3 graph with only distance/mask calculation fused."""

        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Expected positions {(self.num_atoms, 3)}, got {positions.shape}"
            )
        if positions.device != self.device:
            raise ValueError(
                f"Positions must be on {self.device}, got {positions.device}"
            )

        with torch.no_grad():
            distance_sqr, valid = self._compute_distance_sqr_and_valid(
                positions
            )
            cutoff_sqr = self.cutoff * self.cutoff
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
            centres = (
                torch.arange(
                    self.num_atoms, device=self.device, dtype=torch.long
                )
                .view(-1, 1)
                .expand(-1, self.neighbors_per_atom)
                .reshape(-1)
            )
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
        record = super().stats()
        record.update(
            {
                "fixed_builder_distance_backend": "triton",
                "kernel_fusion_stage": "KF1",
                "kernel_fusion_name": self.fusion_name,
                "kernel_fusion_triton_block_size": self.triton_block_size,
            }
        )
        return record


def _make_triton_builder(
    owner,
    *,
    dummy_atoms: int,
    max_neighbors: int,
    degeneracy_tolerance: float,
    triton_block_size: int,
) -> TritonDistanceFixedShapePBCNeighborBuilder:
    """Create KF1 builder over an Opt3 owner's existing static outputs."""

    core = owner.core if hasattr(owner, "core") else owner
    assert core.static_edge_index is not None
    assert core.static_cell_offsets is not None
    return TritonDistanceFixedShapePBCNeighborBuilder(
        num_atoms=owner.num_atoms,
        cell=core.static_batch.cell.reshape(-1, 3, 3)[0],
        pbc=_pbc_vector(core.static_batch, owner.device),
        cutoff=float(core.model.backbone.cutoff),
        neighbors_per_atom=owner.neighbors_per_atom,
        dummy_atoms=dummy_atoms,
        max_neighbors=max_neighbors,
        degeneracy_tolerance=degeneracy_tolerance,
        output_edge_index=core.static_edge_index,
        output_cell_offsets=core.static_cell_offsets,
        triton_block_size=triton_block_size,
    )


class ESENKF1FixedBuilderModelCUDAGraphEvaluator(
    ESENFixedBuilderModelCUDAGraphEvaluator
):
    """KF1 control: fused builder eager, Opt2 model-only graph unchanged."""

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
        triton_block_size: int = 256,
    ) -> None:
        super().__init__(
            eager_evaluator,
            neighbors_per_atom=neighbors_per_atom,
            dummy_atoms=dummy_atoms,
            capture_warmup=capture_warmup,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
            replay_energy_atol=replay_energy_atol,
            replay_force_atol=replay_force_atol,
        )
        self.fixed_builder = _make_triton_builder(
            self,
            dummy_atoms=dummy_atoms,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
            triton_block_size=triton_block_size,
        )


class ESENKF1WholeStepCUDAGraphMD(ESENWholeStepCUDAGraphMD):
    """Opt4 KF1: Opt3 whole-step graph with the fused distance kernel."""

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
        triton_block_size: int = 256,
    ) -> None:
        super().__init__(
            state,
            eager_evaluator,
            integrator,
            neighbors_per_atom=neighbors_per_atom,
            dummy_atoms=dummy_atoms,
            capture_warmup=capture_warmup,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        self.fixed_builder = _make_triton_builder(
            self,
            dummy_atoms=dummy_atoms,
            max_neighbors=max_neighbors,
            degeneracy_tolerance=degeneracy_tolerance,
            triton_block_size=triton_block_size,
        )
