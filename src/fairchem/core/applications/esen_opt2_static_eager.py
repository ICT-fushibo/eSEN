"""Capture-compatible eager control for eSEN model-only CUDA Graph MD.

This evaluator applies every static-shape adaptation required by opt2:

* dynamic real-neighbor construction remains outside the model;
* the real graph is copied into one fixed edge capacity;
* padding uses disconnected dummy sink atoms;
* the deterministic auxiliary rotation reference and device-resident index
  tensors are enabled;
* the same padded energy head and real-only element references are used.

Unlike opt2, model forward, conservative-force autograd, and denormalization
execute eagerly on every call.  Comparing this control with opt2 isolates the
CUDA Graph replay contribution without changing the existing opt1 evaluator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fairchem.core.applications.esen_cuda_graph import (
    CUDAGraphCapacityError,
    ESENModelCUDAGraphEvaluator,
)
from fairchem.core.applications.esen_gpu_md import ESENEnergyForceEvaluator


class ESENOpt2StaticEagerEvaluator(ESENModelCUDAGraphEvaluator):
    """Run the opt2 fixed-capacity model path without CUDA Graph capture."""

    def __init__(
        self,
        eager_evaluator: ESENEnergyForceEvaluator,
        *,
        edge_capacity: int,
        dummy_atoms: int = 32,
        setup_warmup: int = 3,
        replay_energy_atol: float = 0.0,
        replay_force_atol: float = 1e-6,
    ) -> None:
        super().__init__(
            eager_evaluator,
            edge_capacity=edge_capacity,
            dummy_atoms=dummy_atoms,
            capture_warmup=setup_warmup,
            replay_energy_atol=replay_energy_atol,
            replay_force_atol=replay_force_atol,
        )
        self.setup_warmup = int(setup_warmup)
        self.prepared = False
        self.production_calls = 0
        self.production_capacity_misses = 0
        self.production_min_edges: int | None = None
        self.production_max_edges: int | None = None

    def prepare(self, positions: Tensor) -> None:
        """Initialize fixed inputs and run untimed eager stability checks."""

        if self.prepared:
            raise RuntimeError("Static eager evaluator has already been prepared")
        sample_graph = self._build_real_graph(positions)
        self._staticize(sample_graph)
        self.model.backbone.otf_graph = False

        for _ in range(self.setup_warmup):
            self._static_forward()

        first_forces, first_energy = self._static_forward()
        first_forces = first_forces.clone()
        first_energy = first_energy.clone()
        second_forces, second_energy = self._static_forward()
        second_forces = second_forces.clone()
        second_energy = second_energy.clone()
        torch.cuda.synchronize(self.device)

        self.replay_stability_force_max_abs_error = float(
            (second_forces - first_forces).abs().max().item()
        )
        self.replay_stability_energy_abs_error = float(
            (second_energy - first_energy).abs().max().item()
        )
        self.replay_stability_passed = not (
            self.replay_stability_force_max_abs_error > self.replay_force_atol
            or self.replay_stability_energy_abs_error > self.replay_energy_atol
        )
        self.prepared = True

    def reset_production_stats(self) -> None:
        self.production_calls = 0
        self.production_capacity_misses = 0
        self.production_min_edges = None
        self.production_max_edges = None

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        if not self.prepared:
            raise RuntimeError("Static eager evaluator must be prepared first")
        graph = self._build_real_graph(positions)
        num_edges = int(graph["edge_index"].shape[1])
        self.production_calls += 1
        if num_edges > self.edge_capacity:
            self.production_capacity_misses += 1
            raise CUDAGraphCapacityError(num_edges, self.edge_capacity)
        self._staticize(graph)
        forces, energy = self._static_forward()
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
        return forces, energy

    def stats(self) -> dict[str, int | float | bool | None]:
        max_padding_fraction = (
            (self.edge_capacity - self.production_min_edges) / self.edge_capacity
            if self.production_min_edges is not None
            else None
        )
        return {
            "static_eager_production_calls": self.production_calls,
            "static_eager_capacity_misses": self.production_capacity_misses,
            "static_eager_edge_capacity": self.edge_capacity,
            "static_eager_min_real_edges": self.production_min_edges,
            "static_eager_max_real_edges": self.production_max_edges,
            "static_eager_max_padding_fraction": max_padding_fraction,
            "static_eager_dummy_atoms": self.dummy_atoms,
            "static_eager_setup_warmup": self.setup_warmup,
            "static_eager_device_index_tensor_count": (
                self.capture_index_tensor_count
            ),
            "static_eager_repeat_stability_pass": self.replay_stability_passed,
            "static_eager_repeat_energy_abs_error_eV": (
                self.replay_stability_energy_abs_error
            ),
            "static_eager_repeat_force_max_abs_error_eV_per_A": (
                self.replay_stability_force_max_abs_error
            ),
            "static_eager_repeat_energy_atol_eV": self.replay_energy_atol,
            "static_eager_repeat_force_atol_eV_per_A": self.replay_force_atol,
            "cuda_graph_capture_count": 0,
            "cuda_graph_production_capture_count": 0,
            "cuda_graph_production_replays": 0,
        }
