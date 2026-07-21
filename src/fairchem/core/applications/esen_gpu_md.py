"""GPU-resident eager molecular dynamics for eSEN.

This module deliberately does not use ``torch.compile``, CUDA Graphs, or
custom fused kernels.  It is the eager GPU-resident control path on which
those optimizations can later be layered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ase import Atoms, units
from torch import Tensor

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.datasets import data_list_collater


def _resolve_model_output(outputs: dict[str, Any], key: str) -> Tensor:
    """Resolve either a pass-through Hydra output or a nested head output."""

    value = outputs.get(key)
    if isinstance(value, Tensor):
        return value
    for head_output in outputs.values():
        if isinstance(head_output, dict) and isinstance(head_output.get(key), Tensor):
            return head_output[key]
    raise KeyError(f"Model output {key!r} not found in {list(outputs)}")


def configure_esen_energy_force_inference(model):
    """Configure a loaded eSEN HydraModel for energy/force-only inference."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise TypeError("Expected an eSEN HydraModel with a backbone")
    if hasattr(backbone, "regress_stress"):
        backbone.regress_stress = False
    if hasattr(backbone, "activation_checkpointing"):
        backbone.activation_checkpointing = False

    output_heads = getattr(model, "output_heads", {})
    for head in output_heads.values():
        if hasattr(head, "regress_stress"):
            head.regress_stress = False
    return model


class ESENEnergyForceEvaluator:
    """Load eSEN through OCPCalculator and evaluate E/F on persistent GPU data.

    OCPCalculator remains the checkpoint/configuration authority.  The hot path
    bypasses ASE conversion and ``trainer.predict`` and calls the loaded Hydra
    model directly.  Model parameters are frozen while gradients with respect
    to the persistent model-position tensor remain enabled for conservative
    forces.
    """

    def __init__(
        self,
        atoms: Atoms,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        seed: int | None = None,
        disable_amp: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for eSEN GPU-resident MD")

        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=self.device.type == "cpu",
            seed=seed,
            only_output=["energy", "forces"],
            disable_amp=disable_amp,
        )
        self.trainer = self.calculator.trainer
        self.model = configure_esen_energy_force_inference(
            self.trainer._unwrapped_model
        )

        data_object = self.calculator.a2g.convert(atoms)
        self.batch = data_list_collater([data_object], otf_graph=True).to(self.device)
        self.model_positions = self.batch.pos.detach().clone()
        self.model_positions.requires_grad_(True)
        self.batch.pos = self.model_positions
        self.model_dtype = self.model_positions.dtype
        self.num_atoms = int(len(atoms))

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Return denormalized ``(forces, energy)`` without leaving the device."""

        if positions.shape != self.model_positions.shape:
            raise ValueError(
                f"Expected positions with shape {tuple(self.model_positions.shape)}, "
                f"got {tuple(positions.shape)}"
            )
        if positions.device != self.device:
            raise ValueError(
                f"Positions must be on {self.device}, got {positions.device}"
            )

        # The copy performs the FP64 MD-state -> FP32 model-input conversion on
        # device while preserving the address and leaf status of model_positions.
        with torch.no_grad():
            self.model_positions.copy_(positions)

        # The model's E/F head calls autograd.grad internally.  Global
        # inference_mode must not be used for this conservative-force model.
        with torch.enable_grad():
            raw_outputs = self.model(self.batch)
            raw_energy = _resolve_model_output(raw_outputs, "energy")
            raw_forces = _resolve_model_output(raw_outputs, "forces")
            energy = self.trainer._denorm_preds("energy", raw_energy, self.batch)
            forces = self.trainer._denorm_preds("forces", raw_forces, self.batch)

        forces = forces.reshape(self.num_atoms, 3).detach()
        energy = energy.reshape(-1)[0].detach()
        return forces, energy


@dataclass
class GPUMDState:
    """Mutable MD state whose tensors remain on one device."""

    positions: Tensor
    momenta: Tensor
    forces: Tensor | None = None
    potential_energy: Tensor | None = None

    def clone(self) -> "GPUMDState":
        return GPUMDState(
            positions=self.positions.clone(),
            momenta=self.momenta.clone(),
            forces=None if self.forces is None else self.forces.clone(),
            potential_energy=(
                None
                if self.potential_energy is None
                else self.potential_energy.clone()
            ),
        )

    def restore_(self, other: "GPUMDState") -> None:
        self.positions.copy_(other.positions)
        self.momenta.copy_(other.momenta)
        self.forces = None if other.forces is None else other.forces.clone()
        self.potential_energy = (
            None
            if other.potential_energy is None
            else other.potential_energy.clone()
        )


class GPUIntegrator:
    """GPU Velocity-Verlet/Berendsen NVT equations matching unconstrained ASE."""

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float,
        temperature_K: float,
        taut_fs: float,
        fix_com: bool = True,
        degrees_of_freedom: int | None = None,
    ) -> None:
        if timestep_fs <= 0 or temperature_K <= 0 or taut_fs <= 0:
            raise ValueError("timestep, temperature, and taut must be positive")
        if masses.ndim not in {1, 2}:
            raise ValueError(
                f"masses must have shape [N] or [N, 1], got {masses.shape}"
            )
        self.masses = masses.reshape(-1, 1)
        self.dt = float(timestep_fs) * units.fs
        self.temperature_target = float(temperature_K)
        self.taut = float(taut_fs) * units.fs
        self.fix_com = bool(fix_com)
        self.degrees_of_freedom = (
            int(degrees_of_freedom)
            if degrees_of_freedom is not None
            else 3 * int(self.masses.numel())
        )
        if self.degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be positive")

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return (0.5 * momenta.square() / self.masses).sum()

    def temperature(self, momenta: Tensor) -> Tensor:
        return 2.0 * self.kinetic_energy(momenta) / (
            self.degrees_of_freedom * units.kB
        )

    def scale_velocities(self, momenta: Tensor) -> Tensor:
        old_temperature = self.temperature(momenta).clamp_min(1e-12)
        scale = torch.sqrt(
            1.0
            + (self.temperature_target / old_temperature - 1.0)
            * (self.dt / self.taut)
        )
        return momenta * torch.clamp(scale, min=0.9, max=1.1)

    def step(
        self,
        state: GPUMDState,
        force_fn: ESENEnergyForceEvaluator,
    ) -> None:
        momenta = self.scale_velocities(state.momenta)

        if state.forces is None:
            model_forces, energy = force_fn(state.positions)
            state.forces = model_forces.to(dtype=state.positions.dtype)
            state.potential_energy = energy

        momenta = momenta + 0.5 * self.dt * state.forces
        if self.fix_com:
            momenta = momenta - momenta.sum(dim=0, keepdim=True) / float(
                momenta.shape[0]
            )

        positions = state.positions + self.dt * momenta / self.masses
        model_forces, energy = force_fn(positions)
        forces = model_forces.to(dtype=positions.dtype)
        momenta = momenta + 0.5 * self.dt * forces

        state.positions = positions
        state.momenta = momenta
        state.forces = forces
        state.potential_energy = energy


class GPUResidentMD:
    """Eager Python MD loop with all numerical state resident on the GPU."""

    def __init__(
        self,
        state: GPUMDState,
        evaluator: ESENEnergyForceEvaluator,
        integrator: GPUIntegrator,
    ) -> None:
        self.state = state
        self.evaluator = evaluator
        self.integrator = integrator
        self.nsteps = 0

    def evaluate(self) -> tuple[Tensor, Tensor]:
        forces, energy = self.evaluator(self.state.positions)
        self.state.forces = forces.to(dtype=self.state.positions.dtype)
        self.state.potential_energy = energy
        return self.state.forces, energy

    def run(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps and self.state.forces is None:
            self.evaluate()
        for _ in range(steps):
            self.integrator.step(self.state, self.evaluator)
            self.nsteps += 1
