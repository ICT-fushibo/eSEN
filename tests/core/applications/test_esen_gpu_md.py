from __future__ import annotations

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.nvtberendsen import NVTBerendsen

from fairchem.core.applications.esen_gpu_md import (
    GPUIntegrator,
    GPUMDState,
    GPUResidentMD,
)


class ConstantForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, forces: np.ndarray) -> None:
        super().__init__()
        self._forces = forces

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": self._forces.copy()}


class ConstantForceEvaluator:
    def __init__(self, forces: torch.Tensor) -> None:
        self.forces = forces

    def __call__(self, positions: torch.Tensor):
        return self.forces.to(positions), positions.new_zeros(())


def test_gpu_integrator_matches_one_ase_nvt_step_on_cpu():
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]])
    momenta = np.array([[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]])
    forces = np.array([[0.03, -0.02, 0.01], [-0.04, 0.02, -0.01]])
    masses = np.array([12.0, 16.0])

    atoms = Atoms("CO", positions=positions, masses=masses)
    atoms.set_momenta(momenta)
    atoms.calc = ConstantForceCalculator(forces)
    ase_md = NVTBerendsen(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=300.0,
        taut=100.0 * units.fs,
        fixcm=True,
    )
    ase_md.run(1)

    state = GPUMDState(
        positions=torch.tensor(positions, dtype=torch.float64),
        momenta=torch.tensor(momenta, dtype=torch.float64),
    )
    evaluator = ConstantForceEvaluator(torch.tensor(forces, dtype=torch.float64))
    integrator = GPUIntegrator(
        torch.tensor(masses, dtype=torch.float64),
        timestep_fs=1.0,
        temperature_K=300.0,
        taut_fs=100.0,
        fix_com=True,
        degrees_of_freedom=atoms.get_number_of_degrees_of_freedom(),
    )
    md = GPUResidentMD(state, evaluator, integrator)
    md.run(1)

    np.testing.assert_allclose(
        state.positions.numpy(), atoms.get_positions(), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        state.momenta.numpy(), atoms.get_momenta(), rtol=1e-13, atol=1e-13
    )


def test_gpu_md_state_restore_clears_optional_values():
    initial = GPUMDState(torch.ones(2, 3), torch.full((2, 3), 2.0))
    state = GPUMDState(
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        forces=torch.ones(2, 3),
        potential_energy=torch.ones(()),
    )
    state.restore_(initial)
    torch.testing.assert_close(state.positions, initial.positions)
    torch.testing.assert_close(state.momenta, initial.momenta)
    assert state.forces is None
    assert state.potential_energy is None
