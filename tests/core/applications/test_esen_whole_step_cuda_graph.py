from __future__ import annotations

import torch

from fairchem.core.applications.esen_gpu_md import (
    GPUIntegrator,
)
from fairchem.core.applications.esen_whole_step_cuda_graph import (
    _branchless_nvt_momentum_finish,
    _branchless_nvt_position_proposal,
)


def _integrator():
    return GPUIntegrator(
        torch.tensor([12.0, 16.0], dtype=torch.float64),
        timestep_fs=1.0,
        temperature_K=300.0,
        taut_fs=100.0,
        fix_com=True,
        degrees_of_freedom=6,
    )


def test_advance_zero_only_evaluates_current_positions():
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]], dtype=torch.float64
    )
    momenta = torch.tensor(
        [[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]],
        dtype=torch.float64,
    )
    forces = torch.tensor(
        [[0.03, -0.02, 0.01], [-0.04, 0.02, -0.01]],
        dtype=torch.float64,
    )
    integrator = _integrator()
    half, evaluated = _branchless_nvt_position_proposal(
        positions, momenta, forces, integrator, positions.new_zeros(())
    )
    final_momenta = _branchless_nvt_momentum_finish(
        momenta,
        half,
        forces,
        integrator,
        positions.new_zeros(()),
    )

    torch.testing.assert_close(evaluated, positions, rtol=0, atol=0)
    torch.testing.assert_close(final_momenta, momenta, rtol=0, atol=0)


def test_advance_one_matches_gpu_integrator_tensor_equations():
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]], dtype=torch.float64
    )
    momenta = torch.tensor(
        [[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]],
        dtype=torch.float64,
    )
    old_forces = torch.tensor(
        [[0.03, -0.02, 0.01], [-0.04, 0.02, -0.01]],
        dtype=torch.float64,
    )
    new_forces = torch.tensor(
        [[0.02, -0.01, 0.04], [-0.03, 0.01, -0.02]],
        dtype=torch.float64,
    )
    integrator = _integrator()
    half, evaluated = _branchless_nvt_position_proposal(
        positions,
        momenta,
        old_forces,
        integrator,
        positions.new_ones(()),
    )
    final_momenta = _branchless_nvt_momentum_finish(
        momenta,
        half,
        new_forces,
        integrator,
        positions.new_ones(()),
    )

    expected_half = (
        integrator.scale_velocities(momenta)
        + 0.5 * integrator.dt * old_forces
    )
    expected_half = expected_half - expected_half.sum(
        dim=0, keepdim=True
    ) / 2.0
    expected_positions = (
        positions
        + integrator.dt * expected_half / integrator.masses
    )
    expected_momenta = (
        expected_half + 0.5 * integrator.dt * new_forces
    )

    torch.testing.assert_close(evaluated, expected_positions)
    torch.testing.assert_close(final_momenta, expected_momenta)
