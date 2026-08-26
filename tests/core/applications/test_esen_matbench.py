from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.nose_hoover_chain import NoseHooverChainNVT

from fairchem.core.applications.esen_matbench import (
    MatbenchNHCIntegrator,
    MatbenchNHCWholeStepCUDAGraphMD,
    MatbenchTrajectoryRecorder,
    matched_trajectory_window,
    read_matbench_systems,
)


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, spring: float = 0.01):
        super().__init__()
        self.spring = spring

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions())
        self.results = {
            "energy": 0.5 * self.spring * np.square(positions).sum(),
            "forces": -self.spring * positions,
        }


def test_matbench_metric_window_matches_short_prediction_duration():
    window = matched_trajectory_window(
        reference_frames=16_000,
        prediction_frames=1_001,
        reference_dt_fs=0.5,
        prediction_dt_fs=2.5,
    )
    assert window == {
        "reference_frames_available": 16_000,
        "prediction_frames_available": 1_001,
        "reference_frames_used": 5_001,
        "prediction_frames_used": 1_001,
        "reference_stride": 5,
        "prediction_stride": 1,
        "matched_duration_fs": 2_500.0,
    }


def test_matbench_reader_validates_schema_and_frame_zero(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "reference.h5"
    positions = np.array(
        [
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
            [[0.2, 0.2, 0.3], [1.0, 1.2, 1.3]],
        ],
        dtype=np.float64,
    )
    cell = np.broadcast_to(np.diag([4.0, 5.0, 6.0]), (2, 3, 3)).copy()
    with h5py.File(path, "w") as handle:
        group = handle.create_group("toy")
        group.attrs["schema"] = 1
        group.attrs["dt_fs"] = 0.5
        group.attrs["temperature_kelvin"] = 300.0
        group.create_dataset("atomic_numbers", data=[6, 8])
        group.create_dataset("positions", data=positions)
        group.create_dataset("cell", data=cell)
        group.create_dataset("pbc", data=[True, True, True])

    systems = read_matbench_systems(path)
    assert [system.name for system in systems] == ["toy"]
    system = systems[0]
    np.testing.assert_array_equal(system.initial_positions, positions[0])
    np.testing.assert_array_equal(system.cell, cell[0])
    assert system.reference_frames == 2
    assert system.reference_dt_fs == 0.5
    assert system.temperature_kelvin == 300.0
    assert system.reference_has_stress is False
    assert system.atoms().get_pbc().all()


def test_matbench_reader_rejects_variable_cell(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "variable.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("toy")
        group.attrs.update(schema=1, dt_fs=0.5, temperature_kelvin=300.0)
        group.create_dataset("atomic_numbers", data=[6])
        group.create_dataset("positions", data=np.zeros((2, 1, 3)))
        group.create_dataset(
            "cell",
            data=np.array(
                [np.eye(3), np.diag([1.0, 1.0, 1.1])], dtype=np.float64
            ),
        )
        group.create_dataset("pbc", data=[True, True, True])
    with pytest.raises(ValueError, match="variable cells"):
        read_matbench_systems(path)


def test_matbench_recorder_writes_sampled_frame_schema(tmp_path):
    h5py = pytest.importorskip("h5py")
    recorder = MatbenchTrajectoryRecorder(
        n_atoms=1,
        steps=20,
        record_interval=10,
        cell=np.eye(3),
    )
    for step in (0, 10, 20):
        recorder.append(
            step,
            np.full((1, 3), step, dtype=np.float64),
            np.full((1, 3), step + 1, dtype=np.float64),
            np.full((1, 3), step + 2, dtype=np.float32),
            float(step),
        )
    path = tmp_path / "prediction.h5"
    recorder.write(
        path,
        atomic_numbers=np.array([6]),
        pbc=np.array([True, True, True]),
        temperature_kelvin=300.0,
        backend="opt3",
    )
    with h5py.File(path, "r") as handle:
        assert bool(handle.attrs["complete"])
        assert handle.attrs["dt_fs"] == 2.5
        assert handle.attrs["completed_frames"] == 3
        np.testing.assert_array_equal(handle["md_step"][:], [0, 10, 20])
        assert handle["positions"].shape == (3, 1, 3)
        assert handle["cell"].shape == (3, 3, 3)


def test_matbench_nhc_matches_ase_for_one_harmonic_step():
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]], dtype=np.float64)
    momenta = np.array([[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]])
    masses = np.array([12.0, 16.0])
    spring = 0.01

    atoms = Atoms("CO", positions=positions, masses=masses)
    atoms.set_momenta(momenta)
    atoms.calc = HarmonicCalculator(spring)
    ase_md = NoseHooverChainNVT(
        atoms,
        timestep=0.25 * units.fs,
        temperature_K=300.0,
        tdamp=25.0 * units.fs,
        tchain=3,
        tloop=1,
    )
    ase_md.step()
    ase_thermostat = ase_md._thermostat

    state_positions = torch.tensor(positions, dtype=torch.float64)
    state_momenta = torch.tensor(momenta, dtype=torch.float64)
    state = type("State", (), {})()
    state.positions = state_positions
    state.momenta = state_momenta
    state.forces = None
    state.potential_energy = None

    integrator = MatbenchNHCIntegrator(
        torch.tensor(masses, dtype=torch.float64),
        timestep_fs=0.25,
        temperature_K=300.0,
        thermostat_time_fs=25.0,
    )

    def force_fn(current_positions):
        return -spring * current_positions, 0.5 * spring * current_positions.square().sum()

    integrator.step(state, force_fn)
    # NumPy and Torch use different exp implementations across supported
    # CPU/Torch builds.  Their one-step FP64 round-off can reach a few e-9
    # even though the NHC factorization and state updates are identical.
    # This remains far tighter than any model/MD validation tolerance while
    # avoiding a bit-level cross-library assertion.
    for actual, expected in (
        (state.positions.detach().numpy(), atoms.get_positions()),
        (state.momenta.detach().numpy(), atoms.get_momenta()),
        (integrator.eta.detach().numpy(), ase_thermostat._eta),
        (integrator.p_eta.detach().numpy(), ase_thermostat._p_eta),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_matbench_nhc_parameters_match_ase():
    masses = torch.tensor([12.0, 16.0], dtype=torch.float64)
    integrator = MatbenchNHCIntegrator(
        masses,
        timestep_fs=0.25,
        temperature_K=300.0,
        thermostat_time_fs=25.0,
    )
    kT = units.kB * 300.0
    tdamp = 25.0 * units.fs
    expected = np.array([6.0 * kT * tdamp**2, kT * tdamp**2, kT * tdamp**2])
    np.testing.assert_allclose(integrator.Q.numpy(), expected, rtol=1e-15, atol=1e-15)


def test_matbench_whole_step_initial_replay_arms_production_without_advancing():
    class FakeGraph:
        def __init__(self):
            self.calls = 0

        def replay(self):
            self.calls += 1

    whole = object.__new__(MatbenchNHCWholeStepCUDAGraphMD)
    whole.graph = FakeGraph()
    whole.advance = torch.ones((), dtype=torch.float64)
    whole.forces = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    whole.potential_energy = torch.tensor(4.0, dtype=torch.float32)
    whole.production_replays = 0
    whole.total_replays = 0

    forces, energy = whole.evaluate_initial()

    assert whole.graph.calls == 1
    assert whole.production_replays == 1
    assert whole.total_replays == 1
    assert whole.advance.item() == 1.0
    torch.testing.assert_close(forces, whole.forces)
    torch.testing.assert_close(energy, whole.potential_energy)
