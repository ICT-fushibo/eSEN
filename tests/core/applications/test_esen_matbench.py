from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from fairchem.core.applications.esen_matbench import (
    MatbenchNHCIntegrator,
    MatbenchNHCWholeStepCUDAGraphMD,
    MatbenchCanonicalNoseHooverChainNVT,
    MatbenchTrajectoryRecorder,
    matched_trajectory_window,
    read_matbench_systems,
)
from fairchem.core.applications.esen_whole_step_cuda_graph import (
    ElasticWholeStepCUDAGraphController,
    WholeStepTransactionSnapshot,
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
    ase_md = MatbenchCanonicalNoseHooverChainNVT(
        atoms,
        timestep=0.25 * units.fs,
        temperature_K=300.0,
        tdamp=25.0 * units.fs,
        tchain=3,
        tloop=1,
    )
    # The adapter must not call the installed ASE version's integrate_nhc;
    # ASE 3.24 and 3.28 implement different coefficient scaling there.
    def reject_version_dependent_integrator(*_args, **_kwargs):
        raise AssertionError("version-dependent ASE integrate_nhc was called")

    ase_md._thermostat.integrate_nhc = reject_version_dependent_integrator
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
    # The canonical adapter removes ASE-version dependence from the baseline.
    # NumPy and Torch may still differ by a few FP64 ulps, so this is strict
    # numerical equivalence rather than a bitwise assertion.
    for actual, expected in (
        (state.positions.detach().numpy(), atoms.get_positions()),
        (state.momenta.detach().numpy(), atoms.get_momenta()),
        (integrator.eta.detach().numpy(), ase_thermostat._eta),
        (integrator.p_eta.detach().numpy(), ase_thermostat._p_eta),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


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


def test_rob1_snapshot_restores_complete_md_and_nhc_state():
    class FakeIntegrator:
        def __init__(self):
            self.eta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
            self.p_eta = torch.tensor([1.1, 1.2, 1.3], dtype=torch.float64)

    whole = type("Whole", (), {})()
    whole.positions = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    whole.momenta = whole.positions + 10.0
    whole.forces = whole.positions + 20.0
    whole.potential_energy = torch.tensor(31.0, dtype=torch.float32)
    whole.step_counter = torch.tensor(7, dtype=torch.long)
    whole.advance = torch.tensor(1.0, dtype=torch.float64)
    whole.integrator = FakeIntegrator()

    snapshot = WholeStepTransactionSnapshot(whole)
    snapshot.save_from_(whole)
    saved_addresses = snapshot.addresses()

    for tensor in (
        whole.positions,
        whole.momenta,
        whole.forces,
        whole.potential_energy,
        whole.step_counter,
        whole.advance,
        whole.integrator.eta,
        whole.integrator.p_eta,
    ):
        tensor.add_(100)

    snapshot.restore_into_(whole)

    torch.testing.assert_close(
        whole.positions, torch.arange(6, dtype=torch.float64).reshape(2, 3)
    )
    torch.testing.assert_close(whole.momenta, whole.positions + 10.0)
    torch.testing.assert_close(whole.forces, whole.positions + 20.0)
    assert whole.potential_energy.item() == 31.0
    assert whole.step_counter.item() == 7
    assert whole.advance.item() == 1.0
    torch.testing.assert_close(
        whole.integrator.eta,
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
    )
    torch.testing.assert_close(
        whole.integrator.p_eta,
        torch.tensor([1.1, 1.2, 1.3], dtype=torch.float64),
    )
    assert snapshot.addresses() == saved_addresses
    assert snapshot.addresses_stable


def test_rob1_third_step_overflow_rolls_back_and_replays_transaction():
    class FakeIntegrator:
        def __init__(self):
            self.eta = torch.zeros(2, dtype=torch.float64)
            self.p_eta = torch.zeros(2, dtype=torch.float64)

    class FakeBuilder:
        def __init__(self):
            self.injected = False
            self.local_step = 0
            self.misses = 0

        def reset_window_stats(self):
            self.local_step = 0
            self.misses = 0

        def observe_step(self):
            self.local_step += 1
            if self.local_step == 3 and not self.injected:
                self.injected = True
                self.misses = 1

        def window_stats(self):
            return {
                "fixed_builder_window_capacity_misses": self.misses,
                "fixed_builder_window_overflow_dummy_only_replays": self.misses,
                "fixed_builder_window_maximum_included_neighbors_by_atom": [5],
            }

    class FakeWhole:
        device = torch.device("cpu")

        def __init__(self):
            self.positions = torch.zeros((1, 3), dtype=torch.float64)
            self.momenta = torch.zeros((1, 3), dtype=torch.float64)
            self.forces = torch.zeros((1, 3), dtype=torch.float64)
            self.potential_energy = torch.zeros((), dtype=torch.float32)
            self.step_counter = torch.zeros((), dtype=torch.long)
            self.advance = torch.ones((), dtype=torch.float64)
            self.integrator = FakeIntegrator()
            self.fixed_builder = FakeBuilder()

        def step(self):
            self.fixed_builder.observe_step()
            self.positions.add_(1.0)
            self.momenta.add_(2.0)
            self.forces.fill_(3.0)
            self.potential_energy.add_(4.0)
            self.step_counter.add_(1)
            self.integrator.eta.add_(5.0)
            self.integrator.p_eta.add_(6.0)
            return self.forces, self.potential_energy

    whole = FakeWhole()
    controller = object.__new__(ElasticWholeStepCUDAGraphController)
    controller.whole = whole
    controller.snapshot = WholeStepTransactionSnapshot(whole)
    controller.attempted_replays = 0
    controller.committed_replays = 0
    controller.discarded_replays = 0
    controller.committed_physical_steps = 0
    controller.rollback_count = 0
    controller.retried_physical_steps = 0
    controller.detected_overflow_replays = 0

    def promote_and_restore(demand, *, transaction_steps):
        assert demand == [5]
        assert transaction_steps == 10
        controller.snapshot.restore_into_(whole)

    controller._promote_and_recapture = promote_and_restore
    controller.run_steps(10)

    # The first ten attempted steps were discarded.  Final state is exactly
    # the state obtained from ten clean reference steps, including NHC state.
    torch.testing.assert_close(
        whole.positions, torch.full((1, 3), 10.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        whole.momenta, torch.full((1, 3), 20.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        whole.forces, torch.full((1, 3), 3.0, dtype=torch.float64)
    )
    assert whole.potential_energy.item() == 40.0
    assert whole.step_counter.item() == 10
    torch.testing.assert_close(
        whole.integrator.eta, torch.full((2,), 50.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        whole.integrator.p_eta,
        torch.full((2,), 60.0, dtype=torch.float64),
    )
    assert controller.attempted_replays == 20
    assert controller.committed_replays == 10
    assert controller.discarded_replays == 10
    assert controller.rollback_count == 1
    assert controller.retried_physical_steps == 10
