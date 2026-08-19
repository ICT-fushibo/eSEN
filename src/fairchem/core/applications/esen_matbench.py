"""Matbench/DynaMat-only NHC execution helpers for eSEN.

This module is deliberately separate from the production eSEN MD backends.  The
existing Opt1/Opt2/Opt3 paths use the project Berendsen integrator and must keep
that behaviour.  Matbench uses an ASE Nose-Hoover chain (NHC), so this file
contains the independent FP64 integrator and the corresponding whole-step graph
adapter used only by the Matbench runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ase import Atoms, units
from ase.md.nose_hoover_chain import FOURTH_ORDER_COEFFS
from torch import Tensor

from fairchem.core.applications.esen_gpu_md import (
    ESENEnergyForceEvaluator,
    GPUMDState,
)
from fairchem.core.applications.esen_whole_step_cuda_graph import (
    ESENWholeStepCUDAGraphMD,
)


MATBENCH_STEPS = 80_000
MATBENCH_RECORD_INTERVAL = 10
MATBENCH_TIMESTEP_FS = 0.25
MATBENCH_THERMOSTAT_TIME_FS = 25.0
MATBENCH_SEED = 0
MATBENCH_TCHAIN = 3
MATBENCH_TLOOP = 1
MATBENCH_DUMMY_ATOMS = 32


@dataclass(frozen=True)
class MatbenchSystem:
    """Initial frame and metadata for one public DynaMat system."""

    name: str
    atomic_numbers: np.ndarray
    initial_positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    reference_frames: int
    reference_dt_fs: float
    temperature_kelvin: float
    reference_has_stress: bool

    def atoms(self) -> Atoms:
        return Atoms(
            numbers=self.atomic_numbers.copy(),
            positions=self.initial_positions.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc.copy(),
        )


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Matbench HDF5 support requires h5py in the active environment"
        ) from exc
    return h5py


def read_matbench_systems(
    path: str | Path,
    systems: Sequence[str] | None = None,
) -> list[MatbenchSystem]:
    """Read only the first frame and metadata from the public HDF5 file.

    The cell is intentionally checked over all reference frames.  Matbench NVT
    trajectories use a fixed cell, which is also required by the fixed-shape
    neighbor graph in Opt3.
    """

    h5py = _require_h5py()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    requested = None if systems is None else set(systems)
    result: list[MatbenchSystem] = []
    with h5py.File(path, "r") as handle:
        names = sorted(
            name
            for name, value in handle.items()
            if isinstance(value, h5py.Group)
        )
        if requested is not None:
            missing = requested - set(names)
            if missing:
                raise KeyError(
                    f"Unknown Matbench systems {sorted(missing)}; "
                    f"available systems are {names}"
                )
            names = [name for name in names if name in requested]
        if not names:
            raise ValueError(f"No Matbench systems found in {path}")

        for name in names:
            group = handle[name]
            schema = int(group.attrs.get("schema", -1))
            if schema != 1:
                raise ValueError(f"{name}: unsupported HDF5 schema {schema}")
            required = {"atomic_numbers", "positions", "cell", "pbc"}
            missing = required - set(group.keys())
            if missing:
                raise ValueError(f"{name}: missing datasets {sorted(missing)}")
            for attr in ("dt_fs", "temperature_kelvin"):
                if attr not in group.attrs:
                    raise ValueError(f"{name}: missing attribute {attr}")

            atomic_numbers = np.asarray(group["atomic_numbers"][:], dtype=np.int64)
            if atomic_numbers.ndim != 1 or atomic_numbers.size < 1:
                raise ValueError(f"{name}: atomic_numbers must be a non-empty vector")
            positions = group["positions"]
            cell_dataset = group["cell"]
            pbc = np.asarray(group["pbc"][:], dtype=bool).reshape(-1)
            if positions.ndim != 3 or positions.shape[2] != 3:
                raise ValueError(
                    f"{name}: positions must be [frames, atoms, 3], got "
                    f"{positions.shape}"
                )
            if positions.shape[0] < 1:
                raise ValueError(f"{name}: positions must contain frame zero")
            if positions.shape[1] != len(atomic_numbers):
                raise ValueError(
                    f"{name}: positions atom count {positions.shape[1]} does not "
                    f"match atomic_numbers {len(atomic_numbers)}"
                )
            if cell_dataset.shape != (positions.shape[0], 3, 3):
                raise ValueError(
                    f"{name}: cell must be [frames, 3, 3], got {cell_dataset.shape}"
                )
            if pbc.shape != (3,) or not bool(np.all(pbc)):
                raise ValueError(f"{name}: Matbench systems must be 3D periodic")
            first_cell = np.asarray(cell_dataset[0], dtype=np.float64)
            if not np.isfinite(first_cell).all() or abs(np.linalg.det(first_cell)) <= 0:
                raise ValueError(f"{name}: invalid initial cell")
            # Cells are small enough to check directly and this catches accidental
            # use of a variable-cell trajectory before it reaches Opt3 capture.
            if not np.allclose(cell_dataset[:], first_cell[None, :, :]):
                raise ValueError(f"{name}: variable cells are not supported")
            first_positions = np.asarray(positions[0], dtype=np.float64)
            if not np.isfinite(first_positions).all():
                raise ValueError(f"{name}: initial positions contain NaN/Inf")

            result.append(
                MatbenchSystem(
                    name=name,
                    atomic_numbers=atomic_numbers,
                    initial_positions=first_positions,
                    cell=first_cell,
                    pbc=pbc,
                    reference_frames=int(positions.shape[0]),
                    reference_dt_fs=float(group.attrs["dt_fs"]),
                    temperature_kelvin=float(group.attrs["temperature_kelvin"]),
                    reference_has_stress="stress" in group,
                )
            )
    return result


class MatbenchTrajectoryRecorder:
    """Fixed-size host recorder for the 8,001 Matbench sampled frames."""

    def __init__(
        self,
        *,
        n_atoms: int,
        steps: int,
        record_interval: int,
        cell: np.ndarray,
        timestep_fs: float = MATBENCH_TIMESTEP_FS,
    ) -> None:
        if steps < 1 or record_interval < 1 or steps % record_interval:
            raise ValueError("steps must be divisible by a positive record interval")
        if timestep_fs <= 0:
            raise ValueError("timestep_fs must be positive")
        self.n_atoms = int(n_atoms)
        self.steps = int(steps)
        self.record_interval = int(record_interval)
        self.timestep_fs = float(timestep_fs)
        self.n_frames = steps // record_interval + 1
        self.cell = np.asarray(cell, dtype=np.float64).copy()
        self.md_step = np.empty(self.n_frames, dtype=np.int64)
        self.positions = np.empty(
            (self.n_frames, self.n_atoms, 3), dtype=np.float64
        )
        self.momenta = np.empty(
            (self.n_frames, self.n_atoms, 3), dtype=np.float64
        )
        self.forces = np.empty(
            (self.n_frames, self.n_atoms, 3), dtype=np.float32
        )
        self.energy = np.empty(self.n_frames, dtype=np.float64)
        self._next = 0

    @property
    def completed_frames(self) -> int:
        return self._next

    def append(
        self,
        step: int,
        positions: np.ndarray,
        momenta: np.ndarray,
        forces: np.ndarray,
        energy: float,
    ) -> None:
        expected = self._next * self.record_interval
        if step != expected:
            raise ValueError(f"Expected sampled MD step {expected}, got {step}")
        if self._next >= self.n_frames:
            raise RuntimeError("Too many trajectory frames")
        positions = np.asarray(positions, dtype=np.float64)
        momenta = np.asarray(momenta, dtype=np.float64)
        forces = np.asarray(forces, dtype=np.float32)
        if positions.shape != (self.n_atoms, 3):
            raise ValueError(f"Invalid positions shape {positions.shape}")
        if momenta.shape != (self.n_atoms, 3):
            raise ValueError(f"Invalid momenta shape {momenta.shape}")
        if forces.shape != (self.n_atoms, 3):
            raise ValueError(f"Invalid forces shape {forces.shape}")
        self.md_step[self._next] = step
        self.positions[self._next] = positions
        self.momenta[self._next] = momenta
        self.forces[self._next] = forces
        self.energy[self._next] = float(energy)
        self._next += 1

    def finalize(self) -> None:
        if self._next != self.n_frames:
            raise RuntimeError(
                f"Recorded {self._next} frames, expected {self.n_frames}"
            )

    def write(
        self,
        path: str | Path,
        *,
        atomic_numbers: np.ndarray,
        pbc: np.ndarray,
        temperature_kelvin: float,
        backend: str,
        metadata: dict[str, Any] | None = None,
        allow_partial: bool = False,
    ) -> None:
        if not allow_partial:
            self.finalize()
        if self._next == 0:
            raise RuntimeError("Cannot write an empty trajectory")
        h5py = _require_h5py()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = 1
            handle.attrs["trajectory_kind"] = "esen_matbench_prediction"
            handle.attrs["dt_fs"] = self.timestep_fs * self.record_interval
            handle.attrs["record_interval"] = self.record_interval
            handle.attrs["steps"] = self.steps
            handle.attrs["temperature_kelvin"] = float(temperature_kelvin)
            handle.attrs["backend"] = backend
            handle.attrs["stress_status"] = "not_computed"
            handle.attrs["complete"] = bool(self._next == self.n_frames)
            handle.attrs["completed_frames"] = self._next
            for key, value in (metadata or {}).items():
                if isinstance(value, (str, int, float, bool, np.number)):
                    handle.attrs[key] = value
            handle.create_dataset("atomic_numbers", data=np.asarray(atomic_numbers))
            handle.create_dataset("pbc", data=np.asarray(pbc, dtype=bool))
            n_frames = self._next
            handle.create_dataset(
                "cell", data=np.broadcast_to(self.cell, (n_frames, 3, 3))
            )
            handle.create_dataset("md_step", data=self.md_step[:n_frames])
            handle.create_dataset("positions", data=self.positions[:n_frames])
            handle.create_dataset("momenta", data=self.momenta[:n_frames])
            handle.create_dataset("forces", data=self.forces[:n_frames])
            handle.create_dataset("energy", data=self.energy[:n_frames])
        temporary.replace(path)


class MatbenchNHCIntegrator:
    """FP64 GPU implementation of ASE's Nose-Hoover-chain integrator."""

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float = MATBENCH_TIMESTEP_FS,
        temperature_K: float,
        thermostat_time_fs: float = MATBENCH_THERMOSTAT_TIME_FS,
        tchain: int = MATBENCH_TCHAIN,
        tloop: int = MATBENCH_TLOOP,
    ) -> None:
        if masses.ndim not in {1, 2}:
            raise ValueError(f"masses must be [N] or [N, 1], got {masses.shape}")
        if timestep_fs <= 0 or temperature_K <= 0 or thermostat_time_fs <= 0:
            raise ValueError("NHC parameters must be positive")
        if tchain < 1 or tloop < 1:
            raise ValueError("tchain and tloop must be positive")
        self.masses = masses.reshape(-1, 1).to(dtype=torch.float64)
        self.dt = float(timestep_fs) * units.fs
        self.temperature_target = float(temperature_K)
        self.temperature_K = float(temperature_K)
        self.thermostat_time_fs = float(thermostat_time_fs)
        self.tchain = int(tchain)
        self.tloop = int(tloop)
        self.num_atoms = int(self.masses.shape[0])
        self.kT = float(units.kB) * self.temperature_target
        tdamp = float(thermostat_time_fs) * units.fs
        q = torch.full(
            (self.tchain,),
            self.kT * tdamp * tdamp,
            dtype=torch.float64,
            device=self.masses.device,
        )
        q[0] = 3.0 * self.num_atoms * self.kT * tdamp * tdamp
        self.Q = q
        self.eta = torch.zeros_like(q)
        self.p_eta = torch.zeros_like(q)
        self.coefficients = torch.as_tensor(
            list(FOURTH_ORDER_COEFFS), dtype=torch.float64, device=q.device
        )

    def clone_thermostat_state(self) -> tuple[Tensor, Tensor]:
        return self.eta.clone(), self.p_eta.clone()

    @torch.no_grad()
    def restore_thermostat_state_(self, eta: Tensor, p_eta: Tensor) -> None:
        self.eta.copy_(eta)
        self.p_eta.copy_(p_eta)

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return (0.5 * momenta.square() / self.masses).sum()

    def temperature(self, momenta: Tensor) -> Tensor:
        return 2.0 * self.kinetic_energy(momenta) / (
            (3 * self.num_atoms) * units.kB
        )

    def _integrate_p_eta_j(
        self,
        momenta: Tensor,
        p_eta: list[Tensor],
        j: int,
        delta2: Tensor | float,
        delta4: Tensor | float,
    ) -> None:
        if j < self.tchain - 1:
            p_eta[j] = p_eta[j] * torch.exp(
                -delta4 * p_eta[j + 1] / self.Q[j + 1]
            )
        if j == 0:
            g_j = (momenta.square() / self.masses).sum() - (
                3 * self.num_atoms * self.kT
            )
        else:
            g_j = p_eta[j - 1].square() / self.Q[j - 1] - self.kT
        p_eta[j] = p_eta[j] + delta2 * g_j
        if j < self.tchain - 1:
            p_eta[j] = p_eta[j] * torch.exp(
                -delta4 * p_eta[j + 1] / self.Q[j + 1]
            )

    def integrate_nhc(
        self,
        momenta: Tensor,
        eta: Tensor,
        p_eta: Tensor,
        delta: float | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the pure NHC update without mutating persistent state."""

        p = momenta
        eta_out = eta
        p_eta_out = p_eta
        for _ in range(self.tloop):
            for coefficient in self.coefficients:
                sub_delta = coefficient * delta / self.tloop
                delta2 = sub_delta / 2.0
                delta4 = sub_delta / 4.0
                p_eta_list = [p_eta_out[index] for index in range(self.tchain)]
                for index in range(self.tchain - 1, -1, -1):
                    self._integrate_p_eta_j(
                        p, p_eta_list, index, delta2, delta4
                    )
                eta_out = eta_out + sub_delta * (
                    torch.stack(p_eta_list) / self.Q
                )
                p = p * torch.exp(-sub_delta * p_eta_list[0] / self.Q[0])
                for index in range(self.tchain):
                    self._integrate_p_eta_j(
                        p, p_eta_list, index, delta2, delta4
                    )
                p_eta_out = torch.stack(p_eta_list)
        return p, eta_out, p_eta_out

    def step(self, state: GPUMDState, force_fn: ESENEnergyForceEvaluator) -> None:
        """Perform one ASE-order NHC step for GPUResidentMD."""

        half_momenta, eta_half, p_eta_half = self.integrate_nhc(
            state.momenta, self.eta, self.p_eta, self.dt / 2.0
        )
        if state.forces is None:
            model_forces, energy = force_fn(state.positions)
            state.forces = model_forces.to(dtype=state.positions.dtype)
            state.potential_energy = energy
        half_momenta = half_momenta + 0.5 * self.dt * state.forces
        positions = state.positions + self.dt * half_momenta / self.masses
        model_forces, energy = force_fn(positions)
        forces = model_forces.to(dtype=positions.dtype)
        half_momenta = half_momenta + 0.5 * self.dt * forces
        momenta, eta_final, p_eta_final = self.integrate_nhc(
            half_momenta, eta_half, p_eta_half, self.dt / 2.0
        )
        state.positions = positions
        state.momenta = momenta
        state.forces = forces
        state.potential_energy = energy
        self.eta = eta_final
        self.p_eta = p_eta_final


class MatbenchNHCWholeStepCUDAGraphMD(ESENWholeStepCUDAGraphMD):
    """Opt3 whole-step graph with NHC state captured in the graph."""

    def __init__(self, *args, **kwargs) -> None:
        integrator = args[2] if len(args) >= 3 else kwargs.get("integrator")
        if not isinstance(integrator, MatbenchNHCIntegrator):
            raise TypeError("Matbench whole-step graph requires MatbenchNHCIntegrator")
        super().__init__(*args, **kwargs)
        self.initial_eta, self.initial_p_eta = integrator.clone_thermostat_state()

    @torch.no_grad()
    def restore_state_(self, state: GPUMDState) -> None:
        super().restore_state_(state)
        self.integrator.restore_thermostat_state_(
            self.initial_eta, self.initial_p_eta
        )

    @torch.no_grad()
    def evaluate_initial(self) -> tuple[Tensor, Tensor]:
        """Evaluate frame zero without advancing any MD state.

        Explicitly arm the graph's ``advance=0`` input instead of relying on
        whatever value a setup path left there.  This keeps frame zero equal to
        the supplied initial structure and makes the production replay count
        exactly ``steps + 1``.
        """

        if self.graph is None:
            raise RuntimeError("Capture must complete before replay")
        self.advance.zero_()
        self.graph.replay()
        self.production_replays += 1
        self.total_replays += 1
        self.advance.fill_(1.0)
        return self.forces, self.potential_energy

    def _graph_body(self) -> None:
        integrator: MatbenchNHCIntegrator = self.integrator
        with torch.no_grad():
            old_momenta = self.momenta
            old_eta = integrator.eta
            old_p_eta = integrator.p_eta
            half_momenta, eta_half, p_eta_half = integrator.integrate_nhc(
                old_momenta,
                old_eta,
                old_p_eta,
                integrator.dt / 2.0,
            )
            half_momenta = half_momenta + 0.5 * integrator.dt * self.forces
            advanced_positions = (
                self.positions + integrator.dt * half_momenta / integrator.masses
            )
            evaluation_positions = self.positions + self.advance * (
                advanced_positions - self.positions
            )
            graph_step = self.step_counter + self.advance.to(torch.long)
            self.core.static_positions[: self.num_atoms].copy_(evaluation_positions)
            self.fixed_builder.build(
                self.core.static_positions[: self.num_atoms], step=graph_step
            )

        model_forces, model_energy = self.core._static_forward()

        with torch.no_grad():
            forces = model_forces.to(dtype=self.positions.dtype)
            half_momenta = half_momenta + 0.5 * integrator.dt * forces
            advanced_momenta, eta_final, p_eta_final = integrator.integrate_nhc(
                half_momenta,
                eta_half,
                p_eta_half,
                integrator.dt / 2.0,
            )
            final_momenta = old_momenta + self.advance * (
                advanced_momenta - old_momenta
            )
            final_eta = old_eta + self.advance * (eta_final - old_eta)
            final_p_eta = old_p_eta + self.advance * (p_eta_final - old_p_eta)
            self.positions.copy_(evaluation_positions)
            self.momenta.copy_(final_momenta)
            self.forces.copy_(forces)
            self.potential_energy.copy_(model_energy)
            integrator.eta.copy_(final_eta)
            integrator.p_eta.copy_(final_p_eta)
            self.step_counter.add_(self.advance.to(torch.long))

    def reset_production(self, initial_state: GPUMDState) -> None:
        super().reset_production(initial_state)
        self.integrator.restore_thermostat_state_(
            self.initial_eta, self.initial_p_eta
        )


def initialize_matbench_atoms(
    system: MatbenchSystem,
    *,
    seed: int = MATBENCH_SEED,
) -> Atoms:
    """Build the official Matbench initial state, including seeded momenta."""

    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = system.atoms()
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=system.temperature_kelvin,
        rng=np.random.default_rng(seed),
    )
    return atoms


def as_numpy_state(
    state: GPUMDState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if state.forces is None or state.potential_energy is None:
        raise RuntimeError("MD state does not contain forces and energy")
    return (
        state.positions.detach().cpu().numpy(),
        state.momenta.detach().cpu().numpy(),
        state.forces.detach().cpu().numpy(),
        float(state.potential_energy.detach().cpu().item()),
    )
