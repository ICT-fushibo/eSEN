#!/usr/bin/env python3
"""Run eSEN baseline/Opt1/Opt2/Opt3 on the public Matbench MD dataset.

This runner is intentionally independent from the existing Cu/H2O and Opt4
benchmarks.  It uses the official DynaMat NHC protocol and writes sampled
trajectories to HDF5 so the public RDF/ADF/vDOS metrics can be evaluated later.
The current eSEN energy-force path does not calculate stress, so pressure and
private energy/force RMSE values are reported as unavailable rather than being
silently inferred.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import types
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent
DEFAULT_REFERENCE = (
    ROOT
    / "matbench-discovery-data"
    / "md"
    / "2026-06-29-dynamat-v1.0-reference-trajectories.h5"
)
DEFAULT_MATBENCH_REPO = ROOT / "matbench-discovery"
DEFAULT_CHECKPOINT = REPO / "esen_30m_oam.pt"

BACKENDS = ("baseline", "opt1", "opt2", "opt3")
STEPS = 80_000
RECORD_INTERVAL = 10
TIMESTEP_FS = 0.25
THERMOSTAT_TIME_FS = 25.0
SEED = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--reference-h5", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--matbench-repo", type=Path, default=DEFAULT_MATBENCH_REPO)
    parser.add_argument("--published-yaml", type=Path)
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", help="Physical GPU index; normally use CUDA_VISIBLE_DEVICES")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--record-interval", type=int, default=RECORD_INTERVAL)
    parser.add_argument("--timestep-fs", type=float, default=TIMESTEP_FS)
    parser.add_argument(
        "--thermostat-time-fs", type=float, default=THERMOSTAT_TIME_FS
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--probe-steps", type=int, default=50)
    parser.add_argument("--neighbor-margin", type=float, default=0.10)
    parser.add_argument("--neighbor-slot-step", type=int, default=8)
    parser.add_argument("--edge-step", type=int, default=256)
    parser.add_argument("--dummy-atoms", type=int, default=32)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--max-neighbors", type=int, default=300)
    parser.add_argument("--degeneracy-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate public Matbench metrics after trajectories are complete",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.record_interval < 1:
        parser.error("--steps and --record-interval must be positive")
    if args.steps % args.record_interval:
        parser.error("--steps must be divisible by --record-interval")
    if args.timestep_fs <= 0 or args.thermostat_time_fs <= 0:
        parser.error("NHC time parameters must be positive")
    if args.seed != SEED:
        parser.error(f"Matbench seed is fixed at {SEED}")
    if args.probe_steps < 0 or args.neighbor_margin < 0:
        parser.error("probe steps and neighbor margin must be non-negative")
    if args.neighbor_slot_step < 1 or args.edge_step < 1 or args.dummy_atoms < 1:
        parser.error("neighbor and dummy parameters must be positive")
    if args.capture_warmup < 0 or args.max_neighbors < 1:
        parser.error("capture warmup and max neighbors must be valid")
    if args.published_yaml is None:
        args.published_yaml = (
            args.matbench_repo / "models" / "esen" / "esen-30m-oam.yml"
        )
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def _append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _device_used(torch_module, device) -> int | None:
    try:
        free, total = torch_module.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError):
        return None
    return int(total - free)


def _runtime_metadata(torch_module) -> dict[str, Any]:
    """Return scalar HDF5-safe runtime provenance for one prediction file."""

    metadata: dict[str, Any] = {
        "torch_version": str(torch_module.__version__),
        "ase_version": "unknown",
    }
    try:
        import ase

        metadata["ase_version"] = str(ase.__version__)
    except Exception:
        pass
    if torch_module.cuda.is_available():
        try:
            properties = torch_module.cuda.get_device_properties(0)
            metadata["gpu_name"] = str(properties.name)
            uuid = getattr(properties, "uuid", None)
            if uuid is not None:
                metadata["gpu_uuid"] = str(uuid)
        except Exception:
            pass
    return metadata


def _capture_torch_rng(torch_module) -> dict[str, Any]:
    return {
        "cpu": torch_module.get_rng_state(),
        "cuda": torch_module.cuda.get_rng_state_all(),
    }


def _restore_torch_rng(torch_module, state: dict[str, Any]) -> None:
    torch_module.set_rng_state(state["cpu"])
    torch_module.cuda.set_rng_state_all(state["cuda"])


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in text


def _is_capacity_overflow(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "CUDAGraphCapacityError" or (
        "capacity" in str(exc).lower() and "neighbor" in str(exc).lower()
    )


def _record_gpu(recorder, step: int, state, as_numpy_state) -> None:
    positions, momenta, forces, energy = as_numpy_state(state)
    recorder.append(step, positions, momenta, forces, energy)


def _make_gpu_state(torch_module, atoms, device):
    from fairchem.core.applications.esen_gpu_md import GPUMDState

    return GPUMDState(
        positions=torch_module.as_tensor(
            atoms.get_positions(), dtype=torch_module.float64, device=device
        ).clone(),
        momenta=torch_module.as_tensor(
            atoms.get_momenta(), dtype=torch_module.float64, device=device
        ).clone(),
    )


def _make_nhc(torch_module, atoms, args, device):
    from fairchem.core.applications.esen_matbench import MatbenchNHCIntegrator

    masses = torch_module.as_tensor(
        atoms.get_masses(), dtype=torch_module.float64, device=device
    )
    return MatbenchNHCIntegrator(
        masses,
        timestep_fs=args.timestep_fs,
        temperature_K=float(atoms.info["matbench_temperature_kelvin"]),
        thermostat_time_fs=args.thermostat_time_fs,
    )


def _new_recorder(system, args):
    from fairchem.core.applications.esen_matbench import MatbenchTrajectoryRecorder

    return MatbenchTrajectoryRecorder(
        n_atoms=len(system.atomic_numbers),
        steps=args.steps,
        record_interval=args.record_interval,
        cell=system.cell,
        timestep_fs=args.timestep_fs,
    )


def _run_baseline(system, args, checkpoint, recorder):
    import torch
    from ase import units
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
    from fairchem.core import OCPCalculator
    from fairchem.core.applications.esen_gpu_md import (
        configure_esen_energy_force_inference,
    )
    from fairchem.core.applications.esen_matbench import initialize_matbench_atoms

    atoms = initialize_matbench_atoms(system, seed=args.seed)
    atoms.info["matbench_temperature_kelvin"] = system.temperature_kelvin
    calc = OCPCalculator(
        checkpoint_path=checkpoint,
        cpu=False,
        seed=args.seed,
        only_output=["energy", "forces"],
        disable_amp=True,
    )
    configure_esen_energy_force_inference(calc.trainer._unwrapped_model)
    atoms.calc = calc
    dynamics = NoseHooverChainNVT(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=system.temperature_kelvin,
        tdamp=args.thermostat_time_fs * units.fs,
        tchain=3,
        tloop=1,
    )
    recorded_steps: set[int] = set()

    def record() -> None:
        step = int(dynamics.nsteps)
        if step in recorded_steps or step % args.record_interval:
            return
        recorder.append(
            step,
            atoms.get_positions(),
            atoms.get_momenta(),
            atoms.get_forces(),
            float(atoms.get_potential_energy()),
        )
        recorded_steps.add(step)

    dynamics.attach(record, interval=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    record()
    dynamics.run(args.steps)
    record()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    recorder.finalize()
    return {
        "rollout_wall_time_s": elapsed,
        "setup_wall_time_s": 0.0,
        "capture_wall_time_s": 0.0,
        "probe_wall_time_s": 0.0,
        "graph_stats": {},
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available()
            else None
        ),
    }


def _run_gpu(system, args, checkpoint, recorder, backend):
    import torch
    from fairchem.core.applications.esen_fixed_neighbor import (
        maximum_neighbors_in_graph,
        neighbor_capacity_from_probe,
    )
    from fairchem.core.applications.esen_gpu_md import (
        ESENEnergyForceEvaluator,
        GPUResidentMD,
    )
    from fairchem.core.applications.esen_cuda_graph import (
        ESENModelCUDAGraphEvaluator,
        edge_capacity_from_probe,
    )
    from fairchem.core.applications.esen_matbench import (
        MatbenchNHCIntegrator,
        MatbenchNHCWholeStepCUDAGraphMD,
        as_numpy_state,
        initialize_matbench_atoms,
    )

    device = torch.device("cuda:0")
    atoms = initialize_matbench_atoms(system, seed=args.seed)
    atoms.info["matbench_temperature_kelvin"] = system.temperature_kelvin
    evaluator = ESENEnergyForceEvaluator(
        atoms, checkpoint, device=device, seed=args.seed, disable_amp=True
    )
    state = _make_gpu_state(torch, atoms, device)
    integrator = MatbenchNHCIntegrator(
        torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=device),
        timestep_fs=args.timestep_fs,
        temperature_K=system.temperature_kelvin,
        thermostat_time_fs=args.thermostat_time_fs,
    )
    initial_state = state.clone()
    initial_thermostat = integrator.clone_thermostat_state()
    setup_started = time.perf_counter()
    probe_started = time.perf_counter()
    setup_rng = _capture_torch_rng(torch)
    graph_stats: dict[str, Any] = {}
    graph = None
    dynamics = None
    whole = None
    eager_initial_forces, eager_initial_energy = evaluator(state.positions)
    eager_initial_forces = eager_initial_forces.detach().clone()
    eager_initial_energy = eager_initial_energy.detach().clone()
    torch.cuda.synchronize()
    _restore_torch_rng(torch, setup_rng)
    cg_initial_forces = None
    cg_initial_energy = None

    if backend == "opt1":
        probe_elapsed = 0.0
    elif backend == "opt2":
        probe_edges = [
            int(evaluator.build_neighbor_graph(state.positions)["edge_index"].shape[1])
        ]
        probe_dynamics = GPUResidentMD(state, evaluator, integrator)
        for _ in range(args.probe_steps):
            probe_dynamics.run(1)
            probe_edges.append(
                int(
                    evaluator.build_neighbor_graph(state.positions)[
                        "edge_index"
                    ].shape[1]
                )
            )
        torch.cuda.synchronize()
        probe_elapsed = time.perf_counter() - probe_started
        state.restore_(initial_state)
        integrator.restore_thermostat_state_(*initial_thermostat)
        _restore_torch_rng(torch, setup_rng)
        edge_capacity = edge_capacity_from_probe(
            max(probe_edges), margin=args.neighbor_margin, edge_step=args.edge_step
        )
        graph = ESENModelCUDAGraphEvaluator(
            evaluator,
            edge_capacity=edge_capacity,
            dummy_atoms=args.dummy_atoms,
            capture_warmup=args.capture_warmup,
            replay_energy_atol=0.0,
            replay_force_atol=1e-6,
        )
        graph.capture(state.positions)
        graph.reset_production_stats()
        dynamics = GPUResidentMD(state, graph, integrator)
        graph_stats = graph.stats()
        graph_stats.update(
            {
                "matbench_probe_max_edges": max(probe_edges),
                "matbench_edge_capacity": edge_capacity,
            }
        )
    else:
        probe_degrees = [
            maximum_neighbors_in_graph(
                evaluator.build_neighbor_graph(state.positions)["edge_index"],
                len(atoms),
            )
        ]
        probe_dynamics = GPUResidentMD(state, evaluator, integrator)
        for _ in range(args.probe_steps):
            probe_dynamics.run(1)
            probe_degrees.append(
                maximum_neighbors_in_graph(
                    evaluator.build_neighbor_graph(state.positions)["edge_index"],
                    len(atoms),
                )
            )
        torch.cuda.synchronize()
        probe_elapsed = time.perf_counter() - probe_started
        state.restore_(initial_state)
        integrator.restore_thermostat_state_(*initial_thermostat)
        _restore_torch_rng(torch, setup_rng)
        neighbors_per_atom = neighbor_capacity_from_probe(
            max(probe_degrees),
            margin=args.neighbor_margin,
            slot_step=args.neighbor_slot_step,
        )
        whole = MatbenchNHCWholeStepCUDAGraphMD(
            state,
            evaluator,
            integrator,
            neighbors_per_atom=neighbors_per_atom,
            dummy_atoms=args.dummy_atoms,
            capture_warmup=args.capture_warmup,
            max_neighbors=args.max_neighbors,
            degeneracy_tolerance=args.degeneracy_tolerance,
        )
        whole.capture(initial_state)
        whole.reset_production(initial_state)
        graph_stats = whole.stats()
        graph_stats.update(
            {
                "matbench_probe_max_neighbors_per_atom": max(probe_degrees),
                "matbench_neighbors_per_atom": neighbors_per_atom,
            }
        )

    # One un-timed initial graph replay supplies the numerical audit values.
    # Resetting production state afterwards keeps the measured sequence at one
    # initial replay plus exactly ``steps`` production steps.
    if backend == "opt2":
        assert graph is not None
        cg_initial_forces, cg_initial_energy = graph(state.positions)
        cg_initial_forces = cg_initial_forces.detach().clone()
        cg_initial_energy = cg_initial_energy.detach().clone()
        graph.reset_production_stats()
    elif backend == "opt3":
        assert whole is not None
        whole.reset_production(initial_state)
        cg_initial_forces, cg_initial_energy = whole.evaluate_initial()
        cg_initial_forces = cg_initial_forces.detach().clone()
        cg_initial_energy = cg_initial_energy.detach().clone()
        whole.reset_production(initial_state)
    # Keep the audit copies off device during the timed rollout.
    eager_initial_forces = eager_initial_forces.cpu()
    eager_initial_energy = eager_initial_energy.cpu()
    if cg_initial_forces is not None and cg_initial_energy is not None:
        cg_initial_forces = cg_initial_forces.cpu()
        cg_initial_energy = cg_initial_energy.cpu()
    setup_elapsed = time.perf_counter() - setup_started
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if backend == "opt1":
        dynamics = GPUResidentMD(state, evaluator, integrator)

    start = time.perf_counter()
    if backend in {"opt1", "opt2"}:
        assert dynamics is not None
        dynamics.evaluate()
        _record_gpu(recorder, 0, state, as_numpy_state)
        for step in range(1, args.steps + 1):
            dynamics.run(1)
            if step % args.record_interval == 0:
                _record_gpu(recorder, step, state, as_numpy_state)
        if backend == "opt2":
            assert graph is not None
            graph_stats = graph.stats()
    else:
        assert whole is not None
        whole.evaluate_initial()
        _record_gpu(recorder, 0, whole.state_view(), as_numpy_state)
        for step in range(1, args.steps + 1):
            whole.step()
            if step % args.record_interval == 0:
                _record_gpu(recorder, step, whole.state_view(), as_numpy_state)
        graph_stats = whole.stats()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    initial_validation: dict[str, Any] = {}
    if cg_initial_forces is not None and cg_initial_energy is not None:
        energy_error = float(
            (cg_initial_energy - eager_initial_energy).abs().max().item()
        )
        force_error = float(
            (cg_initial_forces - eager_initial_forces).abs().max().item()
        )
        energy_per_atom = energy_error / float(len(atoms))
        initial_validation = {
            "initial_eager_energy_abs_error_eV": energy_error,
            "initial_eager_energy_abs_error_eV_per_atom": energy_per_atom,
            "initial_eager_force_max_abs_error_eV_per_A": force_error,
            "initial_graph_validation_pass": bool(
                energy_per_atom < 1e-5 and force_error < 2e-4
            ),
        }
    recorder.finalize()
    return {
        "rollout_wall_time_s": elapsed,
        "setup_wall_time_s": setup_elapsed,
        "capture_wall_time_s": graph_stats.get("cuda_graph_capture_wall_time_s", 0.0),
        "probe_wall_time_s": probe_elapsed,
        "graph_stats": graph_stats,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "device_used_after_setup_gib": (
            None
            if _device_used(torch, device) is None
            else _device_used(torch, device) / 1024**3
        ),
        **initial_validation,
    }


def _load_public_metric_functions(matbench_repo: Path):
    """Load the public Matbench reader/metrics without its optional UI stack.

    Recent matbench-discovery releases import ``pymatviz`` from the metrics
    package initializer.  eSEN's inference environment does not need that UI
    dependency, so when the normal import fails we load the same ``md.py``
    implementation directly and provide only the metadata helpers that are not
    used by ``evaluate_md_system``.
    """

    try:
        from matbench_discovery.md import read_reference_trajectory
        from matbench_discovery.metrics.md import evaluate_md_system
        return read_reference_trajectory, evaluate_md_system
    except (ImportError, SyntaxError, NameError, TypeError) as exc:
        normal_import_error = exc

    package_root = matbench_repo / "matbench_discovery"
    metric_path = package_root / "metrics" / "md.py"
    if not metric_path.is_file():
        raise RuntimeError(
            "Public metric evaluation requires matbench-discovery on PYTHONPATH"
        ) from normal_import_error

    def load_compat_source(module_name: str, source_path: Path):
        source = source_path.read_text(encoding="utf-8")
        if module_name.endswith(".metrics.md"):
            # This is a runtime type-alias assignment rather than an annotation;
            # Python 3.9 cannot evaluate the PEP 604 expression.
            source = source.replace(
                "TrajectoryLike = Trajectory | Sequence[Atoms]",
                "TrajectoryLike = Any",
            )
        module = types.ModuleType(module_name)
        module.__file__ = str(source_path)
        module.__package__ = module_name.rpartition(".")[0]
        sys.modules[module_name] = module
        # Matbench source checkouts used with eSEN may predate their current
        # Python requirement and omit postponed annotations.  The source code
        # is otherwise compatible; postpone annotations while loading it.
        code = compile(
            "from __future__ import annotations\n" + source,
            str(source_path),
            "exec",
        )
        exec(code, module.__dict__)
        return module

    # Load the lightweight trajectory module, then replace only the metrics
    # package initializer that pulls in pymatviz/enums.
    trajectory_module = load_compat_source(
        "matbench_discovery.trajectory", package_root / "trajectory.py"
    )
    data_stub = types.ModuleType("matbench_discovery.data")
    data_stub.commented_map_with_units = lambda values, units: values
    data_stub.make_file_ref = lambda path, **kwargs: path
    data_stub.update_yaml_file = lambda *args, **kwargs: None
    sys.modules["matbench_discovery.data"] = data_stub
    hpc_stub = types.ModuleType("matbench_discovery.hpc")
    hpc_stub.COST_PROVENANCE_KEYS = (
        "hardware",
        "run_time_sec",
        "max_rss_gb",
        "max_gpu_mem_gb",
    )
    sys.modules["matbench_discovery.hpc"] = hpc_stub

    metrics_package = types.ModuleType("matbench_discovery.metrics")
    metrics_package.__path__ = [str(package_root / "metrics")]
    sys.modules["matbench_discovery.metrics"] = metrics_package
    module_name = "matbench_discovery.metrics.md"
    sys.modules.pop(module_name, None)
    metric_module = load_compat_source(module_name, metric_path)
    setattr(metrics_package, "md", metric_module)
    trajectory_type = importlib.import_module(
        "matbench_discovery.trajectory"
    ).Trajectory

    def read_reference_trajectory(path: str, system_name: str):
        import h5py

        with h5py.File(path, "r") as handle:
            if system_name not in handle:
                raise KeyError(f"{system_name!r} not found in reference {path!r}")
            group = handle[system_name]
            trajectory = trajectory_type.read_from_h5_group(group)
            return (
                trajectory,
                float(group.attrs["dt_fs"]),
                float(group.attrs["temperature_kelvin"]),
            )

    return read_reference_trajectory, metric_module.evaluate_md_system


def _read_prediction(path: Path):
    try:
        from matbench_discovery.trajectory import Trajectory
    except ImportError as exc:
        raise RuntimeError(
            "Public metric evaluation requires matbench-discovery on PYTHONPATH"
        ) from exc
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"Incomplete prediction trajectory: {path}")
        return Trajectory(
            atomic_numbers=handle["atomic_numbers"][:],
            positions=handle["positions"][:],
            cell=handle["cell"][:],
            pbc=handle["pbc"][:].astype(bool),
            energy=handle["energy"][:],
            forces=handle["forces"][:],
            md_step=handle["md_step"][:],
        )


def _evaluate_public_metrics(args, systems, result_rows, output_dir):
    if not args.statistics:
        disabled = {"status": "disabled"}
        with (output_dir / "matbench_esen_metrics.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["status"], delimiter="\t")
            writer.writeheader()
            writer.writerow(disabled)
        return disabled
    try:
        read_reference_trajectory, evaluate_md_system = _load_public_metric_functions(
            args.matbench_repo.resolve()
        )
    except (ImportError, RuntimeError, SyntaxError, NameError, TypeError) as exc:
        unavailable = {
            "status": "unavailable",
            "reason": (
                "matbench-discovery Python package is not importable; "
                f"set PYTHONPATH to {args.matbench_repo}"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
        with (output_dir / "matbench_esen_metrics.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["status", "reason", "error"], delimiter="\t"
            )
            writer.writeheader()
            writer.writerow(unavailable)
        return unavailable
    metrics_root = output_dir / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {"status": "computed"}
    all_metric_rows: list[dict[str, Any]] = []
    for backend in args.backend:
        rows = []
        for system in systems:
            success = next(
                (
                    row
                    for row in result_rows
                    if row.get("backend") == backend
                    and row.get("system") == system.name
                    and row.get("status") == "success"
                ),
                None,
            )
            if success is None:
                continue
            prediction_path = Path(success["trajectory_path"])
            try:
                reference, reference_dt, temperature = read_reference_trajectory(
                    str(args.reference_h5), system.name
                )
                prediction = _read_prediction(prediction_path)
                values = evaluate_md_system(
                    reference,
                    prediction,
                    ref_time_step_fs=reference_dt,
                    pred_time_step_fs=args.timestep_fs * args.record_interval,
                    progress_label=f"{backend}/{system.name}",
                )
            except Exception as exc:
                # A single malformed/short trajectory should not discard the
                # timing and trajectories of the remaining systems.
                rows.append(
                    {
                        "backend": backend,
                        "system": system.name,
                        "metric_status": "error",
                        "metric_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            rows.append(
                {
                    "backend": backend,
                    "system": system.name,
                    "temperature_kelvin": temperature,
                    "n_atoms": system.atomic_numbers.size,
                    "metric_status": "computed",
                    **values,
                }
            )
        csv_path = metrics_root / f"{backend}-per-system.csv"
        if rows:
            all_metric_rows.extend(rows)
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            public_keys = (
                "rdf_error",
                "adf_error",
                "vdos_error",
                "pressure_mae",
                "pressure_wasserstein",
                "pressure_error",
            )
            summary: dict[str, Any] = {
                "backend": backend,
                "n_completed_systems": sum(
                    row.get("metric_status") == "computed" for row in rows
                ),
                "n_requested_systems": len(systems),
                "n_metric_errors": sum(
                    row.get("metric_status") == "error" for row in rows
                ),
            }
            summary["n_systems"] = summary["n_completed_systems"]
            for key in public_keys:
                finite = [
                    float(row[key])
                    for row in rows
                    if row.get(key) is not None and np.isfinite(row[key])
                ]
                summary[key] = float(np.mean(finite)) if finite else None
            summary["pressure_status"] = (
                "available"
                if summary["pressure_mae"] is not None
                else "unavailable_stress_not_computed"
            )
            aggregate[backend] = summary
            _write_json(metrics_root / f"{backend}-aggregate.json", summary)
    if all_metric_rows:
        fields = list(dict.fromkeys(key for row in all_metric_rows for key in row))
        with (output_dir / "matbench_esen_metrics.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(all_metric_rows)
    return aggregate


def _load_published(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing_yaml", "path": str(path)}
    try:
        import yaml
    except ImportError:
        return {"status": "pyyaml_unavailable", "path": str(path)}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    md = payload.get("metrics", {}).get("md", {})
    return {
        "status": "published_reference",
        "model_name": payload.get("model_name"),
        "model_key": payload.get("model_key"),
        "model_params": payload.get("model_params"),
        "yaml_path": str(path.resolve()),
        **{
            key: md.get(key)
            for key in (
                "hardware",
                "run_time_sec",
                "energy_rmse",
                "force_rmse",
                "rdf_error",
                "adf_error",
                "vdos_error",
                "pressure_mae",
                "pressure_wasserstein",
                "pressure_error",
                "n_systems",
            )
        },
        "private_label_note": (
            "energy/force RMSE and pressure values are published references; "
            "the public HDF5 has no energy/force labels and this runner does not "
            "compute predicted stress."
        ),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# eSEN Matbench MD report",
        "",
        "The rollout uses the official NHC protocol. Public HDF5 data does not "
        "contain energy/force labels; predicted stress is not computed in this run.",
        "",
        "## Protocol",
        "",
        "```json",
        json.dumps(report["protocol"], indent=2),
        "```",
        "",
        "## Speed",
        "",
        "| system | backend | seconds/step | steps/second | speedup vs baseline | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["runs"]:
        speed = row.get("speedup_vs_baseline")
        lines.append(
            f"| {row.get('system')} | {row.get('backend')} | "
            f"{row.get('seconds_per_step', '')} | "
            f"{row.get('steps_per_second', '')} | "
            f"{'' if speed is None else f'{speed:.4f}'} | {row.get('status')} |"
        )
    lines += [
        "",
        "## Public metrics",
        "",
        "```json",
        json.dumps(report["public_metrics"], indent=2),
        "```",
        "",
        "## Published eSEN-30M-OAM reference",
        "",
        "```json",
        json.dumps(report["published_esen_30m_oam"], indent=2),
        "```",
        "",
        "Pressure and private energy/force metrics are intentionally marked "
        "unavailable until the independent stress/private-label pass is added.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(ROOT))
    # The Matbench repository is a source checkout whose import package lives
    # one level below the checkout root (matbench-discovery/matbench_discovery).
    sys.path.insert(0, str(args.matbench_repo.resolve()))

    import torch
    from fairchem.core.applications.esen_matbench import (
        initialize_matbench_atoms,
        read_matbench_systems,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the eSEN Matbench runner")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    systems = read_matbench_systems(args.reference_h5, args.systems)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}; use --overwrite"
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference_hash = sha256(args.reference_h5)
    checkpoint_hash = sha256(args.checkpoint)
    _write_json(
        args.output_dir / "run_metadata.json",
        {
            "schema": 1,
            "benchmark": "matbench-dynamat-v1.0",
            "backend": list(args.backend),
            "reference_h5": str(args.reference_h5.resolve()),
            "reference_sha256": reference_hash,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "protocol": {
                "integrator": "NoseHooverChainNVT",
                "tchain": 3,
                "tloop": 1,
                "steps": args.steps,
                "record_interval": args.record_interval,
                "timestep_fs": args.timestep_fs,
                "predicted_frame_dt_fs": args.timestep_fs * args.record_interval,
                "thermostat_time_fs": args.thermostat_time_fs,
                "seed": args.seed,
                "warmup_steps": 0,
            },
            "systems": [
                {
                    "name": system.name,
                    "atoms": int(system.atomic_numbers.size),
                    "temperature_kelvin": system.temperature_kelvin,
                    "reference_dt_fs": system.reference_dt_fs,
                    "reference_frames": system.reference_frames,
                    "reference_has_stress": system.reference_has_stress,
                }
                for system in systems
            ],
            "environment": {
                "host": platform.node(),
                "python": platform.python_version(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                **_runtime_metadata(torch),
            },
        },
    )
    run_rows: list[dict[str, Any]] = []
    status_path = args.output_dir / "run_status.tsv"
    for backend in args.backend:
        for system in systems:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            process_started = time.perf_counter()
            run_name = f"{backend}/{system.name}"
            trajectory_path = (
                args.output_dir / "trajectories" / backend / f"{system.name}.h5"
            )
            result_path = args.output_dir / "runs" / backend / f"{system.name}.json"
            recorder = None
            record: dict[str, Any] = {
                "backend": backend,
                "system": system.name,
                "atoms": int(system.atomic_numbers.size),
                "temperature_kelvin": system.temperature_kelvin,
                "reference_dt_fs": system.reference_dt_fs,
                "reference_frames": system.reference_frames,
                "reference_has_stress": system.reference_has_stress,
                "status": "error",
                "exit_code": 1,
            }
            try:
                if trajectory_path.is_file() and not args.overwrite:
                    raise FileExistsError(trajectory_path)
                recorder = _new_recorder(system, args)
                if backend == "baseline":
                    execution = _run_baseline(
                        system, args, args.checkpoint, recorder
                    )
                else:
                    execution = _run_gpu(
                        system, args, args.checkpoint, recorder, backend
                    )
                serialization_started = time.perf_counter()
                recorder.write(
                    trajectory_path,
                    atomic_numbers=system.atomic_numbers,
                    pbc=system.pbc,
                    temperature_kelvin=system.temperature_kelvin,
                    backend=backend,
                    metadata={
                        "seed": args.seed,
                        "checkpoint_sha256": checkpoint_hash,
                        "host": platform.node(),
                        **_runtime_metadata(torch),
                    },
                )
                serialization_elapsed = time.perf_counter() - serialization_started
                graph_stats = execution.pop("graph_stats", {})
                record.update(execution)
                record["graph_stats"] = graph_stats
                record["trajectory_path"] = str(trajectory_path.resolve())
                record["trajectory_frames"] = recorder.completed_frames
                record["rollout_seconds_per_step"] = (
                    execution["rollout_wall_time_s"] / args.steps
                )
                record["seconds_per_step"] = record["rollout_seconds_per_step"]
                record["steps_per_second"] = (
                    args.steps / execution["rollout_wall_time_s"]
                )
                record["serialization_wall_time_s"] = serialization_elapsed
                record["process_wall_time_s"] = time.perf_counter() - process_started
                record["status"] = "success"
                record["exit_code"] = 0
                if backend in {"opt2", "opt3"}:
                    record["graph_invariants_pass"] = bool(
                        graph_stats.get("cuda_graph_capture_count", 0) == 1
                        and graph_stats.get("cuda_graph_production_capture_count", 0)
                        == 0
                        and graph_stats.get("cuda_graph_production_replays", 0)
                        == args.steps + 1
                        and graph_stats.get("cuda_graph_capacity_misses", 0) == 0
                        and graph_stats.get("cuda_graph_hit_rate", 0.0) == 1.0
                    )
                    if not record["graph_invariants_pass"]:
                        if graph_stats.get("cuda_graph_capacity_misses", 0):
                            record["status"] = "capacity_overflow"
                            record["exit_code"] = 45
                        else:
                            record["status"] = "graph_invariant_failed"
                            record["exit_code"] = 43
            except BaseException as exc:
                if _is_oom(exc):
                    record["status"] = "oom"
                    record["exit_code"] = 42
                elif _is_capacity_overflow(exc):
                    record["status"] = "capacity_overflow"
                    record["exit_code"] = 45
                else:
                    record["status"] = "error"
                    record["exit_code"] = 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                if recorder is not None and recorder.completed_frames:
                    try:
                        recorder.write(
                            trajectory_path,
                            atomic_numbers=system.atomic_numbers,
                            pbc=system.pbc,
                            temperature_kelvin=system.temperature_kelvin,
                            backend=backend,
                            metadata={"seed": args.seed, "partial": True},
                            allow_partial=True,
                        )
                        record["trajectory_path"] = str(trajectory_path.resolve())
                        record["trajectory_frames"] = recorder.completed_frames
                    except Exception as write_exc:
                        record["partial_trajectory_error"] = str(write_exc)
                # Release the failed model/graph and cached CUDA blocks before
                # continuing with the next independent system.
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                record["process_wall_time_s"] = time.perf_counter() - process_started
            _write_json(result_path, record)
            _append_tsv(
                status_path,
                {
                    "backend": backend,
                    "system": system.name,
                    "temperature_K": system.temperature_kelvin,
                    "status": record["status"],
                    "exit_code": record["exit_code"],
                    "seconds_per_step": record.get("seconds_per_step"),
                    "steps_per_second": record.get("steps_per_second"),
                    "process_wall_time_s": record.get("process_wall_time_s"),
                },
            )
            run_rows.append(record)
            print(
                f"{record['status']}: {run_name} "
                f"s/step={record.get('seconds_per_step', '')}",
                flush=True,
            )

    baseline_times = {
        row["system"]: row.get("rollout_wall_time_s")
        for row in run_rows
        if row.get("backend") == "baseline" and row.get("status") == "success"
    }
    for row in run_rows:
        baseline_time = baseline_times.get(row["system"])
        current = row.get("rollout_wall_time_s")
        row["speedup_vs_baseline"] = (
            float(baseline_time) / float(current)
            if baseline_time is not None and current not in (None, 0)
            else None
        )

    public_metrics = _evaluate_public_metrics(
        args, systems, run_rows, args.output_dir
    )
    report = {
        "schema": 1,
        "benchmark": "matbench-dynamat-v1.0",
        "reference_h5": str(args.reference_h5.resolve()),
        "reference_sha256": reference_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "protocol": {
            "integrator": "NoseHooverChainNVT",
            "tchain": 3,
            "tloop": 1,
            "steps": args.steps,
            "record_interval": args.record_interval,
            "timestep_fs": args.timestep_fs,
            "predicted_frame_dt_fs": args.timestep_fs * args.record_interval,
            "thermostat_time_fs": args.thermostat_time_fs,
            "seed": args.seed,
            "warmup_steps": 0,
            "dtype": "FP64 MD state / FP32 eSEN model",
        },
        "systems": [
            {
                "name": system.name,
                "atoms": int(system.atomic_numbers.size),
                "temperature_kelvin": system.temperature_kelvin,
                "reference_dt_fs": system.reference_dt_fs,
                "reference_frames": system.reference_frames,
                "reference_has_stress": system.reference_has_stress,
            }
            for system in systems
        ],
        "runs": run_rows,
        "public_metrics": public_metrics,
        "public_metric_units": {
            "rdf_error": "%",
            "adf_error": "%",
            "vdos_error": "%",
            "pressure_mae": "GPa",
            "pressure_wasserstein": "GPa",
            "pressure_error": "%",
        },
        "published_esen_30m_oam": _load_published(args.published_yaml),
        "comparability": {
            "complete_matrix": all(
                row.get("status") == "success" for row in run_rows
            ),
            "public_metrics_comparable_only_when_all_17_systems_complete": True,
            "pressure_status": "unavailable_stress_not_computed",
            "private_energy_force_status": "unavailable_public_reference_has_no_labels",
            "speed_comparison_scope": "same GPU, same software, same NHC protocol",
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_mps_pipe_directory": os.environ.get(
                "CUDA_MPS_PIPE_DIRECTORY", ""
            ),
            **_runtime_metadata(torch),
        },
    }
    _write_json(args.output_dir / "matbench_esen_report.json", report)
    _write_markdown(report, args.output_dir / "matbench_esen_report.md")

    speedup_path = args.output_dir / "matbench_esen_speedups.tsv"
    speed_rows = []
    for row in run_rows:
        speed_rows.append(
            {
                "system": row["system"],
                "backend": row["backend"],
                "seconds_per_step": row.get("seconds_per_step"),
                "steps_per_second": row.get("steps_per_second"),
                "speedup_vs_baseline": row.get("speedup_vs_baseline"),
                "status": row["status"],
            }
        )
    if speed_rows:
        with speedup_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(speed_rows[0]), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(speed_rows)

    failures = [row for row in run_rows if row.get("status") != "success"]
    print(f"Results: {args.output_dir.resolve()}")
    print(f"completed={len(run_rows) - len(failures)} failures={len(failures)}")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
