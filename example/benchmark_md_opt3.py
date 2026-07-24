#!/usr/bin/env python3
"""Benchmark eSEN opt3 fixed-builder/model-CG and whole-step CUDA Graph.

The timed region contains the initial force evaluation and all requested NVT
steps.  Probing, graph capture, standard warmup, validation, hashing, and I/O
are excluded.  Validation and capacity failures are reported only after the
trajectory and its JSON result have been completed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from benchmark_md_gpu import (
    EXPECTED_ATOMS,
    append_tsv,
    capture_rng_state,
    git_commit,
    package_version,
    restore_rng_state,
    sha256,
)
from md_energy_reference import (
    REQUIRED_SEED,
    checkpoint_energy_fields,
    compare_checkpoint_energies,
    load_baseline_reference,
    reached_energy_checkpoints,
    seed_everything,
    validate_reference_metadata,
)


REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("fixed-builder-model-cg", "whole-step-cg"),
        required=True,
    )
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO / "esen_30m_oam.pt"
    )
    parser.add_argument("--system", required=True, choices=EXPECTED_ATOMS)
    parser.add_argument("--output-dir", type=Path, default=REPO / "example/md_out")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--taut", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=REQUIRED_SEED)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--baseline-result", type=Path, default=None)
    parser.add_argument("--missing-baseline-reference", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=50)
    parser.add_argument("--neighbor-margin", type=float, default=0.10)
    parser.add_argument("--neighbor-slot-step", type=int, default=8)
    parser.add_argument("--dummy-atoms", type=int, default=32)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--max-neighbors", type=int, default=300)
    parser.add_argument("--degeneracy-tolerance", type=float, default=0.01)
    parser.add_argument("--energy-per-atom-atol", type=float, default=1e-5)
    parser.add_argument("--force-max-atol", type=float, default=2e-4)
    parser.add_argument("--replay-energy-atol", type=float, default=0.0)
    parser.add_argument("--replay-force-atol", type=float, default=1e-6)
    parser.add_argument("--md-dtype", choices=("float64",), default="float64")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0 or args.probe_steps < 0:
        parser.error("steps must be positive and warmup/probe steps non-negative")
    if args.seed != REQUIRED_SEED:
        parser.error(f"--seed must be {REQUIRED_SEED}")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.timestep <= 0 or args.temperature <= 0 or args.taut <= 0:
        parser.error("NVT parameters must be positive")
    if args.neighbor_margin < 0 or args.neighbor_slot_step < 1:
        parser.error("invalid neighbor capacity parameters")
    if args.dummy_atoms < 1 or args.capture_warmup < 0:
        parser.error("invalid CUDA Graph setup parameters")
    if args.max_neighbors < 1 or args.degeneracy_tolerance < 0:
        parser.error("invalid neighbor pruning parameters")
    if args.energy_per_atom_atol < 0 or args.force_max_atol < 0:
        parser.error("engineering tolerances must be non-negative")
    if args.baseline_result is not None and args.missing_baseline_reference:
        parser.error("baseline path and missing-reference flag are exclusive")
    return args


def _device_memory_used(torch_module, device) -> int | None:
    try:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError):
        return None
    return int(total_bytes - free_bytes)


def _engineering_energy_validation(
    checkpoint_energies: dict[int, float],
    baseline: dict[str, object] | None,
    *,
    atoms: int,
    atol_per_atom: float,
) -> tuple[dict[str, object], bool | None]:
    fields: dict[str, object] = {
        "engineering_energy_per_atom_atol_eV": atol_per_atom,
    }
    if baseline is None:
        for step in (1, 50, 100, 1000):
            fields[f"energy_abs_error_step_{step}_eV_per_atom"] = None
        fields["engineering_energy_validation_pass"] = None
        fields["engineering_energy_validation_status"] = "missing_reference"
        return fields, None

    passed = True
    for step in (1, 50, 100, 1000):
        current = checkpoint_energies.get(step)
        reference = baseline.get(f"energy_step_{step}_eV")
        error = (
            None
            if current is None or reference is None
            else abs(float(current) - float(reference)) / atoms
        )
        fields[f"energy_abs_error_step_{step}_eV_per_atom"] = error
        if step in (1, 50) and current is not None:
            passed &= error is not None and error < atol_per_atom
    fields["engineering_energy_validation_pass"] = passed
    fields["engineering_energy_validation_status"] = (
        "passed" if passed else "failed"
    )
    return fields, passed


def main() -> int:
    process_start = time.perf_counter()
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(REQUIRED_SEED):
        raise RuntimeError(f"Launch with PYTHONHASHSEED={REQUIRED_SEED}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(REPO / "src"))

    import torch
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from fairchem.core.applications.esen_fixed_neighbor import (
        maximum_neighbors_in_graph,
        neighbor_capacity_from_probe,
    )
    from fairchem.core.applications.esen_gpu_md import (
        ESENEnergyForceEvaluator,
        GPUIntegrator,
        GPUMDState,
        GPUResidentMD,
    )
    from fairchem.core.applications.esen_whole_step_cuda_graph import (
        ESENFixedBuilderModelCUDAGraphEvaluator,
        ESENWholeStepCUDAGraphMD,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.structure.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("structure or checkpoint file is missing")
    seed_everything(torch, args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    atoms = read(args.structure)
    if len(atoms) != EXPECTED_ATOMS[args.system]:
        raise ValueError(
            f"{args.system} expects {EXPECTED_ATOMS[args.system]} atoms, "
            f"got {len(atoms)}"
        )
    if not np.all(atoms.pbc):
        raise ValueError("Periodic boundary conditions are required")

    checkpoint_hash = sha256(args.checkpoint)
    structure_hash = sha256(args.structure)
    baseline_reference = None
    if args.baseline_result is not None:
        baseline_reference = load_baseline_reference(args.baseline_result)
        validate_reference_metadata(
            baseline_reference,
            {
                "system": args.system,
                "atoms": len(atoms),
                "temperature_K": args.temperature,
                "timestep_fs": args.timestep,
                "taut_fs": args.taut,
                "seed": args.seed,
                "repeat": args.repeat,
                "checkpoint_sha256": checkpoint_hash,
                "structure_sha256": structure_hash,
            },
        )

    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=args.temperature, force_temp=True, rng=rng
    )
    Stationary(atoms)
    device = torch.device("cuda:0")
    state = GPUMDState(
        positions=torch.as_tensor(
            atoms.get_positions(), dtype=torch.float64, device=device
        ).clone(),
        momenta=torch.as_tensor(
            atoms.get_momenta(), dtype=torch.float64, device=device
        ).clone(),
    )
    masses = torch.as_tensor(
        atoms.get_masses(), dtype=torch.float64, device=device
    )
    evaluator = ESENEnergyForceEvaluator(
        atoms,
        args.checkpoint,
        device=device,
        seed=args.seed,
        disable_amp=True,
    )
    integrator = GPUIntegrator(
        masses,
        timestep_fs=args.timestep,
        temperature_K=args.temperature,
        taut_fs=args.taut,
        fix_com=True,
        degrees_of_freedom=atoms.get_number_of_degrees_of_freedom(),
    )
    eager_md = GPUResidentMD(state, evaluator, integrator)
    initial_state = state.clone()

    setup_start = time.perf_counter()
    probe_rng = capture_rng_state(torch)
    probe_degrees = []
    initial_graph = evaluator.build_neighbor_graph(state.positions)
    probe_degrees.append(
        maximum_neighbors_in_graph(initial_graph["edge_index"], len(atoms))
    )
    for _ in range(args.probe_steps):
        eager_md.run(1)
        graph = evaluator.build_neighbor_graph(state.positions)
        probe_degrees.append(
            maximum_neighbors_in_graph(graph["edge_index"], len(atoms))
        )
    torch.cuda.synchronize()
    probe_max_neighbors = max(probe_degrees)
    neighbor_capacity = neighbor_capacity_from_probe(
        probe_max_neighbors,
        margin=args.neighbor_margin,
        slot_step=args.neighbor_slot_step,
    )
    del initial_graph
    if args.probe_steps:
        del graph
    state.restore_(initial_state)
    eager_md.nsteps = 0
    restore_rng_state(torch, probe_rng)
    torch.cuda.empty_cache()

    eager_initial_forces, eager_initial_energy = evaluator(state.positions)
    eager_initial_forces = eager_initial_forces.detach().clone()
    eager_initial_energy = eager_initial_energy.detach().clone()
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    device_used_before_capture = _device_memory_used(torch, device)
    fixed_evaluator = None
    whole_md = None
    if args.backend == "fixed-builder-model-cg":
        fixed_evaluator = ESENFixedBuilderModelCUDAGraphEvaluator(
            evaluator,
            neighbors_per_atom=neighbor_capacity,
            dummy_atoms=args.dummy_atoms,
            capture_warmup=args.capture_warmup,
            max_neighbors=args.max_neighbors,
            degeneracy_tolerance=args.degeneracy_tolerance,
            replay_energy_atol=args.replay_energy_atol,
            replay_force_atol=args.replay_force_atol,
        )
        torch.cuda.empty_cache()
        fixed_evaluator.capture(state.positions)
        dynamics = GPUResidentMD(state, fixed_evaluator, integrator)
    else:
        whole_md = ESENWholeStepCUDAGraphMD(
            state,
            evaluator,
            integrator,
            neighbors_per_atom=neighbor_capacity,
            dummy_atoms=args.dummy_atoms,
            capture_warmup=args.capture_warmup,
            max_neighbors=args.max_neighbors,
            degeneracy_tolerance=args.degeneracy_tolerance,
        )
        torch.cuda.empty_cache()
        whole_md.capture(initial_state)
        dynamics = None
    torch.cuda.synchronize()
    device_used_after_capture = _device_memory_used(torch, device)
    setup_wall_time = time.perf_counter() - setup_start

    # Standard three-step warmup is trajectory-neutral and excluded from timing.
    if args.backend == "fixed-builder-model-cg":
        assert dynamics is not None
        dynamics.run(args.warmup_steps)
        torch.cuda.synchronize()
        state.restore_(initial_state)
        dynamics.nsteps = 0
        assert fixed_evaluator is not None
        fixed_evaluator.reset_production_stats()
        initial_forces_device, initial_energy_device = dynamics.evaluate()
    else:
        assert whole_md is not None
        whole_md.reset_production(initial_state)
        whole_md.evaluate_initial()
        for _ in range(args.warmup_steps):
            whole_md.step()
        torch.cuda.synchronize()
        whole_md.reset_production(initial_state)
        initial_forces_device, initial_energy_device = whole_md.evaluate_initial()
    torch.cuda.synchronize()

    initial_energy = float(initial_energy_device.item())
    initial_force_error = float(
        (initial_forces_device - eager_initial_forces).abs().max().item()
    )
    initial_energy_error = abs(
        initial_energy - float(eager_initial_energy.item())
    )
    initial_energy_error_per_atom = initial_energy_error / len(atoms)
    force_validation_pass = initial_force_error < args.force_max_atol

    # Reset after validation.  The timed region starts from the original state.
    if args.backend == "fixed-builder-model-cg":
        assert dynamics is not None and fixed_evaluator is not None
        state.restore_(initial_state)
        dynamics.nsteps = 0
        fixed_evaluator.reset_production_stats()
    else:
        assert whole_md is not None
        whole_md.reset_production(initial_state)

    checkpoint_tensors: dict[int, torch.Tensor] = {}
    checkpoints = set(reached_energy_checkpoints(args.steps))
    torch.cuda.reset_peak_memory_stats()
    device_used_before_timing = _device_memory_used(torch, device)
    torch.cuda.synchronize()
    timed_start = time.perf_counter()
    if args.backend == "fixed-builder-model-cg":
        assert dynamics is not None
        completed = 0
        for checkpoint in reached_energy_checkpoints(args.steps):
            dynamics.run(checkpoint - completed)
            assert state.potential_energy is not None
            checkpoint_tensors[checkpoint] = state.potential_energy.detach().clone()
            completed = checkpoint
        if completed < args.steps:
            dynamics.run(args.steps - completed)
        final_state = state
    else:
        assert whole_md is not None
        whole_md.evaluate_initial()
        for step in range(1, args.steps + 1):
            _, energy = whole_md.step()
            if step in checkpoints:
                checkpoint_tensors[step] = energy.detach().clone()
        final_state = whole_md.state_view()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - timed_start
    device_used_after_timing = _device_memory_used(torch, device)
    checkpoint_energies = {
        step: float(value.item()) for step, value in checkpoint_tensors.items()
    }

    if final_state.forces is None or final_state.potential_energy is None:
        raise RuntimeError("MD completed without forces or energy")
    finite = (
        torch.isfinite(final_state.positions).all()
        & torch.isfinite(final_state.momenta).all()
        & torch.isfinite(final_state.forces).all()
        & torch.isfinite(final_state.potential_energy).all()
    )
    if not bool(finite.item()):
        raise FloatingPointError("Final MD state contains NaN or Inf")

    if fixed_evaluator is not None:
        graph_stats = fixed_evaluator.stats()
    else:
        assert whole_md is not None
        graph_stats = whole_md.stats()
    capacity_overflow = int(graph_stats["cuda_graph_capacity_misses"]) > 0
    expected_replays = args.steps + 1
    graph_invariants_pass = (
        graph_stats["cuda_graph_capture_count"] == 1
        and graph_stats["cuda_graph_production_capture_count"] == 0
        and graph_stats["cuda_graph_production_replays"] == expected_replays
        and graph_stats["cuda_graph_hit_rate"] == 1.0
        and graph_stats.get("cuda_graph_replay_stability_pass", True)
    )

    legacy_fields: dict[str, object] = {}
    legacy_pass: bool | None = None
    if baseline_reference is not None:
        legacy_fields, legacy_pass = compare_checkpoint_energies(
            checkpoint_energies, baseline_reference
        )
    engineering_fields, energy_engineering_pass = (
        _engineering_energy_validation(
            checkpoint_energies,
            baseline_reference,
            atoms=len(atoms),
            atol_per_atom=args.energy_per_atom_atol,
        )
    )
    non_energy_validation_pass = (
        force_validation_pass
        and graph_invariants_pass
        and not capacity_overflow
    )
    engineering_pass: bool | None
    if energy_engineering_pass is None and non_energy_validation_pass:
        engineering_pass = None
    else:
        engineering_pass = bool(
            non_energy_validation_pass
            and energy_engineering_pass is not False
        )
    numerical_failures = []
    if not force_validation_pass:
        numerical_failures.append(
            "initial force max error exceeds engineering tolerance"
        )
    if energy_engineering_pass is False:
        numerical_failures.append(
            "1-step or 50-step per-atom energy error exceeds tolerance"
        )
    if not graph_invariants_pass:
        numerical_failures.append("CUDA Graph replay invariants failed")

    backend_name = (
        "esen_gpu_resident_fixed_builder_model_cg"
        if args.backend == "fixed-builder-model-cg"
        else "esen_gpu_resident_whole_step_cg"
    )
    suffix = (
        "esen_fixed_builder_model_cg"
        if args.backend == "fixed-builder-model-cg"
        else "esen_whole_step_cg"
    )
    run_name = args.run_name or (
        f"{args.system}_{args.temperature:g}K_{args.steps}step_{suffix}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "backend": backend_name,
        "run_name": run_name,
        "system": args.system,
        "atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "steps": args.steps,
        "force_evaluations_timed": args.steps + 1,
        "warmup_steps": args.warmup_steps,
        "temperature_K": args.temperature,
        "timestep_fs": args.timestep,
        "taut_fs": args.taut,
        "seed": args.seed,
        "repeat": args.repeat,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "kernel_fusion": False,
        "cuda_graph": True,
        "cuda_graph_scope": (
            "model_only_fixed_builder"
            if args.backend == "fixed-builder-model-cg"
            else "whole_nvt_step"
        ),
        "cuda_graph_neighbor_build_outside": (
            args.backend == "fixed-builder-model-cg"
        ),
        "gpu_resident_md": True,
        "neighbor_builder": "fixed_shape_radius_graph_pbc",
        "md_state_dtype": "float64",
        "model_dtype": str(evaluator.model_dtype).removeprefix("torch."),
        "probe_steps": args.probe_steps,
        "probe_max_neighbors_per_atom": probe_max_neighbors,
        "neighbor_capacity_per_atom": neighbor_capacity,
        "neighbor_capacity_margin": args.neighbor_margin,
        "neighbor_slot_step": args.neighbor_slot_step,
        "max_neighbors": args.max_neighbors,
        "degeneracy_tolerance": args.degeneracy_tolerance,
        "dummy_atoms": args.dummy_atoms,
        "capture_warmup": args.capture_warmup,
        "setup_wall_time_s": setup_wall_time,
        "cg_setup_wall_time_s": setup_wall_time,
        "md_wall_time_s": elapsed,
        "elapsed_s": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "milliseconds_per_step": 1000.0 * elapsed / args.steps,
        "steps_per_second": args.steps / elapsed,
        "process_wall_time_s": time.perf_counter() - process_start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "device_used_before_timing_gib": (
            None
            if device_used_before_timing is None
            else device_used_before_timing / 1024**3
        ),
        "device_used_after_timing_gib": (
            None
            if device_used_after_timing is None
            else device_used_after_timing / 1024**3
        ),
        "device_used_before_capture_gib": (
            None
            if device_used_before_capture is None
            else device_used_before_capture / 1024**3
        ),
        "device_used_after_capture_gib": (
            None
            if device_used_after_capture is None
            else device_used_after_capture / 1024**3
        ),
        "capture_total_device_used_delta_gib": (
            None
            if (
                device_used_before_capture is None
                or device_used_after_capture is None
            )
            else (
                device_used_after_capture - device_used_before_capture
            )
            / 1024**3
        ),
        "initial_energy_eV": initial_energy,
        "initial_eager_energy_abs_error_eV": initial_energy_error,
        "initial_eager_energy_abs_error_eV_per_atom": (
            initial_energy_error_per_atom
        ),
        "initial_eager_force_max_abs_error_eV_per_A": initial_force_error,
        "engineering_force_max_atol_eV_per_A": args.force_max_atol,
        "engineering_force_validation_pass": force_validation_pass,
        "legacy_energy_validation_pass": legacy_pass,
        "engineering_validation_pass": engineering_pass,
        "engineering_validation_status": (
            "missing_reference"
            if engineering_pass is None
            else ("passed" if engineering_pass else "failed")
        ),
        "energy_validation_pass": legacy_pass,
        "energy_validation_status": (
            "missing_reference"
            if baseline_reference is None
            else ("passed" if legacy_pass else "failed")
        ),
        "numerical_validation_pass": engineering_pass,
        "numerical_validation_status": (
            "missing_reference"
            if engineering_pass is None
            else ("passed" if engineering_pass else "failed")
        ),
        "numerical_validation_failures": numerical_failures,
        "capacity_overflow": capacity_overflow,
        "graph_invariants_pass": graph_invariants_pass,
        "final_energy_eV": float(final_state.potential_energy.item()),
        "final_max_force_eV_per_A": float(final_state.forces.abs().max().item()),
        "final_temperature_K": float(
            integrator.temperature(final_state.momenta).item()
        ),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_capability": ".".join(
            map(str, torch.cuda.get_device_capability(0))
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "fairchem_core_version": package_version("fairchem-core"),
        "ase_version": package_version("ase"),
        "repo_commit": git_commit(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "structure": str(args.structure.resolve()),
        "structure_sha256": structure_hash,
        "baseline_result": (
            "" if args.baseline_result is None else str(args.baseline_result.resolve())
        ),
        "baseline_result_sha256": (
            ""
            if args.baseline_result is None
            else sha256(args.baseline_result)
        ),
        "baseline_reference_status": (
            "available"
            if baseline_reference is not None
            else (
                "missing"
                if args.missing_baseline_reference
                else "not_requested"
            )
        ),
    }
    record.update(graph_stats)
    record.update(checkpoint_energy_fields(checkpoint_energies))
    record.update(legacy_fields)
    record.update(engineering_fields)
    json_path = args.output_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_tsv(args.output_dir / "summary.tsv", record)
    print(json.dumps(record, indent=2))
    print(f"Result: {json_path}")
    print(f"Summary: {args.output_dir / 'summary.tsv'}")

    if capacity_overflow:
        print(
            "BENCHMARK_STATUS=capacity_overflow: MD completed with truncated "
            "fixed neighbor slots",
            file=sys.stderr,
        )
        return 45
    if engineering_pass is False:
        print(
            "BENCHMARK_STATUS=validation_failed: MD completed; "
            + " | ".join(numerical_failures),
            file=sys.stderr,
        )
        return 43
    return 0


def entrypoint() -> int:
    try:
        return main()
    except BaseException as exc:
        message = str(exc).lower()
        if exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in message:
            print(f"BENCHMARK_STATUS=oom: {exc}", file=sys.stderr)
            return 42
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())
