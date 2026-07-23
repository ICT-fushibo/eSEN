#!/usr/bin/env python3
"""Benchmark the opt2 fixed-capacity eSEN path without CUDA Graph capture.

This is an ablation control, not a replacement for opt1.  It retains every
opt2 static-shape adaptation while executing model forward, conservative-force
autograd, and denormalization eagerly on every force evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO / "esen_30m_oam.pt"
    )
    parser.add_argument("--system", required=True, choices=EXPECTED_ATOMS)
    parser.add_argument("--output-dir", type=Path, default=REPO / "example/md_out")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--taut", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=REQUIRED_SEED)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--baseline-result", type=Path, default=None)
    parser.add_argument("--missing-baseline-reference", action="store_true")
    parser.add_argument(
        "--md-dtype", choices=("float64", "float32"), default="float64"
    )
    parser.add_argument("--validate-official", action="store_true")
    parser.add_argument("--energy-atol", type=float, default=1e-4)
    parser.add_argument("--force-atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--probe-steps", type=int, default=50)
    parser.add_argument("--capacity-margin", type=float, default=0.10)
    parser.add_argument("--edge-step", type=int, default=256)
    parser.add_argument("--dummy-atoms", type=int, default=32)
    parser.add_argument("--setup-warmup", type=int, default=3)
    parser.add_argument("--repeat-energy-atol", type=float, default=0.0)
    parser.add_argument("--repeat-force-atol", type=float, default=1e-6)
    parser.add_argument("--original-eager-energy-atol", type=float, default=1e-8)
    parser.add_argument("--original-eager-force-atol", type=float, default=2e-4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.warmup_steps < 0 or args.probe_steps < 0 or args.setup_warmup < 0:
        parser.error("warmup and probe counts must be non-negative")
    if args.timestep <= 0 or args.temperature <= 0 or args.taut <= 0:
        parser.error("--timestep, --temperature, and --taut must be positive")
    if args.repeat < 1 or args.edge_step < 1 or args.dummy_atoms < 1:
        parser.error("repeat, edge-step, and dummy-atoms must be positive")
    if args.capacity_margin < 0:
        parser.error("--capacity-margin must be non-negative")
    if args.seed != REQUIRED_SEED:
        parser.error(f"--seed must be {REQUIRED_SEED}")
    if args.baseline_result is not None and args.missing_baseline_reference:
        parser.error(
            "--baseline-result and --missing-baseline-reference are mutually exclusive"
        )
    tolerances = (
        args.repeat_energy_atol,
        args.repeat_force_atol,
        args.original_eager_energy_atol,
        args.original_eager_force_atol,
        args.energy_atol,
        args.force_atol,
        args.rtol,
    )
    if any(value < 0 for value in tolerances):
        parser.error("validation tolerances must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(REQUIRED_SEED):
        raise RuntimeError(
            f"Launch the benchmark with PYTHONHASHSEED={REQUIRED_SEED}"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(REPO / "src"))

    import torch
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from fairchem.core.applications.esen_cuda_graph import edge_capacity_from_probe
    from fairchem.core.applications.esen_gpu_md import (
        ESENEnergyForceEvaluator,
        GPUIntegrator,
        GPUMDState,
        GPUResidentMD,
    )
    from fairchem.core.applications.esen_opt2_static_eager import (
        ESENOpt2StaticEagerEvaluator,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.structure.is_file():
        raise FileNotFoundError(args.structure)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    seed_everything(torch, args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    atoms = read(args.structure)
    expected_atoms = EXPECTED_ATOMS[args.system]
    if len(atoms) != expected_atoms:
        raise ValueError(
            f"{args.system} must contain {expected_atoms} atoms, got {len(atoms)}"
        )
    if not np.all(atoms.pbc):
        raise ValueError(f"Periodic boundary conditions are required: {atoms.pbc}")
    if not math.isfinite(atoms.get_volume()) or atoms.get_volume() <= 0:
        raise ValueError(f"Invalid cell volume: {atoms.get_volume()}")

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
    md_dtype = torch.float64 if args.md_dtype == "float64" else torch.float32
    state = GPUMDState(
        positions=torch.as_tensor(
            atoms.get_positions(), dtype=md_dtype, device=device
        ).clone(),
        momenta=torch.as_tensor(
            atoms.get_momenta(), dtype=md_dtype, device=device
        ).clone(),
    )
    masses = torch.as_tensor(atoms.get_masses(), dtype=md_dtype, device=device)

    original_evaluator = ESENEnergyForceEvaluator(
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
    dynamics = GPUResidentMD(state, original_evaluator, integrator)
    initial_state = state.clone()
    numerical_validation_failures: list[str] = []

    setup_start = time.perf_counter()
    rng_state = capture_rng_state(torch)
    probe_edge_counts = [
        int(
            original_evaluator.build_neighbor_graph(state.positions)[
                "edge_index"
            ].shape[1]
        )
    ]
    for _ in range(args.probe_steps):
        dynamics.run(1)
        probe_edge_counts.append(
            int(
                original_evaluator.build_neighbor_graph(state.positions)[
                    "edge_index"
                ].shape[1]
            )
        )
    torch.cuda.synchronize()
    probe_max_edges = max(probe_edge_counts)
    edge_capacity = edge_capacity_from_probe(
        probe_max_edges, margin=args.capacity_margin, edge_step=args.edge_step
    )

    state.restore_(initial_state)
    dynamics.nsteps = 0
    restore_rng_state(torch, rng_state)
    original_forces, original_energy = original_evaluator(state.positions)
    original_forces = original_forces.detach().clone()
    original_energy = original_energy.detach().clone()
    torch.cuda.synchronize()

    static_evaluator = ESENOpt2StaticEagerEvaluator(
        original_evaluator,
        edge_capacity=edge_capacity,
        dummy_atoms=args.dummy_atoms,
        setup_warmup=args.setup_warmup,
        replay_energy_atol=args.repeat_energy_atol,
        replay_force_atol=args.repeat_force_atol,
    )
    static_evaluator.prepare(state.positions)
    if not static_evaluator.replay_stability_passed:
        numerical_validation_failures.append(
            "Static eager identical-input stability failed: "
            f"energy_error={static_evaluator.replay_stability_energy_abs_error}, "
            f"energy_atol={static_evaluator.replay_energy_atol}, "
            f"force_error={static_evaluator.replay_stability_force_max_abs_error}, "
            f"force_atol={static_evaluator.replay_force_atol}"
        )
    dynamics.evaluator = static_evaluator
    setup_wall_time_s = time.perf_counter() - setup_start

    if args.warmup_steps:
        dynamics.run(args.warmup_steps)
        torch.cuda.synchronize()
    state.restore_(initial_state)
    dynamics.nsteps = 0

    initial_forces_device, initial_energy_device = dynamics.evaluate()
    torch.cuda.synchronize()
    initial_forces = initial_forces_device.detach().cpu().numpy()
    initial_energy = float(initial_energy_device.item())
    if not np.isfinite(initial_forces).all() or not math.isfinite(initial_energy):
        raise FloatingPointError("Initial static-eager prediction contains NaN/Inf")

    initial_energy_error = abs(initial_energy - float(original_energy.item()))
    initial_force_error = float(
        (initial_forces_device - original_forces).abs().max().item()
    )
    initial_validation_pass = bool(
        initial_energy_error <= args.original_eager_energy_atol
        and initial_force_error <= args.original_eager_force_atol
    )
    if initial_energy_error > args.original_eager_energy_atol:
        numerical_validation_failures.append(
            "Opt2 static eager and original opt1-path eager initial energies differ: "
            f"error={initial_energy_error}, "
            f"atol={args.original_eager_energy_atol}"
        )
    if initial_force_error > args.original_eager_force_atol:
        numerical_validation_failures.append(
            "Opt2 static eager and original opt1-path eager initial forces differ: "
            f"error={initial_force_error}, "
            f"atol={args.original_eager_force_atol}"
        )

    official_energy_error = None
    official_force_error = None
    official_energy_pass = None
    official_force_pass = None
    if args.validate_official:
        original_otf_graph = original_evaluator.model.backbone.otf_graph
        original_evaluator.model.backbone.otf_graph = True
        try:
            original_evaluator.calculator.reset()
            atoms.calc = original_evaluator.calculator
            official_forces = atoms.get_forces()
            official_energy = float(atoms.get_potential_energy())
        finally:
            original_evaluator.model.backbone.otf_graph = original_otf_graph
        official_force_error = float(
            np.max(np.abs(initial_forces - official_forces))
        )
        official_energy_error = abs(initial_energy - official_energy)
        official_force_pass = bool(
            np.allclose(
                initial_forces,
                official_forces,
                rtol=args.rtol,
                atol=args.force_atol,
            )
        )
        official_energy_pass = bool(
            np.allclose(
                initial_energy,
                official_energy,
                rtol=args.rtol,
                atol=args.energy_atol,
            )
        )
        if not official_force_pass:
            numerical_validation_failures.append(
                "Static eager and OCPCalculator initial forces differ: "
                f"max_abs_error={official_force_error}"
            )
        if not official_energy_pass:
            numerical_validation_failures.append(
                "Static eager and OCPCalculator initial energies differ: "
                f"abs_error={official_energy_error}"
            )

    state.restore_(initial_state)
    dynamics.nsteps = 0
    del initial_state
    static_evaluator.reset_production_stats()

    checkpoint_energy_tensors: dict[int, torch.Tensor] = {}
    completed_steps = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for checkpoint_step in reached_energy_checkpoints(args.steps):
        dynamics.run(checkpoint_step - completed_steps)
        if state.potential_energy is None:
            raise RuntimeError(f"No potential energy at step {checkpoint_step}")
        checkpoint_energy_tensors[checkpoint_step] = (
            state.potential_energy.detach().clone()
        )
        completed_steps = checkpoint_step
    if completed_steps < args.steps:
        dynamics.run(args.steps - completed_steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checkpoint_energies = {
        step: float(value.item()) for step, value in checkpoint_energy_tensors.items()
    }

    if state.forces is None or state.potential_energy is None:
        raise RuntimeError("Static eager MD completed without force or energy")
    finite = (
        torch.isfinite(state.positions).all()
        & torch.isfinite(state.momenta).all()
        & torch.isfinite(state.forces).all()
        & torch.isfinite(state.potential_energy).all()
    )
    if not bool(finite.item()):
        raise FloatingPointError("Final static-eager MD state contains NaN/Inf")

    stats = static_evaluator.stats()
    expected_calls = args.steps + 1
    if stats["static_eager_production_calls"] != expected_calls:
        raise RuntimeError(
            "Unexpected static-eager call count: "
            f"expected={expected_calls}, "
            f"actual={stats['static_eager_production_calls']}"
        )
    if stats["static_eager_capacity_misses"] != 0:
        raise RuntimeError("Successful static-eager run recorded a capacity miss")

    final_energy = float(state.potential_energy.item())
    final_max_force = float(state.forces.abs().max().item())
    final_temperature = float(integrator.temperature(state.momenta).item())
    run_name = args.run_name or (
        f"{args.system}_{args.temperature:g}K_{args.steps}step_"
        "esen_opt2_static_eager"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "backend": "esen_gpu_resident_opt2_static_eager",
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
        "outputs": "energy,forces",
        "parameters_frozen": True,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "cuda_graph": False,
        "cuda_graph_scope": "none",
        "kernel_fusion": False,
        "gpu_resident_md": True,
        "opt2_capture_compatible_control": True,
        "fixed_edge_capacity": True,
        "dummy_padding": True,
        "fixed_rotation_reference": True,
        "neighbor_builder": "fairchem_radius_graph_pbc",
        "neighbor_build_outside_model": True,
        "md_state_dtype": str(state.positions.dtype).removeprefix("torch."),
        "model_dtype": str(original_evaluator.model_dtype).removeprefix("torch."),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "md_wall_time_s": elapsed,
        "elapsed_s": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "milliseconds_per_step": 1000.0 * elapsed / args.steps,
        "steps_per_second": args.steps / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "initial_max_force_eV_per_A": float(np.abs(initial_forces).max()),
        "final_max_force_eV_per_A": final_max_force,
        "final_temperature_K": final_temperature,
        "cell_volume_A3": float(atoms.get_volume()),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "fairchem_core_version": package_version("fairchem-core"),
        "ase_version": package_version("ase"),
        "repo_commit": git_commit(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "structure": str(args.structure.resolve()),
        "structure_sha256": structure_hash,
        "official_validation": args.validate_official,
        "baseline_result": (
            "" if args.baseline_result is None else str(args.baseline_result.resolve())
        ),
        "baseline_result_sha256": (
            "" if args.baseline_result is None else sha256(args.baseline_result)
        ),
        "probe_steps": args.probe_steps,
        "probe_max_edges": probe_max_edges,
        "capacity_margin": args.capacity_margin,
        "edge_step": args.edge_step,
        "edge_capacity": edge_capacity,
        "setup_wall_time_s": setup_wall_time_s,
        "original_eager_initial_energy_abs_error_eV": initial_energy_error,
        "original_eager_initial_force_max_abs_error_eV_per_A": initial_force_error,
        "original_eager_initial_validation_pass": initial_validation_pass,
        "original_eager_energy_atol_eV": args.original_eager_energy_atol,
        "original_eager_force_atol_eV_per_A": args.original_eager_force_atol,
        "official_initial_energy_abs_error_eV": official_energy_error,
        "official_initial_force_max_abs_error_eV_per_A": official_force_error,
        "official_initial_energy_validation_pass": official_energy_pass,
        "official_initial_force_validation_pass": official_force_pass,
    }
    record.update(stats)
    record.update(checkpoint_energy_fields(checkpoint_energies))

    validation_pass: bool | None = None
    if baseline_reference is None:
        record["energy_validation_status"] = (
            "missing_reference"
            if args.missing_baseline_reference
            else "not_requested"
        )
        record["energy_validation_pass"] = None
    else:
        validation_fields, validation_pass = compare_checkpoint_energies(
            checkpoint_energies, baseline_reference
        )
        record.update(validation_fields)
        record["energy_validation_status"] = (
            "passed" if validation_pass else "failed"
        )
        if not validation_pass:
            numerical_validation_failures.append(
                "Trajectory energy validation failed at step 1 or 50"
            )

    record["numerical_validation_pass"] = not numerical_validation_failures
    record["numerical_validation_status"] = (
        "passed" if not numerical_validation_failures else "failed"
    )
    record["numerical_validation_failures"] = numerical_validation_failures

    json_path = args.output_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_tsv(args.output_dir / "summary.tsv", record)
    print(json.dumps(record, indent=2))
    print(f"Result: {json_path}")
    print(f"Summary: {args.output_dir / 'summary.tsv'}")
    if numerical_validation_failures:
        print(
            "BENCHMARK_STATUS=validation_failed: MD completed; "
            + " | ".join(numerical_validation_failures),
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
        if exc.__class__.__name__ == "CUDAGraphCapacityError":
            print(f"BENCHMARK_STATUS=capacity_overflow: {exc}", file=sys.stderr)
            return 45
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())
