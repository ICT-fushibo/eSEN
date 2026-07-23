#!/usr/bin/env python3
"""Benchmark GPU-resident eager or model-CUDA-Graph eSEN molecular dynamics.

Neither backend uses ``torch.compile``, AMP, TF32, or custom kernel fusion.
The timed region includes the initial force evaluation and the requested NVT
MD steps, but excludes model loading, probing, capture, warm-up, validation,
hashing, and I/O.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

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
EXPECTED_ATOMS = {
    "Cu32": 32,
    "Cu64": 64,
    "Cu192": 192,
    "Cu512": 512,
    "Cu1024": 1024,
    "H2O32": 96,
    "H2O60": 180,
    "H2O192": 576,
    "H2O512": 1536,
    "H2O1024": 3072,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("gpu-eager", "model-cg"),
        default="gpu-eager",
        help="GPU-resident eager execution or model-only CUDA Graph replay",
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
    parser.add_argument("--timestep", type=float, default=1.0, help="fs")
    parser.add_argument("--temperature", type=float, required=True, help="K")
    parser.add_argument("--taut", type=float, default=100.0, help="fs")
    parser.add_argument("--seed", type=int, default=REQUIRED_SEED)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=None,
        help="Matching baseline JSON used for energy-error validation",
    )
    parser.add_argument(
        "--missing-baseline-reference",
        action="store_true",
        help="Record an expected but unavailable/OOM baseline without skipping",
    )
    parser.add_argument(
        "--md-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="GPU dtype for positions, momenta, masses, and integration",
    )
    parser.add_argument(
        "--validate-official",
        action="store_true",
        help="Compare the initial direct-GPU result with OCPCalculator before timing",
    )
    parser.add_argument("--energy-atol", type=float, default=1e-4)
    parser.add_argument("--force-atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--cg-probe-steps", type=int, default=50)
    parser.add_argument("--cg-capacity-margin", type=float, default=0.10)
    parser.add_argument("--cg-edge-step", type=int, default=256)
    parser.add_argument("--cg-dummy-atoms", type=int, default=32)
    parser.add_argument("--cg-capture-warmup", type=int, default=3)
    parser.add_argument("--cg-eager-energy-atol", type=float, default=1e-8)
    parser.add_argument("--cg-eager-force-atol", type=float, default=2e-4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.timestep <= 0 or args.temperature <= 0 or args.taut <= 0:
        parser.error("--timestep, --temperature, and --taut must be positive")
    if args.seed != REQUIRED_SEED:
        parser.error(f"--seed must be {REQUIRED_SEED}")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.baseline_result is not None and args.missing_baseline_reference:
        parser.error(
            "--baseline-result and --missing-baseline-reference are mutually exclusive"
        )
    if args.cg_probe_steps < 0:
        parser.error("--cg-probe-steps must be non-negative")
    if args.cg_capacity_margin < 0:
        parser.error("--cg-capacity-margin must be non-negative")
    if args.cg_edge_step < 1 or args.cg_dummy_atoms < 1:
        parser.error("--cg-edge-step and --cg-dummy-atoms must be positive")
    if args.cg_capture_warmup < 0:
        parser.error("--cg-capture-warmup must be non-negative")
    if args.cg_eager_energy_atol < 0 or args.cg_eager_force_atol < 0:
        parser.error("CUDA Graph eager-comparison tolerances must be non-negative")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def capture_rng_state(torch_module):
    """Snapshot every benchmark RNG so an untimed probe is trajectory-neutral."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": torch_module.cuda.get_rng_state_all(),
    }


def restore_rng_state(torch_module, state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def append_tsv(path: Path, record: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


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
    from fairchem.core.applications.esen_gpu_md import (
        ESENEnergyForceEvaluator,
        GPUIntegrator,
        GPUMDState,
        GPUResidentMD,
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
            f"{args.system} must contain {expected_atoms} atoms, got {len(atoms)} "
            f"from {args.structure}"
        )
    if not np.all(atoms.pbc):
        raise ValueError(f"Periodic boundary conditions are required: pbc={atoms.pbc}")
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
        atoms,
        temperature_K=args.temperature,
        force_temp=True,
        rng=rng,
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
    masses = torch.as_tensor(
        atoms.get_masses(), dtype=md_dtype, device=device
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
    dynamics = GPUResidentMD(state, evaluator, integrator)
    initial_state = state.clone()

    model_cg_evaluator = None
    cg_probe_max_edges = None
    cg_edge_capacity = None
    cg_setup_wall_time_s = 0.0
    cg_eager_initial_forces = None
    cg_eager_initial_energy = None
    cg_initial_energy_abs_error_eV = None
    cg_initial_force_max_abs_error_eV_per_A = None

    if args.backend == "model-cg":
        from fairchem.core.applications.esen_cuda_graph import (
            CUDAGraphValidationError,
            ESENModelCUDAGraphEvaluator,
            edge_capacity_from_probe,
        )

        setup_start = time.perf_counter()
        rng_state = capture_rng_state(torch)
        probe_edge_counts = [
            int(
                evaluator.build_neighbor_graph(state.positions)["edge_index"].shape[1]
            )
        ]
        for _ in range(args.cg_probe_steps):
            dynamics.run(1)
            probe_edge_counts.append(
                int(
                    evaluator.build_neighbor_graph(state.positions)[
                        "edge_index"
                    ].shape[1]
                )
            )
        torch.cuda.synchronize()
        cg_probe_max_edges = max(probe_edge_counts)
        cg_edge_capacity = edge_capacity_from_probe(
            cg_probe_max_edges,
            margin=args.cg_capacity_margin,
            edge_step=args.cg_edge_step,
        )

        state.restore_(initial_state)
        dynamics.nsteps = 0
        restore_rng_state(torch, rng_state)

        # Keep one untimed eager result for a direct padded/replay correctness
        # check before the trajectory-level baseline validation.
        cg_eager_initial_forces, cg_eager_initial_energy = evaluator(
            state.positions
        )
        cg_eager_initial_forces = cg_eager_initial_forces.detach().clone()
        cg_eager_initial_energy = cg_eager_initial_energy.detach().clone()
        torch.cuda.synchronize()

        model_cg_evaluator = ESENModelCUDAGraphEvaluator(
            evaluator,
            edge_capacity=cg_edge_capacity,
            dummy_atoms=args.cg_dummy_atoms,
            capture_warmup=args.cg_capture_warmup,
        )
        model_cg_evaluator.capture(state.positions)
        dynamics.evaluator = model_cg_evaluator
        cg_setup_wall_time_s = time.perf_counter() - setup_start

    # Warm up CUDA libraries, allocator state, the OTF graph builder, and model.
    if args.warmup_steps:
        dynamics.run(args.warmup_steps)
        torch.cuda.synchronize()
    state.restore_(initial_state)
    dynamics.nsteps = 0

    # Untimed direct-path correctness result.
    initial_forces_device, initial_energy_device = dynamics.evaluate()
    torch.cuda.synchronize()
    initial_forces = initial_forces_device.detach().cpu().numpy()
    initial_energy = float(initial_energy_device.item())
    if not np.isfinite(initial_forces).all() or not math.isfinite(initial_energy):
        raise FloatingPointError("Initial direct-GPU prediction contains NaN or Inf")

    if model_cg_evaluator is not None:
        assert cg_eager_initial_forces is not None
        assert cg_eager_initial_energy is not None
        cg_initial_energy_abs_error_eV = abs(
            initial_energy - float(cg_eager_initial_energy.item())
        )
        cg_initial_force_max_abs_error_eV_per_A = float(
            (
                initial_forces_device - cg_eager_initial_forces
            ).abs().max().item()
        )
        if cg_initial_energy_abs_error_eV > args.cg_eager_energy_atol:
            raise CUDAGraphValidationError(
                "Model CUDA Graph and GPU eager initial energies differ: "
                f"error={cg_initial_energy_abs_error_eV}, "
                f"atol={args.cg_eager_energy_atol}"
            )
        if cg_initial_force_max_abs_error_eV_per_A > args.cg_eager_force_atol:
            raise CUDAGraphValidationError(
                "Model CUDA Graph and GPU eager initial forces differ: "
                f"error={cg_initial_force_max_abs_error_eV_per_A}, "
                f"atol={args.cg_eager_force_atol}"
            )

    if args.validate_official:
        original_otf_graph = evaluator.model.backbone.otf_graph
        evaluator.model.backbone.otf_graph = True
        try:
            evaluator.calculator.reset()
            atoms.calc = evaluator.calculator
            official_forces = atoms.get_forces()
            official_energy = float(atoms.get_potential_energy())
        finally:
            evaluator.model.backbone.otf_graph = original_otf_graph
        np.testing.assert_allclose(
            initial_forces,
            official_forces,
            rtol=args.rtol,
            atol=args.force_atol,
            err_msg="Direct-GPU and OCPCalculator forces differ",
        )
        np.testing.assert_allclose(
            initial_energy,
            official_energy,
            rtol=args.rtol,
            atol=args.energy_atol,
            err_msg="Direct-GPU and OCPCalculator energies differ",
        )

    # Restore once more.  Setting forces=None intentionally matches the
    # existing ASE benchmark, whose timed dynamics performs its initial force
    # evaluation after calc.reset().
    state.restore_(initial_state)
    dynamics.nsteps = 0
    del initial_state
    if model_cg_evaluator is not None:
        model_cg_evaluator.reset_production_stats()

    checkpoint_energy_tensors: dict[int, torch.Tensor] = {}
    completed_steps = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for checkpoint_step in reached_energy_checkpoints(args.steps):
        dynamics.run(checkpoint_step - completed_steps)
        if state.potential_energy is None:
            raise RuntimeError(f"No potential energy at MD step {checkpoint_step}")
        # Keep the value on the GPU so checkpoint reporting does not introduce
        # a device synchronization into the timed MD loop.
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
        raise RuntimeError("GPU MD completed without forces or potential energy")
    finite = (
        torch.isfinite(state.positions).all()
        & torch.isfinite(state.momenta).all()
        & torch.isfinite(state.forces).all()
        & torch.isfinite(state.potential_energy).all()
    )
    if not bool(finite.item()):
        raise FloatingPointError("Final GPU MD state contains NaN or Inf")

    if model_cg_evaluator is not None:
        cg_stats = model_cg_evaluator.stats()
        expected_replays = args.steps + 1
        if cg_stats["cuda_graph_production_replays"] != expected_replays:
            raise RuntimeError(
                "Unexpected model CUDA Graph replay count: "
                f"expected={expected_replays}, "
                f"actual={cg_stats['cuda_graph_production_replays']}"
            )
        if cg_stats["cuda_graph_capacity_misses"] != 0:
            raise RuntimeError("Successful CUDA Graph run recorded a capacity miss")
        if cg_stats["cuda_graph_production_capture_count"] != 0:
            raise RuntimeError("CUDA Graph was captured inside the production region")
    else:
        cg_stats = None

    final_energy = float(state.potential_energy.item())
    final_max_force = float(state.forces.abs().max().item())
    final_temperature = float(integrator.temperature(state.momenta).item())
    backend_name = (
        "esen_gpu_resident_model_cg"
        if args.backend == "model-cg"
        else "esen_gpu_resident_eager"
    )
    backend_suffix = "esen_model_cg" if args.backend == "model-cg" else "esen_gpu_eager"
    run_name = args.run_name or (
        f"{args.system}_{args.temperature:g}K_{args.steps}step_{backend_suffix}"
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
        "outputs": "energy,forces",
        "parameters_frozen": True,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "cuda_graph": args.backend == "model-cg",
        "cuda_graph_scope": "model_only" if args.backend == "model-cg" else "none",
        "cuda_graph_neighbor_build_outside": args.backend == "model-cg",
        "kernel_fusion": False,
        "stress_computed": False,
        "activation_checkpointing": False,
        "gpu_resident_md": True,
        "neighbor_builder": "fairchem_radius_graph_pbc",
        "md_state_dtype": str(state.positions.dtype).removeprefix("torch."),
        "model_dtype": str(evaluator.model_dtype).removeprefix("torch."),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
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
        "energy_reference_role": "candidate",
        "baseline_result": (
            "" if args.baseline_result is None else str(args.baseline_result.resolve())
        ),
        "baseline_result_sha256": (
            "" if args.baseline_result is None else sha256(args.baseline_result)
        ),
        "cg_probe_steps": args.cg_probe_steps if args.backend == "model-cg" else 0,
        "cg_probe_max_edges": cg_probe_max_edges,
        "cg_capacity_margin": (
            args.cg_capacity_margin if args.backend == "model-cg" else None
        ),
        "cg_edge_step": args.cg_edge_step if args.backend == "model-cg" else None,
        "cg_edge_capacity": cg_edge_capacity,
        "cg_setup_wall_time_s": (
            cg_setup_wall_time_s if args.backend == "model-cg" else None
        ),
        "cg_initial_energy_abs_error_eV": cg_initial_energy_abs_error_eV,
        "cg_eager_energy_atol_eV": args.cg_eager_energy_atol,
        "cg_initial_force_max_abs_error_eV_per_A": (
            cg_initial_force_max_abs_error_eV_per_A
        ),
        "cg_eager_force_atol_eV_per_A": args.cg_eager_force_atol,
    }
    if cg_stats is not None:
        record.update(cg_stats)
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

    json_path = args.output_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_tsv(args.output_dir / "summary.tsv", record)
    print(json.dumps(record, indent=2))
    print(f"Result: {json_path}")
    print(f"Summary: {args.output_dir / 'summary.tsv'}")
    if validation_pass is False:
        print(
            "BENCHMARK_STATUS=validation_failed: energy error exceeded the "
            "1-step or 50-step limit",
            file=sys.stderr,
        )
        return 43
    return 0


def entrypoint() -> int:
    try:
        return main()
    except BaseException as exc:  # Classify CUDA OOM for the batch driver.
        message = str(exc).lower()
        if exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in message:
            print(f"BENCHMARK_STATUS=oom: {exc}", file=sys.stderr)
            return 42
        if exc.__class__.__name__ == "CUDAGraphCapacityError":
            print(f"BENCHMARK_STATUS=capacity_overflow: {exc}", file=sys.stderr)
            return 45
        if exc.__class__.__name__ == "CUDAGraphValidationError":
            print(f"BENCHMARK_STATUS=validation_failed: {exc}", file=sys.stderr)
            return 43
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())
