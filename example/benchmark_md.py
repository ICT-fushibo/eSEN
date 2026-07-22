#!/usr/bin/env python3
"""Benchmark eSEN ASE molecular dynamics without CUDA Graph or torch.compile.

The timed region contains only ASE NVT MD steps. Model loading, warm-up,
checkpoint hashing, validation, and result I/O are deliberately excluded.
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
import subprocess
import sys
import time

import numpy as np

from md_energy_reference import (
    REQUIRED_SEED,
    checkpoint_energy_fields,
    reached_energy_checkpoints,
    seed_everything,
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
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO / "esen_30m_oam.pt",
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
        "--outputs",
        nargs="+",
        default=["energy", "forces"],
        help=(
            "Checkpoint outputs to retain. Use '--outputs all' for checkpoint "
            "defaults."
        ),
    )
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


def append_tsv(path: Path, record: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    args = parse_args()

    if os.environ.get("PYTHONHASHSEED") != str(REQUIRED_SEED):
        raise RuntimeError(
            f"Launch the benchmark with PYTHONHASHSEED={REQUIRED_SEED}"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Import the checkout when fairchem-core has not been installed editable.
    sys.path.insert(0, str(REPO / "src"))

    import torch
    from ase import units
    from ase.io import read
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from fairchem.core import OCPCalculator
    from fairchem.core.applications.esen_gpu_md import (
        configure_esen_energy_force_inference,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.structure.is_file():
        raise FileNotFoundError(args.structure)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    seed_everything(torch, args.seed)

    # Precision is part of the baseline contract.
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

    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=args.temperature,
        force_temp=True,
        rng=rng,
    )
    Stationary(atoms)

    initial_positions = atoms.get_positions().copy()
    initial_momenta = atoms.get_momenta().copy()
    initial_cell = atoms.get_cell().copy()

    only_output = None if args.outputs == ["all"] else args.outputs
    calc = OCPCalculator(
        checkpoint_path=args.checkpoint,
        cpu=False,
        seed=args.seed,
        only_output=only_output,
        disable_amp=True,
    )
    loaded_model = calc.trainer._unwrapped_model
    energy_force_only = only_output is not None and "stress" not in only_output
    if energy_force_only:
        configure_esen_energy_force_inference(loaded_model)
    else:
        loaded_model.eval()
        for parameter in loaded_model.parameters():
            parameter.requires_grad_(False)
        if hasattr(loaded_model.backbone, "activation_checkpointing"):
            loaded_model.backbone.activation_checkpointing = False
    atoms.calc = calc

    # Untimed warm-up initializes CUDA libraries and allocator state.
    if args.warmup_steps:
        warmup = NVTBerendsen(
            atoms,
            timestep=args.timestep * units.fs,
            temperature_K=args.temperature,
            taut=args.taut * units.fs,
        )
        warmup.run(args.warmup_steps)
        torch.cuda.synchronize()

    # Production always starts from the same state, independent of warm-up count.
    atoms.set_cell(initial_cell, scale_atoms=False)
    atoms.set_positions(initial_positions)
    atoms.set_momenta(initial_momenta)
    calc.reset()

    # One untimed correctness call also catches bad structures/checkpoints early.
    initial_forces = atoms.get_forces()
    if not np.isfinite(initial_forces).all():
        raise FloatingPointError("Initial force prediction contains NaN or Inf")
    initial_energy = float(atoms.get_potential_energy())
    if not math.isfinite(initial_energy):
        raise FloatingPointError("Initial energy prediction contains NaN or Inf")

    calc.reset()
    dynamics = NVTBerendsen(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        taut=args.taut * units.fs,
    )

    checkpoint_hash = sha256(args.checkpoint)
    structure_hash = sha256(args.structure)
    checkpoint_energies: dict[int, float] = {}
    completed_steps = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for checkpoint_step in reached_energy_checkpoints(args.steps):
        dynamics.run(checkpoint_step - completed_steps)
        # OCPCalculator already returned energy together with the forces for
        # this step, so this reads the ASE cache without another model call.
        checkpoint_energies[checkpoint_step] = float(atoms.get_potential_energy())
        completed_steps = checkpoint_step
    if completed_steps < args.steps:
        dynamics.run(args.steps - completed_steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    final_forces = atoms.get_forces()
    final_energy = float(atoms.get_potential_energy())
    if not np.isfinite(final_forces).all() or not math.isfinite(final_energy):
        raise FloatingPointError("Final MD prediction contains NaN or Inf")

    run_name = args.run_name or (
        f"{args.system}_{args.temperature:g}K_{args.steps}step_esen_baseline"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {
        "backend": "esen_ocpcalculator_eager",
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
        "outputs": "all" if only_output is None else ",".join(only_output),
        "parameters_frozen": True,
        "stress_computed": not energy_force_only,
        "activation_checkpointing": False,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "cuda_graph": False,
        "kernel_fusion": False,
        "gpu_resident_md": False,
        "neighbor_builder": "fairchem_radius_graph_pbc",
        "md_state_dtype": "float64",
        "model_dtype": str(next(loaded_model.parameters()).dtype).removeprefix(
            "torch."
        ),
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
        "final_max_force_eV_per_A": float(np.abs(final_forces).max()),
        "final_temperature_K": float(atoms.get_temperature()),
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
        "energy_reference_role": "baseline",
    }
    record.update(checkpoint_energy_fields(checkpoint_energies))

    json_path = args.output_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_tsv(args.output_dir / "summary.tsv", record)

    print(json.dumps(record, indent=2))
    print(f"Result: {json_path}")
    print(f"Summary: {args.output_dir / 'summary.tsv'}")


def entrypoint() -> int:
    try:
        main()
    except BaseException as exc:
        message = str(exc).lower()
        if exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in message:
            print(f"BENCHMARK_STATUS=oom: {exc}", file=sys.stderr)
            return 42
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
