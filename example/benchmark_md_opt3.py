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
        choices=(
            "fixed-builder-model-cg",
            "whole-step-cg",
            "fixed-builder-model-cg-kf1",
            "whole-step-cg-kf1",
            "fixed-builder-model-cg-opt4",
            "whole-step-cg-opt4",
        ),
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
    parser.add_argument(
        "--neighbor-capacity-policy",
        "--opt4-neighbor-capacity-policy",
        dest="neighbor_capacity_policy",
        choices=(
            "uniform",
            "species",
            "atom",
            "auto",
            "auto-safe",
            "elastic",
        ),
        default="uniform",
        help=(
            "Static neighbor-slot allocation. 'uniform' preserves Opt3/Opt4 "
            "behavior; 'species' assigns each element its probed maximum; "
            "'atom' assigns rounded probe bounds per atom; 'auto' uses atom "
            "slots only when their capacity reduction reaches the configured "
            "threshold; 'auto-safe' additionally promotes every atom by the "
            "configured number of guard buckets before applying the threshold."
        ),
    )
    parser.add_argument(
        "--neighbor-auto-min-reduction",
        type=float,
        default=0.05,
        help=(
            "Minimum fractional edge-capacity reduction required for the "
            "'auto' policy to select per-atom slots (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--neighbor-auto-guard-slots",
        type=int,
        default=1,
        help=(
            "Additional neighbor-slot buckets used by 'auto-safe' before its "
            "minimum-reduction decision (default: 1)."
        ),
    )
    parser.add_argument("--rob1", action="store_true")
    parser.add_argument("--rob1-window-steps", type=int, default=10)
    parser.add_argument("--rob1-max-retries", type=int, default=2)
    parser.add_argument("--cap2-compact-slot-step", type=int, default=4)
    parser.add_argument("--cap2-compact-margin", type=float, default=0.0)
    parser.add_argument("--cap2-min-reduction", type=float, default=0.05)
    parser.add_argument("--cap2-test-capacity-limit", type=int, default=0)
    parser.add_argument("--dummy-atoms", type=int, default=32)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--max-neighbors", type=int, default=300)
    parser.add_argument("--degeneracy-tolerance", type=float, default=0.01)
    parser.add_argument("--energy-per-atom-atol", type=float, default=1e-5)
    parser.add_argument("--force-max-atol", type=float, default=2e-4)
    parser.add_argument("--replay-energy-atol", type=float, default=0.0)
    parser.add_argument("--replay-force-atol", type=float, default=1e-6)
    parser.add_argument("--triton-block-size", type=int, default=256)
    parser.add_argument("--model-fusions", default="")
    parser.add_argument("--fusion-stage", default="")
    parser.add_argument(
        "--tf32-mode",
        choices=("off", "on"),
        default="off",
        help=(
            "Explicit process-wide TF32 policy. The default 'off' preserves "
            "Opt3/Opt4 behavior; 'on' is only for the independent PREC1 "
            "performance experiment."
        ),
    )
    parser.add_argument(
        "--external-profiler",
        action="store_true",
        help=(
            "Call cudaProfilerStart/Stop around the timed production region. "
            "This is intended for Nsight capture-range=cudaProfilerApi runs."
        ),
    )
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
    if args.neighbor_auto_guard_slots < 1:
        parser.error("neighbor auto-safe guard slots must be positive")
    if args.rob1_window_steps < 1:
        parser.error("ROB1 window steps must be positive")
    if not 0 <= args.rob1_max_retries <= 2:
        parser.error("ROB1 max retries must be between zero and two")
    if args.cap2_compact_slot_step < 1 or args.cap2_compact_margin < 0:
        parser.error("invalid CAP2 compact capacity parameters")
    if not 0.0 <= args.cap2_min_reduction <= 1.0:
        parser.error("CAP2 minimum reduction must be between zero and one")
    if args.cap2_test_capacity_limit < 0:
        parser.error("CAP2 test capacity limit must be non-negative")
    if (
        not np.isfinite(args.neighbor_auto_min_reduction)
        or not 0.0 <= args.neighbor_auto_min_reduction <= 1.0
    ):
        parser.error("neighbor auto minimum reduction must be between 0 and 1")
    if args.dummy_atoms < 1 or args.capture_warmup < 0:
        parser.error("invalid CUDA Graph setup parameters")
    if args.max_neighbors < 1 or args.degeneracy_tolerance < 0:
        parser.error("invalid neighbor pruning parameters")
    if args.energy_per_atom_atol < 0 or args.force_max_atol < 0:
        parser.error("engineering tolerances must be non-negative")
    if not 32 <= args.triton_block_size <= 1024:
        parser.error("triton block size must be between 32 and 1024")
    if args.triton_block_size & (args.triton_block_size - 1):
        parser.error("triton block size must be a power of two")
    if args.baseline_result is not None and args.missing_baseline_reference:
        parser.error("baseline path and missing-reference flag are exclusive")
    if args.neighbor_capacity_policy == "elastic" and not args.rob1:
        parser.error("elastic capacity requires --rob1")
    if args.rob1 and args.neighbor_capacity_policy != "elastic":
        parser.error("--rob1 is only valid with elastic capacity")
    if args.neighbor_capacity_policy == "elastic" and (
        args.backend != "whole-step-cg-opt4"
    ):
        parser.error("CAP2/ROB1 is only supported by whole-step-cg-opt4")
    return args


def _device_memory_used(torch_module, device) -> int | None:
    try:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError):
        return None
    return int(total_bytes - free_bytes)


def _start_external_profiler(torch_module) -> None:
    result = torch_module.cuda.cudart().cudaProfilerStart()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStart failed: {result}")


def _stop_external_profiler(torch_module) -> None:
    result = torch_module.cuda.cudart().cudaProfilerStop()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStop failed: {result}")


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
        atom_neighbor_capacities_from_probe,
        auto_neighbor_capacities_from_probe,
        elastic_neighbor_capacities_from_probe,
        neighbor_counts_in_graph,
        neighbor_capacity_from_probe,
        species_neighbor_capacities_from_probe,
    )
    from fairchem.core.applications.esen_gpu_md import (
        ESENEnergyForceEvaluator,
        GPUIntegrator,
        GPUMDState,
        GPUResidentMD,
    )
    from fairchem.core.applications.esen_whole_step_cuda_graph import (
        ElasticWholeStepCUDAGraphController,
        ESENFixedBuilderModelCUDAGraphEvaluator,
        ESENWholeStepCUDAGraphMD,
    )
    from fairchem.core.applications.esen_opt4_kernel_fusion import (
        ESENKF1FixedBuilderModelCUDAGraphEvaluator,
        ESENKF1WholeStepCUDAGraphMD,
        ESENOpt4FixedBuilderModelCUDAGraphEvaluator,
        ESENOpt4WholeStepCUDAGraphMD,
    )
    from fairchem.core.applications.esen_opt4_model_fusion import (
        parse_model_fusions,
    )
    from fairchem.core.applications.esen_precision import (
        configure_tf32,
        verify_tf32,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.structure.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("structure or checkpoint file is missing")
    seed_everything(torch, args.seed)
    precision_metadata = configure_tf32(torch, args.tf32_mode)
    fixed_builder_backend = args.backend.startswith("fixed-builder-model-cg")
    kf1_backend = args.backend.endswith("-kf1")
    opt4_backend = args.backend.endswith("-opt4")
    selected_model_fusions = parse_model_fusions(args.model_fusions)
    if opt4_backend and not selected_model_fusions:
        raise ValueError("Opt4 backend requires at least one --model-fusions item")
    if not opt4_backend and selected_model_fusions:
        raise ValueError("--model-fusions is only valid for an Opt4 backend")
    if opt4_backend and not args.fusion_stage:
        raise ValueError("Opt4 backend requires --fusion-stage")

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
    initial_graph = evaluator.build_neighbor_graph(state.positions)
    probe_max_degrees = neighbor_counts_in_graph(
        initial_graph["edge_index"], len(atoms)
    )
    for _ in range(args.probe_steps):
        eager_md.run(1)
        graph = evaluator.build_neighbor_graph(state.positions)
        probe_max_degrees = torch.maximum(
            probe_max_degrees,
            neighbor_counts_in_graph(graph["edge_index"], len(atoms)),
        )
    torch.cuda.synchronize()
    probe_max_neighbors = int(probe_max_degrees.max().item())
    uniform_neighbor_capacity = neighbor_capacity_from_probe(
        probe_max_neighbors,
        margin=args.neighbor_margin,
        slot_step=args.neighbor_slot_step,
    )
    auto_candidate_capacities, auto_candidate_reduction = (
        auto_neighbor_capacities_from_probe(
            probe_max_degrees,
            margin=args.neighbor_margin,
            slot_step=args.neighbor_slot_step,
            minimum_reduction=args.neighbor_auto_min_reduction,
        )
    )
    safe_auto_candidate_capacities, safe_auto_candidate_reduction = (
        auto_neighbor_capacities_from_probe(
            probe_max_degrees,
            margin=args.neighbor_margin,
            slot_step=args.neighbor_slot_step,
            minimum_reduction=args.neighbor_auto_min_reduction,
            guard_slots=args.neighbor_auto_guard_slots,
        )
    )
    unprotected_atom_capacities = atom_neighbor_capacities_from_probe(
        probe_max_degrees,
        margin=args.neighbor_margin,
        slot_step=args.neighbor_slot_step,
    )
    safe_atom_capacities = tuple(
        min(
            uniform_neighbor_capacity,
            capacity
            + args.neighbor_auto_guard_slots * args.neighbor_slot_step,
        )
        for capacity in unprotected_atom_capacities
    )
    safe_effective_capacities = (
        safe_auto_candidate_capacities
        if safe_auto_candidate_capacities is not None
        else (uniform_neighbor_capacity,) * len(atoms)
    )
    cap2_compact_selected = False
    cap2_reduction_vs_safe = 0.0
    cap2_compact_capacities = None
    if args.neighbor_capacity_policy == "species":
        neighbor_capacities = species_neighbor_capacities_from_probe(
            probe_max_degrees,
            atoms.numbers,
            margin=args.neighbor_margin,
            slot_step=args.neighbor_slot_step,
        )
    elif args.neighbor_capacity_policy == "atom":
        neighbor_capacities = atom_neighbor_capacities_from_probe(
            probe_max_degrees,
            margin=args.neighbor_margin,
            slot_step=args.neighbor_slot_step,
        )
    elif args.neighbor_capacity_policy == "auto":
        neighbor_capacities = auto_candidate_capacities
    elif args.neighbor_capacity_policy == "auto-safe":
        neighbor_capacities = safe_auto_candidate_capacities
    elif args.neighbor_capacity_policy == "elastic":
        (
            neighbor_capacities,
            cap2_compact_selected,
            cap2_reduction_vs_safe,
            cap2_compact_capacities,
        ) = elastic_neighbor_capacities_from_probe(
            probe_max_degrees,
            safe_effective_capacities,
            margin=args.cap2_compact_margin,
            slot_step=args.cap2_compact_slot_step,
            minimum_reduction=args.cap2_min_reduction,
        )
        if args.cap2_test_capacity_limit:
            neighbor_capacities = tuple(
                max(1, min(value, args.cap2_test_capacity_limit))
                for value in neighbor_capacities
            )
    else:
        neighbor_capacities = None
    effective_neighbor_capacity_policy = (
        (
            "atom-safe"
            if args.neighbor_capacity_policy == "auto-safe"
            else (
                "elastic-compact"
                if args.neighbor_capacity_policy == "elastic"
                and cap2_compact_selected
                else (
                    "elastic-auto-safe"
                    if args.neighbor_capacity_policy == "elastic"
                    else "atom"
                )
            )
        )
        if neighbor_capacities is not None
        and args.neighbor_capacity_policy
        in {"atom", "auto", "auto-safe", "elastic"}
        else args.neighbor_capacity_policy
    )
    if (
        args.neighbor_capacity_policy in {"auto", "auto-safe", "elastic"}
        and neighbor_capacities is None
    ):
        effective_neighbor_capacity_policy = "uniform"
    effective_neighbor_capacities = (
        neighbor_capacities
        if neighbor_capacities is not None
        else (uniform_neighbor_capacity,) * len(atoms)
    )
    neighbor_capacity = max(effective_neighbor_capacities)
    neighbor_edge_capacity = sum(effective_neighbor_capacities)
    uniform_edge_capacity = len(atoms) * uniform_neighbor_capacity
    probe_max_degrees_cpu = probe_max_degrees.detach().to(device="cpu")
    atomic_numbers_cpu = torch.as_tensor(atoms.numbers, dtype=torch.long)
    neighbor_capacity_by_species = {}
    for atomic_number in torch.unique(atomic_numbers_cpu, sorted=True):
        mask = atomic_numbers_cpu == atomic_number
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        species_capacities = [
            effective_neighbor_capacities[int(index)] for index in indices
        ]
        unique_species_capacities = sorted(set(species_capacities))
        neighbor_capacity_by_species[str(int(atomic_number))] = {
            "atomic_number": int(atomic_number),
            "atom_count": int(mask.sum().item()),
            "probe_max_neighbors": int(
                probe_max_degrees_cpu[mask].max().item()
            ),
            "capacity_per_atom": (
                unique_species_capacities[0]
                if len(unique_species_capacities) == 1
                else None
            ),
            "capacity_min": min(species_capacities),
            "capacity_max": max(species_capacities),
            "capacity_mean": sum(species_capacities) / len(species_capacities),
            "capacity_histogram": {
                str(capacity): species_capacities.count(capacity)
                for capacity in unique_species_capacities
            },
        }
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
    if fixed_builder_backend:
        if kf1_backend:
            evaluator_class = ESENKF1FixedBuilderModelCUDAGraphEvaluator
        elif opt4_backend:
            evaluator_class = ESENOpt4FixedBuilderModelCUDAGraphEvaluator
        else:
            evaluator_class = ESENFixedBuilderModelCUDAGraphEvaluator
        fixed_kwargs = {}
        if kf1_backend:
            fixed_kwargs["triton_block_size"] = args.triton_block_size
        if opt4_backend:
            fixed_kwargs.update(
                model_fusions=",".join(selected_model_fusions),
                fusion_stage=args.fusion_stage,
            )
        fixed_evaluator = evaluator_class(
            evaluator,
            neighbors_per_atom=neighbor_capacity,
            neighbor_capacities=neighbor_capacities,
            neighbor_capacity_policy=effective_neighbor_capacity_policy,
            dummy_atoms=args.dummy_atoms,
            capture_warmup=args.capture_warmup,
            max_neighbors=args.max_neighbors,
            degeneracy_tolerance=args.degeneracy_tolerance,
            replay_energy_atol=args.replay_energy_atol,
            replay_force_atol=args.replay_force_atol,
            **fixed_kwargs,
        )
        torch.cuda.empty_cache()
        fixed_evaluator.capture(state.positions)
        dynamics = GPUResidentMD(state, fixed_evaluator, integrator)
    else:
        if kf1_backend:
            whole_class = ESENKF1WholeStepCUDAGraphMD
        elif opt4_backend:
            whole_class = ESENOpt4WholeStepCUDAGraphMD
        else:
            whole_class = ESENWholeStepCUDAGraphMD
        whole_kwargs = {}
        if kf1_backend:
            whole_kwargs["triton_block_size"] = args.triton_block_size
        if opt4_backend:
            whole_kwargs.update(
                model_fusions=",".join(selected_model_fusions),
                fusion_stage=args.fusion_stage,
            )
        whole_kwargs.update(
            dummy_atoms=args.dummy_atoms,
            capture_warmup=args.capture_warmup,
            max_neighbors=args.max_neighbors,
            degeneracy_tolerance=args.degeneracy_tolerance,
        )
        if args.neighbor_capacity_policy == "elastic":
            whole_md = ElasticWholeStepCUDAGraphController(
                state,
                evaluator,
                integrator,
                whole_class=whole_class,
                atomic_numbers=atoms.numbers,
                neighbor_capacities=effective_neighbor_capacities,
                max_promotions=args.rob1_max_retries,
                whole_kwargs=whole_kwargs,
            )
        else:
            whole_md = whole_class(
                state,
                evaluator,
                integrator,
                neighbors_per_atom=neighbor_capacity,
                neighbor_capacities=neighbor_capacities,
                neighbor_capacity_policy=effective_neighbor_capacity_policy,
                **whole_kwargs,
            )
        torch.cuda.empty_cache()
        whole_md.capture(initial_state)
        dynamics = None
    torch.cuda.synchronize()
    device_used_after_capture = _device_memory_used(torch, device)
    setup_wall_time = time.perf_counter() - setup_start

    adaptive = args.neighbor_capacity_policy == "elastic"
    # Standard warmup is trajectory-neutral and excluded from timing.  CAP2's
    # initial force transaction is intentionally not preflighted when warmup
    # is zero: forced-low-capacity smoke tests must exercise rollback inside
    # the measured production path rather than silently promoting in setup.
    if fixed_builder_backend:
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
        if adaptive:
            if args.warmup_steps:
                whole_md.evaluate_initial()
                for _ in range(args.warmup_steps):
                    whole_md.step()
                torch.cuda.synchronize()
                whole_md.reset_production(initial_state)
            initial_forces_device = None
            initial_energy_device = None
        else:
            whole_md.evaluate_initial()
            for _ in range(args.warmup_steps):
                whole_md.step()
            torch.cuda.synchronize()
            whole_md.reset_production(initial_state)
            initial_forces_device, initial_energy_device = (
                whole_md.evaluate_initial()
            )
    torch.cuda.synchronize()

    # Reset after validation.  The timed region starts from the original state.
    if fixed_builder_backend:
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
    if args.external_profiler:
        _start_external_profiler(torch)
    timed_start = time.perf_counter()
    if fixed_builder_backend:
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
        timed_initial_forces, timed_initial_energy = whole_md.evaluate_initial()
        if adaptive:
            initial_forces_device = timed_initial_forces.detach().clone()
            initial_energy_device = timed_initial_energy.detach().clone()
            completed = 0
            while completed < args.steps:
                transaction_steps = min(
                    args.rob1_window_steps, args.steps - completed
                )
                local_offsets = [
                    checkpoint - completed
                    for checkpoint in checkpoints
                    if completed < checkpoint <= completed + transaction_steps
                ]
                _, _, committed_checkpoints = (
                    whole_md.run_steps_with_checkpoints(
                        transaction_steps, local_offsets
                    )
                )
                for offset, value in committed_checkpoints.items():
                    checkpoint_tensors[completed + offset] = value
                completed += transaction_steps
        else:
            for step in range(1, args.steps + 1):
                _, energy = whole_md.step()
                if step in checkpoints:
                    checkpoint_tensors[step] = energy.detach().clone()
        final_state = whole_md.state_view()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - timed_start
    if args.external_profiler:
        _stop_external_profiler(torch)
    device_used_after_timing = _device_memory_used(torch, device)
    assert initial_forces_device is not None
    assert initial_energy_device is not None
    initial_energy = float(initial_energy_device.item())
    initial_force_error = float(
        (initial_forces_device - eager_initial_forces).abs().max().item()
    )
    initial_energy_error = abs(
        initial_energy - float(eager_initial_energy.item())
    )
    initial_energy_error_per_atom = initial_energy_error / len(atoms)
    force_validation_pass = initial_force_error < args.force_max_atol
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
    graph_invariants_pass = bool(
        graph_stats["cuda_graph_capture_count"] == 1
        and graph_stats["cuda_graph_production_replays"] == expected_replays
        and not capacity_overflow
        and graph_stats["cuda_graph_hit_rate"] == 1.0
        and graph_stats.get("cuda_graph_replay_stability_pass", True)
        and (
            (
                graph_stats.get("rob1_committed_physical_steps") == args.steps
                and graph_stats.get("rob1_unrecovered_overflows") == 0
                and graph_stats.get("rob1_snapshot_addresses_stable", False)
                and graph_stats.get("cuda_graph_production_capture_count", 0)
                == graph_stats.get("cuda_graph_recovery_capture_count", 0)
            )
            if adaptive
            else graph_stats["cuda_graph_production_capture_count"] == 0
        )
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

    if fixed_builder_backend:
        if kf1_backend:
            backend_name = "esen_gpu_resident_fixed_builder_model_cg_kf1"
            suffix = "esen_fixed_builder_model_cg_kf1"
        elif opt4_backend:
            backend_name = "esen_gpu_resident_fixed_builder_model_cg_opt4"
            suffix = f"esen_fixed_builder_model_cg_{args.fusion_stage}"
        else:
            backend_name = "esen_gpu_resident_fixed_builder_model_cg"
            suffix = "esen_fixed_builder_model_cg"
    else:
        if kf1_backend:
            backend_name = "esen_gpu_resident_whole_step_cg_kf1"
            suffix = "esen_whole_step_cg_kf1"
        elif opt4_backend:
            backend_name = "esen_gpu_resident_whole_step_cg_opt4"
            suffix = f"esen_whole_step_cg_{args.fusion_stage}"
        else:
            backend_name = "esen_gpu_resident_whole_step_cg"
            suffix = "esen_whole_step_cg"
    run_name = args.run_name or (
        f"{args.system}_{args.temperature:g}K_{args.steps}step_{suffix}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    precision_metadata = verify_tf32(torch, args.tf32_mode)
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
        **precision_metadata,
        "torch_compile": False,
        "external_profiler_range": args.external_profiler,
        "kernel_fusion": kf1_backend or opt4_backend,
        "kernel_fusion_stage": (
            "KF1" if kf1_backend else (args.fusion_stage if opt4_backend else "KF0")
        ),
        "kernel_fusion_name": (
            "triton_pbc_distance_cutoff_mask"
            if kf1_backend
            else (",".join(selected_model_fusions) if opt4_backend else "none")
        ),
        "model_fusions": ",".join(selected_model_fusions),
        "triton_block_size": (
            args.triton_block_size if kf1_backend else None
        ),
        "cuda_graph": True,
        "cuda_graph_scope": (
            "model_only_fixed_builder"
            if fixed_builder_backend
            else "whole_nvt_step"
        ),
        "cuda_graph_neighbor_build_outside": (
            fixed_builder_backend
        ),
        "gpu_resident_md": True,
        "neighbor_builder": (
            "fixed_shape_radius_graph_pbc_triton_kf1"
            if kf1_backend
            else "fixed_shape_radius_graph_pbc"
        ),
        "md_state_dtype": "float64",
        "model_dtype": str(evaluator.model_dtype).removeprefix("torch."),
        "probe_steps": args.probe_steps,
        "probe_max_neighbors_per_atom": probe_max_neighbors,
        "neighbor_capacity_per_atom": neighbor_capacity,
        "neighbor_capacity_policy": args.neighbor_capacity_policy,
        "neighbor_capacity_policy_requested": args.neighbor_capacity_policy,
        "neighbor_capacity_policy_effective": effective_neighbor_capacity_policy,
        "neighbor_capacity_auto_min_reduction": (
            args.neighbor_auto_min_reduction
        ),
        "neighbor_capacity_auto_guard_slots": (
            args.neighbor_auto_guard_slots
            if args.neighbor_capacity_policy == "auto-safe"
            else 0
        ),
        "neighbor_capacity_auto_guard_neighbors": (
            args.neighbor_auto_guard_slots * args.neighbor_slot_step
            if args.neighbor_capacity_policy == "auto-safe"
            else 0
        ),
        "neighbor_capacity_auto_candidate_edge_capacity": sum(
            safe_atom_capacities
            if args.neighbor_capacity_policy == "auto-safe"
            else unprotected_atom_capacities
        ),
        "neighbor_capacity_auto_candidate_reduction_vs_uniform": (
            safe_auto_candidate_reduction
            if args.neighbor_capacity_policy == "auto-safe"
            else auto_candidate_reduction
        ),
        "neighbor_capacity_auto_unprotected_reduction_vs_uniform": (
            auto_candidate_reduction
        ),
        "neighbor_capacity_auto_unprotected_edge_capacity": sum(
            unprotected_atom_capacities
        ),
        "neighbor_capacity_auto_safe_reduction_vs_uniform": (
            safe_auto_candidate_reduction
        ),
        "neighbor_capacity_auto_safe_edge_capacity": sum(
            safe_atom_capacities
        ),
        "neighbor_capacity_auto_selected": (
            args.neighbor_capacity_policy in {"auto", "auto-safe"}
            and effective_neighbor_capacity_policy in {"atom", "atom-safe"}
        ),
        "neighbor_capacity_auto_safe_selected": (
            args.neighbor_capacity_policy == "auto-safe"
            and effective_neighbor_capacity_policy == "atom-safe"
        ),
        "cap2_enabled": adaptive,
        "cap2_compact_selected": cap2_compact_selected,
        "cap2_compact_slot_step": args.cap2_compact_slot_step,
        "cap2_compact_margin": args.cap2_compact_margin,
        "cap2_min_reduction": args.cap2_min_reduction,
        "cap2_compact_reduction_vs_auto_safe": cap2_reduction_vs_safe,
        "cap2_compact_edge_capacity": (
            None
            if cap2_compact_capacities is None
            else sum(cap2_compact_capacities)
        ),
        "cap2_auto_safe_edge_capacity": sum(safe_effective_capacities),
        "cap2_test_capacity_limit": args.cap2_test_capacity_limit,
        "rob1_enabled": adaptive,
        "rob1_window_steps": args.rob1_window_steps if adaptive else None,
        "rob1_max_retries": args.rob1_max_retries if adaptive else None,
        "neighbor_capacity_by_species": neighbor_capacity_by_species,
        "neighbor_edge_capacity": neighbor_edge_capacity,
        "neighbor_uniform_capacity_per_atom": uniform_neighbor_capacity,
        "neighbor_uniform_edge_capacity": uniform_edge_capacity,
        "neighbor_capacity_reduction_vs_uniform": (
            (uniform_edge_capacity - neighbor_edge_capacity)
            / uniform_edge_capacity
        ),
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
        "performance_sample_eligible": bool(
            graph_invariants_pass and not capacity_overflow
        ),
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
        "source_bundle_sha256": os.environ.get(
            "SOURCE_BUNDLE_SHA256", ""
        ),
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
    # Energy and force tolerances are recorded for later analysis and do not
    # invalidate a completed performance sample.  CUDA Graph invariants and
    # capacity are structural/runtime conditions and remain hard failures.
    if not graph_invariants_pass:
        print(
            "BENCHMARK_STATUS=graph_validation_failed: MD completed; "
            + " | ".join(numerical_failures),
            file=sys.stderr,
        )
        return 43
    if numerical_failures:
        print(
            "BENCHMARK_STATUS=validation_warning: MD completed; "
            + " | ".join(numerical_failures),
            file=sys.stderr,
        )
    return 0


def entrypoint() -> int:
    try:
        return main()
    except BaseException as exc:
        message = str(exc).lower()
        if exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in message:
            print(f"BENCHMARK_STATUS=oom: {exc}", file=sys.stderr)
            return 42
        if exc.__class__.__name__ == "UnsupportedFusionConfigError":
            print(
                f"BENCHMARK_STATUS=unsupported_fusion_config: {exc}",
                file=sys.stderr,
            )
            return 46
        if exc.__class__.__name__ == "UnrecoveredCapacityOverflow":
            print(
                f"BENCHMARK_STATUS=unrecovered_capacity_overflow: {exc}",
                file=sys.stderr,
            )
            return 45
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())
