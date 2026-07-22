"""Shared reproducibility and energy-reference helpers for eSEN MD benchmarks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_SEED = 42
ENERGY_CHECKPOINT_STEPS = (1, 50, 100, 1000)
ENERGY_ERROR_LIMITS_EV = {1: 1.0e-8, 50: 1.0e-6}


def seed_everything(torch_module: Any, seed: int) -> None:
    """Seed every RNG used by the benchmark process."""

    if seed != REQUIRED_SEED:
        raise ValueError(f"Benchmark seed must be {REQUIRED_SEED}, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True


def reached_energy_checkpoints(total_steps: int) -> tuple[int, ...]:
    """Return reference checkpoints reached by a trajectory of ``total_steps``."""

    return tuple(step for step in ENERGY_CHECKPOINT_STEPS if step <= total_steps)


def energy_field(step: int) -> str:
    return f"energy_step_{step}_eV"


def baseline_energy_field(step: int) -> str:
    return f"baseline_energy_step_{step}_eV"


def energy_error_field(step: int) -> str:
    return f"energy_abs_error_step_{step}_eV"


def checkpoint_energy_fields(
    energies: dict[int, float],
) -> dict[str, float | None]:
    """Flatten checkpoint energies into stable JSON/TSV field names."""

    return {
        energy_field(step): energies.get(step) for step in ENERGY_CHECKPOINT_STEPS
    }


def load_baseline_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline reference not found: {path}")
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("backend") != "esen_ocpcalculator_eager":
        raise ValueError(
            f"Expected an esen_ocpcalculator_eager reference, got "
            f"{reference.get('backend')!r} from {path}"
        )
    return reference


def validate_reference_metadata(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Reject a baseline produced with a different physical configuration."""

    exact_fields = (
        "system",
        "atoms",
        "seed",
        "repeat",
        "checkpoint_sha256",
        "structure_sha256",
    )
    float_fields = ("temperature_K", "timestep_fs", "taut_fs")
    mismatches: list[str] = []
    for field in exact_fields:
        if reference.get(field) != current.get(field):
            mismatches.append(
                f"{field}: baseline={reference.get(field)!r}, "
                f"current={current.get(field)!r}"
            )
    for field in float_fields:
        try:
            matches = float(reference[field]) == float(current[field])
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            mismatches.append(
                f"{field}: baseline={reference.get(field)!r}, "
                f"current={current.get(field)!r}"
            )
    if mismatches:
        raise ValueError(
            "Baseline reference metadata mismatch: " + "; ".join(mismatches)
        )


def compare_checkpoint_energies(
    current_energies: dict[int, float],
    reference: dict[str, Any],
) -> tuple[dict[str, float | bool | None], bool]:
    """Compare total structure energies and enforce the 1/50-step limits."""

    fields: dict[str, float | bool | None] = {
        "energy_step_1_atol_eV": ENERGY_ERROR_LIMITS_EV[1],
        "energy_step_50_atol_eV": ENERGY_ERROR_LIMITS_EV[50],
    }
    validation_pass = True
    for step in ENERGY_CHECKPOINT_STEPS:
        baseline_value = reference.get(energy_field(step))
        current_value = current_energies.get(step)
        fields[baseline_energy_field(step)] = baseline_value
        if baseline_value is None or current_value is None:
            fields[energy_error_field(step)] = None
            if step in ENERGY_ERROR_LIMITS_EV and current_value is not None:
                validation_pass = False
            continue
        error = abs(float(current_value) - float(baseline_value))
        fields[energy_error_field(step)] = error
        if step in ENERGY_ERROR_LIMITS_EV:
            # The requested contract is strict: error must be less than the limit.
            validation_pass &= error < ENERGY_ERROR_LIMITS_EV[step]
    fields["energy_validation_pass"] = validation_pass
    return fields, validation_pass
