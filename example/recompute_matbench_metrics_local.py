#!/usr/bin/env python3
"""Recompute Matbench MD observables from an existing eSEN rollout.

This utility is intentionally independent of the eSEN/Fairchem runtime.  It
loads the public Matbench metric implementation directly, so a CPU-only
workstation can evaluate completed trajectory HDF5 files even when the
rollout environment does not have torch-geometric installed.  The source
rollout directory is never modified.

The command copies the run JSON files and trajectories into a new directory,
rewrites trajectory paths to the local copies, computes the official RDF,
ADF, vDOS and (when both trajectories contain stress) pressure metrics, and
writes compact alignment records against the baseline backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


BACKENDS = ("baseline", "opt1", "opt2", "opt3", "opt4")
PUBLIC_METRIC_KEYS = (
    "rdf_error",
    "adf_error",
    "vdos_error",
    "pressure_mae",
    "pressure_wasserstein",
    "pressure_error",
)
ALIGNMENT_THRESHOLDS = {
    "rdf_error": 0.1,
    "adf_error": 0.1,
    "vdos_error": 0.5,
    "pressure_mae": 0.02,
    "pressure_wasserstein": 0.02,
    "pressure_error": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-h5", type=Path, required=True)
    parser.add_argument("--matbench-repo", type=Path, required=True)
    parser.add_argument(
        "--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_runner(repo_root: Path):
    """Load eSEN's compatibility loader without importing the Fairchem stack."""

    runner_path = Path(__file__).resolve().parent / "run_esen_matbench.py"
    spec = importlib.util.spec_from_file_location("esen_matbench_runner_local", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # The compatibility loader expects the source checkout on sys.path, but
    # does not import the package initializer when the current checkout targets
    # a newer Python than the rollout environment.
    sys.path.insert(0, str(repo_root.resolve()))
    read_reference, _evaluate = runner._load_public_metric_functions(repo_root.resolve())
    metric_module = sys.modules["matbench_discovery.metrics.md"]
    trajectory_module = sys.modules["matbench_discovery.trajectory"]
    return runner, read_reference, metric_module, trajectory_module.Trajectory


def _read_system_metadata(reference_h5: Path, requested: set[str]) -> list[Any]:
    import h5py

    systems: list[Any] = []
    with h5py.File(reference_h5, "r") as handle:
        names = sorted(name for name, value in handle.items() if isinstance(value, h5py.Group))
        missing = requested - set(names)
        if missing:
            raise KeyError(f"Missing systems in reference HDF5: {sorted(missing)}")
        for name in names:
            if name not in requested:
                continue
            group = handle[name]
            systems.append(
                SimpleNamespace(
                    name=name,
                    atomic_numbers=np.asarray(group["atomic_numbers"][:], dtype=np.int64),
                    reference_frames=int(group["positions"].shape[0]),
                    reference_dt_fs=float(group.attrs["dt_fs"]),
                    temperature_kelvin=float(group.attrs["temperature_kelvin"]),
                    reference_has_stress="stress" in group,
                )
            )
    if not systems:
        raise ValueError("No requested systems found in reference HDF5")
    return systems


def _copy_rollout(source: Path, output: Path, backends: list[str]) -> list[dict[str, Any]]:
    """Copy successful run records and HDF5 files, returning rewritten rows."""

    rows: list[dict[str, Any]] = []
    for backend in backends:
        run_dir = source / backend / "runs" / backend
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for result_path in sorted(run_dir.glob("*.json")):
            row = json.loads(result_path.read_text(encoding="utf-8"))
            if row.get("status") != "success":
                continue
            system = str(row["system"])
            source_traj = source / backend / "trajectories" / backend / f"{system}.h5"
            if not source_traj.is_file():
                # Accept an already-local path as a fallback, but never copy a
                # path outside the source tree silently when the expected layout
                # is present.
                candidate = Path(str(row.get("trajectory_path", "")))
                if not candidate.is_file():
                    raise FileNotFoundError(source_traj)
                source_traj = candidate
            target_traj = output / "trajectories" / backend / source_traj.name
            target_traj.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_traj, target_traj)
            rewritten = dict(row)
            rewritten["source_trajectory_path"] = str(source_traj.resolve())
            rewritten["trajectory_path"] = str(target_traj.resolve())
            rewritten["metrics_recomputed_locally"] = True
            target_json = output / "runs" / backend / result_path.name
            _write_json(target_json, rewritten)
            rows.append(rewritten)

        # Keep the original status/log metadata available beside the new records.
        for metadata_name in ("run_status.tsv", "run_metadata.json"):
            source_metadata = source / backend / metadata_name
            if source_metadata.is_file():
                target_metadata = output / backend / metadata_name
                target_metadata.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_metadata, target_metadata)
    if not rows:
        raise FileNotFoundError("No successful run JSON files were found")
    return rows


def _metric_row(
    *,
    backend: str,
    system: Any,
    reference: Any,
    prediction: Any,
    reference_dt: float,
    prediction_dt: float,
    metric_module: Any,
    reference_cache: dict[tuple[str, int, float], dict[str, Any]],
) -> dict[str, Any]:
    matched_frame_counts = metric_module.matched_frame_counts
    n_ref_use, n_pred_use = matched_frame_counts(
        n_ref_frames=reference.n_frames,
        n_pred_frames=prediction.n_frames,
        ref_time_step_fs=reference_dt,
        pred_time_step_fs=prediction_dt,
    )
    if n_ref_use < 4 or n_pred_use < 4:
        raise ValueError(f"Trajectories too short after time matching: {n_ref_use=}, {n_pred_use=}")
    ref = reference[:n_ref_use]
    pred = prediction[:n_pred_use]
    # Some rollout writers preserve the fixed cell at a slightly different
    # floating-point precision than the public reference.  The official code
    # derives the RDF cutoff from the reference only, which can exceed the
    # prediction's minimum-image radius by a few ulps.  Use the common valid
    # radius so both histograms remain well-defined and comparable.
    ref_mic_radius = float(metric_module.min_image_radius(ref.cell, ref.pbc))
    pred_mic_radius = float(metric_module.min_image_radius(pred.cell, pred.pbc))
    safe_r_max = min(ref_mic_radius, pred_mic_radius) * (1.0 - 1e-12)
    cache_key = (system.name, n_ref_use, round(safe_r_max, 12))
    if cache_key not in reference_cache:
        print(f"{backend}/{system.name}: computing cached reference observables", flush=True)
        radii, g_ref = metric_module.calc_rdf(ref, n_bins=500, r_max=safe_r_max)
        angles, adf_ref = metric_module.calc_adf(ref)
        ref_velocities = metric_module.calc_velocities(ref, time_step_fs=reference_dt)
        freqs_ref, vdos_ref = metric_module.calc_vdos(
            ref_velocities, time_step_fs=reference_dt
        )
        ref_pressures = (
            metric_module.get_trajectory_pressures(ref)
            if ref.stress is not None
            else None
        )
        reference_cache[cache_key] = {
            "radii": radii,
            "g_ref": g_ref,
            "angles": angles,
            "adf_ref": adf_ref,
            "freqs_ref": freqs_ref,
            "vdos_ref": vdos_ref,
            "pressures_ref": ref_pressures,
        }
    cached = reference_cache[cache_key]
    print(f"{backend}/{system.name}: computing prediction observables", flush=True)
    radii = cached["radii"]
    _, g_pred = metric_module.calc_rdf(pred, n_bins=500, r_max=safe_r_max)
    _, adf_pred = metric_module.calc_adf(pred)
    pred_velocities = metric_module.calc_velocities(pred, time_step_fs=prediction_dt)
    freqs_pred, vdos_pred = metric_module.calc_vdos(
        pred_velocities, time_step_fs=prediction_dt
    )
    values: dict[str, Any] = {
        "rdf_error": float(
            metric_module.calc_rdf_error(cached["radii"], cached["g_ref"], g_pred)
        ),
        "adf_error": float(
            metric_module.calc_adf_error(cached["angles"], cached["adf_ref"], adf_pred)
        ),
        "vdos_error": float(
            metric_module.calc_vdos_error(
                cached["freqs_ref"], cached["vdos_ref"], freqs_pred, vdos_pred
            )
        ),
    }
    pressures_ref = cached["pressures_ref"]
    if pressures_ref is not None and pred.stress is not None:
        values.update(
            metric_module.calc_pressure_metrics(
                pressures_ref, metric_module.get_trajectory_pressures(pred)
            )
        )
    else:
        values.update({key: float("nan") for key in PUBLIC_METRIC_KEYS[3:]})
    # Keep the alignment metadata identical to the runner's
    # ``matched_trajectory_window`` helper.  The public metric module itself
    # only exposes matched_frame_counts.
    from fractions import Fraction

    ref_fraction = Fraction(str(reference_dt))
    pred_fraction = Fraction(str(prediction_dt))
    common_denominator = math.lcm(ref_fraction.denominator, pred_fraction.denominator)
    ref_ticks = ref_fraction.numerator * (common_denominator // ref_fraction.denominator)
    pred_ticks = pred_fraction.numerator * (common_denominator // pred_fraction.denominator)
    common_ticks = math.lcm(ref_ticks, pred_ticks)
    ref_stride = common_ticks // ref_ticks
    pred_stride = common_ticks // pred_ticks
    common_intervals = min(
        (reference.n_frames - 1) // ref_stride,
        (prediction.n_frames - 1) // pred_stride,
    )
    matched_duration = float(Fraction(common_intervals * common_ticks, common_denominator))
    return {
        "backend": backend,
        "system": system.name,
        "temperature_kelvin": system.temperature_kelvin,
        "n_atoms": int(system.atomic_numbers.size),
        "metric_status": "computed",
        "reference_frames_available": int(reference.n_frames),
        "prediction_frames_available": int(prediction.n_frames),
        "reference_frames_used": int(n_ref_use),
        "prediction_frames_used": int(n_pred_use),
        "reference_stride": int(ref_stride),
        "prediction_stride": int(pred_stride),
        "matched_duration_fs": matched_duration,
        **values,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, delimiter: str = "\t") -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(value) for key, value in row.items()})


def _aggregate(rows: list[dict[str, Any]], backend: str, n_systems: int) -> dict[str, Any]:
    selected = [row for row in rows if row.get("backend") == backend]
    summary: dict[str, Any] = {
        "backend": backend,
        "n_completed_systems": sum(row.get("metric_status") == "computed" for row in selected),
        "n_requested_systems": n_systems,
        "n_metric_errors": sum(row.get("metric_status") == "error" for row in selected),
    }
    summary["n_systems"] = summary["n_completed_systems"]
    for key in PUBLIC_METRIC_KEYS:
        finite = [float(row[key]) for row in selected if row.get(key) is not None and np.isfinite(row.get(key))]
        summary[key] = float(np.mean(finite)) if finite else None
    summary["pressure_status"] = "available" if summary["pressure_mae"] is not None else "unavailable_stress_not_computed"
    return summary


def _make_alignment(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(row["system"], row["backend"]): row for row in rows if row.get("metric_status") == "computed"}
    alignment_rows: list[dict[str, Any]] = []
    system_status: dict[str, bool] = {}
    for system in sorted({row["system"] for row in rows}):
        baseline = by_key.get((system, "baseline"))
        if baseline is None:
            continue
        for backend in BACKENDS:
            current = by_key.get((system, backend))
            if current is None:
                continue
            out: dict[str, Any] = {"system": system, "backend": backend}
            passes: list[bool] = []
            for key in PUBLIC_METRIC_KEYS:
                base_value = baseline.get(key)
                current_value = current.get(key)
                out[f"baseline_{key}"] = base_value
                out[f"{backend}_{key}"] = current_value
                if base_value is None or current_value is None or not np.isfinite(base_value) or not np.isfinite(current_value):
                    out[f"{key}_abs_delta"] = None
                    out[f"{key}_threshold"] = ALIGNMENT_THRESHOLDS[key]
                    out[f"{key}_within_threshold"] = "unavailable"
                else:
                    delta = abs(float(current_value) - float(base_value))
                    passed = delta <= ALIGNMENT_THRESHOLDS[key]
                    passes.append(passed)
                    out[f"{key}_abs_delta"] = delta
                    out[f"{key}_threshold"] = ALIGNMENT_THRESHOLDS[key]
                    out[f"{key}_within_threshold"] = passed
            out["alignment_pass"] = all(passes) if passes else True
            alignment_rows.append(out)
            if backend == "opt4":
                system_status[system] = bool(out["alignment_pass"])
    return alignment_rows, {
        "opt4_system_alignment": system_status,
        "opt4_all_systems_within_threshold": all(system_status.values()) if system_status else False,
        "thresholds": ALIGNMENT_THRESHOLDS,
    }


def main() -> int:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    reference_h5 = args.reference_h5.resolve()
    matbench_repo = args.matbench_repo.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not reference_h5.is_file():
        raise FileNotFoundError(reference_h5)
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    runner, read_reference, metric_module, _Trajectory = _load_runner(matbench_repo)
    rows = _copy_rollout(source, output, args.backend)
    systems = _read_system_metadata(reference_h5, {str(row["system"]) for row in rows})
    system_map = {system.name: system for system in systems}
    reference_cache: dict[tuple[str, int, float], dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in rows:
        backend = str(row["backend"])
        system_name = str(row["system"])
        system = system_map[system_name]
        try:
            reference, reference_dt, temperature = read_reference(str(reference_h5), system_name)
            prediction_path = Path(row["trajectory_path"])
            prediction = runner._read_prediction(prediction_path)
            # The saved-frame interval is part of the HDF5 schema.  Falling
            # back to the 10-step x 0.25 fs protocol keeps older records
            # readable when the attribute is absent.
            import h5py

            with h5py.File(prediction_path, "r") as handle:
                prediction_dt = float(handle.attrs.get("dt_fs", 2.5))
            result = _metric_row(
                backend=backend,
                system=system,
                reference=reference,
                prediction=prediction,
                reference_dt=float(reference_dt),
                prediction_dt=prediction_dt,
                metric_module=metric_module,
                reference_cache=reference_cache,
            )
            result["temperature_kelvin"] = float(temperature)
        except Exception as exc:
            result = {
                "backend": backend,
                "system": system_name,
                "metric_status": "error",
                "metric_error": f"{type(exc).__name__}: {exc}",
            }
            print(f"ERROR {backend}/{system_name}: {result['metric_error']}", file=sys.stderr, flush=True)
        metric_rows.append(result)

    _write_tsv(output / "matbench_esen_metrics.tsv", metric_rows)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    aggregates: dict[str, Any] = {"status": "computed"}
    for backend in args.backend:
        per_system = [row for row in metric_rows if row.get("backend") == backend]
        _write_tsv(metrics_dir / f"{backend}-per-system.csv", per_system, delimiter=",")
        summary = _aggregate(metric_rows, backend, len(systems))
        aggregates[backend] = summary
        _write_json(metrics_dir / f"{backend}-aggregate.json", summary)

    speed_rows: list[dict[str, Any]] = []
    for row in rows:
        speed_rows.append(
            {
                "system": row.get("system"),
                "backend": row.get("backend"),
                "seconds_per_step": row.get("seconds_per_step"),
                "steps_per_second": row.get("steps_per_second"),
                "speedup_vs_baseline": row.get("speedup_vs_baseline"),
                "status": row.get("status"),
            }
        )
    _write_tsv(output / "matbench_5backend_speedups.tsv", speed_rows)
    alignment_rows, alignment_summary = _make_alignment(metric_rows)
    _write_tsv(output / "matbench_5backend_physical_alignment.tsv", alignment_rows)

    # Preserve the rollout protocol from the first backend report when present,
    # while clearly marking this artifact as a local metrics-only recomputation.
    source_report = next(
        (source / backend / "matbench_esen_report.json" for backend in args.backend if (source / backend / "matbench_esen_report.json").is_file()),
        None,
    )
    protocol = {}
    opt4 = {}
    if source_report is not None:
        original_report = json.loads(source_report.read_text(encoding="utf-8"))
        protocol = original_report.get("protocol", {})
        opt4 = original_report.get("opt4", {})
    report = {
        "schema": 1,
        "benchmark": "matbench-dynamat-v1.0",
        "source_rollout_directory": str(source),
        "reference_h5": str(reference_h5),
        "reference_sha256": _sha256(reference_h5),
        "protocol": protocol,
        "opt4": opt4,
        "runs": rows,
        "public_metrics": aggregates,
        "physical_alignment": alignment_summary,
        "metrics_only_recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics_compute_wall_time_s": time.perf_counter() - started,
        "private_energy_force_status": "unavailable_public_reference_has_no_labels",
    }
    _write_json(output / "matbench_esen_report.json", report)
    _write_json(
        output / "matbench_5backend_aggregate.json",
        {
            "source_rollout_directory": str(source),
            "backends": list(args.backend),
            "systems": [system.name for system in systems],
            "public_metrics": aggregates,
            "physical_alignment": alignment_summary,
            "computed_locally": True,
        },
    )
    md_lines = [
        "# Local Matbench 10k physical statistics",
        "",
        f"Source rollout: `{source}`",
        f"Reference: `{reference_h5}`",
        "",
        "Metrics use the official Matbench DynaMat RDF, ADF, vDOS and pressure definitions.",
        "The reference trajectory is cached per system; the original rollout is unchanged.",
        "",
        "## Speed",
        "",
        "| system | backend | s/step | steps/s | speedup vs baseline |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in speed_rows:
        md_lines.append(
            f"| {row['system']} | {row['backend']} | {row.get('seconds_per_step', '')} | "
            f"{row.get('steps_per_second', '')} | {row.get('speedup_vs_baseline', '')} |"
        )
    md_lines += ["", "## Physical statistics", "", "See `matbench_esen_metrics.tsv` and `matbench_5backend_physical_alignment.tsv`.", ""]
    for backend in args.backend:
        summary = aggregates[backend]
        md_lines.append(
            f"- **{backend}**: RDF {summary['rdf_error']}, ADF {summary['adf_error']}, "
            f"vDOS {summary['vdos_error']}, pressure MAE {summary['pressure_mae']}"
        )
    md_lines += ["", f"Opt4 baseline-alignment pass: `{alignment_summary['opt4_all_systems_within_threshold']}`", ""]
    (output / "matbench_esen_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Local metrics results: {output}")
    print(f"runs={len(rows)} systems={len(systems)} metric_errors={sum(row.get('metric_status') == 'error' for row in metric_rows)}")
    print(f"compute_wall_time_s={time.perf_counter() - started:.3f}")
    return 1 if any(row.get("metric_status") == "error" for row in metric_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
