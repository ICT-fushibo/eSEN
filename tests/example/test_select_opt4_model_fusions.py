from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "example" / "select_opt4_model_fusions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("select_opt4_model_fusions", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    stage: str,
    repeat: int,
    seconds: float,
    *,
    scope: str = "whole-step",
    peak_reserved_gib: float = 1.0,
    system: str = "Cu32",
    temperature: int = 300,
    tf32: bool | None = None,
) -> dict[str, object]:
    if scope == "model-only":
        base_backend = "esen_gpu_resident_model_cg"
        fused_backend = "esen_gpu_resident_model_cg_opt4"
    else:
        base_backend = "esen_gpu_resident_whole_step_cg"
        fused_backend = "esen_gpu_resident_whole_step_cg_opt4"
    record = {
        "backend": base_backend if stage.endswith("_base") else fused_backend,
        "kernel_fusion_stage": stage,
        "system": system,
        "temperature_K": temperature,
        "repeat": repeat,
        "seconds_per_step": seconds,
        "engineering_validation_pass": True,
        "graph_invariants_pass": True,
        "cuda_graph_capacity_misses": 0,
        "cuda_graph_capture_count": 1,
        "peak_reserved_gib": peak_reserved_gib,
    }
    if tf32 is not None:
        record.update(
            {
                "tf32": tf32,
                "tf32_mode_requested": "on" if tf32 else "off",
                "tf32_matmul_allowed": tf32,
                "tf32_cudnn_allowed": tf32,
                "float32_matmul_precision": "high" if tf32 else "highest",
                "tf32_config_verified": True,
            }
        )
    return record


def _write_status_tsv(root: Path, scope: str, stage: str, status: str) -> None:
    header = (
        "scope\tvariant\tfusion_stage\tmodel_fusions\tsystem\t"
        "temperature_K\trepeat\trun_name\tstatus\texit_code\t"
        "process_wall_time_s\n"
    )
    rows = [
        f"{scope}\tcandidate\t{stage}\tgather-wigner\tCu32\t300\t{repeat}\t"
        f"run_{repeat}\t{status}\t0\t1.0\n"
        for repeat in range(1, 6)
    ]
    (root / "run_status.tsv").write_text(header + "".join(rows), encoding="utf-8")


def _status_row(
    scope: str,
    variant: str,
    stage: str,
    system: str,
    temperature: int,
    repeat: int,
    status: str,
) -> str:
    exit_code = 0 if status in {"success", "validation_failed"} else 42
    return (
        f"{scope}\t{variant}\t{stage}\tgather-wigner\t{system}\t"
        f"{temperature}\t{repeat}\trun_{stage}_{system}_{repeat}\t{status}\t"
        f"{exit_code}\t1.0\n"
    )


def test_selector_accepts_stable_candidate_despite_validation_status(
    tmp_path, monkeypatch
):
    # Energy/force-vs-baseline errors are telemetry only: a validation_failed
    # shell status must not block a structurally healthy, stably faster
    # candidate under the current acceptance policy.
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for repeat in range(1, 6):
        records = (
            _record("KF2_base", repeat, 1.0),
            _record("KF2", repeat, 0.98),
        )
        for index, record in enumerate(records):
            (result_dir / f"{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
    _write_status_tsv(tmp_path, "whole-step", "KF2", "validation_failed")
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT), "--input-dir", str(tmp_path),
            "--scope", "whole-step",
            "--base-stage", "KF2_base", "--candidate-stage", "KF2",
            "--candidate-fusion", "gather-wigner", "--output", str(output),
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["accepted"] is True
    assert result["accepted_after"] == ["gather-wigner"]


def test_selector_rejects_regression(tmp_path, monkeypatch):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for repeat in range(1, 6):
        for index, record in enumerate(
            (_record("KF3_base", repeat, 1.0), _record("KF3", repeat, 1.02))
        ):
            (result_dir / f"{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT), "--input-dir", str(tmp_path),
            "--scope", "whole-step",
            "--base-stage", "KF3_base", "--candidate-stage", "KF3",
            "--candidate-fusion", "reverse-scatter", "--output", str(output),
        ],
    )
    assert module.main() == 1
    assert json.loads(output.read_text(encoding="utf-8"))["accepted"] is False


def test_selector_supports_model_only_scope(tmp_path, monkeypatch):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for repeat in range(1, 6):
        for index, record in enumerate(
            (
                _record(
                    "KF2_base", repeat, 1.0, scope="model-only"
                ),
                _record("KF2", repeat, 0.98, scope="model-only"),
            )
        ):
            (result_dir / f"{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
    _write_status_tsv(tmp_path, "model-only", "KF2", "success")
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT), "--input-dir", str(tmp_path),
            "--scope", "model-only",
            "--base-stage", "KF2_base", "--candidate-stage", "KF2",
            "--candidate-fusion", "gather-wigner", "--output", str(output),
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["scope"] == "model-only"
    assert result["accepted"] is True


def test_selector_rejects_peak_reserved_guardrail(tmp_path, monkeypatch):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for repeat in range(1, 4):
        records = (
            _record("KF9", repeat, 1.0, peak_reserved_gib=2.0),
            _record("KF10", repeat, 0.9, peak_reserved_gib=3.25),
        )
        for index, record in enumerate(records):
            (result_dir / f"{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
    _write_status_tsv(tmp_path, "whole-step", "KF10", "success")
    # Keep status coverage aligned with the three result pairs used here.
    status = (tmp_path / "run_status.tsv").read_text(encoding="utf-8")
    (tmp_path / "run_status.tsv").write_text(
        "\n".join(status.splitlines()[:4]) + "\n", encoding="utf-8"
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT),
            "--input-dir",
            str(tmp_path),
            "--scope",
            "whole-step",
            "--base-stage",
            "KF9",
            "--candidate-stage",
            "KF10",
            "--candidate-fusion",
            "so2-gate-bridge",
            "--min-paired-repeats",
            "3",
            "--min-faster-directions",
            "3",
            "--maximum-peak-reserved-increase-gib",
            "1.0",
            "--output",
            str(output),
        ],
    )
    assert module.main() == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["reserved_memory_ok"] is False


def test_selector_focus_system_supports_multiple_temperatures(
    tmp_path, monkeypatch
):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    status_rows = []
    for temperature in (300, 800):
        for repeat in range(1, 4):
            records = (
                _record(
                    "KF12",
                    repeat,
                    1.0,
                    system="H2O192",
                    temperature=temperature,
                ),
                _record(
                    "KF12CAP1SAFE",
                    repeat,
                    0.9,
                    system="H2O192",
                    temperature=temperature,
                ),
            )
            for index, record in enumerate(records):
                (result_dir / f"{temperature}_{repeat}_{index}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
            status_rows.append(
                "whole-step\tcandidate\tKF12CAP1SAFE\tauto-safe-capacity\t"
                f"H2O192\t{temperature}\t{repeat}\trun_{temperature}_{repeat}"
                "\tsuccess\t0\t1.0\n"
            )
    header = (
        "scope\tvariant\tfusion_stage\tmodel_fusions\tsystem\t"
        "temperature_K\trepeat\trun_name\tstatus\texit_code\t"
        "process_wall_time_s\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        header + "".join(status_rows), encoding="utf-8"
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT),
            "--input-dir",
            str(tmp_path),
            "--scope",
            "whole-step",
            "--base-stage",
            "KF12",
            "--candidate-stage",
            "KF12CAP1SAFE",
            "--candidate-fusion",
            "auto-safe-capacity",
            "--focus-systems",
            "H2O192",
            "--min-paired-repeats",
            "3",
            "--min-faster-directions",
            "3",
            "--output",
            str(output),
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["focus_stable"] is True
    assert len(result["comparisons"]) == 2


def test_selector_requires_verified_tf32_pair(tmp_path, monkeypatch):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for repeat in range(1, 4):
        for index, record in enumerate(
            (
                _record("OPT4V3_FP32", repeat, 1.0, tf32=False),
                _record("PREC1_TF32", repeat, 0.8, tf32=True),
            )
        ):
            (result_dir / f"{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
    _write_status_tsv(tmp_path, "whole-step", "PREC1_TF32", "success")
    status = (tmp_path / "run_status.tsv").read_text(encoding="utf-8")
    (tmp_path / "run_status.tsv").write_text(
        "\n".join(status.splitlines()[:4]) + "\n", encoding="utf-8"
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT),
            "--input-dir",
            str(tmp_path),
            "--scope",
            "whole-step",
            "--base-stage",
            "OPT4V3_FP32",
            "--candidate-stage",
            "PREC1_TF32",
            "--candidate-fusion",
            "tf32",
            "--require-tf32-pair",
            "--min-paired-repeats",
            "3",
            "--min-faster-directions",
            "3",
            "--output",
            str(output),
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["accepted"] is True
    assert result["precision_configuration_ok"] is True

    candidate = json.loads((result_dir / "1_1.json").read_text(encoding="utf-8"))
    candidate["tf32_matmul_allowed"] = False
    (result_dir / "1_1.json").write_text(json.dumps(candidate), encoding="utf-8")
    assert module.main() == 1
    assert json.loads(output.read_text(encoding="utf-8"))[
        "precision_configuration_ok"
    ] is False


def test_selector_can_require_tf32_on_for_both_stages(tmp_path, monkeypatch):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for repeat in range(1, 4):
        for index, record in enumerate(
            (
                _record("PREC1_TF32", repeat, 1.0, tf32=True),
                _record("KF13_PREC1_TF32", repeat, 0.95, tf32=True),
            )
        ):
            (result_dir / f"{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
    _write_status_tsv(tmp_path, "whole-step", "KF13_PREC1_TF32", "success")
    status = (tmp_path / "run_status.tsv").read_text(encoding="utf-8")
    (tmp_path / "run_status.tsv").write_text(
        "\n".join(status.splitlines()[:4]) + "\n", encoding="utf-8"
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT),
            "--input-dir",
            str(tmp_path),
            "--scope",
            "whole-step",
            "--base-stage",
            "PREC1_TF32",
            "--candidate-stage",
            "KF13_PREC1_TF32",
            "--candidate-fusion",
            "so3-weight-cache",
            "--expected-tf32-mode",
            "on",
            "--min-paired-repeats",
            "3",
            "--min-faster-directions",
            "3",
            "--output",
            str(output),
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["accepted"] is True
    assert result["expected_tf32_mode"] == "on"
    assert result["precision_configuration_ok"] is True


def test_selector_excludes_symmetric_oom_pairs_from_coverage(
    tmp_path, monkeypatch
):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    status_rows = []
    for repeat in range(1, 4):
        for index, record in enumerate(
            (
                _record("OPT4V4_FP32", repeat, 1.0),
                _record("KF16_FP32", repeat, 0.9),
            )
        ):
            (result_dir / f"Cu32_{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        status_rows.extend(
            (
                _status_row(
                    "whole-step", "base", "OPT4V4_FP32", "Cu32", 300,
                    repeat, "success",
                ),
                _status_row(
                    "whole-step", "candidate", "KF16_FP32", "Cu32", 300,
                    repeat, "success",
                ),
                _status_row(
                    "whole-step", "base", "OPT4V4_FP32", "H2O512", 300,
                    repeat, "oom",
                ),
                _status_row(
                    "whole-step", "candidate", "KF16_FP32", "H2O512", 300,
                    repeat, "oom",
                ),
            )
        )
    header = (
        "scope\tvariant\tfusion_stage\tmodel_fusions\tsystem\t"
        "temperature_K\trepeat\trun_name\tstatus\texit_code\t"
        "process_wall_time_s\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        header + "".join(status_rows), encoding="utf-8"
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT), "--input-dir", str(tmp_path),
            "--scope", "whole-step",
            "--base-stage", "OPT4V4_FP32",
            "--candidate-stage", "KF16_FP32",
            "--candidate-fusion", "wigner-so2-tiled-backward",
            "--min-paired-repeats", "3",
            "--min-faster-directions", "3",
            "--output", str(output),
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["accepted"] is True
    assert result["coverage_ok"] is True
    assert result["status_ok"] is True
    assert result["partial_coverage"] is True
    assert result["symmetric_oom_count"] == 3
    assert result["candidate_result_count"] == 3


def test_selector_rejects_asymmetric_oom(tmp_path, monkeypatch):
    module = _load_module()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    status_rows = []
    for repeat in range(1, 4):
        for index, record in enumerate(
            (
                _record("OPT4V4_FP32", repeat, 1.0),
                _record("KF16_FP32", repeat, 0.9),
                _record(
                    "OPT4V4_FP32", repeat, 2.0, system="H2O512"
                ),
            )
        ):
            (result_dir / f"{repeat}_{index}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        status_rows.extend(
            (
                _status_row(
                    "whole-step", "base", "OPT4V4_FP32", "Cu32", 300,
                    repeat, "success",
                ),
                _status_row(
                    "whole-step", "candidate", "KF16_FP32", "Cu32", 300,
                    repeat, "success",
                ),
                _status_row(
                    "whole-step", "base", "OPT4V4_FP32", "H2O512", 300,
                    repeat, "success",
                ),
                _status_row(
                    "whole-step", "candidate", "KF16_FP32", "H2O512", 300,
                    repeat, "oom",
                ),
            )
        )
    header = (
        "scope\tvariant\tfusion_stage\tmodel_fusions\tsystem\t"
        "temperature_K\trepeat\trun_name\tstatus\texit_code\t"
        "process_wall_time_s\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        header + "".join(status_rows), encoding="utf-8"
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT), "--input-dir", str(tmp_path),
            "--scope", "whole-step",
            "--base-stage", "OPT4V4_FP32",
            "--candidate-stage", "KF16_FP32",
            "--candidate-fusion", "wigner-so2-tiled-backward",
            "--min-paired-repeats", "3",
            "--min-faster-directions", "3",
            "--output", str(output),
        ],
    )
    assert module.main() == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["accepted"] is False
    assert result["coverage_ok"] is False
    assert result["status_ok"] is False
    assert result["symmetric_oom_count"] == 0
