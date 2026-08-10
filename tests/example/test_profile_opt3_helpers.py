from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys


EXAMPLE = Path(__file__).resolve().parents[2] / "example"
sys.path.insert(0, str(EXAMPLE))
MODULE_PATH = EXAMPLE / "profile_opt3.py"
SPEC = importlib.util.spec_from_file_location("profile_opt3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_two_graph_replay_contract():
    stats = {
        "cuda_graph_capture_count": 2,
        "cuda_graph_production_replays": 1001,
        "cuda_graph_production_calls": 1001,
        "cuda_graph_builder_production_replays": 1001,
        "cuda_graph_model_production_replays": 1001,
        "cuda_graph_capacity_misses": 0,
        "cuda_graph_replay_output_addresses_stable": True,
    }

    fields = PROFILE._graph_invariants(
        "builder-cg-model-cg", stats, steps=1000
    )

    assert fields["graph_invariants_pass"] is True


def test_graph_contract_rejects_production_recapture():
    stats = {
        "cuda_graph_capture_count": 1,
        "cuda_graph_production_capture_count": 1,
        "cuda_graph_production_replays": 1001,
        "cuda_graph_production_calls": 1001,
        "cuda_graph_capacity_misses": 0,
        "cuda_graph_replay_output_addresses_stable": True,
    }

    fields = PROFILE._graph_invariants(
        "whole-step-cg", stats, steps=1000
    )

    assert fields["graph_invariants_pass"] is False


def test_profile_tsv_expands_schema_without_misaligned_rows(tmp_path: Path):
    path = tmp_path / "profile_runs.tsv"
    PROFILE.append_profile_tsv(path, {"backend": "first", "a": 1})
    PROFILE.append_profile_tsv(path, {"backend": "second", "b": 2})

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert rows == [
        {"backend": "first", "a": "1", "b": ""},
        {"backend": "second", "a": "", "b": "2"},
    ]
