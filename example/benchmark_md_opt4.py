#!/usr/bin/env python3
"""Opt4-only entry point for eSEN kernel-fusion benchmarks.

The implementation reuses the validated Opt3 measurement and validation
pipeline, but only permits explicitly named KF1 backends.  Baseline through
Opt3 backend behavior remains available through their original scripts.
"""

from __future__ import annotations

import sys

from benchmark_md_opt3 import entrypoint


ALLOWED_BACKENDS = {
    "fixed-builder-model-cg-kf1",
    "whole-step-cg-kf1",
}


def _selected_backend(argv: list[str]) -> str | None:
    for index, argument in enumerate(argv):
        if argument == "--backend" and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith("--backend="):
            return argument.split("=", 1)[1]
    return None


if __name__ == "__main__":
    backend = _selected_backend(sys.argv[1:])
    help_requested = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
    if not help_requested and backend not in ALLOWED_BACKENDS:
        choices = ", ".join(sorted(ALLOWED_BACKENDS))
        raise SystemExit(f"Opt4 backend must be one of: {choices}")
    raise SystemExit(entrypoint())
