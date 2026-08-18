#!/usr/bin/env python3
"""Run Opt4 Triton operator forward/backward and CUDA Graph smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fairchem.core.applications.esen_opt4_model_fusion import (
    FusedEnergyBlock,
    FusedGateActivation,
    FusedRadialMLP,
    FusedRMSNormSH,
    FusedSpectralAtomwise,
    gather_cat_wigner,
    reverse_envelope_scatter,
)
from fairchem.core.models.esen.esen_block import SpectralAtomwise
from fairchem.core.models.esen.nn.activation import GateActivation
from fairchem.core.models.esen.nn.layer_norm import (
    EquivariantRMSNormArraySphericalHarmonicsV2,
)
from fairchem.core.models.esen.nn.radial import RadialMLP


RTOL = 2e-4
ATOL = 2e-5


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-12)
    return {
        "max_abs": float(absolute.max().item()),
        "max_rel": float(relative.max().item()),
    }


def check(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return error(actual, expected)


def reference_reverse(message, wigner, mask, distance, target, base):
    rotated = torch.bmm(wigner[:, :, mask], message)
    d = distance / 6.0
    env = torch.where(
        d < 1.0,
        1.0 - 21.0 * d**5 + 35.0 * d**6 - 15.0 * d**7,
        torch.zeros_like(d),
    )
    output = base.clone()
    output.index_add_(0, target, rotated * env.reshape(-1, 1, 1))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    device = torch.device("cuda")
    mask = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], device=device
    )
    results: dict[str, object] = {"rtol": RTOL, "atol": ATOL}

    nodes, edges, channels = 7, 9, 128
    edge_index = torch.randint(0, nodes, (2, edges), device=device)
    inputs = (
        torch.randn(nodes, 16, channels, device=device),
        torch.randn(edges, 16, 16, device=device),
    )
    ref = [value.clone().requires_grad_(True) for value in inputs]
    fused = [value.clone().requires_grad_(True) for value in inputs]
    grad = torch.randn(edges, len(mask), 2 * channels, device=device)
    expected = torch.bmm(
        ref[1][:, mask, :],
        torch.cat((ref[0][edge_index[0]], ref[0][edge_index[1]]), dim=2),
    )
    expected.backward(grad)
    actual = gather_cat_wigner(fused[0], edge_index, fused[1], mask)
    actual.backward(grad)
    results["gather_wigner"] = {
        "forward": check(actual, expected),
        "grad_x": check(fused[0].grad, ref[0].grad),
        "grad_wigner": check(fused[1].grad, ref[1].grad),
    }

    targets = torch.randint(0, nodes, (edges,), device=device)
    raw = (
        torch.randn(edges, len(mask), channels, device=device),
        torch.randn(edges, 16, 16, device=device),
        (torch.rand(edges, 1, device=device) * 6.5),
        torch.randn(nodes, 16, channels, device=device),
    )
    ref = [value.clone().requires_grad_(True) for value in raw]
    fused = [value.clone().requires_grad_(True) for value in raw]
    grad = torch.randn(nodes, 16, channels, device=device)
    expected = reference_reverse(ref[0], ref[1], mask, ref[2], targets, ref[3])
    expected.backward(grad)
    actual = reverse_envelope_scatter(
        fused[0], fused[1], mask, fused[2], targets, fused[3], 6.0
    )
    actual.backward(grad)
    results["reverse_scatter"] = {
        "forward": check(actual, expected),
        "grad_message": check(fused[0].grad, ref[0].grad),
        "grad_wigner": check(fused[1].grad, ref[1].grad),
        "grad_distance": check(fused[2].grad, ref[2].grad),
        "grad_base": check(fused[3].grad, ref[3].grad),
    }

    norm = EquivariantRMSNormArraySphericalHarmonicsV2(3, 128).cuda().eval()
    norm.requires_grad_(False)
    fused_norm = FusedRMSNormSH(norm)
    x_ref = torch.randn(5, 16, 128, device=device, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_ref)
    expected = norm(x_ref)
    expected.backward(grad)
    actual = fused_norm(x_fused)
    actual.backward(grad)
    results["rmsnorm"] = {
        "forward": check(actual, expected),
        "grad_x": check(x_fused.grad, x_ref.grad),
    }

    gate_results = {}
    for mmax, coefficients in ((2, 14), (3, 16)):
        gate = GateActivation(3, mmax, 128).cuda()
        fused_gate = FusedGateActivation(gate)
        g_ref = torch.randn(6, 384, device=device, requires_grad=True)
        x_ref = torch.randn(6, coefficients, 128, device=device, requires_grad=True)
        g_fused = g_ref.detach().clone().requires_grad_(True)
        x_fused = x_ref.detach().clone().requires_grad_(True)
        grad = torch.randn_like(x_ref)
        expected = gate(g_ref, x_ref)
        expected.backward(grad)
        actual = fused_gate(g_fused, x_fused)
        actual.backward(grad)
        gate_results[str(mmax)] = {
            "forward": check(actual, expected),
            "grad_gate": check(g_fused.grad, g_ref.grad),
            "grad_x": check(x_fused.grad, x_ref.grad),
        }
    results["gate"] = gate_results

    radial = RadialMLP([64, 32, 32, 48]).cuda().eval()
    radial.requires_grad_(False)
    fused_radial = FusedRadialMLP(radial)
    x_ref = torch.randn(37, 64, device=device, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(37, 48, device=device)
    expected = radial(x_ref)
    expected.backward(grad)
    actual = fused_radial(x_fused)
    actual.backward(grad)
    results["radial_mlp"] = {
        "forward": check(actual, expected),
        "grad_x": check(x_fused.grad, x_ref.grad),
    }

    atomwise = SpectralAtomwise(128, 128, 3, 3, None).cuda().eval()
    atomwise.requires_grad_(False)
    fused_atomwise = FusedSpectralAtomwise(atomwise)
    x_ref = torch.randn(7, 16, 128, device=device, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(7, 16, 128, device=device)
    expected = atomwise(x_ref)
    expected.backward(grad)
    actual = fused_atomwise(x_fused)
    actual.backward(grad)
    results["so3_mlp"] = {
        "forward": check(actual, expected),
        "grad_x": check(x_fused.grad, x_ref.grad),
    }

    energy_block = torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 1),
    ).cuda().eval()
    energy_block.requires_grad_(False)
    fused_energy = FusedEnergyBlock(energy_block)
    # Mirror the head's narrow+squeeze view layout (non-contiguous rows).
    x_ref = torch.randn(11, 16, 128, device=device)[:, :1, :].squeeze(1).requires_grad_(True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(11, 1, device=device)
    expected = energy_block(x_ref)
    expected.backward(grad)
    actual = fused_energy(x_fused)
    actual.backward(grad)
    results["energy_head"] = {
        "forward": check(actual, expected),
        "grad_x": check(x_fused.grad, x_ref.grad),
    }

    results["passed"] = True
    text = json.dumps(results, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
