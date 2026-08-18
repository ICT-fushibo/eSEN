from __future__ import annotations

import pytest
import torch

from fairchem.core.applications.esen_opt4_model_fusion import (
    FusedEnergyBlock,
    FusedGateActivation,
    FusedRadialMLP,
    FusedRMSNormSH,
    FusedSpectralAtomwise,
    UnsupportedFusionConfigError,
    gather_cat_wigner,
    model_fusion_available,
    parse_model_fusions,
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
CUDA_TRITON = torch.cuda.is_available() and model_fusion_available()


def test_parse_model_fusions_is_ordered_and_strict():
    assert parse_model_fusions("gate,gather-wigner,gate") == (
        "gather-wigner",
        "gate",
    )
    with pytest.raises(UnsupportedFusionConfigError, match="unknown"):
        parse_model_fusions("unknown")


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_gather_wigner_forward_and_input_gradients_match_torch():
    torch.manual_seed(42)
    device = torch.device("cuda")
    nodes, edges, channels = 7, 9, 128
    mask = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        device=device,
    )
    edge_index = torch.randint(0, nodes, (2, edges), device=device)
    x_ref = torch.randn(nodes, 16, channels, device=device, requires_grad=True)
    w_ref = torch.randn(edges, 16, 16, device=device, requires_grad=True)
    # Match layouts seen inside the real eSEN forward, where intermediate
    # views are not guaranteed to be contiguous.
    x_fused = (
        x_ref.detach().transpose(1, 2).contiguous().transpose(1, 2)
        .requires_grad_(True)
    )
    w_fused = (
        w_ref.detach().transpose(1, 2).contiguous().transpose(1, 2)
        .requires_grad_(True)
    )
    assert not x_fused.is_contiguous()
    assert not w_fused.is_contiguous()
    grad = torch.randn(edges, len(mask), 2 * channels, device=device)

    gathered = torch.cat((x_ref[edge_index[0]], x_ref[edge_index[1]]), dim=2)
    expected = torch.bmm(w_ref[:, mask, :], gathered)
    expected.backward(grad)
    actual = gather_cat_wigner(x_fused, edge_index, w_fused, mask)
    actual.backward(grad)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(w_fused.grad, w_ref.grad, rtol=RTOL, atol=ATOL)


def _reference_reverse(message, wigner, mask, distance, target, base, cutoff, scale):
    rotated = torch.bmm(wigner[:, :, mask], message)
    x = distance / cutoff
    envelope = torch.where(
        x < 1.0,
        1.0 - 21.0 * x**5 + 35.0 * x**6 - 15.0 * x**7,
        torch.zeros_like(x),
    )
    out = base.clone()
    out.index_add_(0, target, rotated * (envelope * scale).view(-1, 1, 1))
    return out


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_reverse_envelope_scatter_forward_and_gradients_match_torch():
    torch.manual_seed(42)
    device = torch.device("cuda")
    nodes, edges, channels = 6, 11, 128
    mask = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        device=device,
    )
    target = torch.randint(0, nodes, (edges,), device=device)
    values = [
        torch.randn(edges, len(mask), channels, device=device),
        torch.randn(edges, 16, 16, device=device),
        (torch.rand(edges, device=device) * 6.5).reshape(-1, 1),
        torch.randn(nodes, 16, channels, device=device),
    ]
    ref = [value.detach().clone().requires_grad_(True) for value in values]
    fused = [
        values[0].detach().transpose(1, 2).contiguous().transpose(1, 2)
        .requires_grad_(True),
        values[1].detach().transpose(1, 2).contiguous().transpose(1, 2)
        .requires_grad_(True),
        torch.cat((values[2].detach(), values[2].detach()), dim=1)[:, :1]
        .requires_grad_(True),
        values[3].detach().transpose(1, 2).contiguous().transpose(1, 2)
        .requires_grad_(True),
    ]
    assert all(not value.is_contiguous() for value in fused)
    grad = torch.randn(nodes, 16, channels, device=device)
    expected = _reference_reverse(
        ref[0], ref[1], mask, ref[2], target, ref[3], 6.0, 0.2
    )
    expected.backward(grad)
    actual = reverse_envelope_scatter(
        fused[0], fused[1], mask, fused[2], target, fused[3], 6.0, 0.2
    )
    actual.backward(grad)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    for actual_input, expected_input in zip(fused, ref):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=RTOL, atol=ATOL
        )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_rmsnorm_forward_and_input_gradient_match_torch():
    torch.manual_seed(42)
    original = EquivariantRMSNormArraySphericalHarmonicsV2(3, 128).cuda().eval()
    original.requires_grad_(False)
    fused = FusedRMSNormSH(original)
    x_ref = torch.randn(5, 16, 128, device="cuda", requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_ref)
    expected = original(x_ref)
    expected.backward(grad)
    actual = fused(x_fused)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
@pytest.mark.parametrize("mmax,coefficients", [(2, 14), (3, 16)])
def test_gate_forward_and_both_input_gradients_match_torch(mmax, coefficients):
    torch.manual_seed(42)
    original = GateActivation(3, mmax, 128).cuda()
    fused = FusedGateActivation(original)
    gate_ref = torch.randn(6, 3 * 128, device="cuda", requires_grad=True)
    x_ref = torch.randn(6, coefficients, 128, device="cuda", requires_grad=True)
    gate_fused = gate_ref.detach().clone().requires_grad_(True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_ref)
    expected = original(gate_ref, x_ref)
    expected.backward(grad)
    actual = fused(gate_fused, x_fused)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(gate_fused.grad, gate_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_fused_gate_forward_backward_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    original = GateActivation(3, 2, 128).cuda()
    fused = FusedGateActivation(original)
    gate = torch.randn(8, 384, device="cuda", requires_grad=True)
    x = torch.randn(8, 14, 128, device="cuda", requires_grad=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.autograd.grad(fused(gate, x).sum(), (gate, x))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fused(gate, x)
        gradients = torch.autograd.grad(output.sum(), (gate, x))
    addresses = (output.data_ptr(), gradients[0].data_ptr(), gradients[1].data_ptr())
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == (
        output.data_ptr(), gradients[0].data_ptr(), gradients[1].data_ptr()
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_radial_mlp_forward_and_input_gradient_match_torch():
    torch.manual_seed(42)
    original = RadialMLP([64, 32, 32, 48]).cuda().eval()
    original.requires_grad_(False)
    fused = FusedRadialMLP(original)
    x_ref = torch.randn(37, 64, device="cuda", requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(37, 48, device="cuda")
    expected = original(x_ref)
    expected.backward(grad)
    actual = fused(x_fused)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so3_mlp_forward_and_input_gradient_match_torch():
    torch.manual_seed(42)
    original = SpectralAtomwise(128, 128, 3, 3, None).cuda().eval()
    original.requires_grad_(False)
    fused = FusedSpectralAtomwise(original)
    x_ref = torch.randn(7, 16, 128, device="cuda", requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(7, 16, 128, device="cuda")
    expected = original(x_ref)
    expected.backward(grad)
    actual = fused(x_fused)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_energy_mlp_forward_and_input_gradient_match_torch():
    torch.manual_seed(42)
    original = torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 1),
    ).cuda().eval()
    original.requires_grad_(False)
    fused = FusedEnergyBlock(original)
    x_ref = torch.randn(11, 16, 128, device="cuda")[:, :1, :].squeeze(1).requires_grad_(True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(11, 1, device="cuda")
    expected = original(x_ref)
    expected.backward(grad)
    actual = fused(x_fused)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_radial_mlp_forward_backward_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    original = RadialMLP([64, 32, 32, 48]).cuda().eval()
    original.requires_grad_(False)
    fused = FusedRadialMLP(original)
    x = torch.randn(24, 64, device="cuda", requires_grad=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.autograd.grad(fused(x).sum(), (x,))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fused(x)
        gradients = torch.autograd.grad(output.sum(), (x,))
    addresses = (output.data_ptr(), gradients[0].data_ptr())
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == (output.data_ptr(), gradients[0].data_ptr())
