from __future__ import annotations

import copy

import pytest
import torch

from fairchem.core.applications.esen_opt4_model_fusion import (
    FusedEnergyBlock,
    FusedGateActivation,
    FusedRadialMLP,
    FusedRMSNormSH,
    FusedSO2Convolution,
    FusedSO2GateBridge,
    FusedSpectralAtomwise,
    FrozenSO3Linear,
    SO2BlockLinear,
    UnsupportedFusionConfigError,
    _SO2BlockEpilogue,
    _SO2BlockGateBridge,
    _SO2Epilogue,
    _SO2GateBridge,
    _SO2Prepare,
    _SO2PrepareBackwardReduce,
    _WignerSO2Prepare,
    gather_cat_wigner,
    model_fusion_available,
    parse_model_fusions,
    reverse_envelope_scatter,
    wigner_so2_prepare,
    _energy_head_candidates,
    configure_esen_30m_model_fusions,
)
from fairchem.core.models.esen.esen_block import SpectralAtomwise
from fairchem.core.models.esen.nn.activation import GateActivation
from fairchem.core.models.esen.nn.layer_norm import (
    EquivariantRMSNormArraySphericalHarmonicsV2,
)
from fairchem.core.models.esen.nn.radial import RadialMLP
from fairchem.core.models.esen.common.so3 import CoefficientMapping
from fairchem.core.models.esen.nn.so2_layers import SO2_Convolution
from fairchem.core.models.esen.nn.so3_layers import SO3_Linear


RTOL = 2e-4
ATOL = 2e-5
CUDA_TRITON = torch.cuda.is_available() and model_fusion_available()


def _so2_indices(mapping):
    to_m = mapping.to_m.argmax(dim=1).to(dtype=torch.long)
    l_to_m = torch.empty_like(to_m)
    l_to_m[to_m] = torch.arange(14, device=to_m.device)
    return to_m, l_to_m


def _reference_so2_prepare(x, radial, mapping):
    channels = x.shape[2]
    mapped = torch.einsum("nac,ba->nbc", x, mapping.to_m)
    m0 = mapped[:, :4] * radial[:, : 4 * channels].reshape(-1, 4, channels)
    m1 = mapped[:, 4:10].reshape(-1, 2, 3 * channels)
    m1 = m1 * radial[:, 4 * channels : 7 * channels].reshape(
        -1, 1, 3 * channels
    )
    m2 = mapped[:, 10:14].reshape(-1, 2, 2 * channels)
    m2 = m2 * radial[:, 7 * channels :].reshape(-1, 1, 2 * channels)
    return (
        m0.reshape(-1, 4 * channels),
        m1,
        m2,
    )


def _reference_so2_epilogue(m0, m1, m2, mapping, extra_channels):
    channels = m1.shape[2] // 6

    def combine(value, coefficients):
        half = coefficients * channels
        real = value[:, :, :half]
        imag = value[:, :, half:]
        combined_real = real[:, 0:1] - imag[:, 1:2]
        combined_imag = real[:, 1:2] + imag[:, 0:1]
        return torch.cat((combined_real, combined_imag), dim=1).reshape(
            -1, 2 * coefficients, channels
        )

    gating = m0[:, :extra_channels]
    m_order = torch.cat(
        (
            m0[:, extra_channels:].reshape(-1, 4, channels),
            combine(m1, 3),
            combine(m2, 2),
        ),
        dim=1,
    )
    return torch.einsum("nac,ab->nbc", m_order, mapping.to_m), gating


def _reference_so2_gate_bridge(m0, m1, m2, mapping):
    l_order, gating = _reference_so2_epilogue(m0, m1, m2, mapping, 384)
    gate = torch.sigmoid(gating).reshape(-1, 3, 128)
    expand = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        device=m0.device,
        dtype=torch.long,
    )
    activated = torch.cat(
        (
            torch.nn.functional.silu(l_order[:, :1]),
            l_order[:, 1:] * gate.index_select(1, expand),
        ),
        dim=1,
    )
    m_order = torch.einsum("nac,ba->nbc", activated, mapping.to_m)
    return (
        m_order[:, :4].reshape(-1, 512),
        m_order[:, 4:10].reshape(-1, 2, 384),
        m_order[:, 10:14].reshape(-1, 2, 256),
    )


def _reference_so2_block_epilogue(m0, m1, m2, mapping, extra_channels):
    channels = m1.shape[1] // 6
    gating = m0[:, :extra_channels]
    m_order = torch.cat(
        (
            m0[:, extra_channels:].reshape(-1, 4, channels),
            m1.reshape(-1, 6, channels),
            m2.reshape(-1, 4, channels),
        ),
        dim=1,
    )
    return torch.einsum("nac,ab->nbc", m_order, mapping.to_m), gating


def _reference_so2_block_gate_bridge(m0, m1, m2, mapping):
    l_order, gating = _reference_so2_block_epilogue(
        m0, m1, m2, mapping, 384
    )
    gate = torch.sigmoid(gating).reshape(-1, 3, 128)
    expand = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        device=m0.device,
        dtype=torch.long,
    )
    activated = torch.cat(
        (
            torch.nn.functional.silu(l_order[:, :1]),
            l_order[:, 1:] * gate.index_select(1, expand),
        ),
        dim=1,
    )
    m_order = torch.einsum("nac,ba->nbc", activated, mapping.to_m)
    return (
        m_order[:, :4].reshape(-1, 512),
        m_order[:, 4:10].reshape(-1, 2, 384),
        m_order[:, 10:14].reshape(-1, 2, 256),
    )


def _m_degree_index(mapping):
    degree_l = torch.tensor(
        [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        device=mapping.to_m.device,
        dtype=torch.long,
    )
    return degree_l.index_select(0, mapping.to_m.argmax(dim=1))


def test_energy_head_discovery_supports_moduledict_and_legacy_layouts():
    energy = torch.nn.Linear(2, 1)
    other = torch.nn.Linear(2, 1)
    model = torch.nn.Module()
    model.output_heads = torch.nn.ModuleDict({"energy": energy, "other": other})
    assert _energy_head_candidates(model) == [energy, other]

    legacy = torch.nn.Module()
    legacy.head = energy
    assert _energy_head_candidates(legacy) == [energy]


def test_parse_model_fusions_is_ordered_and_strict():
    assert parse_model_fusions("gate,gather-wigner,gate") == (
        "gather-wigner",
        "gate",
    )
    with pytest.raises(UnsupportedFusionConfigError, match="unknown"):
        parse_model_fusions("unknown")
    assert parse_model_fusions("so2-epilogue,rmsnorm") == (
        "rmsnorm",
        "so2-epilogue",
    )
    assert parse_model_fusions("so2-gate-bridge,so2-epilogue") == (
        "so2-epilogue",
        "so2-gate-bridge",
    )
    with pytest.raises(UnsupportedFusionConfigError, match="requires so2-epilogue"):
        parse_model_fusions("so2-gate-bridge")
    assert parse_model_fusions(
        "wigner-so2-bridge,so2-gate-bridge,so2-epilogue"
    ) == (
        "so2-epilogue",
        "so2-gate-bridge",
        "wigner-so2-bridge",
    )
    with pytest.raises(UnsupportedFusionConfigError, match="requires so2-epilogue"):
        parse_model_fusions("wigner-so2-bridge")
    with pytest.raises(UnsupportedFusionConfigError, match="subsumes gather-wigner"):
        parse_model_fusions(
            "gather-wigner,so2-epilogue,so2-gate-bridge,wigner-so2-bridge"
        )
    assert parse_model_fusions(
        "so2-block-gemm,so2-gate-bridge,so2-epilogue"
    ) == (
        "so2-epilogue",
        "so2-gate-bridge",
        "so2-block-gemm",
    )
    with pytest.raises(UnsupportedFusionConfigError, match="requires so2-epilogue"):
        parse_model_fusions("so2-block-gemm")
    assert parse_model_fusions(
        "so2-prepare-backward-reduce,so2-gate-bridge,so2-epilogue"
    ) == (
        "so2-epilogue",
        "so2-gate-bridge",
        "so2-prepare-backward-reduce",
    )
    with pytest.raises(
        UnsupportedFusionConfigError,
        match="requires so2-epilogue and so2-gate-bridge",
    ):
        parse_model_fusions("so2-epilogue,so2-prepare-backward-reduce")
    assert parse_model_fusions(
        "so3-weight-cache,so2-block-gemm,so2-gate-bridge,so2-epilogue"
    ) == (
        "so2-epilogue",
        "so2-gate-bridge",
        "so2-block-gemm",
        "so3-weight-cache",
    )
    with pytest.raises(UnsupportedFusionConfigError, match="do not combine"):
        parse_model_fusions("so3-mlp,so3-weight-cache")


def test_frozen_so3_linear_forward_and_input_gradient_match_reference():
    torch.manual_seed(42)
    original = SO3_Linear(128, 128, lmax=3).eval()
    original.requires_grad_(False)
    cached = FrozenSO3Linear(original)
    # Exercise the wrapper boundary with a non-contiguous logical [N,16,128].
    x_ref = torch.randn(3, 128, 16).transpose(1, 2).requires_grad_(True)
    x_cached = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(3, 16, 128)

    expected = original(x_ref)
    expected.backward(grad)
    address = cached.expanded_weight.data_ptr()
    actual = cached(x_cached)
    actual.backward(grad)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x_cached.grad, x_ref.grad, rtol=0, atol=0)
    assert cached.expanded_weight.data_ptr() == address
    assert cached.expanded_weight.shape == (16, 128, 128)
    assert cached.cached_weight_bytes == 16 * 128 * 128 * 4


def test_frozen_so3_linear_rejects_trainable_parameters():
    original = SO3_Linear(128, 128, lmax=3)
    with pytest.raises(UnsupportedFusionConfigError, match="requires frozen"):
        FrozenSO3Linear(original)


def test_so2_block_linear_forward_and_input_gradient_match_reference():
    torch.manual_seed(42)
    linear = torch.nn.Linear(7, 10, bias=False)
    block = SO2BlockLinear(linear)
    x_ref = torch.randn(5, 2, 7, requires_grad=True)
    x_actual = x_ref.detach().clone().requires_grad_(True)
    raw = linear(x_ref)
    w1_real = raw[:, 0, :5]
    w2_real = raw[:, 0, 5:]
    w1_imag = raw[:, 1, :5]
    w2_imag = raw[:, 1, 5:]
    expected = torch.cat(
        (w1_real - w2_imag, w2_real + w1_imag), dim=1
    )
    actual = block(x_actual)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_actual.grad, x_ref.grad, rtol=RTOL, atol=ATOL)


def test_so2_epilogue_rejects_non_permutation_mapping():
    mapping = CoefficientMapping(3, 2)
    mapping.to_m[1].zero_()
    mapping.to_m[1, 0] = 1.0
    reference = SO2_Convolution(128, 128, 3, 2, mapping)
    with pytest.raises(UnsupportedFusionConfigError, match="permutation"):
        FusedSO2Convolution(reference)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
@pytest.mark.parametrize(
    "prepare", (_SO2Prepare, _SO2PrepareBackwardReduce),
    ids=("atomic", "edge-local-reduce"),
)
def test_so2_prepare_forward_and_gradients_match_torch(prepare):
    torch.manual_seed(42)
    device = torch.device("cuda")
    mapping = CoefficientMapping(3, 2).to(device)
    to_m, _ = _so2_indices(mapping)
    edges, channels = 5, 256
    x_values = torch.randn(edges, 14, channels, device=device)
    x_values[1] = x_values[0]
    x_values[-1].zero_()
    radial_values = torch.randn(edges, 9 * channels, device=device)
    x_ref = x_values.detach().clone().requires_grad_(True)
    radial_ref = radial_values.detach().clone().requires_grad_(True)
    x_actual = (
        x_values.detach().transpose(1, 2).contiguous().transpose(1, 2)
        .requires_grad_(True)
    )
    radial_storage = torch.empty(
        edges, 9 * channels + 1, device=device, dtype=radial_values.dtype
    )
    radial_storage[:, : 9 * channels].copy_(radial_values)
    radial_actual = radial_storage[:, : 9 * channels].detach().requires_grad_(True)
    assert not x_actual.is_contiguous()
    assert not radial_actual.is_contiguous()
    grad = (
        torch.randn(edges, 4 * channels, device=device),
        torch.randn(edges, 2, 3 * channels, device=device),
        torch.randn(edges, 2, 2 * channels, device=device),
    )

    expected = _reference_so2_prepare(x_ref, radial_ref, mapping)
    torch.autograd.backward(expected, grad)
    actual = prepare.apply(
        x_actual, radial_actual, to_m, True, 9 * channels
    )
    torch.autograd.backward(actual, grad)

    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(
            actual_part, expected_part, rtol=RTOL, atol=ATOL
        )
    torch.testing.assert_close(x_actual.grad, x_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        radial_actual.grad, radial_ref.grad, rtol=RTOL, atol=ATOL
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_prepare_backward_reduce_accepts_empty_edges():
    mapping = CoefficientMapping(3, 2).cuda()
    to_m, _ = _so2_indices(mapping)
    x = torch.empty(0, 14, 256, device="cuda", requires_grad=True)
    radial = torch.empty(0, 2304, device="cuda", requires_grad=True)
    outputs = _SO2PrepareBackwardReduce.apply(
        x, radial, to_m, True, 2304
    )
    sum(output.sum() for output in outputs).backward()
    assert [tuple(output.shape) for output in outputs] == [
        (0, 1024),
        (0, 2, 768),
        (0, 2, 512),
    ]
    assert x.grad is not None and x.grad.shape == x.shape
    assert radial.grad is not None and radial.grad.shape == radial.shape


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_prepare_backward_reduce_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    mapping = CoefficientMapping(3, 2).cuda()
    to_m, _ = _so2_indices(mapping)
    x = torch.randn(16, 14, 256, device="cuda", requires_grad=True)
    radial = torch.randn(16, 2304, device="cuda", requires_grad=True)
    inputs = (x, radial)

    def forward():
        return _SO2PrepareBackwardReduce.apply(
            x, radial, to_m, True, 2304
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            outputs = forward()
            torch.autograd.grad(sum(output.sum() for output in outputs), inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = forward()
        gradients = torch.autograd.grad(
            sum(output.sum() for output in outputs), inputs
        )
    addresses = tuple(output.data_ptr() for output in outputs) + tuple(
        gradient.data_ptr() for gradient in gradients
    )
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == tuple(output.data_ptr() for output in outputs) + tuple(
        gradient.data_ptr() for gradient in gradients
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_epilogue_forward_and_gradients_match_torch():
    torch.manual_seed(42)
    device = torch.device("cuda")
    mapping = CoefficientMapping(3, 2).to(device)
    _, l_to_m = _so2_indices(mapping)
    edges, channels, extra = 3, 128, 384
    values = (
        torch.randn(edges, 4 * channels + extra, device=device),
        torch.randn(edges, 2, 6 * channels, device=device),
        torch.randn(edges, 2, 4 * channels, device=device),
    )
    reference_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    actual_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    grad_out = torch.randn(edges, 14, channels, device=device)
    grad_gate = torch.randn(edges, extra, device=device)

    expected = _reference_so2_epilogue(
        *reference_inputs, mapping, extra
    )
    torch.autograd.backward(expected, (grad_out, grad_gate))
    actual = _SO2Epilogue.apply(*actual_inputs, l_to_m, extra)
    torch.autograd.backward(actual, (grad_out, grad_gate))

    torch.testing.assert_close(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)
    for actual_input, expected_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=RTOL, atol=ATOL
        )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_block_epilogue_forward_and_gradients_match_torch():
    torch.manual_seed(42)
    mapping = CoefficientMapping(3, 2).cuda()
    _, l_to_m = _so2_indices(mapping)
    values = (
        torch.randn(4, 896, device="cuda"),
        torch.randn(4, 768, device="cuda"),
        torch.randn(4, 512, device="cuda"),
    )
    reference_inputs = [
        value.detach().clone().requires_grad_(True) for value in values
    ]
    actual_inputs = [
        value.detach().clone().requires_grad_(True) for value in values
    ]
    gradients = (
        torch.randn(4, 14, 128, device="cuda"),
        torch.randn(4, 384, device="cuda"),
    )
    expected = _reference_so2_block_epilogue(
        *reference_inputs, mapping, 384
    )
    torch.autograd.backward(expected, gradients)
    actual = _SO2BlockEpilogue.apply(*actual_inputs, l_to_m, 384)
    torch.autograd.backward(actual, gradients)
    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(
            actual_part, expected_part, rtol=RTOL, atol=ATOL
        )
    for actual_input, reference_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, rtol=RTOL, atol=ATOL
        )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_block_gate_bridge_forward_and_gradients_match_torch():
    torch.manual_seed(42)
    mapping = CoefficientMapping(3, 2).cuda()
    values = (
        torch.randn(5, 896, device="cuda"),
        torch.randn(5, 768, device="cuda"),
        torch.randn(5, 512, device="cuda"),
    )
    values[0][-1].zero_()
    values[1][-1].zero_()
    values[2][-1].zero_()
    reference_inputs = [
        value.detach().clone().requires_grad_(True) for value in values
    ]
    actual_inputs = [
        value.detach().clone().requires_grad_(True) for value in values
    ]
    gradients = (
        torch.randn(5, 512, device="cuda"),
        torch.randn(5, 2, 384, device="cuda"),
        torch.randn(5, 2, 256, device="cuda"),
    )
    expected = _reference_so2_block_gate_bridge(*reference_inputs, mapping)
    torch.autograd.backward(expected, gradients)
    actual = _SO2BlockGateBridge.apply(
        *actual_inputs, _m_degree_index(mapping)
    )
    torch.autograd.backward(actual, gradients)
    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(
            actual_part, expected_part, rtol=RTOL, atol=ATOL
        )
    for actual_input, reference_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, rtol=RTOL, atol=ATOL
        )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_wigner_so2_prepare_forward_and_gradients_match_torch():
    torch.manual_seed(42)
    device = torch.device("cuda")
    mapping = CoefficientMapping(3, 2).to(device)
    to_m, _ = _so2_indices(mapping)
    nodes, edges = 6, 7
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 5, 5], [1, 1, 3, 3, 4, 0, 0]],
        device=device,
        dtype=torch.long,
    )
    out_mask = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14],
        device=device,
        dtype=torch.long,
    )
    x = torch.randn(nodes, 16, 128, device=device)
    wigner = torch.randn(edges, 16, 16, device=device)
    radial = torch.randn(edges, 2304, device=device)
    reference_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (x, wigner, radial)
    ]
    x_storage = torch.empty(nodes, 128, 17, device=device)
    x_storage[:, :, :16].copy_(x.transpose(1, 2))
    x_actual = x_storage[:, :, :16].transpose(1, 2).detach().requires_grad_(True)
    w_actual = (
        wigner.transpose(1, 2).contiguous().transpose(1, 2)
        .detach()
        .requires_grad_(True)
    )
    radial_storage = torch.empty(edges, 2305, device=device)
    radial_storage[:, :2304].copy_(radial)
    radial_actual = radial_storage[:, :2304].detach().requires_grad_(True)
    actual_inputs = [x_actual, w_actual, radial_actual]
    assert not all(value.is_contiguous() for value in actual_inputs)

    gathered = torch.cat(
        (
            reference_inputs[0][edge_index[0]],
            reference_inputs[0][edge_index[1]],
        ),
        dim=2,
    )
    rotated = torch.bmm(reference_inputs[1][:, out_mask, :], gathered)
    expected = _reference_so2_prepare(
        rotated, reference_inputs[2], mapping
    )
    gradients = (
        torch.randn(edges, 1024, device=device),
        torch.randn(edges, 2, 768, device=device),
        torch.randn(edges, 2, 512, device=device),
    )
    torch.autograd.backward(expected, gradients)
    actual = wigner_so2_prepare(
        x_actual,
        edge_index,
        w_actual,
        out_mask,
        radial_actual,
        to_m,
    )
    torch.autograd.backward(actual, gradients)

    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(
            actual_part, expected_part, rtol=RTOL, atol=ATOL
        )
    for actual_input, expected_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=RTOL, atol=ATOL
        )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_wigner_so2_prepare_accepts_empty_edges():
    mapping = CoefficientMapping(3, 2).cuda()
    to_m, _ = _so2_indices(mapping)
    x = torch.randn(3, 16, 128, device="cuda", requires_grad=True)
    edge_index = torch.empty(2, 0, device="cuda", dtype=torch.long)
    wigner = torch.empty(0, 16, 16, device="cuda", requires_grad=True)
    radial = torch.empty(0, 2304, device="cuda", requires_grad=True)
    out_mask = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14],
        device="cuda",
    )
    outputs = _WignerSO2Prepare.apply(
        x, edge_index, wigner, out_mask, radial, to_m
    )
    sum(output.sum() for output in outputs).backward()
    assert [tuple(output.shape) for output in outputs] == [
        (0, 1024),
        (0, 2, 768),
        (0, 2, 512),
    ]
    assert x.grad is not None and torch.count_nonzero(x.grad) == 0
    assert wigner.grad is not None and wigner.grad.shape == wigner.shape
    assert radial.grad is not None and radial.grad.shape == radial.shape


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_wigner_so2_prepare_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    mapping = CoefficientMapping(3, 2).cuda()
    to_m, _ = _so2_indices(mapping)
    x = torch.randn(8, 16, 128, device="cuda", requires_grad=True)
    edge_index = torch.randint(0, 8, (2, 16), device="cuda")
    wigner = torch.randn(16, 16, 16, device="cuda", requires_grad=True)
    radial = torch.randn(16, 2304, device="cuda", requires_grad=True)
    out_mask = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14],
        device="cuda",
    )
    inputs = (x, wigner, radial)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            outputs = wigner_so2_prepare(
                x, edge_index, wigner, out_mask, radial, to_m
            )
            torch.autograd.grad(sum(output.sum() for output in outputs), inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = wigner_so2_prepare(
            x, edge_index, wigner, out_mask, radial, to_m
        )
        gradients = torch.autograd.grad(
            sum(output.sum() for output in outputs), inputs
        )
    addresses = tuple(output.data_ptr() for output in outputs) + tuple(
        gradient.data_ptr() for gradient in gradients
    )
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == tuple(output.data_ptr() for output in outputs) + tuple(
        gradient.data_ptr() for gradient in gradients
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_gate_bridge_forward_and_gradients_match_torch():
    torch.manual_seed(42)
    device = torch.device("cuda")
    mapping = CoefficientMapping(3, 2).to(device)
    edges = 5
    values = (
        torch.randn(edges, 896, device=device),
        torch.randn(edges, 2, 768, device=device),
        torch.randn(edges, 2, 512, device=device),
    )
    # Repeated and zero rows cover dummy-capacity-like inputs.
    for value in values:
        value[1].copy_(value[0])
        value[-1].zero_()
    reference_inputs = [
        value.detach().clone().requires_grad_(True) for value in values
    ]
    actual_inputs = []
    for value in values:
        storage = torch.empty(
            *value.shape[:-1], value.shape[-1] + 1,
            device=device,
            dtype=value.dtype,
        )
        storage[..., : value.shape[-1]].copy_(value)
        actual = storage[..., : value.shape[-1]].detach().requires_grad_(True)
        assert not actual.is_contiguous()
        actual_inputs.append(actual)
    grad = (
        torch.randn(edges, 512, device=device),
        torch.randn(edges, 2, 384, device=device),
        torch.randn(edges, 2, 256, device=device),
    )

    expected = _reference_so2_gate_bridge(*reference_inputs, mapping)
    torch.autograd.backward(expected, grad)
    actual = _SO2GateBridge.apply(
        *actual_inputs, _m_degree_index(mapping)
    )
    torch.autograd.backward(actual, grad)

    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(
            actual_part, expected_part, rtol=RTOL, atol=ATOL
        )
    for actual_input, expected_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=RTOL, atol=ATOL
        )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_gate_bridge_accepts_empty_edges():
    mapping = CoefficientMapping(3, 2).cuda()
    values = (
        torch.empty(0, 896, device="cuda", requires_grad=True),
        torch.empty(0, 2, 768, device="cuda", requires_grad=True),
        torch.empty(0, 2, 512, device="cuda", requires_grad=True),
    )
    outputs = _SO2GateBridge.apply(*values, _m_degree_index(mapping))
    sum(output.sum() for output in outputs).backward()
    assert [tuple(output.shape) for output in outputs] == [
        (0, 512),
        (0, 2, 384),
        (0, 2, 256),
    ]
    assert all(
        value.grad is not None and value.grad.shape == value.shape
        for value in values
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_gate_bridge_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    mapping = CoefficientMapping(3, 2).cuda()
    values = (
        torch.randn(24, 896, device="cuda", requires_grad=True),
        torch.randn(24, 2, 768, device="cuda", requires_grad=True),
        torch.randn(24, 2, 512, device="cuda", requires_grad=True),
    )
    degree = _m_degree_index(mapping)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            outputs = _SO2GateBridge.apply(*values, degree)
            torch.autograd.grad(sum(output.sum() for output in outputs), values)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = _SO2GateBridge.apply(*values, degree)
        gradients = torch.autograd.grad(
            sum(output.sum() for output in outputs), values
        )
    addresses = tuple(output.data_ptr() for output in outputs) + tuple(
        gradient.data_ptr() for gradient in gradients
    )
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == tuple(output.data_ptr() for output in outputs) + tuple(
        gradient.data_ptr() for gradient in gradients
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_gate_bridge_module_validates_and_runs_30m_pair():
    mapping = CoefficientMapping(3, 2).cuda()
    conv1 = SO2_Convolution(
        256,
        128,
        3,
        2,
        mapping,
        internal_weights=False,
        edge_channels_list=[32, 64],
        extra_m0_output_channels=384,
    ).cuda()
    conv2 = SO2_Convolution(
        128, 128, 3, 2, mapping, internal_weights=True
    ).cuda()
    bridge = FusedSO2GateBridge(
        FusedSO2Convolution(conv1), FusedSO2Convolution(conv2)
    )
    outputs = bridge(
        torch.randn(1, 896, device="cuda"),
        torch.randn(1, 2, 768, device="cuda"),
        torch.randn(1, 2, 512, device="cuda"),
    )
    assert [tuple(output.shape) for output in outputs] == [
        (1, 512),
        (1, 2, 384),
        (1, 2, 256),
    ]


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_epilogue_external_radial_forward_and_gradients_match_torch():
    torch.manual_seed(42)
    device = torch.device("cuda")
    mapping = CoefficientMapping(3, 2).to(device)
    reference = SO2_Convolution(
        256,
        128,
        3,
        2,
        mapping,
        internal_weights=False,
        edge_channels_list=[32, 64],
        extra_m0_output_channels=384,
    ).cuda().eval()
    fused = FusedSO2Convolution(copy.deepcopy(reference).cuda().eval())
    reference.requires_grad_(False)
    fused.requires_grad_(False)
    edges = 13
    x_ref = torch.randn(edges, 14, 256, device=device, requires_grad=True)
    edge_ref = torch.randn(edges, 32, device=device, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    edge_fused = edge_ref.detach().clone().requires_grad_(True)
    grad_out = torch.randn(edges, 14, 128, device=device)
    grad_gate = torch.randn(edges, 384, device=device)

    expected, expected_gate = reference(x_ref, edge_ref)
    torch.autograd.backward((expected, expected_gate), (grad_out, grad_gate))
    actual, actual_gate = fused(x_fused, edge_fused)
    torch.autograd.backward((actual, actual_gate), (grad_out, grad_gate))

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(edge_fused.grad, edge_ref.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
@pytest.mark.parametrize("prepare_backward_reduce", (False, True))
def test_so2_block_gemm_external_radial_forward_and_gradients_match_torch(
    prepare_backward_reduce,
):
    torch.manual_seed(42)
    mapping = CoefficientMapping(3, 2).cuda()
    reference = SO2_Convolution(
        256,
        128,
        3,
        2,
        mapping,
        internal_weights=False,
        edge_channels_list=[32, 64],
        extra_m0_output_channels=384,
    ).cuda().eval()
    fused = FusedSO2Convolution(
        copy.deepcopy(reference),
        block_gemm=True,
        prepare_backward_reduce=prepare_backward_reduce,
    ).cuda().eval()
    reference.requires_grad_(False)
    fused.requires_grad_(False)
    x_ref = torch.randn(11, 14, 256, device="cuda", requires_grad=True)
    edge_ref = torch.randn(11, 32, device="cuda", requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    edge_fused = edge_ref.detach().clone().requires_grad_(True)
    grad_out = torch.randn(11, 14, 128, device="cuda")
    grad_gate = torch.randn(11, 384, device="cuda")

    expected = reference(x_ref, edge_ref)
    torch.autograd.backward(expected, (grad_out, grad_gate))
    actual = fused(x_fused, edge_fused)
    torch.autograd.backward(actual, (grad_out, grad_gate))

    torch.testing.assert_close(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        edge_fused.grad, edge_ref.grad, rtol=RTOL, atol=ATOL
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_block_gemm_kf10_pair_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    mapping = CoefficientMapping(3, 2).cuda()
    conv1 = FusedSO2Convolution(
        SO2_Convolution(
            256,
            128,
            3,
            2,
            mapping,
            internal_weights=False,
            edge_channels_list=[32, 64],
            extra_m0_output_channels=384,
        ).cuda(),
        block_gemm=True,
    ).eval()
    conv2 = FusedSO2Convolution(
        SO2_Convolution(
            128, 128, 3, 2, mapping, internal_weights=True
        ).cuda(),
        block_gemm=True,
    ).eval()
    conv1.requires_grad_(False)
    conv2.requires_grad_(False)
    bridge = FusedSO2GateBridge(conv1, conv2)
    x = torch.randn(18, 14, 256, device="cuda", requires_grad=True)
    x_edge = torch.randn(18, 32, device="cuda", requires_grad=True)
    inputs = (x, x_edge)

    def forward():
        m0, m1, m2 = conv1.prepare_and_linear(x, x_edge)
        prepared = bridge(m0, m1, m2)
        return conv2.epilogue(*conv2.linear_from_prepared(*prepared))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = forward()
            torch.autograd.grad(output.sum(), inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = forward()
        gradients = torch.autograd.grad(output.sum(), inputs)
    addresses = (output.data_ptr(),) + tuple(
        gradient.data_ptr() for gradient in gradients
    )
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == (output.data_ptr(),) + tuple(
        gradient.data_ptr() for gradient in gradients
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_epilogue_accepts_empty_edge_set():
    device = torch.device("cuda")
    mapping = CoefficientMapping(3, 2).to(device)
    original = SO2_Convolution(
        128, 128, 3, 2, mapping, internal_weights=True
    ).cuda().eval()
    fused = FusedSO2Convolution(copy.deepcopy(original).cuda().eval())
    fused.requires_grad_(False)
    x_fused = torch.empty(0, 14, 128, device=device, requires_grad=True)
    x_edge = torch.empty(0, 0, device=device)

    # The upstream SO2 implementation uses ``reshape(0, -1)`` and therefore
    # cannot serve as an empty-edge reference.  The fused contract is still
    # well-defined: all outputs and input gradients have zero rows.
    actual = fused(x_fused, x_edge)
    actual.sum().backward()

    assert actual.shape == (0, 14, 128)
    assert x_fused.grad is not None
    assert x_fused.grad.shape == x_fused.shape


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so2_epilogue_internal_weights_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    device = torch.device("cuda")
    mapping = CoefficientMapping(3, 2).to(device)
    reference = SO2_Convolution(
        128, 128, 3, 2, mapping, internal_weights=True
    ).cuda().eval()
    fused = FusedSO2Convolution(copy.deepcopy(reference).cuda().eval())
    reference.requires_grad_(False)
    fused.requires_grad_(False)
    x = torch.randn(24, 14, 128, device=device, requires_grad=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = fused(x, x.new_empty((x.shape[0], 0)))
            torch.autograd.grad(output.sum(), (x,), retain_graph=False)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fused(x, x.new_empty((x.shape[0], 0)))
        gradient = torch.autograd.grad(output.sum(), (x,))[0]
    addresses = (output.data_ptr(), gradient.data_ptr())
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == (output.data_ptr(), gradient.data_ptr())


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


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_so3_mlp_forward_backward_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    original = SpectralAtomwise(128, 128, 3, 3, None).cuda().eval()
    original.requires_grad_(False)
    fused = FusedSpectralAtomwise(original)
    x = torch.randn(8, 16, 128, device="cuda", requires_grad=True)
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


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_frozen_so3_linear_forward_backward_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    original = SO3_Linear(128, 128, lmax=3).cuda().eval()
    original.requires_grad_(False)
    cached = FrozenSO3Linear(original)
    x = torch.randn(8, 16, 128, device="cuda", requires_grad=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.autograd.grad(cached(x).sum(), (x,))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = cached(x)
        gradients = torch.autograd.grad(output.sum(), (x,))
    addresses = (
        cached.expanded_weight.data_ptr(),
        output.data_ptr(),
        gradients[0].data_ptr(),
    )
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert addresses == (
        cached.expanded_weight.data_ptr(),
        output.data_ptr(),
        gradients[0].data_ptr(),
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_configure_so3_weight_cache_replaces_all_20_linears():
    class Backbone(torch.nn.Module):
        lmax = 3
        mmax = 2
        sphere_channels = 128
        hidden_channels = 128
        num_layers = 10
        act_type = "gate"
        norm_type = "rms_norm_sh"
        mlp_type = "spectral"
        use_envelope = True

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            for _ in range(10):
                block = torch.nn.Module()
                block.atom_wise = SpectralAtomwise(128, 128, 3, 3, None)
                self.blocks.append(block)

    model = torch.nn.Module()
    model.backbone = Backbone().cuda().eval()
    model.requires_grad_(False)
    metadata = configure_esen_30m_model_fusions(
        model, "so3-weight-cache"
    )
    assert metadata.so3_weight_cache_replacements == 20
    assert metadata.so3_weight_cache_expanded_weight_count == 20
    assert metadata.so3_weight_cache_bytes == 20 * 16 * 128 * 128 * 4
    assert all(
        isinstance(block.atom_wise.so3_linear_1, FrozenSO3Linear)
        and isinstance(block.atom_wise.so3_linear_2, FrozenSO3Linear)
        for block in model.backbone.blocks
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA and Triton")
def test_energy_mlp_forward_backward_is_cuda_graph_capture_safe():
    torch.manual_seed(42)
    original = torch.nn.Sequential(
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 1),
    ).cuda().eval()
    original.requires_grad_(False)
    fused = FusedEnergyBlock(original)
    x = torch.randn(8, 128, device="cuda", requires_grad=True)
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
