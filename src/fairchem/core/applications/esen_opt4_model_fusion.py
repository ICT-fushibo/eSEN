"""30M-specialized Triton model fusions for eSEN Opt4.

The public eSEN modules are deliberately left untouched.  ``configure_*``
replaces modules only on the already-loaded, inference-only model instance
owned by an Opt4 evaluator.  All fused operations implement the first-order
input gradients required by eSEN's conservative-force head.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import torch
from torch import Tensor, nn

from fairchem.core.models.esen.esen_block import Edgewise, SpectralAtomwise
from fairchem.core.models.esen.nn.activation import GateActivation
from fairchem.core.models.esen.nn.embedding import EdgeDegreeEmbedding
from fairchem.core.models.esen.nn.layer_norm import (
    EquivariantRMSNormArraySphericalHarmonicsV2,
)
from fairchem.core.models.esen.nn.radial import RadialMLP
from fairchem.core.models.esen.nn.so2_layers import SO2_Convolution
from fairchem.core.models.esen.nn.so3_layers import SO3_Linear

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU-only import tests
    triton = None
    tl = None


FUSION_KERNEL_VERSION = "opt4-model-fusion-v2"
SO2_FUSION_KERNEL_VERSION = "opt4-model-fusion-v3-so2"
SO2_GATE_BRIDGE_KERNEL_VERSION = "opt4-model-fusion-v4-so2-gate"
WIGNER_SO2_BRIDGE_KERNEL_VERSION = "opt4-model-fusion-v5-wigner-so2"
SO2_BLOCK_GEMM_VERSION = "opt4-model-fusion-v6-so2-block-gemm"
SO3_WEIGHT_CACHE_VERSION = "opt4-model-fusion-v7-so3-weight-cache"
SO2_PREPARE_BACKWARD_REDUCE_VERSION = (
    "opt4-model-fusion-v8-so2-prepare-backward-reduce"
)
WIGNER_SO2_HYBRID_KERNEL_VERSION = "opt4-model-fusion-v9-wigner-so2-hybrid"
SO2_KERNEL_BLOCK = 512
SUPPORTED_FUSIONS = (
    "gather-wigner",
    "reverse-scatter",
    "rmsnorm",
    "gate",
    "radial-mlp",
    "so3-mlp",
    "energy-head",
    "so2-epilogue",
    "so2-gate-bridge",
    "wigner-so2-bridge",
    "wigner-so2-hybrid",
    "so2-block-gemm",
    "so3-weight-cache",
    "so2-prepare-backward-reduce",
)


class UnsupportedFusionConfigError(RuntimeError):
    """The requested Opt4 kernels cannot safely run on this model."""


def model_fusion_available() -> bool:
    return triton is not None


def parse_model_fusions(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        requested = [item.strip() for item in value.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in value if str(item).strip()]
    unknown = sorted(set(requested).difference(SUPPORTED_FUSIONS))
    if unknown:
        raise UnsupportedFusionConfigError(
            "Unknown Opt4 model fusion(s): " + ", ".join(unknown)
        )
    requested_set = set(requested)
    if (
        "so2-gate-bridge" in requested_set
        and "so2-epilogue" not in requested_set
    ):
        raise UnsupportedFusionConfigError(
            "so2-gate-bridge requires so2-epilogue to be requested explicitly"
        )
    if "wigner-so2-bridge" in requested_set and not {
        "so2-epilogue",
        "so2-gate-bridge",
    }.issubset(requested_set):
        raise UnsupportedFusionConfigError(
            "wigner-so2-bridge requires so2-epilogue and so2-gate-bridge "
            "to be requested explicitly"
        )
    if {
        "gather-wigner",
        "wigner-so2-bridge",
    }.issubset(requested_set):
        raise UnsupportedFusionConfigError(
            "wigner-so2-bridge subsumes gather-wigner; request only the bridge"
        )
    if "wigner-so2-hybrid" in requested_set and not {
        "so2-epilogue",
        "so2-gate-bridge",
        "so2-prepare-backward-reduce",
    }.issubset(requested_set):
        raise UnsupportedFusionConfigError(
            "wigner-so2-hybrid requires so2-epilogue, so2-gate-bridge and "
            "so2-prepare-backward-reduce to be requested explicitly"
        )
    if "wigner-so2-hybrid" in requested_set and {
        "gather-wigner",
        "wigner-so2-bridge",
    }.intersection(requested_set):
        raise UnsupportedFusionConfigError(
            "wigner-so2-hybrid is mutually exclusive with gather-wigner "
            "and wigner-so2-bridge"
        )
    if (
        "so2-block-gemm" in requested_set
        and "so2-epilogue" not in requested_set
    ):
        raise UnsupportedFusionConfigError(
            "so2-block-gemm requires so2-epilogue to be requested explicitly"
        )
    if (
        "so2-prepare-backward-reduce" in requested_set
        and not {"so2-epilogue", "so2-gate-bridge"}.issubset(requested_set)
    ):
        raise UnsupportedFusionConfigError(
            "so2-prepare-backward-reduce requires so2-epilogue and "
            "so2-gate-bridge to be requested explicitly"
        )
    if {"so3-mlp", "so3-weight-cache"}.issubset(requested_set):
        raise UnsupportedFusionConfigError(
            "so3-mlp already materializes expanded SO3 weights; "
            "do not combine it with so3-weight-cache"
        )
    return tuple(name for name in SUPPORTED_FUSIONS if name in requested_set)


if triton is not None:

    @triton.jit
    def _gather_wigner_forward_kernel(
        x_ptr,
        source_ptr,
        target_ptr,
        wigner_ptr,
        mask_ptr,
        out_ptr,
        num_edges: tl.constexpr,
        reduced_coefficients: tl.constexpr,
        channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        output_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        reduced = tl.program_id(1)
        channel2 = tl.arange(0, block_channels)
        channel_active = channel2 < output_channels
        source_half = channel2 < channels
        channel = channel2 % channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        full_out = tl.load(mask_ptr + reduced)
        acc = tl.zeros((block_channels,), tl.float32)
        for full_in in range(full_coefficients):
            w = tl.load(
                wigner_ptr
                + edge * full_coefficients * full_coefficients
                + full_out * full_coefficients
                + full_in
            )
            feature = tl.load(
                x_ptr
                + node * full_coefficients * channels
                + full_in * channels
                + channel,
                mask=channel_active,
                other=0.0,
            )
            acc += w * feature
        tl.store(
            out_ptr
            + edge * reduced_coefficients * output_channels
            + reduced * output_channels
            + channel2,
            acc,
            mask=channel_active,
        )

    @triton.jit
    def _gather_wigner_backward_x_kernel(
        grad_out_ptr,
        source_ptr,
        target_ptr,
        wigner_ptr,
        mask_ptr,
        grad_x_ptr,
        num_edges: tl.constexpr,
        reduced_coefficients: tl.constexpr,
        channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        output_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        channel2 = tl.arange(0, block_channels)
        channel_active = channel2 < output_channels
        source_half = channel2 < channels
        channel = channel2 % channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        for full_in in range(full_coefficients):
            acc = tl.zeros((block_channels,), tl.float32)
            for reduced in range(reduced_coefficients):
                full_out = tl.load(mask_ptr + reduced)
                w = tl.load(
                    wigner_ptr
                    + edge * full_coefficients * full_coefficients
                    + full_out * full_coefficients
                    + full_in
                )
                grad = tl.load(
                    grad_out_ptr
                    + edge * reduced_coefficients * output_channels
                    + reduced * output_channels
                    + channel2,
                    mask=channel_active,
                    other=0.0,
                )
                acc += w * grad
            tl.atomic_add(
                grad_x_ptr
                + node * full_coefficients * channels
                + full_in * channels
                + channel,
                acc,
                mask=channel_active,
            )

    @triton.jit
    def _gather_wigner_backward_w_kernel(
        grad_out_ptr,
        x_ptr,
        source_ptr,
        target_ptr,
        mask_ptr,
        grad_wigner_ptr,
        num_edges: tl.constexpr,
        reduced_coefficients: tl.constexpr,
        channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        output_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        reduced = tl.program_id(1)
        full_in = tl.program_id(2)
        channel2 = tl.arange(0, block_channels)
        channel_active = channel2 < output_channels
        source_half = channel2 < channels
        channel = channel2 % channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        grad = tl.load(
            grad_out_ptr
            + edge * reduced_coefficients * output_channels
            + reduced * output_channels
            + channel2,
            mask=channel_active,
            other=0.0,
        )
        feature = tl.load(
            x_ptr
            + node * full_coefficients * channels
            + full_in * channels
            + channel,
            mask=channel_active,
            other=0.0,
        )
        full_out = tl.load(mask_ptr + reduced)
        tl.store(
            grad_wigner_ptr
            + edge * full_coefficients * full_coefficients
            + full_out * full_coefficients
            + full_in,
            tl.sum(grad * feature),
        )

    @triton.jit
    def _reverse_scatter_forward_kernel(
        message_ptr,
        wigner_ptr,
        mask_ptr,
        distance_ptr,
        target_ptr,
        out_ptr,
        num_edges: tl.constexpr,
        reduced_coefficients: tl.constexpr,
        channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        cutoff: tl.constexpr,
        scale: tl.constexpr,
        node_offset: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        full_out = tl.program_id(1)
        channel = tl.arange(0, block_channels)
        channel_active = channel < channels
        distance = tl.load(distance_ptr + edge) / cutoff
        d2 = distance * distance
        d4 = d2 * d2
        d5 = d4 * distance
        envelope = 1.0 - 21.0 * d5 + 35.0 * d5 * distance - 15.0 * d5 * d2
        envelope = tl.where(distance < 1.0, envelope, 0.0) * scale
        acc = tl.zeros((block_channels,), tl.float32)
        for reduced in range(reduced_coefficients):
            full_in = tl.load(mask_ptr + reduced)
            w = tl.load(
                wigner_ptr
                + edge * full_coefficients * full_coefficients
                + full_out * full_coefficients
                + full_in
            )
            message = tl.load(
                message_ptr
                + edge * reduced_coefficients * channels
                + reduced * channels
                + channel,
                mask=channel_active,
                other=0.0,
            )
            acc += w * message
        target = tl.load(target_ptr + edge) - node_offset
        tl.atomic_add(
            out_ptr
            + target * full_coefficients * channels
            + full_out * channels
            + channel,
            envelope * acc,
            mask=channel_active,
        )

    @triton.jit
    def _reverse_scatter_backward_message_kernel(
        grad_out_ptr,
        wigner_ptr,
        mask_ptr,
        distance_ptr,
        target_ptr,
        grad_message_ptr,
        num_edges: tl.constexpr,
        reduced_coefficients: tl.constexpr,
        channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        cutoff: tl.constexpr,
        scale: tl.constexpr,
        node_offset: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        reduced = tl.program_id(1)
        channel = tl.arange(0, block_channels)
        channel_active = channel < channels
        distance = tl.load(distance_ptr + edge) / cutoff
        d2 = distance * distance
        d4 = d2 * d2
        d5 = d4 * distance
        envelope = 1.0 - 21.0 * d5 + 35.0 * d5 * distance - 15.0 * d5 * d2
        envelope = tl.where(distance < 1.0, envelope, 0.0) * scale
        target = tl.load(target_ptr + edge) - node_offset
        full_in = tl.load(mask_ptr + reduced)
        acc = tl.zeros((block_channels,), tl.float32)
        for full_out in range(full_coefficients):
            w = tl.load(
                wigner_ptr
                + edge * full_coefficients * full_coefficients
                + full_out * full_coefficients
                + full_in
            )
            grad = tl.load(
                grad_out_ptr
                + target * full_coefficients * channels
                + full_out * channels
                + channel,
                mask=channel_active,
                other=0.0,
            )
            acc += w * grad
        tl.store(
            grad_message_ptr
            + edge * reduced_coefficients * channels
            + reduced * channels
            + channel,
            envelope * acc,
            mask=channel_active,
        )

    @triton.jit
    def _reverse_scatter_backward_w_kernel(
        grad_out_ptr,
        message_ptr,
        mask_ptr,
        distance_ptr,
        target_ptr,
        grad_wigner_ptr,
        num_edges: tl.constexpr,
        reduced_coefficients: tl.constexpr,
        channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        cutoff: tl.constexpr,
        scale: tl.constexpr,
        node_offset: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        full_out = tl.program_id(1)
        reduced = tl.program_id(2)
        channel = tl.arange(0, block_channels)
        channel_active = channel < channels
        distance = tl.load(distance_ptr + edge) / cutoff
        d2 = distance * distance
        d4 = d2 * d2
        d5 = d4 * distance
        envelope = 1.0 - 21.0 * d5 + 35.0 * d5 * distance - 15.0 * d5 * d2
        envelope = tl.where(distance < 1.0, envelope, 0.0) * scale
        target = tl.load(target_ptr + edge) - node_offset
        grad = tl.load(
            grad_out_ptr
            + target * full_coefficients * channels
            + full_out * channels
            + channel,
            mask=channel_active,
            other=0.0,
        )
        message = tl.load(
            message_ptr
            + edge * reduced_coefficients * channels
            + reduced * channels
            + channel,
            mask=channel_active,
            other=0.0,
        )
        full_in = tl.load(mask_ptr + reduced)
        tl.store(
            grad_wigner_ptr
            + edge * full_coefficients * full_coefficients
            + full_out * full_coefficients
            + full_in,
            envelope * tl.sum(grad * message),
        )

    @triton.jit
    def _reverse_scatter_backward_distance_kernel(
        grad_out_ptr,
        message_ptr,
        wigner_ptr,
        mask_ptr,
        distance_ptr,
        target_ptr,
        grad_distance_ptr,
        num_edges: tl.constexpr,
        reduced_coefficients: tl.constexpr,
        channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        cutoff: tl.constexpr,
        scale: tl.constexpr,
        node_offset: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        channel = tl.arange(0, block_channels)
        channel_active = channel < channels
        target = tl.load(target_ptr + edge) - node_offset
        total = tl.zeros((block_channels,), tl.float32)
        for full_out in range(full_coefficients):
            rotated = tl.zeros((block_channels,), tl.float32)
            for reduced in range(reduced_coefficients):
                full_in = tl.load(mask_ptr + reduced)
                w = tl.load(
                    wigner_ptr
                    + edge * full_coefficients * full_coefficients
                    + full_out * full_coefficients
                    + full_in
                )
                message = tl.load(
                    message_ptr
                    + edge * reduced_coefficients * channels
                    + reduced * channels
                    + channel,
                    mask=channel_active,
                    other=0.0,
                )
                rotated += w * message
            grad = tl.load(
                grad_out_ptr
                + target * full_coefficients * channels
                + full_out * channels
                + channel,
                mask=channel_active,
                other=0.0,
            )
            total += grad * rotated
        distance = tl.load(distance_ptr + edge) / cutoff
        d2 = distance * distance
        d4 = d2 * d2
        derivative = -105.0 * d4 + 210.0 * d4 * distance - 105.0 * d4 * d2
        derivative = tl.where(distance < 1.0, derivative / cutoff, 0.0) * scale
        tl.store(grad_distance_ptr + edge, derivative * tl.sum(total))

    @triton.jit
    def _rmsnorm_forward_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offset = tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        x = tl.load(x_ptr + row * coefficients * channels + offset, mask=active, other=0.0)
        l0_mean = tl.sum(tl.where(coefficient == 0, x, 0.0)) / channels
        feature = tl.where(coefficient == 0, x - l0_mean, x)
        degree = tl.where(coefficient < 1, 0, tl.where(coefficient < 4, 1, tl.where(coefficient < 9, 2, 3)))
        degree_length = 2 * degree + 1
        balance = 1.0 / degree_length
        norm2 = tl.sum(tl.where(active, feature * feature * balance, 0.0)) / (4.0 * channels)
        inv_norm = tl.rsqrt(norm2 + eps)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=active, other=0.0)
        bias = tl.load(bias_ptr + channel, mask=active & (coefficient == 0), other=0.0)
        out = feature * inv_norm * weight + tl.where(coefficient == 0, bias, 0.0)
        tl.store(out_ptr + row * coefficients * channels + offset, out, mask=active)

    @triton.jit
    def _rmsnorm_backward_kernel(
        grad_out_ptr,
        x_ptr,
        weight_ptr,
        grad_x_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offset = tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        x = tl.load(x_ptr + row * coefficients * channels + offset, mask=active, other=0.0)
        grad_out = tl.load(grad_out_ptr + row * coefficients * channels + offset, mask=active, other=0.0)
        l0_mean = tl.sum(tl.where(coefficient == 0, x, 0.0)) / channels
        feature = tl.where(coefficient == 0, x - l0_mean, x)
        degree = tl.where(coefficient < 1, 0, tl.where(coefficient < 4, 1, tl.where(coefficient < 9, 2, 3)))
        degree_length = 2 * degree + 1
        balance = 1.0 / degree_length
        norm2 = tl.sum(tl.where(active, feature * feature * balance, 0.0)) / (4.0 * channels)
        inv_norm = tl.rsqrt(norm2 + eps)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=active, other=0.0)
        weighted_grad = grad_out * weight
        projection = tl.sum(tl.where(active, weighted_grad * feature, 0.0))
        grad_feature = weighted_grad * inv_norm - balance * feature * projection * inv_norm * inv_norm * inv_norm / (4.0 * channels)
        l0_grad_mean = tl.sum(tl.where(coefficient == 0, grad_feature, 0.0)) / channels
        grad_x = tl.where(coefficient == 0, grad_feature - l0_grad_mean, grad_feature)
        tl.store(grad_x_ptr + row * coefficients * channels + offset, grad_x, mask=active)

    @triton.jit
    def _gate_forward_kernel(
        gate_ptr,
        x_ptr,
        out_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offset = tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        x = tl.load(x_ptr + row * coefficients * channels + offset, mask=active, other=0.0)
        degree = tl.where(coefficient <= 3, 1, tl.where(coefficient <= 8, 2, 3))
        gate = tl.load(gate_ptr + row * 3 * channels + (degree - 1) * channels + channel, mask=active & (coefficient > 0), other=0.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        scalar_sigmoid = 1.0 / (1.0 + tl.exp(-x))
        out = tl.where(coefficient == 0, x * scalar_sigmoid, x * sigmoid)
        tl.store(out_ptr + row * coefficients * channels + offset, out, mask=active)

    @triton.jit
    def _gate_backward_kernel(
        grad_out_ptr,
        gate_ptr,
        x_ptr,
        grad_x_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offset = tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        x = tl.load(x_ptr + row * coefficients * channels + offset, mask=active, other=0.0)
        grad_out = tl.load(grad_out_ptr + row * coefficients * channels + offset, mask=active, other=0.0)
        degree = tl.where(coefficient <= 3, 1, tl.where(coefficient <= 8, 2, 3))
        gate_offset = row * 3 * channels + (degree - 1) * channels + channel
        gate = tl.load(gate_ptr + gate_offset, mask=active & (coefficient > 0), other=0.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        scalar_sigmoid = 1.0 / (1.0 + tl.exp(-x))
        scalar_grad = scalar_sigmoid * (1.0 + x * (1.0 - scalar_sigmoid))
        grad_x = grad_out * tl.where(coefficient == 0, scalar_grad, sigmoid)
        tl.store(grad_x_ptr + row * coefficients * channels + offset, grad_x, mask=active)

    @triton.jit
    def _gate_backward_gate_kernel(
        grad_out_ptr,
        gate_ptr,
        x_ptr,
        grad_gate_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        degree0 = tl.program_id(1)
        channel = tl.arange(0, block)
        active = channel < channels
        gate = tl.load(
            gate_ptr + row * 3 * channels + degree0 * channels + channel,
            mask=active,
            other=0.0,
        )
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        total = tl.zeros((block,), tl.float32)
        for coefficient in range(1, coefficients):
            degree = tl.where(coefficient <= 3, 1, tl.where(coefficient <= 8, 2, 3))
            belongs = degree == degree0 + 1
            offset = row * coefficients * channels + coefficient * channels + channel
            grad = tl.load(grad_out_ptr + offset, mask=active & belongs, other=0.0)
            x = tl.load(x_ptr + offset, mask=active & belongs, other=0.0)
            total += grad * x
        tl.store(
            grad_gate_ptr + row * 3 * channels + degree0 * channels + channel,
            total * sigmoid * (1.0 - sigmoid),
            mask=active,
        )

    @triton.jit
    def _so2_prepare_kernel(
        x_ptr,
        radial_ptr,
        to_m_ptr,
        m0_ptr,
        m1_ptr,
        m2_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        radial_channels: tl.constexpr,
        use_radial: tl.constexpr,
        block: tl.constexpr,
    ):
        """Move l-order coefficients to m-order and apply radial weights.

        The eSEN mapping is a permutation for the supported 30M model.  Keeping
        this operation in one kernel avoids the generic einsum and the three
        separate radial multiplication kernels around the cuBLAS Linear calls.
        """
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        source_coefficient = tl.load(to_m_ptr + coefficient, mask=active, other=0)
        x_offset = (
            row * coefficients * channels
            + source_coefficient * channels
            + channel
        )
        value = tl.load(x_ptr + x_offset, mask=active, other=0.0)
        if use_radial:
            radial_coefficient = tl.where(
                coefficient < 4,
                coefficient,
                tl.where(
                    coefficient < 10,
                    4 + ((coefficient - 4) % 3),
                    7 + ((coefficient - 10) % 2),
                ),
            )
            radial_offset = row * radial_channels + radial_coefficient * channels + channel
            value = value * tl.load(radial_ptr + radial_offset, mask=active, other=0.0)
        m0_mask = active & (coefficient < 4)
        m1_mask = active & (coefficient >= 4) & (coefficient < 10)
        m2_mask = active & (coefficient >= 10)
        tl.store(
            m0_ptr + row * 4 * channels + coefficient * channels + channel,
            value,
            mask=m0_mask,
        )
        tl.store(
            m1_ptr
            + row * 6 * channels
            + (coefficient - 4) * channels
            + channel,
            value,
            mask=m1_mask,
        )
        tl.store(
            m2_ptr
            + row * 4 * channels
            + (coefficient - 10) * channels
            + channel,
            value,
            mask=m2_mask,
        )

    @triton.jit
    def _so2_prepare_backward_kernel(
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        x_ptr,
        radial_ptr,
        to_m_ptr,
        grad_x_ptr,
        grad_radial_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        radial_channels: tl.constexpr,
        use_radial: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        source_coefficient = tl.load(to_m_ptr + coefficient, mask=active, other=0)
        source_offset = (
            row * coefficients * channels
            + source_coefficient * channels
            + channel
        )
        grad_m0 = tl.load(
            grad_m0_ptr
            + row * 4 * channels
            + coefficient * channels
            + channel,
            mask=active & (coefficient < 4),
            other=0.0,
        )
        grad_m1 = tl.load(
            grad_m1_ptr
            + row * 6 * channels
            + (coefficient - 4) * channels
            + channel,
            mask=active & (coefficient >= 4) & (coefficient < 10),
            other=0.0,
        )
        grad_m2 = tl.load(
            grad_m2_ptr
            + row * 4 * channels
            + (coefficient - 10) * channels
            + channel,
            mask=active & (coefficient >= 10),
            other=0.0,
        )
        grad = tl.where(
            coefficient < 4,
            grad_m0,
            tl.where(coefficient < 10, grad_m1, grad_m2),
        )
        x_value = tl.load(x_ptr + source_offset, mask=active, other=0.0)
        if use_radial:
            radial_coefficient = tl.where(
                coefficient < 4,
                coefficient,
                tl.where(
                    coefficient < 10,
                    4 + ((coefficient - 4) % 3),
                    7 + ((coefficient - 10) % 2),
                ),
            )
            radial_offset = row * radial_channels + radial_coefficient * channels + channel
            radial_value = tl.load(radial_ptr + radial_offset, mask=active, other=0.0)
            tl.atomic_add(grad_radial_ptr + radial_offset, grad * x_value, mask=active)
            grad = grad * radial_value
        # ``to_m`` is a permutation, so every input coefficient is written by
        # exactly one output coefficient.  A direct store avoids an unnecessary
        # zero-fill and atomic operation on the feature gradient.
        tl.store(grad_x_ptr + source_offset, grad, mask=active)

    @triton.jit
    def _so2_prepare_backward_reduce_kernel(
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        x_ptr,
        radial_ptr,
        to_m_ptr,
        grad_x_ptr,
        grad_radial_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        radial_channels: tl.constexpr,
        use_radial: tl.constexpr,
        block: tl.constexpr,
    ):
        """Reduce the nine radial gradients per edge without atomics.

        A program owns one edge and one lane owns one feature channel.  The
        14 coefficient contributions are accumulated in registers before the
        nine unique radial entries are written once.  Since no radial address
        is shared by different edges or channels, direct stores are sufficient.
        """
        row = tl.program_id(0)
        channel = tl.arange(0, block)
        active = channel < channels
        row_x = row * coefficients * channels
        row_radial = row * radial_channels

        radial_grad_0 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_1 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_2 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_3 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_4 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_5 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_6 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_7 = tl.zeros((block,), dtype=tl.float32)
        radial_grad_8 = tl.zeros((block,), dtype=tl.float32)

        for coefficient in range(coefficients):
            source_coefficient = tl.load(to_m_ptr + coefficient)
            source_offset = row_x + source_coefficient * channels + channel
            if coefficient < 4:
                grad = tl.load(
                    grad_m0_ptr
                    + row * 4 * channels
                    + coefficient * channels
                    + channel,
                    mask=active,
                    other=0.0,
                )
                radial_coefficient = coefficient
            elif coefficient < 10:
                grad = tl.load(
                    grad_m1_ptr
                    + row * 6 * channels
                    + (coefficient - 4) * channels
                    + channel,
                    mask=active,
                    other=0.0,
                )
                radial_coefficient = 4 + ((coefficient - 4) % 3)
            else:
                grad = tl.load(
                    grad_m2_ptr
                    + row * 4 * channels
                    + (coefficient - 10) * channels
                    + channel,
                    mask=active,
                    other=0.0,
                )
                radial_coefficient = 7 + ((coefficient - 10) % 2)

            if use_radial:
                radial_offset = (
                    row_radial + radial_coefficient * channels + channel
                )
                x_value = tl.load(
                    x_ptr + source_offset, mask=active, other=0.0
                )
                contribution = grad * x_value
                if radial_coefficient == 0:
                    radial_grad_0 += contribution
                elif radial_coefficient == 1:
                    radial_grad_1 += contribution
                elif radial_coefficient == 2:
                    radial_grad_2 += contribution
                elif radial_coefficient == 3:
                    radial_grad_3 += contribution
                elif radial_coefficient == 4:
                    radial_grad_4 += contribution
                elif radial_coefficient == 5:
                    radial_grad_5 += contribution
                elif radial_coefficient == 6:
                    radial_grad_6 += contribution
                elif radial_coefficient == 7:
                    radial_grad_7 += contribution
                else:
                    radial_grad_8 += contribution
                radial_value = tl.load(
                    radial_ptr + radial_offset, mask=active, other=0.0
                )
                grad *= radial_value
            tl.store(grad_x_ptr + source_offset, grad, mask=active)

        if use_radial:
            tl.store(
                grad_radial_ptr + row_radial + 0 * channels + channel,
                radial_grad_0,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 1 * channels + channel,
                radial_grad_1,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 2 * channels + channel,
                radial_grad_2,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 3 * channels + channel,
                radial_grad_3,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 4 * channels + channel,
                radial_grad_4,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 5 * channels + channel,
                radial_grad_5,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 6 * channels + channel,
                radial_grad_6,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 7 * channels + channel,
                radial_grad_7,
                mask=active,
            )
            tl.store(
                grad_radial_ptr + row_radial + 8 * channels + channel,
                radial_grad_8,
                mask=active,
            )

    @triton.jit
    def _wigner_so2_prepare_forward_kernel(
        x_ptr,
        source_ptr,
        target_ptr,
        wigner_ptr,
        out_mask_ptr,
        to_m_ptr,
        radial_ptr,
        m0_ptr,
        m1_ptr,
        m2_ptr,
        node_channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        input_channels: tl.constexpr,
        radial_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Gather, rotate, permute and radial-pack conv1 inputs."""
        edge = tl.program_id(0)
        coefficient = tl.program_id(1)
        channel2 = tl.arange(0, block_channels)
        active = channel2 < input_channels
        source_half = channel2 < node_channels
        channel = channel2 % node_channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        reduced = tl.load(to_m_ptr + coefficient)
        full_out = tl.load(out_mask_ptr + reduced)
        rotated = tl.zeros((block_channels,), tl.float32)
        for full_in in range(full_coefficients):
            weight = tl.load(
                wigner_ptr
                + edge * full_coefficients * full_coefficients
                + full_out * full_coefficients
                + full_in
            )
            feature = tl.load(
                x_ptr
                + node * full_coefficients * node_channels
                + full_in * node_channels
                + channel,
                mask=active,
                other=0.0,
            )
            rotated += weight * feature
        radial_coefficient = tl.where(
            coefficient < 4,
            coefficient,
            tl.where(
                coefficient < 10,
                4 + ((coefficient - 4) % 3),
                7 + ((coefficient - 10) % 2),
            ),
        )
        radial_offset = (
            edge * radial_channels + radial_coefficient * input_channels + channel2
        )
        value = rotated * tl.load(radial_ptr + radial_offset, mask=active, other=0.0)
        tl.store(
            m0_ptr + edge * 4 * input_channels + coefficient * input_channels + channel2,
            value,
            mask=active & (coefficient < 4),
        )
        tl.store(
            m1_ptr
            + edge * 6 * input_channels
            + (coefficient - 4) * input_channels
            + channel2,
            value,
            mask=active & (coefficient >= 4) & (coefficient < 10),
        )
        tl.store(
            m2_ptr
            + edge * 4 * input_channels
            + (coefficient - 10) * input_channels
            + channel2,
            value,
            mask=active & (coefficient >= 10),
        )

    @triton.jit
    def _wigner_so2_hybrid_forward_kernel(
        x_ptr,
        source_ptr,
        target_ptr,
        wigner_ptr,
        out_mask_ptr,
        to_m_ptr,
        radial_ptr,
        m0_ptr,
        m1_ptr,
        m2_ptr,
        gathered_ptr,
        rotated_ptr,
        node_channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        input_channels: tl.constexpr,
        radial_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """KF15 fused forward with cuBLAS-compatible saved intermediates."""
        edge = tl.program_id(0)
        coefficient = tl.program_id(1)
        channel2 = tl.arange(0, block_channels)
        active = channel2 < input_channels
        source_half = channel2 < node_channels
        channel = channel2 % node_channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        source_coefficient = tl.load(to_m_ptr + coefficient)
        full_out = tl.load(out_mask_ptr + source_coefficient)
        rotated = tl.zeros((block_channels,), tl.float32)
        for full_in in range(full_coefficients):
            weight = tl.load(
                wigner_ptr
                + edge * full_coefficients * full_coefficients
                + full_out * full_coefficients
                + full_in
            )
            feature = tl.load(
                x_ptr
                + node * full_coefficients * node_channels
                + full_in * node_channels
                + channel,
                mask=active,
                other=0.0,
            )
            # Only one coefficient program materializes the gathered input.
            # The remaining programs reuse the value in registers for their
            # Wigner dot product without racing on the auxiliary buffer.
            tl.store(
                gathered_ptr
                + edge * full_coefficients * input_channels
                + full_in * input_channels
                + channel2,
                feature,
                mask=active & (coefficient == 0),
            )
            rotated += weight * feature
        # SO2 prepare consumes the Wigner result in l-order.  ``coefficient``
        # is m-order here, so save through the fixed inverse permutation.
        tl.store(
            rotated_ptr
            + edge * 14 * input_channels
            + source_coefficient * input_channels
            + channel2,
            rotated,
            mask=active,
        )
        radial_coefficient = tl.where(
            coefficient < 4,
            coefficient,
            tl.where(
                coefficient < 10,
                4 + ((coefficient - 4) % 3),
                7 + ((coefficient - 10) % 2),
            ),
        )
        radial_offset = (
            edge * radial_channels
            + radial_coefficient * input_channels
            + channel2
        )
        value = rotated * tl.load(
            radial_ptr + radial_offset, mask=active, other=0.0
        )
        tl.store(
            m0_ptr
            + edge * 4 * input_channels
            + coefficient * input_channels
            + channel2,
            value,
            mask=active & (coefficient < 4),
        )
        tl.store(
            m1_ptr
            + edge * 6 * input_channels
            + (coefficient - 4) * input_channels
            + channel2,
            value,
            mask=active & (coefficient >= 4) & (coefficient < 10),
        )
        tl.store(
            m2_ptr
            + edge * 4 * input_channels
            + (coefficient - 10) * input_channels
            + channel2,
            value,
            mask=active & (coefficient >= 10),
        )

    @triton.jit
    def _wigner_so2_prepare_backward_x_kernel(
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        source_ptr,
        target_ptr,
        wigner_ptr,
        out_mask_ptr,
        to_m_ptr,
        radial_ptr,
        grad_x_ptr,
        node_channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        input_channels: tl.constexpr,
        radial_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        channel2 = tl.arange(0, block_channels)
        active = channel2 < input_channels
        source_half = channel2 < node_channels
        channel = channel2 % node_channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        full_in = tl.program_id(1)
        total = tl.zeros((block_channels,), tl.float32)
        for coefficient in range(14):
            if coefficient < 4:
                grad0 = tl.load(
                    grad_m0_ptr
                    + edge * 4 * input_channels
                    + coefficient * input_channels
                    + channel2,
                    mask=active,
                    other=0.0,
                )
                grad = grad0
                radial_coefficient = coefficient
            elif coefficient < 10:
                grad1 = tl.load(
                    grad_m1_ptr
                    + edge * 6 * input_channels
                    + (coefficient - 4) * input_channels
                    + channel2,
                    mask=active,
                    other=0.0,
                )
                grad = grad1
                radial_coefficient = 4 + ((coefficient - 4) % 3)
            else:
                grad2 = tl.load(
                    grad_m2_ptr
                    + edge * 4 * input_channels
                    + (coefficient - 10) * input_channels
                    + channel2,
                    mask=active,
                    other=0.0,
                )
                grad = grad2
                radial_coefficient = 7 + ((coefficient - 10) % 2)
            radial = tl.load(
                radial_ptr
                + edge * radial_channels
                + radial_coefficient * input_channels
                + channel2,
                mask=active,
                other=0.0,
            )
            reduced = tl.load(to_m_ptr + coefficient)
            full_out = tl.load(out_mask_ptr + reduced)
            weight = tl.load(
                wigner_ptr
                + edge * full_coefficients * full_coefficients
                + full_out * full_coefficients
                + full_in
            )
            total += grad * radial * weight
        tl.atomic_add(
            grad_x_ptr
            + node * full_coefficients * node_channels
            + full_in * node_channels
            + channel,
            total,
            mask=active,
        )

    @triton.jit
    def _wigner_so2_prepare_backward_w_kernel(
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        x_ptr,
        source_ptr,
        target_ptr,
        out_mask_ptr,
        to_m_ptr,
        radial_ptr,
        grad_wigner_ptr,
        node_channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        input_channels: tl.constexpr,
        radial_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        coefficient = tl.program_id(1)
        full_in = tl.program_id(2)
        channel2 = tl.arange(0, block_channels)
        active = channel2 < input_channels
        source_half = channel2 < node_channels
        channel = channel2 % node_channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        grad0 = tl.load(
            grad_m0_ptr
            + edge * 4 * input_channels
            + coefficient * input_channels
            + channel2,
            mask=active & (coefficient < 4),
            other=0.0,
        )
        grad1 = tl.load(
            grad_m1_ptr
            + edge * 6 * input_channels
            + (coefficient - 4) * input_channels
            + channel2,
            mask=active & (coefficient >= 4) & (coefficient < 10),
            other=0.0,
        )
        grad2 = tl.load(
            grad_m2_ptr
            + edge * 4 * input_channels
            + (coefficient - 10) * input_channels
            + channel2,
            mask=active & (coefficient >= 10),
            other=0.0,
        )
        grad = tl.where(
            coefficient < 4,
            grad0,
            tl.where(coefficient < 10, grad1, grad2),
        )
        radial_coefficient = tl.where(
            coefficient < 4,
            coefficient,
            tl.where(
                coefficient < 10,
                4 + ((coefficient - 4) % 3),
                7 + ((coefficient - 10) % 2),
            ),
        )
        radial = tl.load(
            radial_ptr
            + edge * radial_channels
            + radial_coefficient * input_channels
            + channel2,
            mask=active,
            other=0.0,
        )
        feature = tl.load(
            x_ptr
            + node * full_coefficients * node_channels
            + full_in * node_channels
            + channel,
            mask=active,
            other=0.0,
        )
        reduced = tl.load(to_m_ptr + coefficient)
        full_out = tl.load(out_mask_ptr + reduced)
        tl.store(
            grad_wigner_ptr
            + edge * full_coefficients * full_coefficients
            + full_out * full_coefficients
            + full_in,
            tl.sum(grad * radial * feature),
        )

    @triton.jit
    def _wigner_so2_prepare_backward_radial_kernel(
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        x_ptr,
        source_ptr,
        target_ptr,
        wigner_ptr,
        out_mask_ptr,
        to_m_ptr,
        grad_radial_ptr,
        node_channels: tl.constexpr,
        full_coefficients: tl.constexpr,
        input_channels: tl.constexpr,
        radial_channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        edge = tl.program_id(0)
        coefficient = tl.program_id(1)
        channel2 = tl.arange(0, block_channels)
        active = channel2 < input_channels
        source_half = channel2 < node_channels
        channel = channel2 % node_channels
        source = tl.load(source_ptr + edge)
        target = tl.load(target_ptr + edge)
        node = tl.where(source_half, source, target)
        reduced = tl.load(to_m_ptr + coefficient)
        full_out = tl.load(out_mask_ptr + reduced)
        rotated = tl.zeros((block_channels,), tl.float32)
        for full_in in range(full_coefficients):
            weight = tl.load(
                wigner_ptr
                + edge * full_coefficients * full_coefficients
                + full_out * full_coefficients
                + full_in
            )
            feature = tl.load(
                x_ptr
                + node * full_coefficients * node_channels
                + full_in * node_channels
                + channel,
                mask=active,
                other=0.0,
            )
            rotated += weight * feature
        grad0 = tl.load(
            grad_m0_ptr
            + edge * 4 * input_channels
            + coefficient * input_channels
            + channel2,
            mask=active & (coefficient < 4),
            other=0.0,
        )
        grad1 = tl.load(
            grad_m1_ptr
            + edge * 6 * input_channels
            + (coefficient - 4) * input_channels
            + channel2,
            mask=active & (coefficient >= 4) & (coefficient < 10),
            other=0.0,
        )
        grad2 = tl.load(
            grad_m2_ptr
            + edge * 4 * input_channels
            + (coefficient - 10) * input_channels
            + channel2,
            mask=active & (coefficient >= 10),
            other=0.0,
        )
        grad = tl.where(
            coefficient < 4,
            grad0,
            tl.where(coefficient < 10, grad1, grad2),
        )
        radial_coefficient = tl.where(
            coefficient < 4,
            coefficient,
            tl.where(
                coefficient < 10,
                4 + ((coefficient - 4) % 3),
                7 + ((coefficient - 10) % 2),
            ),
        )
        tl.atomic_add(
            grad_radial_ptr
            + edge * radial_channels
            + radial_coefficient * input_channels
            + channel2,
            grad * rotated,
            mask=active,
        )

    @triton.jit
    def _so2_epilogue_kernel(
        m0_ptr,
        m1_ptr,
        m2_ptr,
        l_to_m_ptr,
        out_ptr,
        gating_ptr,
        rows: tl.constexpr,
        output_coefficients: tl.constexpr,
        output_channels: tl.constexpr,
        m0_channels: tl.constexpr,
        m1_coefficients: tl.constexpr,
        m2_coefficients: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        main_size = output_coefficients * output_channels
        main_active = offset < main_size
        gate_active = offset >= main_size
        l_coefficient = offset // output_channels
        channel = offset % output_channels
        m_coefficient = tl.load(l_to_m_ptr + l_coefficient, mask=main_active, other=0)

        m0_offset = (
            row * m0_channels
            + extra_channels
            + m_coefficient * output_channels
            + channel
        )
        m0_value = tl.load(m0_ptr + m0_offset, mask=main_active & (m_coefficient < 4), other=0.0)

        m1_local = m_coefficient - 4
        m1_part = m1_local // m1_coefficients
        m1_coefficient = m1_local % m1_coefficients
        m1_half_width = m1_coefficients * output_channels
        m1_linear_width = 2 * m1_half_width
        m1_edge_stride = 2 * m1_linear_width
        m1_base = row * m1_edge_stride + m1_coefficient * output_channels + channel
        m1_row0_first = tl.load(m1_ptr + m1_base, mask=main_active & (m_coefficient >= 4) & (m_coefficient < 10), other=0.0)
        m1_row0_second = tl.load(m1_ptr + m1_base + m1_half_width, mask=main_active & (m_coefficient >= 4) & (m_coefficient < 10), other=0.0)
        m1_row1_first = tl.load(m1_ptr + m1_base + m1_linear_width, mask=main_active & (m_coefficient >= 4) & (m_coefficient < 10), other=0.0)
        m1_row1_second = tl.load(m1_ptr + m1_base + m1_linear_width + m1_half_width, mask=main_active & (m_coefficient >= 4) & (m_coefficient < 10), other=0.0)
        m1_value = tl.where(m1_part == 0, m1_row0_first - m1_row1_second, m1_row1_first + m1_row0_second)

        m2_local = m_coefficient - 10
        m2_part = m2_local // m2_coefficients
        m2_coefficient = m2_local % m2_coefficients
        m2_half_width = m2_coefficients * output_channels
        m2_linear_width = 2 * m2_half_width
        m2_edge_stride = 2 * m2_linear_width
        m2_base = row * m2_edge_stride + m2_coefficient * output_channels + channel
        m2_row0_first = tl.load(m2_ptr + m2_base, mask=main_active & (m_coefficient >= 10), other=0.0)
        m2_row0_second = tl.load(m2_ptr + m2_base + m2_half_width, mask=main_active & (m_coefficient >= 10), other=0.0)
        m2_row1_first = tl.load(m2_ptr + m2_base + m2_linear_width, mask=main_active & (m_coefficient >= 10), other=0.0)
        m2_row1_second = tl.load(m2_ptr + m2_base + m2_linear_width + m2_half_width, mask=main_active & (m_coefficient >= 10), other=0.0)
        m2_value = tl.where(m2_part == 0, m2_row0_first - m2_row1_second, m2_row1_first + m2_row0_second)
        value = tl.where(m_coefficient < 4, m0_value, tl.where(m_coefficient < 10, m1_value, m2_value))
        tl.store(out_ptr + row * main_size + offset, value, mask=main_active)

        if extra_channels > 0:
            gate_offset = offset - main_size
            gate_mask = gate_active & (gate_offset < extra_channels)
            tl.store(
                gating_ptr + row * extra_channels + gate_offset,
                tl.load(
                    m0_ptr + row * m0_channels + gate_offset,
                    mask=gate_mask,
                    other=0.0,
                ),
                mask=gate_mask,
            )

    @triton.jit
    def _so2_epilogue_backward_kernel(
        grad_out_ptr,
        grad_gating_ptr,
        l_to_m_ptr,
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        rows: tl.constexpr,
        output_coefficients: tl.constexpr,
        output_channels: tl.constexpr,
        m0_channels: tl.constexpr,
        m1_coefficients: tl.constexpr,
        m2_coefficients: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        main_size = output_coefficients * output_channels
        main_active = offset < main_size
        gate_active = offset >= main_size
        l_coefficient = offset // output_channels
        channel = offset % output_channels
        m_coefficient = tl.load(l_to_m_ptr + l_coefficient, mask=main_active, other=0)
        grad = tl.load(grad_out_ptr + row * main_size + offset, mask=main_active, other=0.0)

        m0_mask = main_active & (m_coefficient < 4)
        m0_offset = row * m0_channels + extra_channels + m_coefficient * output_channels + channel
        tl.store(grad_m0_ptr + m0_offset, grad, mask=m0_mask)

        m1_mask = main_active & (m_coefficient >= 4) & (m_coefficient < 10)
        m1_local = m_coefficient - 4
        m1_part = m1_local // m1_coefficients
        m1_coefficient = m1_local % m1_coefficients
        m1_half_width = m1_coefficients * output_channels
        m1_linear_width = 2 * m1_half_width
        m1_edge_stride = 2 * m1_linear_width
        m1_base = row * m1_edge_stride + m1_coefficient * output_channels + channel
        m1_row0_first = m1_base
        m1_row0_second = m1_base + m1_half_width
        m1_row1_first = m1_base + m1_linear_width
        m1_row1_second = m1_row1_first + m1_half_width
        m1_real_mask = m1_mask & (m1_part == 0)
        m1_imag_mask = m1_mask & (m1_part == 1)
        tl.store(grad_m1_ptr + m1_row0_first, grad, mask=m1_real_mask)
        tl.store(grad_m1_ptr + m1_row0_second, grad, mask=m1_imag_mask)
        tl.store(grad_m1_ptr + m1_row1_first, grad, mask=m1_imag_mask)
        tl.store(grad_m1_ptr + m1_row1_second, -grad, mask=m1_real_mask)

        m2_mask = main_active & (m_coefficient >= 10)
        m2_local = m_coefficient - 10
        m2_part = m2_local // m2_coefficients
        m2_coefficient = m2_local % m2_coefficients
        m2_half_width = m2_coefficients * output_channels
        m2_linear_width = 2 * m2_half_width
        m2_edge_stride = 2 * m2_linear_width
        m2_base = row * m2_edge_stride + m2_coefficient * output_channels + channel
        m2_row0_first = m2_base
        m2_row0_second = m2_base + m2_half_width
        m2_row1_first = m2_base + m2_linear_width
        m2_row1_second = m2_row1_first + m2_half_width
        m2_real_mask = m2_mask & (m2_part == 0)
        m2_imag_mask = m2_mask & (m2_part == 1)
        tl.store(grad_m2_ptr + m2_row0_first, grad, mask=m2_real_mask)
        tl.store(grad_m2_ptr + m2_row0_second, grad, mask=m2_imag_mask)
        tl.store(grad_m2_ptr + m2_row1_first, grad, mask=m2_imag_mask)
        tl.store(grad_m2_ptr + m2_row1_second, -grad, mask=m2_real_mask)

        if extra_channels > 0:
            gate_offset = offset - main_size
            gate_mask = gate_active & (gate_offset < extra_channels)
            tl.store(
                grad_m0_ptr + row * m0_channels + gate_offset,
                tl.load(
                    grad_gating_ptr + row * extra_channels + gate_offset,
                    mask=gate_mask,
                    other=0.0,
                ),
                mask=gate_mask,
            )

    @triton.jit
    def _so2_block_epilogue_kernel(
        m0_ptr,
        m1_ptr,
        m2_ptr,
        l_to_m_ptr,
        out_ptr,
        gating_ptr,
        rows: tl.constexpr,
        output_coefficients: tl.constexpr,
        output_channels: tl.constexpr,
        m0_channels: tl.constexpr,
        m1_coefficients: tl.constexpr,
        m2_coefficients: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        """Map already-recombined block-GEMM outputs back to l order."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        main_size = output_coefficients * output_channels
        main_active = offset < main_size
        gate_active = offset >= main_size
        l_coefficient = offset // output_channels
        channel = offset % output_channels
        m_coefficient = tl.load(
            l_to_m_ptr + l_coefficient, mask=main_active, other=0
        )

        m0_offset = (
            row * m0_channels
            + extra_channels
            + m_coefficient * output_channels
            + channel
        )
        m0_value = tl.load(
            m0_ptr + m0_offset,
            mask=main_active & (m_coefficient < 4),
            other=0.0,
        )

        m1_local = m_coefficient - 4
        m1_part = m1_local // m1_coefficients
        m1_coefficient = m1_local % m1_coefficients
        m1_half_width = m1_coefficients * output_channels
        m1_offset = (
            row * 2 * m1_half_width
            + m1_part * m1_half_width
            + m1_coefficient * output_channels
            + channel
        )
        m1_value = tl.load(
            m1_ptr + m1_offset,
            mask=main_active
            & (m_coefficient >= 4)
            & (m_coefficient < 10),
            other=0.0,
        )

        m2_local = m_coefficient - 10
        m2_part = m2_local // m2_coefficients
        m2_coefficient = m2_local % m2_coefficients
        m2_half_width = m2_coefficients * output_channels
        m2_offset = (
            row * 2 * m2_half_width
            + m2_part * m2_half_width
            + m2_coefficient * output_channels
            + channel
        )
        m2_value = tl.load(
            m2_ptr + m2_offset,
            mask=main_active & (m_coefficient >= 10),
            other=0.0,
        )
        value = tl.where(
            m_coefficient < 4,
            m0_value,
            tl.where(m_coefficient < 10, m1_value, m2_value),
        )
        tl.store(out_ptr + row * main_size + offset, value, mask=main_active)

        if extra_channels > 0:
            gate_offset = offset - main_size
            gate_mask = gate_active & (gate_offset < extra_channels)
            tl.store(
                gating_ptr + row * extra_channels + gate_offset,
                tl.load(
                    m0_ptr + row * m0_channels + gate_offset,
                    mask=gate_mask,
                    other=0.0,
                ),
                mask=gate_mask,
            )

    @triton.jit
    def _so2_block_epilogue_backward_kernel(
        grad_out_ptr,
        grad_gating_ptr,
        l_to_m_ptr,
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        rows: tl.constexpr,
        output_coefficients: tl.constexpr,
        output_channels: tl.constexpr,
        m0_channels: tl.constexpr,
        m1_coefficients: tl.constexpr,
        m2_coefficients: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        main_size = output_coefficients * output_channels
        main_active = offset < main_size
        gate_active = offset >= main_size
        l_coefficient = offset // output_channels
        channel = offset % output_channels
        m_coefficient = tl.load(
            l_to_m_ptr + l_coefficient, mask=main_active, other=0
        )
        grad = tl.load(
            grad_out_ptr + row * main_size + offset,
            mask=main_active,
            other=0.0,
        )

        m0_offset = (
            row * m0_channels
            + extra_channels
            + m_coefficient * output_channels
            + channel
        )
        tl.store(
            grad_m0_ptr + m0_offset,
            grad,
            mask=main_active & (m_coefficient < 4),
        )

        m1_local = m_coefficient - 4
        m1_part = m1_local // m1_coefficients
        m1_coefficient = m1_local % m1_coefficients
        m1_half_width = m1_coefficients * output_channels
        m1_offset = (
            row * 2 * m1_half_width
            + m1_part * m1_half_width
            + m1_coefficient * output_channels
            + channel
        )
        tl.store(
            grad_m1_ptr + m1_offset,
            grad,
            mask=main_active
            & (m_coefficient >= 4)
            & (m_coefficient < 10),
        )

        m2_local = m_coefficient - 10
        m2_part = m2_local // m2_coefficients
        m2_coefficient = m2_local % m2_coefficients
        m2_half_width = m2_coefficients * output_channels
        m2_offset = (
            row * 2 * m2_half_width
            + m2_part * m2_half_width
            + m2_coefficient * output_channels
            + channel
        )
        tl.store(
            grad_m2_ptr + m2_offset,
            grad,
            mask=main_active & (m_coefficient >= 10),
        )

        if extra_channels > 0:
            gate_offset = offset - main_size
            gate_mask = gate_active & (gate_offset < extra_channels)
            tl.store(
                grad_m0_ptr + row * m0_channels + gate_offset,
                tl.load(
                    grad_gating_ptr + row * extra_channels + gate_offset,
                    mask=gate_mask,
                    other=0.0,
                ),
                mask=gate_mask,
            )

    @triton.jit
    def _so2_block_gate_bridge_kernel(
        m0_ptr,
        m1_ptr,
        m2_ptr,
        m_degree_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        """Gate canonical block-GEMM results directly into conv2 inputs."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        degree = tl.load(m_degree_ptr + coefficient, mask=active, other=0)

        m0_value = tl.load(
            m0_ptr
            + row * (extra_channels + 4 * channels)
            + extra_channels
            + coefficient * channels
            + channel,
            mask=active & (coefficient < 4),
            other=0.0,
        )
        m1_value = tl.load(
            m1_ptr
            + row * 6 * channels
            + (coefficient - 4) * channels
            + channel,
            mask=active & (coefficient >= 4) & (coefficient < 10),
            other=0.0,
        )
        m2_value = tl.load(
            m2_ptr
            + row * 4 * channels
            + (coefficient - 10) * channels
            + channel,
            mask=active & (coefficient >= 10),
            other=0.0,
        )
        value = tl.where(
            coefficient < 4,
            m0_value,
            tl.where(coefficient < 10, m1_value, m2_value),
        )
        gate = tl.load(
            m0_ptr
            + row * (extra_channels + 4 * channels)
            + (degree - 1) * channels
            + channel,
            mask=active & (degree > 0),
            other=0.0,
        )
        gate_sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        scalar_sigmoid = 1.0 / (1.0 + tl.exp(-value))
        activated = tl.where(
            degree == 0,
            value * scalar_sigmoid,
            value * gate_sigmoid,
        )

        tl.store(
            out0_ptr + row * 4 * channels + coefficient * channels + channel,
            activated,
            mask=active & (coefficient < 4),
        )
        tl.store(
            out1_ptr
            + row * 6 * channels
            + (coefficient - 4) * channels
            + channel,
            activated,
            mask=active & (coefficient >= 4) & (coefficient < 10),
        )
        tl.store(
            out2_ptr
            + row * 4 * channels
            + (coefficient - 10) * channels
            + channel,
            activated,
            mask=active & (coefficient >= 10),
        )

    @triton.jit
    def _so2_block_gate_bridge_backward_kernel(
        grad_out0_ptr,
        grad_out1_ptr,
        grad_out2_ptr,
        m0_ptr,
        m1_ptr,
        m2_ptr,
        m_degree_ptr,
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        channel = tl.arange(0, block)
        active = channel < channels
        gate_grad_1 = tl.zeros((block,), tl.float32)
        gate_grad_2 = tl.zeros((block,), tl.float32)
        gate_grad_3 = tl.zeros((block,), tl.float32)

        for coefficient in range(14):
            degree = tl.load(m_degree_ptr + coefficient)
            m0_value = tl.load(
                m0_ptr
                + row * (extra_channels + 4 * channels)
                + extra_channels
                + coefficient * channels
                + channel,
                mask=active & (coefficient < 4),
                other=0.0,
            )
            m1_value = tl.load(
                m1_ptr
                + row * 6 * channels
                + (coefficient - 4) * channels
                + channel,
                mask=active & (coefficient >= 4) & (coefficient < 10),
                other=0.0,
            )
            m2_value = tl.load(
                m2_ptr
                + row * 4 * channels
                + (coefficient - 10) * channels
                + channel,
                mask=active & (coefficient >= 10),
                other=0.0,
            )
            value = tl.where(
                coefficient < 4,
                m0_value,
                tl.where(coefficient < 10, m1_value, m2_value),
            )
            grad_out = tl.load(
                grad_out0_ptr
                + row * 4 * channels
                + coefficient * channels
                + channel,
                mask=active & (coefficient < 4),
                other=0.0,
            )
            grad_out = tl.where(
                coefficient < 4,
                grad_out,
                tl.where(
                    coefficient < 10,
                    tl.load(
                        grad_out1_ptr
                        + row * 6 * channels
                        + (coefficient - 4) * channels
                        + channel,
                        mask=active
                        & (coefficient >= 4)
                        & (coefficient < 10),
                        other=0.0,
                    ),
                    tl.load(
                        grad_out2_ptr
                        + row * 4 * channels
                        + (coefficient - 10) * channels
                        + channel,
                        mask=active & (coefficient >= 10),
                        other=0.0,
                    ),
                ),
            )
            gate = tl.load(
                m0_ptr
                + row * (extra_channels + 4 * channels)
                + (degree - 1) * channels
                + channel,
                mask=active & (degree > 0),
                other=0.0,
            )
            sigmoid = 1.0 / (1.0 + tl.exp(-gate))
            scalar_sigmoid = 1.0 / (1.0 + tl.exp(-value))
            grad_value = tl.where(
                degree == 0,
                grad_out
                * (
                    scalar_sigmoid
                    + value * scalar_sigmoid * (1.0 - scalar_sigmoid)
                ),
                grad_out * sigmoid,
            )
            gate_contribution = (
                grad_out * value * sigmoid * (1.0 - sigmoid)
            )
            gate_grad_1 += tl.where(degree == 1, gate_contribution, 0.0)
            gate_grad_2 += tl.where(degree == 2, gate_contribution, 0.0)
            gate_grad_3 += tl.where(degree == 3, gate_contribution, 0.0)

            tl.store(
                grad_m0_ptr
                + row * (extra_channels + 4 * channels)
                + extra_channels
                + coefficient * channels
                + channel,
                grad_value,
                mask=active & (coefficient < 4),
            )
            tl.store(
                grad_m1_ptr
                + row * 6 * channels
                + (coefficient - 4) * channels
                + channel,
                grad_value,
                mask=active & (coefficient >= 4) & (coefficient < 10),
            )
            tl.store(
                grad_m2_ptr
                + row * 4 * channels
                + (coefficient - 10) * channels
                + channel,
                grad_value,
                mask=active & (coefficient >= 10),
            )

        tl.store(
            grad_m0_ptr + row * (extra_channels + 4 * channels) + channel,
            gate_grad_1,
            mask=active,
        )
        tl.store(
            grad_m0_ptr
            + row * (extra_channels + 4 * channels)
            + channels
            + channel,
            gate_grad_2,
            mask=active,
        )
        tl.store(
            grad_m0_ptr
            + row * (extra_channels + 4 * channels)
            + 2 * channels
            + channel,
            gate_grad_3,
            mask=active,
        )

    @triton.jit
    def _so2_gate_bridge_kernel(
        m0_ptr,
        m1_ptr,
        m2_ptr,
        m_degree_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        rows: tl.constexpr,
        coefficients: tl.constexpr,
        channels: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        """Apply the Edgewise gate while remaining in the m-order layout."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offset = tile * block + tl.arange(0, block)
        active = offset < coefficients * channels
        coefficient = offset // channels
        channel = offset % channels
        degree = tl.load(m_degree_ptr + coefficient, mask=active, other=0)

        m0_offset = (
            row * (extra_channels + 4 * channels)
            + extra_channels
            + coefficient * channels
            + channel
        )
        m0_value = tl.load(
            m0_ptr + m0_offset,
            mask=active & (coefficient < 4),
            other=0.0,
        )

        m1_local = coefficient - 4
        m1_part = m1_local // 3
        m1_coefficient = m1_local % 3
        m1_half_width = 3 * channels
        m1_linear_width = 2 * m1_half_width
        m1_edge_stride = 2 * m1_linear_width
        m1_base = row * m1_edge_stride + m1_coefficient * channels + channel
        m1_mask = active & (coefficient >= 4) & (coefficient < 10)
        m1_row0_first = tl.load(m1_ptr + m1_base, mask=m1_mask, other=0.0)
        m1_row0_second = tl.load(
            m1_ptr + m1_base + m1_half_width, mask=m1_mask, other=0.0
        )
        m1_row1_first = tl.load(
            m1_ptr + m1_base + m1_linear_width, mask=m1_mask, other=0.0
        )
        m1_row1_second = tl.load(
            m1_ptr + m1_base + m1_linear_width + m1_half_width,
            mask=m1_mask,
            other=0.0,
        )
        m1_value = tl.where(
            m1_part == 0,
            m1_row0_first - m1_row1_second,
            m1_row1_first + m1_row0_second,
        )

        m2_local = coefficient - 10
        m2_part = m2_local // 2
        m2_coefficient = m2_local % 2
        m2_half_width = 2 * channels
        m2_linear_width = 2 * m2_half_width
        m2_edge_stride = 2 * m2_linear_width
        m2_base = row * m2_edge_stride + m2_coefficient * channels + channel
        m2_mask = active & (coefficient >= 10)
        m2_row0_first = tl.load(m2_ptr + m2_base, mask=m2_mask, other=0.0)
        m2_row0_second = tl.load(
            m2_ptr + m2_base + m2_half_width, mask=m2_mask, other=0.0
        )
        m2_row1_first = tl.load(
            m2_ptr + m2_base + m2_linear_width, mask=m2_mask, other=0.0
        )
        m2_row1_second = tl.load(
            m2_ptr + m2_base + m2_linear_width + m2_half_width,
            mask=m2_mask,
            other=0.0,
        )
        m2_value = tl.where(
            m2_part == 0,
            m2_row0_first - m2_row1_second,
            m2_row1_first + m2_row0_second,
        )

        value = tl.where(
            coefficient < 4,
            m0_value,
            tl.where(coefficient < 10, m1_value, m2_value),
        )
        gate_offset = (
            row * (extra_channels + 4 * channels)
            + (degree - 1) * channels
            + channel
        )
        gate = tl.load(
            m0_ptr + gate_offset,
            mask=active & (degree > 0),
            other=0.0,
        )
        gate_sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        scalar_sigmoid = 1.0 / (1.0 + tl.exp(-value))
        activated = tl.where(
            degree == 0, value * scalar_sigmoid, value * gate_sigmoid
        )

        tl.store(
            out0_ptr + row * 4 * channels + coefficient * channels + channel,
            activated,
            mask=active & (coefficient < 4),
        )
        tl.store(
            out1_ptr
            + row * 6 * channels
            + (coefficient - 4) * channels
            + channel,
            activated,
            mask=active & (coefficient >= 4) & (coefficient < 10),
        )
        tl.store(
            out2_ptr
            + row * 4 * channels
            + (coefficient - 10) * channels
            + channel,
            activated,
            mask=active & (coefficient >= 10),
        )

    @triton.jit
    def _so2_gate_bridge_backward_kernel(
        grad_out0_ptr,
        grad_out1_ptr,
        grad_out2_ptr,
        m0_ptr,
        m1_ptr,
        m2_ptr,
        m_degree_ptr,
        grad_m0_ptr,
        grad_m1_ptr,
        grad_m2_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        extra_channels: tl.constexpr,
        block: tl.constexpr,
    ):
        """Differentiate the bridge with one non-atomic gate reduction."""
        row = tl.program_id(0)
        channel = tl.arange(0, block)
        active = channel < channels
        gate_grad_1 = tl.zeros((block,), tl.float32)
        gate_grad_2 = tl.zeros((block,), tl.float32)
        gate_grad_3 = tl.zeros((block,), tl.float32)

        for coefficient in range(14):
            degree = tl.load(m_degree_ptr + coefficient)
            m0_value = tl.load(
                m0_ptr
                + row * (extra_channels + 4 * channels)
                + extra_channels
                + coefficient * channels
                + channel,
                mask=active & (coefficient < 4),
                other=0.0,
            )

            m1_local = coefficient - 4
            m1_part = m1_local // 3
            m1_coefficient = m1_local % 3
            m1_half_width = 3 * channels
            m1_linear_width = 2 * m1_half_width
            m1_edge_stride = 2 * m1_linear_width
            m1_base = (
                row * m1_edge_stride + m1_coefficient * channels + channel
            )
            m1_mask = active & (coefficient >= 4) & (coefficient < 10)
            m1_row0_first = tl.load(
                m1_ptr + m1_base, mask=m1_mask, other=0.0
            )
            m1_row0_second = tl.load(
                m1_ptr + m1_base + m1_half_width,
                mask=m1_mask,
                other=0.0,
            )
            m1_row1_first = tl.load(
                m1_ptr + m1_base + m1_linear_width,
                mask=m1_mask,
                other=0.0,
            )
            m1_row1_second = tl.load(
                m1_ptr + m1_base + m1_linear_width + m1_half_width,
                mask=m1_mask,
                other=0.0,
            )
            m1_value = tl.where(
                m1_part == 0,
                m1_row0_first - m1_row1_second,
                m1_row1_first + m1_row0_second,
            )

            m2_local = coefficient - 10
            m2_part = m2_local // 2
            m2_coefficient = m2_local % 2
            m2_half_width = 2 * channels
            m2_linear_width = 2 * m2_half_width
            m2_edge_stride = 2 * m2_linear_width
            m2_base = (
                row * m2_edge_stride + m2_coefficient * channels + channel
            )
            m2_mask = active & (coefficient >= 10)
            m2_row0_first = tl.load(
                m2_ptr + m2_base, mask=m2_mask, other=0.0
            )
            m2_row0_second = tl.load(
                m2_ptr + m2_base + m2_half_width,
                mask=m2_mask,
                other=0.0,
            )
            m2_row1_first = tl.load(
                m2_ptr + m2_base + m2_linear_width,
                mask=m2_mask,
                other=0.0,
            )
            m2_row1_second = tl.load(
                m2_ptr + m2_base + m2_linear_width + m2_half_width,
                mask=m2_mask,
                other=0.0,
            )
            m2_value = tl.where(
                m2_part == 0,
                m2_row0_first - m2_row1_second,
                m2_row1_first + m2_row0_second,
            )
            value = tl.where(
                coefficient < 4,
                m0_value,
                tl.where(coefficient < 10, m1_value, m2_value),
            )

            grad0 = tl.load(
                grad_out0_ptr
                + row * 4 * channels
                + coefficient * channels
                + channel,
                mask=active & (coefficient < 4),
                other=0.0,
            )
            grad1 = tl.load(
                grad_out1_ptr
                + row * 6 * channels
                + (coefficient - 4) * channels
                + channel,
                mask=active & (coefficient >= 4) & (coefficient < 10),
                other=0.0,
            )
            grad2 = tl.load(
                grad_out2_ptr
                + row * 4 * channels
                + (coefficient - 10) * channels
                + channel,
                mask=active & (coefficient >= 10),
                other=0.0,
            )
            grad_out = tl.where(
                coefficient < 4,
                grad0,
                tl.where(coefficient < 10, grad1, grad2),
            )

            gate_offset = (
                row * (extra_channels + 4 * channels)
                + (degree - 1) * channels
                + channel
            )
            gate = tl.load(
                m0_ptr + gate_offset,
                mask=active & (degree > 0),
                other=0.0,
            )
            gate_sigmoid = 1.0 / (1.0 + tl.exp(-gate))
            scalar_sigmoid = 1.0 / (1.0 + tl.exp(-value))
            scalar_derivative = scalar_sigmoid * (
                1.0 + value * (1.0 - scalar_sigmoid)
            )
            grad_value = grad_out * tl.where(
                degree == 0, scalar_derivative, gate_sigmoid
            )
            gate_term = (
                grad_out * value * gate_sigmoid * (1.0 - gate_sigmoid)
            )
            gate_grad_1 += tl.where(degree == 1, gate_term, 0.0)
            gate_grad_2 += tl.where(degree == 2, gate_term, 0.0)
            gate_grad_3 += tl.where(degree == 3, gate_term, 0.0)

            tl.store(
                grad_m0_ptr
                + row * (extra_channels + 4 * channels)
                + extra_channels
                + coefficient * channels
                + channel,
                grad_value,
                mask=active & (coefficient < 4),
            )
            m1_real_mask = m1_mask & (m1_part == 0)
            m1_imag_mask = m1_mask & (m1_part == 1)
            tl.store(
                grad_m1_ptr + m1_base, grad_value, mask=m1_real_mask
            )
            tl.store(
                grad_m1_ptr + m1_base + m1_half_width,
                grad_value,
                mask=m1_imag_mask,
            )
            tl.store(
                grad_m1_ptr + m1_base + m1_linear_width,
                grad_value,
                mask=m1_imag_mask,
            )
            tl.store(
                grad_m1_ptr
                + m1_base
                + m1_linear_width
                + m1_half_width,
                -grad_value,
                mask=m1_real_mask,
            )
            m2_real_mask = m2_mask & (m2_part == 0)
            m2_imag_mask = m2_mask & (m2_part == 1)
            tl.store(
                grad_m2_ptr + m2_base, grad_value, mask=m2_real_mask
            )
            tl.store(
                grad_m2_ptr + m2_base + m2_half_width,
                grad_value,
                mask=m2_imag_mask,
            )
            tl.store(
                grad_m2_ptr + m2_base + m2_linear_width,
                grad_value,
                mask=m2_imag_mask,
            )
            tl.store(
                grad_m2_ptr
                + m2_base
                + m2_linear_width
                + m2_half_width,
                -grad_value,
                mask=m2_real_mask,
            )

        tl.store(
            grad_m0_ptr + row * (extra_channels + 4 * channels) + channel,
            gate_grad_1,
            mask=active,
        )
        tl.store(
            grad_m0_ptr
            + row * (extra_channels + 4 * channels)
            + channels
            + channel,
            gate_grad_2,
            mask=active,
        )
        tl.store(
            grad_m0_ptr
            + row * (extra_channels + 4 * channels)
            + 2 * channels
            + channel,
            gate_grad_3,
            mask=active,
        )

    @triton.jit
    def _radial_mlp_forward_kernel(
        x_ptr,
        w1_ptr,
        b1_ptr,
        g1_ptr,
        be1_ptr,
        w2_ptr,
        b2_ptr,
        g2_ptr,
        be2_ptr,
        w3_ptr,
        b3_ptr,
        out_ptr,
        save_a1_ptr,
        save_hhat1_ptr,
        save_rstd1_ptr,
        save_a2_ptr,
        save_hhat2_ptr,
        save_rstd2_ptr,
        rows: tl.constexpr,
        in_ch: tl.constexpr,
        h1_ch: tl.constexpr,
        h2_ch: tl.constexpr,
        out_ch: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        """Fused Linear+LayerNorm+SiLU+Linear+LayerNorm+SiLU+Linear over rows.

        Weights are row-major ``[out, in]`` like ``nn.Linear.weight``.  FP32
        ``tl.dot`` with ``input_precision="ieee"`` keeps the exact FP32
        arithmetic class of the eSEN inference path (TF32 stays disabled).
        """
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < rows

        j1 = tl.arange(0, h1_ch)
        acc1 = tl.zeros((BLOCK_R, h1_ch), tl.float32)
        for k0 in range(0, in_ch, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            x_t = tl.load(
                x_ptr + r[:, None] * in_ch + k[None, :],
                mask=rmask[:, None] & (k[None, :] < in_ch),
                other=0.0,
            )
            w_t = tl.load(
                w1_ptr + j1[:, None] * in_ch + k[None, :],
                mask=k[None, :] < in_ch,
                other=0.0,
            )
            acc1 = tl.dot(x_t, tl.trans(w_t), acc1, input_precision="ieee")
        h1 = acc1 + tl.load(b1_ptr + j1)[None, :]
        mean1 = tl.sum(h1, axis=1) / h1_ch
        h1c = h1 - mean1[:, None]
        rstd1 = 1.0 / tl.sqrt(tl.sum(h1c * h1c, axis=1) / h1_ch + eps)
        hhat1 = h1c * rstd1[:, None]
        pre1 = hhat1 * tl.load(g1_ptr + j1)[None, :] + tl.load(be1_ptr + j1)[None, :]
        sig1 = 1.0 / (1.0 + tl.exp(-pre1))
        a1 = pre1 * sig1
        tl.store(save_hhat1_ptr + r[:, None] * h1_ch + j1[None, :], hhat1, mask=rmask[:, None])
        tl.store(save_rstd1_ptr + r, rstd1, mask=rmask)
        # Save the preactivation.  The backward derivative of SiLU needs x,
        # not SiLU(x); the activated value is recomputed below for Linear 2.
        tl.store(save_a1_ptr + r[:, None] * h1_ch + j1[None, :], pre1, mask=rmask[:, None])

        j2 = tl.arange(0, h2_ch)
        acc2 = tl.zeros((BLOCK_R, h2_ch), tl.float32)
        for k0 in range(0, h1_ch, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            pre1_t = tl.load(
                save_a1_ptr + r[:, None] * h1_ch + k[None, :],
                mask=rmask[:, None] & (k[None, :] < h1_ch),
                other=0.0,
            )
            sig1_t = 1.0 / (1.0 + tl.exp(-pre1_t))
            a_t = pre1_t * sig1_t
            w_t = tl.load(
                w2_ptr + j2[:, None] * h1_ch + k[None, :],
                mask=(j2[:, None] < h2_ch) & (k[None, :] < h1_ch),
                other=0.0,
            )
            acc2 = tl.dot(a_t, tl.trans(w_t), acc2, input_precision="ieee")
        h2 = acc2 + tl.load(b2_ptr + j2)[None, :]
        mean2 = tl.sum(h2, axis=1) / h2_ch
        h2c = h2 - mean2[:, None]
        rstd2 = 1.0 / tl.sqrt(tl.sum(h2c * h2c, axis=1) / h2_ch + eps)
        hhat2 = h2c * rstd2[:, None]
        pre2 = hhat2 * tl.load(g2_ptr + j2)[None, :] + tl.load(be2_ptr + j2)[None, :]
        sig2 = 1.0 / (1.0 + tl.exp(-pre2))
        a2 = pre2 * sig2
        tl.store(save_hhat2_ptr + r[:, None] * h2_ch + j2[None, :], hhat2, mask=rmask[:, None])
        tl.store(save_rstd2_ptr + r, rstd2, mask=rmask)
        tl.store(save_a2_ptr + r[:, None] * h2_ch + j2[None, :], pre2, mask=rmask[:, None])

        for o0 in range(0, out_ch, BLOCK_O):
            o = o0 + tl.arange(0, BLOCK_O)
            omask = o < out_ch
            acc3 = tl.zeros((BLOCK_R, BLOCK_O), tl.float32)
            for k0 in range(0, h2_ch, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                pre2_t = tl.load(
                    save_a2_ptr + r[:, None] * h2_ch + k[None, :],
                    mask=rmask[:, None] & (k[None, :] < h2_ch),
                    other=0.0,
                )
                sig2_t = 1.0 / (1.0 + tl.exp(-pre2_t))
                a_t = pre2_t * sig2_t
                w_t = tl.load(
                    w3_ptr + o[:, None] * h2_ch + k[None, :],
                    mask=omask[:, None] & (k[None, :] < h2_ch),
                    other=0.0,
                )
                acc3 = tl.dot(a_t, tl.trans(w_t), acc3, input_precision="ieee")
            out_t = acc3 + tl.load(b3_ptr + o, mask=omask, other=0.0)[None, :]
            tl.store(
                out_ptr + r[:, None] * out_ch + o[None, :],
                out_t,
                mask=rmask[:, None] & omask[None, :],
            )

    @triton.jit
    def _radial_mlp_backward_kernel(
        grad_out_ptr,
        save_a1_ptr,
        save_hhat1_ptr,
        save_rstd1_ptr,
        save_a2_ptr,
        save_hhat2_ptr,
        save_rstd2_ptr,
        g1_ptr,
        g2_ptr,
        w1_ptr,
        w2_ptr,
        w3_ptr,
        scratch_g_h2_ptr,
        scratch_g_h1_ptr,
        grad_x_ptr,
        rows: tl.constexpr,
        in_ch: tl.constexpr,
        h1_ch: tl.constexpr,
        h2_ch: tl.constexpr,
        out_ch: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        """Input-gradients only for the fused radial MLP (weights are frozen)."""
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < rows

        j2 = tl.arange(0, h2_ch)
        g_a2 = tl.zeros((BLOCK_R, h2_ch), tl.float32)
        for o0 in range(0, out_ch, BLOCK_O):
            o = o0 + tl.arange(0, BLOCK_O)
            omask = o < out_ch
            gy_t = tl.load(
                grad_out_ptr + r[:, None] * out_ch + o[None, :],
                mask=rmask[:, None] & omask[None, :],
                other=0.0,
            )
            w_t = tl.load(
                w3_ptr + o[:, None] * h2_ch + j2[None, :],
                mask=omask[:, None],
                other=0.0,
            )
            g_a2 = tl.dot(gy_t, w_t, g_a2, input_precision="ieee")
        pre2 = tl.load(
            save_a2_ptr + r[:, None] * h2_ch + j2[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig2 = 1.0 / (1.0 + tl.exp(-pre2))
        g_n2 = g_a2 * (sig2 * (1.0 + pre2 * (1.0 - sig2)))
        hhat2 = tl.load(
            save_hhat2_ptr + r[:, None] * h2_ch + j2[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        rstd2 = tl.load(save_rstd2_ptr + r, mask=rmask, other=0.0)
        gz2 = g_n2 * tl.load(g2_ptr + j2)[None, :]
        mean_gz2 = tl.sum(gz2, axis=1) / h2_ch
        mean_gz2h = tl.sum(gz2 * hhat2, axis=1) / h2_ch
        g_h2 = (gz2 - mean_gz2[:, None] - hhat2 * mean_gz2h[:, None]) * rstd2[:, None]
        tl.store(scratch_g_h2_ptr + r[:, None] * h2_ch + j2[None, :], g_h2, mask=rmask[:, None])

        j1 = tl.arange(0, h1_ch)
        g_a1 = tl.zeros((BLOCK_R, h1_ch), tl.float32)
        for k0 in range(0, h2_ch, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            gh_t = tl.load(
                scratch_g_h2_ptr + r[:, None] * h2_ch + k[None, :],
                mask=rmask[:, None] & (k[None, :] < h2_ch),
                other=0.0,
            )
            # g_a1[r, i] = sum_j g_h2[r, j] * W2[j, i]
            w_t = tl.load(
                w2_ptr + k[:, None] * h1_ch + j1[None, :],
                mask=(k[:, None] < h2_ch) & (j1[None, :] < h1_ch),
                other=0.0,
            )
            g_a1 = tl.dot(gh_t, w_t, g_a1, input_precision="ieee")
        pre1 = tl.load(
            save_a1_ptr + r[:, None] * h1_ch + j1[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig1 = 1.0 / (1.0 + tl.exp(-pre1))
        g_n1 = g_a1 * (sig1 * (1.0 + pre1 * (1.0 - sig1)))
        hhat1 = tl.load(
            save_hhat1_ptr + r[:, None] * h1_ch + j1[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        rstd1 = tl.load(save_rstd1_ptr + r, mask=rmask, other=0.0)
        gz1 = g_n1 * tl.load(g1_ptr + j1)[None, :]
        mean_gz1 = tl.sum(gz1, axis=1) / h1_ch
        mean_gz1h = tl.sum(gz1 * hhat1, axis=1) / h1_ch
        g_h1 = (gz1 - mean_gz1[:, None] - hhat1 * mean_gz1h[:, None]) * rstd1[:, None]
        tl.store(scratch_g_h1_ptr + r[:, None] * h1_ch + j1[None, :], g_h1, mask=rmask[:, None])

        for k0 in range(0, in_ch, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < in_ch
            gh_t = tl.load(
                scratch_g_h1_ptr + r[:, None] * h1_ch + j1[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            w_t = tl.load(
                w1_ptr + j1[:, None] * in_ch + k[None, :],
                mask=kmask[None, :],
                other=0.0,
            )
            acc = tl.dot(gh_t, w_t, input_precision="ieee")
            tl.store(
                grad_x_ptr + r[:, None] * in_ch + k[None, :],
                acc,
                mask=rmask[:, None] & kmask[None, :],
            )

    @triton.jit
    def _so3_mlp_forward_kernel(
        x_ptr,
        gating_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        scratch_h_ptr,
        out_ptr,
        rows: tl.constexpr,
        coeffs: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused SO3_Linear + GateActivation + SO3_Linear per atom.

        Weights are the pre-expanded ``[coeffs, channels, channels]``
        (``SO3_Linear`` per-l weights gathered with ``expand_index`` once at
        configure time; the per-step ``index_select`` disappears entirely).
        Bias is applied to coefficient row 0 only, exactly like ``SO3_Linear``.
        """
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < rows
        c = tl.arange(0, channels)
        for m in range(coeffs):
            acc = tl.zeros((BLOCK_R, channels), tl.float32)
            for k0 in range(0, channels, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                x_t = tl.load(
                    x_ptr + r[:, None] * coeffs * channels + m * channels + k[None, :],
                    mask=rmask[:, None] & (k[None, :] < channels),
                    other=0.0,
                )
                w_t = tl.load(
                    w1_ptr + m * channels * channels + c[:, None] * channels + k[None, :],
                    mask=(c[:, None] < channels) & (k[None, :] < channels),
                    other=0.0,
                )
                acc = tl.dot(x_t, tl.trans(w_t), acc, input_precision="ieee")
            if m == 0:
                h = acc + tl.load(b1_ptr + c)[None, :]
            else:
                h = acc
            tl.store(
                scratch_h_ptr + r[:, None] * coeffs * channels + m * channels + c[None, :],
                h,
                mask=rmask[:, None],
            )
            if m == 0:
                sig = 1.0 / (1.0 + tl.exp(-h))
                gated = h * sig
            else:
                degree = 0 if m <= 3 else (1 if m <= 8 else 2)
                gate = tl.load(
                    gating_ptr + r[:, None] * 3 * channels + degree * channels + c[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
                gated = h * (1.0 / (1.0 + tl.exp(-gate)))
            acc2 = tl.zeros((BLOCK_R, channels), tl.float32)
            for k0 in range(0, channels, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                h_t = tl.load(
                    scratch_h_ptr + r[:, None] * coeffs * channels + m * channels + k[None, :],
                    mask=rmask[:, None] & (k[None, :] < channels),
                    other=0.0,
                )
                if m == 0:
                    sig = 1.0 / (1.0 + tl.exp(-h_t))
                    g_t = h_t * sig
                else:
                    degree = 0 if m <= 3 else (1 if m <= 8 else 2)
                    gate = tl.load(
                        gating_ptr + r[:, None] * 3 * channels + degree * channels + k[None, :],
                        mask=rmask[:, None],
                        other=0.0,
                    )
                    g_t = h_t * (1.0 / (1.0 + tl.exp(-gate)))
                w_t = tl.load(
                    w2_ptr + m * channels * channels + c[:, None] * channels + k[None, :],
                    mask=(c[:, None] < channels) & (k[None, :] < channels),
                    other=0.0,
                )
                acc2 = tl.dot(g_t, tl.trans(w_t), acc2, input_precision="ieee")
            if m == 0:
                out = acc2 + tl.load(b2_ptr + c)[None, :]
            else:
                out = acc2
            tl.store(
                out_ptr + r[:, None] * coeffs * channels + m * channels + c[None, :],
                out,
                mask=rmask[:, None],
            )

    @triton.jit
    def _so3_mlp_backward_kernel(
        grad_out_ptr,
        x_ptr,
        gating_ptr,
        w1_ptr,
        w2_ptr,
        scratch_h_ptr,
        scratch_g_h_ptr,
        grad_gating_ptr,
        grad_x_ptr,
        rows: tl.constexpr,
        coeffs: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < rows
        c = tl.arange(0, channels)
        for m in range(coeffs):
            g_gated = tl.zeros((BLOCK_R, channels), tl.float32)
            for k0 in range(0, channels, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                go_t = tl.load(
                    grad_out_ptr + r[:, None] * coeffs * channels + m * channels + k[None, :],
                    mask=rmask[:, None] & (k[None, :] < channels),
                    other=0.0,
                )
                w_t = tl.load(
                    w2_ptr + m * channels * channels + k[:, None] * channels + c[None, :],
                    mask=(k[:, None] < channels) & (c[None, :] < channels),
                    other=0.0,
                )
                g_gated = tl.dot(go_t, w_t, g_gated, input_precision="ieee")
            h_t = tl.load(
                scratch_h_ptr + r[:, None] * coeffs * channels + m * channels + c[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            if m == 0:
                sig = 1.0 / (1.0 + tl.exp(-h_t))
                g_h = g_gated * (sig * (1.0 + h_t * (1.0 - sig)))
            else:
                degree = 0 if m <= 3 else (1 if m <= 8 else 2)
                gate = tl.load(
                    gating_ptr + r[:, None] * 3 * channels + degree * channels + c[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
                sig = 1.0 / (1.0 + tl.exp(-gate))
                g_h = g_gated * sig
                tl.atomic_add(
                    grad_gating_ptr + r[:, None] * 3 * channels + degree * channels + c[None, :],
                    g_h * h_t * (1.0 - sig),
                    mask=rmask[:, None],
                )
            tl.store(
                scratch_g_h_ptr + r[:, None] * coeffs * channels + m * channels + c[None, :],
                g_h,
                mask=rmask[:, None],
            )
            grad_x_acc = tl.zeros((BLOCK_R, channels), tl.float32)
            for k0 in range(0, channels, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                gh_t = tl.load(
                    scratch_g_h_ptr + r[:, None] * coeffs * channels + m * channels + k[None, :],
                    mask=rmask[:, None] & (k[None, :] < channels),
                    other=0.0,
                )
                w_t = tl.load(
                    w1_ptr + m * channels * channels + k[:, None] * channels + c[None, :],
                    mask=(k[:, None] < channels) & (c[None, :] < channels),
                    other=0.0,
                )
                grad_x_acc = tl.dot(
                    gh_t, w_t, grad_x_acc, input_precision="ieee"
                )
            tl.store(
                grad_x_ptr + r[:, None] * coeffs * channels + m * channels + c[None, :],
                grad_x_acc,
                mask=rmask[:, None],
            )

    @triton.jit
    def _energy_mlp_forward_kernel(
        x_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        w3_ptr,
        b3_ptr,
        save_a1_ptr,
        save_a2_ptr,
        out_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < rows
        j = tl.arange(0, channels)
        acc1 = tl.zeros((BLOCK_R, channels), tl.float32)
        for k0 in range(0, channels, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            x_t = tl.load(
                x_ptr + r[:, None] * channels + k[None, :],
                mask=rmask[:, None] & (k[None, :] < channels),
                other=0.0,
            )
            w_t = tl.load(
                w1_ptr + j[:, None] * channels + k[None, :],
                mask=(j[:, None] < channels) & (k[None, :] < channels),
                other=0.0,
            )
            acc1 = tl.dot(x_t, tl.trans(w_t), acc1, input_precision="ieee")
        h1 = acc1 + tl.load(b1_ptr + j)[None, :]
        sig1 = 1.0 / (1.0 + tl.exp(-h1))
        a1 = h1 * sig1
        tl.store(save_a1_ptr + r[:, None] * channels + j[None, :], h1, mask=rmask[:, None])

        acc2 = tl.zeros((BLOCK_R, channels), tl.float32)
        for k0 in range(0, channels, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            h1_t = tl.load(
                save_a1_ptr + r[:, None] * channels + k[None, :],
                mask=rmask[:, None] & (k[None, :] < channels),
                other=0.0,
            )
            sig1_t = 1.0 / (1.0 + tl.exp(-h1_t))
            a_t = h1_t * sig1_t
            w_t = tl.load(
                w2_ptr + j[:, None] * channels + k[None, :],
                mask=(j[:, None] < channels) & (k[None, :] < channels),
                other=0.0,
            )
            acc2 = tl.dot(a_t, tl.trans(w_t), acc2, input_precision="ieee")
        h2 = acc2 + tl.load(b2_ptr + j)[None, :]
        sig2 = 1.0 / (1.0 + tl.exp(-h2))
        a2 = h2 * sig2
        tl.store(save_a2_ptr + r[:, None] * channels + j[None, :], h2, mask=rmask[:, None])

        w3_t = tl.load(w3_ptr + j)
        out = tl.sum(a2 * w3_t[None, :], axis=1) + tl.load(b3_ptr)
        tl.store(out_ptr + r, out, mask=rmask)

    @triton.jit
    def _energy_mlp_backward_kernel(
        grad_out_ptr,
        save_a1_ptr,
        save_a2_ptr,
        w1_ptr,
        w2_ptr,
        w3_ptr,
        scratch_g_a2_ptr,
        scratch_g_a1_ptr,
        grad_x_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < rows
        j = tl.arange(0, channels)
        gy = tl.load(grad_out_ptr + r, mask=rmask, other=0.0)
        w3_t = tl.load(w3_ptr + j)
        g_a2 = gy[:, None] * w3_t[None, :]
        h2 = tl.load(
            save_a2_ptr + r[:, None] * channels + j[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig2 = 1.0 / (1.0 + tl.exp(-h2))
        g_n2 = g_a2 * (sig2 * (1.0 + h2 * (1.0 - sig2)))
        tl.store(scratch_g_a2_ptr + r[:, None] * channels + j[None, :], g_n2, mask=rmask[:, None])

        g_a1 = tl.zeros((BLOCK_R, channels), tl.float32)
        for k0 in range(0, channels, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            gn_t = tl.load(
                scratch_g_a2_ptr + r[:, None] * channels + k[None, :],
                mask=rmask[:, None] & (k[None, :] < channels),
                other=0.0,
            )
            # g_a1[r, i] = sum_j g_n2[r, j] * W2[j, i]
            w_t = tl.load(
                w2_ptr + k[:, None] * channels + j[None, :],
                mask=(k[:, None] < channels) & (j[None, :] < channels),
                other=0.0,
            )
            g_a1 = tl.dot(gn_t, w_t, g_a1, input_precision="ieee")
        h1 = tl.load(
            save_a1_ptr + r[:, None] * channels + j[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig1 = 1.0 / (1.0 + tl.exp(-h1))
        g_n1 = g_a1 * (sig1 * (1.0 + h1 * (1.0 - sig1)))
        tl.store(scratch_g_a1_ptr + r[:, None] * channels + j[None, :], g_n1, mask=rmask[:, None])

        for k0 in range(0, channels, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            gn_t = tl.load(
                scratch_g_a1_ptr + r[:, None] * channels + j[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            w_t = tl.load(
                w1_ptr + j[:, None] * channels + k[None, :],
                mask=(j[:, None] < channels) & (k[None, :] < channels),
                other=0.0,
            )
            # g_x[r, k] = sum_j g_a1[r, j] * W1[j, k].  The previous
            # implementation loaded a [R, BLOCK_K] slice of g_a1 and formed
            # a [R, channels] result, then attempted to store it into a
            # [R, BLOCK_K] tile.  Load all output channels and keep only the
            # current input-channel tile in the weight matrix.
            acc = tl.dot(gn_t, w_t, input_precision="ieee")
            tl.store(
                grad_x_ptr + r[:, None] * channels + k[None, :],
                acc,
                mask=rmask[:, None],
            )


class _GatherWigner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, edge_index: Tensor, wigner: Tensor, out_mask: Tensor):
        if x.ndim != 3 or x.shape[1:] != (16, 128):
            raise UnsupportedFusionConfigError(
                f"Gather/Wigner expected x=[N,16,128], got {tuple(x.shape)}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise UnsupportedFusionConfigError("Gather/Wigner expected edge_index=[2,E]")
        if wigner.shape != (edge_index.shape[1], 16, 16):
            raise UnsupportedFusionConfigError(
                f"Gather/Wigner received wigner={tuple(wigner.shape)}"
            )
        source_index = edge_index[0].contiguous()
        target_index = edge_index[1].contiguous()
        if source_index.dtype != torch.long or target_index.dtype != torch.long:
            raise UnsupportedFusionConfigError("Gather/Wigner requires int64 edge indices")
        out_mask = out_mask.to(device=x.device, dtype=torch.long).contiguous()
        _require_triton_cuda_fp32(x, source_index, target_index, wigner, out_mask)
        num_edges = edge_index.shape[1]
        reduced = out_mask.numel()
        out = x.new_empty((num_edges, reduced, 2 * x.shape[2]))
        _gather_wigner_forward_kernel[(num_edges, reduced)](
            x, source_index, target_index, wigner, out_mask, out,
            num_edges=num_edges, reduced_coefficients=reduced,
            channels=x.shape[2], full_coefficients=x.shape[1],
            output_channels=2 * x.shape[2], block_channels=256, num_warps=8,
        )
        ctx.save_for_backward(x, source_index, target_index, wigner, out_mask)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, source_index, target_index, wigner, out_mask = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        num_edges = source_index.numel()
        reduced = out_mask.numel()
        grad_x = torch.zeros_like(x)
        grad_wigner = torch.zeros_like(wigner)
        _gather_wigner_backward_x_kernel[(num_edges,)](
            grad_out, source_index, target_index, wigner, out_mask, grad_x,
            num_edges=num_edges, reduced_coefficients=reduced,
            channels=x.shape[2], full_coefficients=x.shape[1],
            output_channels=2 * x.shape[2], block_channels=256, num_warps=8,
        )
        _gather_wigner_backward_w_kernel[(num_edges, reduced, x.shape[1])](
            grad_out, x, source_index, target_index, out_mask, grad_wigner,
            num_edges=num_edges, reduced_coefficients=reduced,
            channels=x.shape[2], full_coefficients=x.shape[1],
            output_channels=2 * x.shape[2], block_channels=256, num_warps=8,
        )
        return grad_x, None, grad_wigner, None


class _ReverseEnvelopeScatter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        message: Tensor,
        wigner_inv: Tensor,
        out_mask: Tensor,
        distance: Tensor,
        target: Tensor,
        base: Tensor,
        cutoff: float,
        scale: float,
        node_offset: int,
        include_base: bool,
    ):
        if message.ndim != 3 or message.shape[1:] != (14, 128):
            raise UnsupportedFusionConfigError(
                f"Reverse/scatter expected message=[E,14,128], got {tuple(message.shape)}"
            )
        if wigner_inv.shape != (message.shape[0], 16, 16):
            raise UnsupportedFusionConfigError("Reverse/scatter expected wigner=[E,16,16]")
        if base.ndim != 3 or base.shape[1:] != (16, 128):
            raise UnsupportedFusionConfigError("Reverse/scatter expected base=[N,16,128]")
        _require_triton_cuda_fp32(message, wigner_inv, distance, base)
        out_mask = out_mask.to(device=message.device, dtype=torch.long).contiguous()
        target = target.contiguous()
        if target.dtype != torch.long:
            raise UnsupportedFusionConfigError("Reverse/scatter requires int64 targets")
        distance = distance.contiguous()
        out = base.clone() if include_base else torch.zeros_like(base)
        num_edges, reduced, channels = message.shape
        _reverse_scatter_forward_kernel[(num_edges, base.shape[1])](
            message, wigner_inv, out_mask, distance, target, out,
            num_edges=num_edges, reduced_coefficients=reduced,
            channels=channels, full_coefficients=base.shape[1],
            cutoff=float(cutoff), scale=float(scale), node_offset=int(node_offset),
            block_channels=128, num_warps=4,
        )
        ctx.save_for_backward(message, wigner_inv, out_mask, distance, target)
        ctx.cutoff = float(cutoff)
        ctx.scale = float(scale)
        ctx.node_offset = int(node_offset)
        ctx.include_base = bool(include_base)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        message, wigner_inv, out_mask, distance, target = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        num_edges, reduced, channels = message.shape
        full = grad_out.shape[1]
        grad_message = torch.empty_like(message)
        grad_wigner = torch.zeros_like(wigner_inv)
        grad_distance_flat = torch.empty_like(distance.reshape(-1))
        common = dict(
            num_edges=num_edges, reduced_coefficients=reduced,
            channels=channels, full_coefficients=full,
            cutoff=ctx.cutoff, scale=ctx.scale, node_offset=ctx.node_offset,
            block_channels=128,
        )
        _reverse_scatter_backward_message_kernel[(num_edges, reduced)](
            grad_out, wigner_inv, out_mask, distance, target, grad_message,
            **common, num_warps=4,
        )
        _reverse_scatter_backward_w_kernel[(num_edges, full, reduced)](
            grad_out, message, out_mask, distance, target, grad_wigner,
            **common, num_warps=4,
        )
        _reverse_scatter_backward_distance_kernel[(num_edges,)](
            grad_out, message, wigner_inv, out_mask, distance, target,
            grad_distance_flat, **common, num_warps=4,
        )
        grad_distance = grad_distance_flat.reshape_as(distance)
        return (
            grad_message, grad_wigner, None, grad_distance, None,
            (grad_out if ctx.include_base else None), None, None, None, None,
        )


class _RMSNormSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor, eps: float):
        _require_triton_cuda_fp32(x, weight, bias)
        out = torch.empty_like(x)
        block = triton.next_power_of_2(x.shape[1] * x.shape[2])
        _rmsnorm_forward_kernel[(x.shape[0],)](
            x, weight, bias, out, rows=x.shape[0], coefficients=x.shape[1],
            channels=x.shape[2], eps=float(eps), block=block, num_warps=8,
        )
        ctx.save_for_backward(x, weight)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, weight = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        block = triton.next_power_of_2(x.shape[1] * x.shape[2])
        _rmsnorm_backward_kernel[(x.shape[0],)](
            grad_out.contiguous(), x, weight, grad_x,
            rows=x.shape[0], coefficients=x.shape[1], channels=x.shape[2],
            eps=ctx.eps, block=block, num_warps=8,
        )
        return grad_x, None, None, None


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gating: Tensor, x: Tensor):
        _require_triton_cuda_fp32(gating, x)
        out = torch.empty_like(x)
        block = triton.next_power_of_2(x.shape[1] * x.shape[2])
        _gate_forward_kernel[(x.shape[0],)](
            gating, x, out, rows=x.shape[0], coefficients=x.shape[1],
            channels=x.shape[2], block=block, num_warps=8,
        )
        ctx.save_for_backward(gating, x)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        gating, x = ctx.saved_tensors
        grad_gating = torch.empty_like(gating)
        grad_x = torch.empty_like(x)
        block = triton.next_power_of_2(x.shape[1] * x.shape[2])
        _gate_backward_kernel[(x.shape[0],)](
            grad_out.contiguous(), gating, x, grad_x,
            rows=x.shape[0], coefficients=x.shape[1], channels=x.shape[2],
            block=block, num_warps=8,
        )
        _gate_backward_gate_kernel[(x.shape[0], 3)](
            grad_out.contiguous(), gating, x, grad_gating,
            rows=x.shape[0], coefficients=x.shape[1], channels=x.shape[2],
            block=triton.next_power_of_2(x.shape[2]), num_warps=4,
        )
        return grad_gating, grad_x


class _WignerSO2Prepare(torch.autograd.Function):
    """KF11 producer bridge for the 30M Edgewise conv1 path."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        edge_index: Tensor,
        wigner: Tensor,
        out_mask: Tensor,
        radial: Tensor,
        to_m_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (16, 128):
            raise UnsupportedFusionConfigError(
                f"Wigner/SO2 bridge expected x=[N,16,128], got {tuple(x.shape)}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 bridge expected edge_index=[2,E]"
            )
        edges = edge_index.shape[1]
        if wigner.shape != (edges, 16, 16):
            raise UnsupportedFusionConfigError(
                f"Wigner/SO2 bridge received wigner={tuple(wigner.shape)}"
            )
        if radial.shape != (edges, 2304):
            raise UnsupportedFusionConfigError(
                f"Wigner/SO2 bridge expected radial=[E,2304], got {tuple(radial.shape)}"
            )
        x = x.contiguous()
        source_index = edge_index[0].contiguous()
        target_index = edge_index[1].contiguous()
        wigner = wigner.contiguous()
        radial = radial.contiguous()
        out_mask = out_mask.to(device=x.device, dtype=torch.long).contiguous()
        to_m_index = to_m_index.to(device=x.device, dtype=torch.long).contiguous()
        _require_triton_cuda_fp32(
            x,
            source_index,
            target_index,
            wigner,
            out_mask,
            radial,
            to_m_index,
        )
        if source_index.dtype != torch.long or target_index.dtype != torch.long:
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 bridge requires int64 edge indices"
            )
        if out_mask.shape != (14,) or to_m_index.shape != (14,):
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 bridge requires [14] mask and permutation tensors"
            )
        m0 = x.new_empty((edges, 1024))
        m1 = x.new_empty((edges, 2, 768))
        m2 = x.new_empty((edges, 2, 512))
        if edges > 0:
            _wigner_so2_prepare_forward_kernel[(edges, 14)](
                x,
                source_index,
                target_index,
                wigner,
                out_mask,
                to_m_index,
                radial,
                m0,
                m1,
                m2,
                node_channels=128,
                full_coefficients=16,
                input_channels=256,
                radial_channels=2304,
                block_channels=256,
                num_warps=8,
            )
        ctx.save_for_backward(
            x,
            source_index,
            target_index,
            wigner,
            out_mask,
            radial,
            to_m_index,
        )
        return m0, m1, m2

    @staticmethod
    def backward(ctx, grad_m0: Tensor, grad_m1: Tensor, grad_m2: Tensor):
        (
            x,
            source_index,
            target_index,
            wigner,
            out_mask,
            radial,
            to_m_index,
        ) = ctx.saved_tensors
        edges = source_index.numel()
        if grad_m0 is None:
            grad_m0 = x.new_zeros((edges, 1024))
        if grad_m1 is None:
            grad_m1 = x.new_zeros((edges, 2, 768))
        if grad_m2 is None:
            grad_m2 = x.new_zeros((edges, 2, 512))
        grad_m0 = grad_m0.contiguous()
        grad_m1 = grad_m1.contiguous()
        grad_m2 = grad_m2.contiguous()
        grad_x = torch.zeros_like(x)
        grad_wigner = torch.zeros_like(wigner)
        grad_radial = torch.zeros_like(radial)
        if edges > 0:
            _wigner_so2_prepare_backward_x_kernel[(edges, 16)](
                grad_m0,
                grad_m1,
                grad_m2,
                source_index,
                target_index,
                wigner,
                out_mask,
                to_m_index,
                radial,
                grad_x,
                node_channels=128,
                full_coefficients=16,
                input_channels=256,
                radial_channels=2304,
                block_channels=256,
                num_warps=8,
            )
            _wigner_so2_prepare_backward_w_kernel[(edges, 14, 16)](
                grad_m0,
                grad_m1,
                grad_m2,
                x,
                source_index,
                target_index,
                out_mask,
                to_m_index,
                radial,
                grad_wigner,
                node_channels=128,
                full_coefficients=16,
                input_channels=256,
                radial_channels=2304,
                block_channels=256,
                num_warps=8,
            )
            _wigner_so2_prepare_backward_radial_kernel[(edges, 14)](
                grad_m0,
                grad_m1,
                grad_m2,
                x,
                source_index,
                target_index,
                wigner,
                out_mask,
                to_m_index,
                grad_radial,
                node_channels=128,
                full_coefficients=16,
                input_channels=256,
                radial_channels=2304,
                block_channels=256,
                num_warps=8,
            )
        return grad_x, None, grad_wigner, None, grad_radial, None


class _SO2Prepare(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        radial: Tensor,
        to_m_index: Tensor,
        use_radial: bool,
        radial_channels: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1] != 14:
            raise UnsupportedFusionConfigError(
                f"SO2 prepare expected x=[E,14,C], got {tuple(x.shape)}"
            )
        if radial.ndim != 2 or radial.shape[0] != x.shape[0]:
            raise UnsupportedFusionConfigError(
                "SO2 prepare expected radial=[E,R] with the same E as x, "
                f"got {tuple(radial.shape)}"
            )
        x = x.contiguous()
        radial = radial.contiguous()
        to_m_index = to_m_index.contiguous()
        _require_triton_cuda_fp32(x, radial, to_m_index)
        if radial.device != x.device or to_m_index.device != x.device:
            raise UnsupportedFusionConfigError(
                "SO2 prepare tensors must be on the same CUDA device"
            )
        if to_m_index.dtype != torch.long or to_m_index.shape != (14,):
            raise UnsupportedFusionConfigError("SO2 prepare requires a [14] int64 permutation")
        if use_radial and radial.shape[1] != radial_channels:
            raise UnsupportedFusionConfigError(
                f"SO2 prepare radial width {radial.shape[1]} != {radial_channels}"
            )
        if not use_radial and radial.shape[1] != 0:
            raise UnsupportedFusionConfigError(
                "SO2 prepare without radial modulation requires radial=[E,0]"
            )
        channels = x.shape[2]
        m0 = x.new_empty((x.shape[0], 4 * channels))
        m1 = x.new_empty((x.shape[0], 2, 3 * channels))
        m2 = x.new_empty((x.shape[0], 2, 2 * channels))
        if x.shape[0] > 0:
            total = 14 * x.shape[2]
            _so2_prepare_kernel[
                (x.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                x, radial, to_m_index, m0, m1, m2,
                rows=x.shape[0], coefficients=14, channels=x.shape[2],
                radial_channels=int(radial_channels), use_radial=bool(use_radial),
                block=SO2_KERNEL_BLOCK, num_warps=4,
            )
        ctx.save_for_backward(x, radial, to_m_index)
        ctx.use_radial = bool(use_radial)
        ctx.radial_channels = int(radial_channels)
        return m0, m1, m2

    @staticmethod
    def backward(ctx, grad_m0: Tensor, grad_m1: Tensor, grad_m2: Tensor):
        x, radial, to_m_index = ctx.saved_tensors
        grad_m0 = grad_m0.contiguous()
        grad_m1 = grad_m1.contiguous()
        grad_m2 = grad_m2.contiguous()
        grad_x = torch.empty_like(x)
        grad_radial = torch.zeros_like(radial)
        if x.shape[0] > 0:
            total = 14 * x.shape[2]
            _so2_prepare_backward_kernel[
                (x.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                grad_m0, grad_m1, grad_m2,
                x, radial, to_m_index, grad_x, grad_radial,
                rows=x.shape[0], coefficients=14, channels=x.shape[2],
                radial_channels=ctx.radial_channels,
                use_radial=ctx.use_radial, block=SO2_KERNEL_BLOCK, num_warps=4,
            )
        return grad_x, (grad_radial if ctx.use_radial else None), None, None, None


def _so2_prepare_backward_reduce_impl(
    grad_m0: Tensor,
    grad_m1: Tensor,
    grad_m2: Tensor,
    x: Tensor,
    radial: Tensor,
    to_m_index: Tensor,
    *,
    use_radial: bool,
    radial_channels: int,
) -> tuple[Tensor, Tensor]:
    """Launch KF14's edge-local reduction outside an autograd context."""
    grad_m0 = grad_m0.contiguous()
    grad_m1 = grad_m1.contiguous()
    grad_m2 = grad_m2.contiguous()
    grad_x = torch.empty_like(x)
    grad_radial = torch.empty_like(radial)
    if x.shape[0] > 0:
        block = triton.next_power_of_2(x.shape[2])
        if block not in (128, 256):
            raise UnsupportedFusionConfigError(
                "SO2 prepare backward reduction supports 128 or 256 channels"
            )
        _so2_prepare_backward_reduce_kernel[(x.shape[0],)](
            grad_m0,
            grad_m1,
            grad_m2,
            x,
            radial,
            to_m_index,
            grad_x,
            grad_radial,
            rows=x.shape[0],
            coefficients=14,
            channels=x.shape[2],
            radial_channels=int(radial_channels),
            use_radial=bool(use_radial),
            block=block,
            num_warps=8 if block == 256 else 4,
        )
    return grad_x, grad_radial


class _SO2PrepareBackwardReduce(_SO2Prepare):
    """KF14 SO2 prepare with an edge-local, non-atomic backward."""

    @staticmethod
    def backward(ctx, grad_m0: Tensor, grad_m1: Tensor, grad_m2: Tensor):
        x, radial, to_m_index = ctx.saved_tensors
        grad_x, grad_radial = _so2_prepare_backward_reduce_impl(
            grad_m0,
            grad_m1,
            grad_m2,
            x,
            radial,
            to_m_index,
            use_radial=ctx.use_radial,
            radial_channels=ctx.radial_channels,
        )
        return grad_x, (grad_radial if ctx.use_radial else None), None, None, None


class _WignerSO2Hybrid(torch.autograd.Function):
    """KF15 fused producer forward with KF14/cuBLAS backward."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        edge_index: Tensor,
        wigner: Tensor,
        out_mask: Tensor,
        radial: Tensor,
        to_m_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (16, 128):
            raise UnsupportedFusionConfigError(
                f"Wigner/SO2 hybrid expected x=[N,16,128], got {tuple(x.shape)}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 hybrid expected edge_index=[2,E]"
            )
        edges = edge_index.shape[1]
        if wigner.shape != (edges, 16, 16):
            raise UnsupportedFusionConfigError(
                f"Wigner/SO2 hybrid received wigner={tuple(wigner.shape)}"
            )
        if radial.shape != (edges, 2304):
            raise UnsupportedFusionConfigError(
                f"Wigner/SO2 hybrid expected radial=[E,2304], got {tuple(radial.shape)}"
            )
        x = x.contiguous()
        source_index = edge_index[0].contiguous()
        target_index = edge_index[1].contiguous()
        wigner = wigner.contiguous()
        radial = radial.contiguous()
        out_mask = out_mask.to(device=x.device, dtype=torch.long).contiguous()
        to_m_index = to_m_index.to(
            device=x.device, dtype=torch.long
        ).contiguous()
        _require_triton_cuda_fp32(
            x,
            source_index,
            target_index,
            wigner,
            out_mask,
            radial,
            to_m_index,
        )
        if source_index.dtype != torch.long or target_index.dtype != torch.long:
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 hybrid requires int64 edge indices"
            )
        if out_mask.shape != (14,) or to_m_index.shape != (14,):
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 hybrid requires [14] mask and permutation tensors"
            )

        m0 = x.new_empty((edges, 1024))
        m1 = x.new_empty((edges, 2, 768))
        m2 = x.new_empty((edges, 2, 512))
        gathered = x.new_empty((edges, 16, 256))
        rotated = x.new_empty((edges, 14, 256))
        if edges > 0:
            _wigner_so2_hybrid_forward_kernel[(edges, 14)](
                x,
                source_index,
                target_index,
                wigner,
                out_mask,
                to_m_index,
                radial,
                m0,
                m1,
                m2,
                gathered,
                rotated,
                node_channels=128,
                full_coefficients=16,
                input_channels=256,
                radial_channels=2304,
                block_channels=256,
                num_warps=8,
            )
        ctx.node_count = x.shape[0]
        ctx.save_for_backward(
            gathered,
            rotated,
            source_index,
            target_index,
            wigner,
            out_mask,
            radial,
            to_m_index,
        )
        return m0, m1, m2

    @staticmethod
    def backward(ctx, grad_m0: Tensor, grad_m1: Tensor, grad_m2: Tensor):
        (
            gathered,
            rotated,
            source_index,
            target_index,
            wigner,
            out_mask,
            radial,
            to_m_index,
        ) = ctx.saved_tensors
        edges = source_index.numel()
        if grad_m0 is None:
            grad_m0 = radial.new_zeros((edges, 1024))
        if grad_m1 is None:
            grad_m1 = radial.new_zeros((edges, 2, 768))
        if grad_m2 is None:
            grad_m2 = radial.new_zeros((edges, 2, 512))

        grad_rotated, grad_radial = _so2_prepare_backward_reduce_impl(
            grad_m0,
            grad_m1,
            grad_m2,
            rotated,
            radial,
            to_m_index,
            use_radial=True,
            radial_channels=2304,
        )
        grad_x = wigner.new_zeros((ctx.node_count, 16, 128))
        grad_wigner = torch.zeros_like(wigner)
        if edges > 0:
            selected_wigner = wigner.index_select(1, out_mask)
            grad_gathered = torch.bmm(
                selected_wigner.transpose(1, 2), grad_rotated
            )
            grad_selected_wigner = torch.bmm(
                grad_rotated, gathered.transpose(1, 2)
            )
            grad_wigner.index_copy_(1, out_mask, grad_selected_wigner)
            grad_x.index_add_(
                0, source_index, grad_gathered[:, :, :128]
            )
            grad_x.index_add_(
                0, target_index, grad_gathered[:, :, 128:]
            )
        return grad_x, None, grad_wigner, None, grad_radial, None


class _SO2Epilogue(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        m0: Tensor,
        m1: Tensor,
        m2: Tensor,
        l_to_m_index: Tensor,
        extra_channels: int,
    ) -> tuple[Tensor, Tensor]:
        if m0.ndim != 2 or m1.ndim != 3 or m2.ndim != 3:
            raise UnsupportedFusionConfigError("SO2 epilogue received invalid Linear shapes")
        if m1.shape[1] != 2 or m2.shape[1] != 2:
            raise UnsupportedFusionConfigError("SO2 epilogue requires real/imag Linear outputs")
        if m0.shape[0] != m1.shape[0] or m0.shape[0] != m2.shape[0]:
            raise UnsupportedFusionConfigError(
                "SO2 epilogue Linear outputs must have the same edge count"
            )
        if int(extra_channels) < 0:
            raise UnsupportedFusionConfigError(
                "SO2 epilogue extra channel count cannot be negative"
            )
        m0 = m0.contiguous()
        m1 = m1.contiguous()
        m2 = m2.contiguous()
        l_to_m_index = l_to_m_index.contiguous()
        _require_triton_cuda_fp32(m0, m1, m2, l_to_m_index)
        if not (
            m0.device == m1.device == m2.device == l_to_m_index.device
        ):
            raise UnsupportedFusionConfigError(
                "SO2 epilogue tensors must be on the same CUDA device"
            )
        if l_to_m_index.dtype != torch.long or l_to_m_index.shape != (14,):
            raise UnsupportedFusionConfigError("SO2 epilogue requires a [14] int64 permutation")
        output_channels = m1.shape[2] // 2 // 3
        if output_channels <= 0:
            raise UnsupportedFusionConfigError(
                "SO2 epilogue output channel count must be positive"
            )
        if m1.shape[2] != 2 * 3 * output_channels:
            raise UnsupportedFusionConfigError("SO2 m=1 Linear output width is unsupported")
        if m2.shape[2] != 2 * 2 * output_channels:
            raise UnsupportedFusionConfigError("SO2 m=2 Linear output width is unsupported")
        if m0.shape[1] != 4 * output_channels + int(extra_channels):
            raise UnsupportedFusionConfigError("SO2 m=0 Linear output width is unsupported")
        out = m0.new_empty((m0.shape[0], 14, output_channels))
        gating = m0.new_empty((m0.shape[0], int(extra_channels)))
        total = 14 * output_channels + int(extra_channels)
        if m0.shape[0] > 0:
            _so2_epilogue_kernel[
                (m0.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                m0, m1, m2, l_to_m_index, out, gating,
                rows=m0.shape[0], output_coefficients=14,
                output_channels=output_channels, m0_channels=m0.shape[1],
                m1_coefficients=3, m2_coefficients=2,
                extra_channels=int(extra_channels), block=SO2_KERNEL_BLOCK,
                num_warps=4,
            )
        ctx.save_for_backward(l_to_m_index)
        ctx.output_channels = int(output_channels)
        ctx.m0_channels = int(m0.shape[1])
        ctx.extra_channels = int(extra_channels)
        return out, gating

    @staticmethod
    def backward(ctx, grad_out: Tensor, grad_gating: Tensor):
        (l_to_m_index,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if grad_gating is None:
            grad_gating = grad_out.new_zeros(
                (grad_out.shape[0], ctx.extra_channels)
            )
        else:
            grad_gating = grad_gating.contiguous()
        output_channels = ctx.output_channels
        grad_m0 = grad_out.new_empty((grad_out.shape[0], ctx.m0_channels))
        grad_m1 = grad_out.new_empty(
            (grad_out.shape[0], 2, 3 * 2 * output_channels)
        )
        grad_m2 = grad_out.new_empty(
            (grad_out.shape[0], 2, 2 * 2 * output_channels)
        )
        total = 14 * output_channels + ctx.extra_channels
        if grad_out.shape[0] > 0:
            _so2_epilogue_backward_kernel[
                (grad_out.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                grad_out, grad_gating, l_to_m_index, grad_m0, grad_m1, grad_m2,
                rows=grad_out.shape[0], output_coefficients=14,
                output_channels=output_channels, m0_channels=ctx.m0_channels,
                m1_coefficients=3, m2_coefficients=2,
                extra_channels=ctx.extra_channels, block=SO2_KERNEL_BLOCK,
                num_warps=4,
            )
        return grad_m0, grad_m1, grad_m2, None, None


class _SO2BlockEpilogue(torch.autograd.Function):
    """Epilogue for canonical real/imag outputs from block-diagonal GEMMs."""

    @staticmethod
    def forward(
        ctx,
        m0: Tensor,
        m1: Tensor,
        m2: Tensor,
        l_to_m_index: Tensor,
        extra_channels: int,
    ) -> tuple[Tensor, Tensor]:
        if m0.ndim != 2 or m1.ndim != 2 or m2.ndim != 2:
            raise UnsupportedFusionConfigError(
                "SO2 block epilogue received invalid Linear shapes"
            )
        if m0.shape[0] != m1.shape[0] or m0.shape[0] != m2.shape[0]:
            raise UnsupportedFusionConfigError(
                "SO2 block epilogue outputs must have the same edge count"
            )
        if int(extra_channels) < 0:
            raise UnsupportedFusionConfigError(
                "SO2 block epilogue extra channel count cannot be negative"
            )
        m0 = m0.contiguous()
        m1 = m1.contiguous()
        m2 = m2.contiguous()
        l_to_m_index = l_to_m_index.contiguous()
        _require_triton_cuda_fp32(m0, m1, m2, l_to_m_index)
        if not (
            m0.device == m1.device == m2.device == l_to_m_index.device
        ):
            raise UnsupportedFusionConfigError(
                "SO2 block epilogue tensors must be on the same CUDA device"
            )
        if l_to_m_index.dtype != torch.long or l_to_m_index.shape != (14,):
            raise UnsupportedFusionConfigError(
                "SO2 block epilogue requires a [14] int64 permutation"
            )
        output_channels = m1.shape[1] // 2 // 3
        if output_channels <= 0 or m1.shape[1] != 2 * 3 * output_channels:
            raise UnsupportedFusionConfigError(
                "SO2 block m=1 Linear output width is unsupported"
            )
        if m2.shape[1] != 2 * 2 * output_channels:
            raise UnsupportedFusionConfigError(
                "SO2 block m=2 Linear output width is unsupported"
            )
        if m0.shape[1] != 4 * output_channels + int(extra_channels):
            raise UnsupportedFusionConfigError(
                "SO2 block m=0 Linear output width is unsupported"
            )
        out = m0.new_empty((m0.shape[0], 14, output_channels))
        gating = m0.new_empty((m0.shape[0], int(extra_channels)))
        total = 14 * output_channels + int(extra_channels)
        if m0.shape[0] > 0:
            _so2_block_epilogue_kernel[
                (m0.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                m0,
                m1,
                m2,
                l_to_m_index,
                out,
                gating,
                rows=m0.shape[0],
                output_coefficients=14,
                output_channels=output_channels,
                m0_channels=m0.shape[1],
                m1_coefficients=3,
                m2_coefficients=2,
                extra_channels=int(extra_channels),
                block=SO2_KERNEL_BLOCK,
                num_warps=4,
            )
        ctx.save_for_backward(l_to_m_index)
        ctx.output_channels = int(output_channels)
        ctx.m0_channels = int(m0.shape[1])
        ctx.extra_channels = int(extra_channels)
        return out, gating

    @staticmethod
    def backward(ctx, grad_out: Tensor, grad_gating: Tensor):
        (l_to_m_index,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if grad_gating is None:
            grad_gating = grad_out.new_zeros(
                (grad_out.shape[0], ctx.extra_channels)
            )
        else:
            grad_gating = grad_gating.contiguous()
        output_channels = ctx.output_channels
        grad_m0 = grad_out.new_empty((grad_out.shape[0], ctx.m0_channels))
        grad_m1 = grad_out.new_empty(
            (grad_out.shape[0], 2 * 3 * output_channels)
        )
        grad_m2 = grad_out.new_empty(
            (grad_out.shape[0], 2 * 2 * output_channels)
        )
        total = 14 * output_channels + ctx.extra_channels
        if grad_out.shape[0] > 0:
            _so2_block_epilogue_backward_kernel[
                (grad_out.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                grad_out,
                grad_gating,
                l_to_m_index,
                grad_m0,
                grad_m1,
                grad_m2,
                rows=grad_out.shape[0],
                output_coefficients=14,
                output_channels=output_channels,
                m0_channels=ctx.m0_channels,
                m1_coefficients=3,
                m2_coefficients=2,
                extra_channels=ctx.extra_channels,
                block=SO2_KERNEL_BLOCK,
                num_warps=4,
            )
        return grad_m0, grad_m1, grad_m2, None, None


class _SO2BlockGateBridge(torch.autograd.Function):
    """KF10 bridge specialized for already-recombined block-GEMM outputs."""

    @staticmethod
    def forward(
        ctx,
        m0: Tensor,
        m1: Tensor,
        m2: Tensor,
        m_degree_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        expected = (
            (m0.shape[0], 896),
            (m0.shape[0], 768),
            (m0.shape[0], 512),
        )
        if (
            tuple(m0.shape) != expected[0]
            or tuple(m1.shape) != expected[1]
            or tuple(m2.shape) != expected[2]
        ):
            raise UnsupportedFusionConfigError(
                "SO2 block gate bridge requires m0=[E,896], m1=[E,768], "
                f"m2=[E,512], got {tuple(m0.shape)}, {tuple(m1.shape)}, "
                f"{tuple(m2.shape)}"
            )
        m0 = m0.contiguous()
        m1 = m1.contiguous()
        m2 = m2.contiguous()
        m_degree_index = m_degree_index.contiguous()
        _require_triton_cuda_fp32(m0, m1, m2, m_degree_index)
        if not (m0.device == m1.device == m2.device == m_degree_index.device):
            raise UnsupportedFusionConfigError(
                "SO2 block gate bridge tensors must be on the same CUDA device"
            )
        if m_degree_index.dtype != torch.long or m_degree_index.shape != (14,):
            raise UnsupportedFusionConfigError(
                "SO2 block gate bridge requires a [14] int64 degree mapping"
            )
        out0 = m0.new_empty((m0.shape[0], 512))
        out1 = m0.new_empty((m0.shape[0], 2, 384))
        out2 = m0.new_empty((m0.shape[0], 2, 256))
        if m0.shape[0] > 0:
            total = 14 * 128
            _so2_block_gate_bridge_kernel[
                (m0.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                m0,
                m1,
                m2,
                m_degree_index,
                out0,
                out1,
                out2,
                rows=m0.shape[0],
                coefficients=14,
                channels=128,
                extra_channels=384,
                block=SO2_KERNEL_BLOCK,
                num_warps=4,
            )
        ctx.save_for_backward(m0, m1, m2, m_degree_index)
        return out0, out1, out2

    @staticmethod
    def backward(ctx, grad_out0: Tensor, grad_out1: Tensor, grad_out2: Tensor):
        m0, m1, m2, m_degree_index = ctx.saved_tensors
        grad_out0 = grad_out0.contiguous()
        grad_out1 = grad_out1.contiguous()
        grad_out2 = grad_out2.contiguous()
        grad_m0 = torch.empty_like(m0)
        grad_m1 = torch.empty_like(m1)
        grad_m2 = torch.empty_like(m2)
        if m0.shape[0] > 0:
            _so2_block_gate_bridge_backward_kernel[(m0.shape[0],)](
                grad_out0,
                grad_out1,
                grad_out2,
                m0,
                m1,
                m2,
                m_degree_index,
                grad_m0,
                grad_m1,
                grad_m2,
                rows=m0.shape[0],
                channels=128,
                extra_channels=384,
                block=128,
                num_warps=4,
            )
        return grad_m0, grad_m1, grad_m2, None


class _SO2GateBridge(torch.autograd.Function):
    """Bridge conv1 Linear outputs directly to conv2 Linear inputs.

    The supported Edgewise layout has 128 channels, 384 degree gates, and a
    fixed 14-coefficient reduced spherical-harmonic representation.  The
    l-to-m permutation before and after GateActivation cancels, so the gate
    can be applied directly to the reconstructed m-order coefficients.
    """

    @staticmethod
    def forward(
        ctx,
        m0: Tensor,
        m1: Tensor,
        m2: Tensor,
        m_degree_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if m0.ndim != 2 or m1.ndim != 3 or m2.ndim != 3:
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge received invalid conv1 Linear output ranks"
            )
        expected = (
            (m0.shape[0], 896),
            (m0.shape[0], 2, 768),
            (m0.shape[0], 2, 512),
        )
        if (
            tuple(m0.shape) != expected[0]
            or tuple(m1.shape) != expected[1]
            or tuple(m2.shape) != expected[2]
        ):
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge requires m0=[E,896], m1=[E,2,768], "
                f"m2=[E,2,512], got {tuple(m0.shape)}, {tuple(m1.shape)}, "
                f"{tuple(m2.shape)}"
            )
        m0 = m0.contiguous()
        m1 = m1.contiguous()
        m2 = m2.contiguous()
        m_degree_index = m_degree_index.contiguous()
        _require_triton_cuda_fp32(m0, m1, m2, m_degree_index)
        if not (m0.device == m1.device == m2.device == m_degree_index.device):
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge tensors must be on the same CUDA device"
            )
        if m_degree_index.dtype != torch.long or m_degree_index.shape != (14,):
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge requires a [14] int64 degree mapping"
            )
        out0 = m0.new_empty((m0.shape[0], 512))
        out1 = m0.new_empty((m0.shape[0], 2, 384))
        out2 = m0.new_empty((m0.shape[0], 2, 256))
        if m0.shape[0] > 0:
            total = 14 * 128
            _so2_gate_bridge_kernel[
                (m0.shape[0], triton.cdiv(total, SO2_KERNEL_BLOCK))
            ](
                m0,
                m1,
                m2,
                m_degree_index,
                out0,
                out1,
                out2,
                rows=m0.shape[0],
                coefficients=14,
                channels=128,
                extra_channels=384,
                block=SO2_KERNEL_BLOCK,
                num_warps=4,
            )
        ctx.save_for_backward(m0, m1, m2, m_degree_index)
        return out0, out1, out2

    @staticmethod
    def backward(ctx, grad_out0: Tensor, grad_out1: Tensor, grad_out2: Tensor):
        m0, m1, m2, m_degree_index = ctx.saved_tensors
        grad_out0 = grad_out0.contiguous()
        grad_out1 = grad_out1.contiguous()
        grad_out2 = grad_out2.contiguous()
        grad_m0 = torch.empty_like(m0)
        grad_m1 = torch.empty_like(m1)
        grad_m2 = torch.empty_like(m2)
        if m0.shape[0] > 0:
            _so2_gate_bridge_backward_kernel[(m0.shape[0],)](
                grad_out0,
                grad_out1,
                grad_out2,
                m0,
                m1,
                m2,
                m_degree_index,
                grad_m0,
                grad_m1,
                grad_m2,
                rows=m0.shape[0],
                channels=128,
                extra_channels=384,
                block=128,
                num_warps=4,
            )
        return grad_m0, grad_m1, grad_m2, None


class _FusedRadialMLP(torch.autograd.Function):
    """One Triton kernel for the whole RadialMLP row chain."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        w1: Tensor,
        b1: Tensor,
        g1: Tensor,
        be1: Tensor,
        w2: Tensor,
        b2: Tensor,
        g2: Tensor,
        be2: Tensor,
        w3: Tensor,
        b3: Tensor,
        in_ch: int,
        h1_ch: int,
        h2_ch: int,
        out_ch: int,
        eps: float,
    ) -> Tensor:
        x = x.contiguous()
        _require_triton_cuda_fp32(x, w1, b1, g1, be1, w2, b2, g2, be2, w3, b3)
        if x.ndim != 2 or x.shape[1] != in_ch:
            raise UnsupportedFusionConfigError(
                f"Fused RadialMLP expected x=[R,{in_ch}], got {tuple(x.shape)}"
            )
        rows = x.shape[0]
        out = x.new_empty((rows, out_ch))
        save_a1 = x.new_empty((rows, h1_ch))
        save_hhat1 = x.new_empty((rows, h1_ch))
        save_rstd1 = x.new_empty((rows,))
        save_a2 = x.new_empty((rows, h2_ch))
        save_hhat2 = x.new_empty((rows, h2_ch))
        save_rstd2 = x.new_empty((rows,))
        if rows > 0:
            grid = (triton.cdiv(rows, 16),)
            _radial_mlp_forward_kernel[grid](
                x, w1, b1, g1, be1, w2, b2, g2, be2, w3, b3, out,
                save_a1, save_hhat1, save_rstd1, save_a2, save_hhat2, save_rstd2,
                rows=rows, in_ch=in_ch, h1_ch=h1_ch, h2_ch=h2_ch,
                out_ch=out_ch, eps=float(eps),
                BLOCK_R=16, BLOCK_K=32, BLOCK_O=256, num_warps=4,
            )
        ctx.save_for_backward(
            x, w1, w2, w3, g1, g2,
            save_a1, save_hhat1, save_rstd1, save_a2, save_hhat2, save_rstd2,
        )
        ctx.in_ch = int(in_ch)
        ctx.h1_ch = int(h1_ch)
        ctx.h2_ch = int(h2_ch)
        ctx.out_ch = int(out_ch)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            x, w1, w2, w3, g1, g2,
            save_a1, save_hhat1, save_rstd1, save_a2, save_hhat2, save_rstd2,
        ) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        scratch_g_h2 = torch.empty_like(save_a2)
        scratch_g_h1 = torch.empty_like(save_a1)
        rows = x.shape[0]
        if rows > 0:
            # The backward path keeps several [BLOCK_R, hidden] dot
            # accumulators live at once.  BLOCK_R=16/BLOCK_O=256 exceeds the
            # H100 shared-memory limit for the 30M radial heads (the forward
            # tile remains unchanged).  Keep BLOCK_R >= 16 because Triton
            # tensor-dot requires that minimum, and reduce the output tile.
            backward_block_r = 16
            backward_block_o = 64
            grid = (triton.cdiv(rows, backward_block_r),)
            _radial_mlp_backward_kernel[grid](
                grad_out,
                save_a1, save_hhat1, save_rstd1,
                save_a2, save_hhat2, save_rstd2,
                g1, g2, w1, w2, w3,
                scratch_g_h2, scratch_g_h1, grad_x,
                rows=rows, in_ch=ctx.in_ch, h1_ch=ctx.h1_ch, h2_ch=ctx.h2_ch,
                out_ch=ctx.out_ch,
                BLOCK_R=backward_block_r,
                BLOCK_K=32,
                BLOCK_O=backward_block_o,
                num_warps=2,
            )
        return grad_x, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class _FusedSO3MLP(torch.autograd.Function):
    """Fused SO3_Linear + gate + SO3_Linear chain for the spectral atomwise."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        gating: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        _require_triton_cuda_fp32(x, gating, w1, b1, w2, b2)
        x = x.contiguous()
        gating = gating.contiguous()
        if x.ndim != 3 or x.shape[1:] != (16, 128):
            raise UnsupportedFusionConfigError(
                f"Fused SO3 MLP expected x=[N,16,128], got {tuple(x.shape)}"
            )
        if gating.shape != (x.shape[0], 3, 128):
            raise UnsupportedFusionConfigError(
                f"Fused SO3 MLP expected gating=[N,3,128], got {tuple(gating.shape)}"
            )
        scratch_h = x.new_empty(x.shape)
        out = x.new_empty(x.shape)
        rows = x.shape[0]
        if rows > 0:
            grid = (triton.cdiv(rows, 16),)
            _so3_mlp_forward_kernel[grid](
                x, gating, w1, b1, w2, b2, scratch_h, out,
                rows=rows, coeffs=16, channels=128,
                BLOCK_R=16, BLOCK_K=32, num_warps=4,
            )
        ctx.save_for_backward(x, gating, w1, w2, scratch_h)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, gating, w1, w2, scratch_h = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        grad_gating = torch.zeros_like(gating)
        scratch_g_h = torch.empty_like(x)
        rows = x.shape[0]
        if rows > 0:
            grid = (triton.cdiv(rows, 16),)
            _so3_mlp_backward_kernel[grid](
                grad_out, x, gating, w1, w2, scratch_h, scratch_g_h,
                grad_gating, grad_x,
                rows=rows, coeffs=16, channels=128,
                BLOCK_R=16, BLOCK_K=32, num_warps=4,
            )
        return grad_x, grad_gating, None, None, None, None


class _FusedEnergyMLP(torch.autograd.Function):
    """Fused energy-head MLP (Linear+SiLU+Linear+SiLU+Linear->1) over nodes."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        w3: Tensor,
        b3: Tensor,
    ) -> Tensor:
        # The head passes a narrow+squeeze view; normalize the layout first.
        x = x.contiguous()
        _require_triton_cuda_fp32(x, w1, b1, w2, b2, w3, b3)
        if x.ndim != 2 or x.shape[1] != 128:
            raise UnsupportedFusionConfigError(
                f"Fused energy MLP expected x=[N,128], got {tuple(x.shape)}"
            )
        rows = x.shape[0]
        save_a1 = x.new_empty(x.shape)
        save_a2 = x.new_empty(x.shape)
        out = x.new_empty((rows, 1))
        if rows > 0:
            grid = (triton.cdiv(rows, 16),)
            _energy_mlp_forward_kernel[grid](
                x, w1, b1, w2, b2, w3, b3, save_a1, save_a2, out,
                rows=rows, channels=128, BLOCK_R=16, BLOCK_K=32, num_warps=4,
            )
        ctx.save_for_backward(x, w1, w2, w3, save_a1, save_a2)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, w1, w2, w3, save_a1, save_a2 = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)
        scratch_g_a2 = torch.empty_like(x)
        scratch_g_a1 = torch.empty_like(x)
        rows = x.shape[0]
        if rows > 0:
            grid = (triton.cdiv(rows, 16),)
            _energy_mlp_backward_kernel[grid](
                grad_out, save_a1, save_a2, w1, w2, w3,
                scratch_g_a2, scratch_g_a1, grad_x,
                rows=rows, channels=128, BLOCK_R=16, BLOCK_K=32, num_warps=4,
            )
        return grad_x, None, None, None, None, None, None


def _require_triton_cuda_fp32(*tensors: Tensor) -> None:
    if triton is None:
        raise UnsupportedFusionConfigError("Opt4 model fusion requires Triton")
    for index, tensor in enumerate(tensors):
        if tensor.device.type != "cuda":
            raise UnsupportedFusionConfigError(
                f"Opt4 model fusion tensor {index} must be CUDA, got {tensor.device}"
            )
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise UnsupportedFusionConfigError(
                f"Opt4 model fusion tensor {index} must be FP32, got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise UnsupportedFusionConfigError(
                "Opt4 model fusion received a non-contiguous internal tensor "
                f"{index}: shape={tuple(tensor.shape)}, stride={tensor.stride()}"
            )


class SO2BlockLinear(nn.Module):
    """Frozen SO2 m>0 Linear expressed as one block-diagonal cuBLAS GEMM."""

    def __init__(self, original: nn.Linear) -> None:
        super().__init__()
        if not isinstance(original, nn.Linear) or original.bias is not None:
            raise UnsupportedFusionConfigError(
                "SO2 block GEMM requires a bias-free nn.Linear"
            )
        if original.out_features % 2:
            raise UnsupportedFusionConfigError(
                "SO2 block GEMM requires an even output width"
            )
        output_half = original.out_features // 2
        w1, w2 = original.weight.detach().split(output_half, dim=0)
        block_weight = torch.cat(
            (
                torch.cat((w1, -w2), dim=1),
                torch.cat((w2, w1), dim=1),
            ),
            dim=0,
        ).contiguous()
        self.in_features = int(original.in_features)
        self.out_features_half = int(output_half)
        self.register_buffer("block_weight", block_weight, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (2, self.in_features):
            raise UnsupportedFusionConfigError(
                "SO2 block GEMM expected x=[E,2,"
                f"{self.in_features}], got {tuple(x.shape)}"
            )
        return torch.nn.functional.linear(
            x.reshape(x.shape[0], 2 * self.in_features), self.block_weight
        )


class FusedSO2Convolution(nn.Module):
    """SO2 convolution with fused m-layout and epilogue operations.

    The three Linear modules remain the original cuBLAS-backed modules.  Only
    the fixed coefficient permutation, radial modulation, complex
    recombination, and inverse permutation are handled by Triton.
    """

    def __init__(
        self,
        original: SO2_Convolution,
        *,
        block_gemm: bool = False,
        prepare_backward_reduce: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(original, SO2_Convolution):
            raise UnsupportedFusionConfigError("Expected an SO2_Convolution")
        if (original.lmax, original.mmax) != (3, 2):
            raise UnsupportedFusionConfigError(
                "FusedSO2Convolution requires lmax=3 and mmax=2"
            )
        external_radial = (
            not original.internal_weights
            and original.sphere_channels == 256
            and original.m_output_channels == 128
            and original.rad_func is not None
            and int(original.extra_m0_output_channels or 0) == 384
        )
        internal_weights = (
            original.internal_weights
            and original.sphere_channels == 128
            and original.m_output_channels == 128
            and original.rad_func is None
            and original.extra_m0_output_channels is None
        )
        if not (external_radial or internal_weights):
            raise UnsupportedFusionConfigError(
                "FusedSO2Convolution only supports the 30M Edgewise "
                "so2_conv_1/so2_conv_2 layouts"
            )
        mapping = original.mappingReduced.to_m.detach()
        if tuple(mapping.shape) != (14, 14):
            raise UnsupportedFusionConfigError(
                f"FusedSO2Convolution requires a [14,14] mapping, got {tuple(mapping.shape)}"
            )
        to_m_index = mapping.argmax(dim=1).to(dtype=torch.long)
        if torch.unique(to_m_index).numel() != 14:
            raise UnsupportedFusionConfigError(
                "FusedSO2Convolution requires a bijective permutation mapping"
            )
        expected = torch.zeros_like(mapping)
        expected.scatter_(1, to_m_index[:, None], 1.0)
        if not torch.equal(mapping, expected):
            raise UnsupportedFusionConfigError(
                "FusedSO2Convolution requires a fixed permutation mapping"
            )
        l_to_m_index = torch.empty_like(to_m_index)
        l_to_m_index[to_m_index] = torch.arange(
            14, device=to_m_index.device, dtype=torch.long
        )

        self.sphere_channels = original.sphere_channels
        self.m_output_channels = original.m_output_channels
        self.lmax = original.lmax
        self.mmax = original.mmax
        self.mappingReduced = original.mappingReduced
        self.internal_weights = original.internal_weights
        self.edge_channels_list = original.edge_channels_list
        self.extra_m0_output_channels = int(original.extra_m0_output_channels or 0)
        self.fc_m0 = original.fc_m0
        self.so2_m_conv = original.so2_m_conv
        self.block_gemm = bool(block_gemm)
        self.prepare_backward_reduce = bool(prepare_backward_reduce)
        self.so2_block_linear = nn.ModuleList()
        if self.block_gemm:
            self.so2_block_linear.extend(
                SO2BlockLinear(module.fc) for module in self.so2_m_conv
            )
        self.rad_func = original.rad_func
        self.register_buffer("to_m_index", to_m_index, persistent=False)
        self.register_buffer("l_to_m_index", l_to_m_index, persistent=False)

        self.radial_channels = sum(
            [
                self.fc_m0.in_features,
                self.so2_m_conv[0].fc.in_features,
                self.so2_m_conv[1].fc.in_features,
            ]
        )
        radial_width = None
        if self.rad_func is not None:
            radial_width = getattr(self.rad_func, "out_channels", None)
            if radial_width is None and hasattr(self.rad_func, "net"):
                radial_width = self.rad_func.net[-1].out_features
        if self.rad_func is not None and radial_width != self.radial_channels:
            raise UnsupportedFusionConfigError(
                "SO2 radial feature width does not match m=0/m=1/m=2 inputs"
            )

    def radial_features(self, x_edge: Tensor) -> Tensor:
        if self.rad_func is None:
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 producer requires conv1 external radial weights"
            )
        radial = self.rad_func(x_edge)
        if radial.ndim != 2 or radial.shape[1] != self.radial_channels:
            raise UnsupportedFusionConfigError(
                "SO2 radial features do not match the fixed 30M conv1 layout"
            )
        return radial

    def prepare(self, x: Tensor, x_edge: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (14, self.sphere_channels):
            raise UnsupportedFusionConfigError(
                f"FusedSO2Convolution expected x=[E,14,{self.sphere_channels}], got {tuple(x.shape)}"
            )
        if self.rad_func is not None:
            radial = self.radial_features(x_edge)
            use_radial = True
        else:
            radial = x.new_empty((x.shape[0], 0))
            use_radial = False
        prepare = (
            _SO2PrepareBackwardReduce
            if self.prepare_backward_reduce
            else _SO2Prepare
        )
        return prepare.apply(
            x.contiguous(), radial, self.to_m_index, use_radial, self.radial_channels
        )

    def linear_from_prepared(
        self, x0: Tensor, x1: Tensor, x2: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.block_gemm:
            return (
                self.fc_m0(x0),
                self.so2_block_linear[0](x1),
                self.so2_block_linear[1](x2),
            )
        return (
            self.fc_m0(x0),
            self.so2_m_conv[0].fc(x1),
            self.so2_m_conv[1].fc(x2),
        )

    def prepare_and_linear(
        self, x: Tensor, x_edge: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.linear_from_prepared(*self.prepare(x, x_edge))

    def epilogue(self, m0: Tensor, m1: Tensor, m2: Tensor):
        epilogue = _SO2BlockEpilogue if self.block_gemm else _SO2Epilogue
        out, gating = epilogue.apply(
            m0,
            m1,
            m2,
            self.l_to_m_index,
            self.extra_m0_output_channels,
        )
        if self.extra_m0_output_channels:
            return out, gating
        return out

    def forward(self, x: Tensor, x_edge: Tensor):
        return self.epilogue(*self.prepare_and_linear(x, x_edge))


class FusedSO2GateBridge(nn.Module):
    """Fused conv1-epilogue, gate activation, and conv2-prepare bridge."""

    def __init__(
        self, conv1: FusedSO2Convolution, conv2: FusedSO2Convolution
    ) -> None:
        super().__init__()
        if not isinstance(conv1, FusedSO2Convolution) or not isinstance(
            conv2, FusedSO2Convolution
        ):
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge requires two fused SO2 convolutions"
            )
        if (
            conv1.sphere_channels != 256
            or conv1.m_output_channels != 128
            or conv1.extra_m0_output_channels != 384
            or conv2.sphere_channels != 128
            or conv2.m_output_channels != 128
            or conv2.extra_m0_output_channels != 0
        ):
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge only supports the 30M Edgewise conv1/conv2 pair"
            )
        if not torch.equal(conv1.to_m_index, conv2.to_m_index):
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge requires matching conv1/conv2 permutations"
            )
        if conv1.block_gemm != conv2.block_gemm:
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge requires matching conv1/conv2 GEMM layouts"
            )
        self.block_gemm = conv1.block_gemm
        l_degree = torch.tensor(
            [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            dtype=torch.long,
            device=conv1.to_m_index.device,
        )
        m_degree = l_degree.index_select(0, conv1.to_m_index)
        expected = torch.tensor(
            [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 2, 3],
            dtype=torch.long,
            device=m_degree.device,
        )
        if not torch.equal(m_degree, expected):
            raise UnsupportedFusionConfigError(
                "SO2 gate bridge received an unsupported coefficient permutation"
            )
        self.register_buffer("m_degree_index", m_degree, persistent=False)

    def forward(
        self, m0: Tensor, m1: Tensor, m2: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        bridge = _SO2BlockGateBridge if self.block_gemm else _SO2GateBridge
        return bridge.apply(
            m0.contiguous(),
            m1.contiguous(),
            m2.contiguous(),
            self.m_degree_index,
        )


def gather_cat_wigner(x: Tensor, edge_index: Tensor, wigner: Tensor, out_mask: Tensor) -> Tensor:
    # The official model can hand us views even though the logical shapes are
    # fixed.  Normalize their layout before entering Triton.  ``contiguous`` is
    # a no-op for the common case and remains differentiable/capture-safe when
    # a real copy is required.
    return _GatherWigner.apply(
        x.contiguous(), edge_index.contiguous(), wigner.contiguous(), out_mask
    )


def wigner_so2_prepare(
    x: Tensor,
    edge_index: Tensor,
    wigner: Tensor,
    out_mask: Tensor,
    radial: Tensor,
    to_m_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the KF11 producer bridge and return conv1 GEMM-ready buffers."""
    return _WignerSO2Prepare.apply(
        x,
        edge_index,
        wigner,
        out_mask,
        radial,
        to_m_index,
    )


def wigner_so2_hybrid(
    x: Tensor,
    edge_index: Tensor,
    wigner: Tensor,
    out_mask: Tensor,
    radial: Tensor,
    to_m_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run KF15's fused forward and KF14/cuBLAS backward."""
    return _WignerSO2Hybrid.apply(
        x,
        edge_index,
        wigner,
        out_mask,
        radial,
        to_m_index,
    )


def reverse_envelope_scatter(
    message: Tensor,
    wigner_inv: Tensor,
    out_mask: Tensor,
    distance: Tensor,
    target: Tensor,
    base: Tensor,
    cutoff: float,
    scale: float = 1.0,
    node_offset: int = 0,
    include_base: bool = True,
) -> Tensor:
    return _ReverseEnvelopeScatter.apply(
        message.contiguous(), wigner_inv.contiguous(), out_mask,
        distance.contiguous(), target.contiguous(), base.contiguous(),
        cutoff, scale, node_offset, include_base,
    )


class FusedGateActivation(nn.Module):
    def __init__(self, original: GateActivation) -> None:
        super().__init__()
        self.lmax = original.lmax
        self.mmax = original.mmax
        self.num_channels = original.num_channels

    def forward(self, gating_scalars: Tensor, input_tensors: Tensor) -> Tensor:
        expected_coefficients = 1 + sum(
            min(2 * degree + 1, 2 * self.mmax + 1)
            for degree in range(1, self.lmax + 1)
        )
        if input_tensors.shape[1:] != (
            expected_coefficients,
            self.num_channels,
        ):
            raise UnsupportedFusionConfigError(
                "Fused GateActivation received an unsupported shape: "
                f"{tuple(input_tensors.shape)}"
            )
        return _Gate.apply(gating_scalars.contiguous(), input_tensors.contiguous())


class FusedRMSNormSH(nn.Module):
    def __init__(self, original: EquivariantRMSNormArraySphericalHarmonicsV2) -> None:
        super().__init__()
        self.lmax = original.lmax
        self.num_channels = original.num_channels
        self.eps = original.eps
        self.affine_weight = original.affine_weight
        self.affine_bias = original.affine_bias

    def forward(self, node_input: Tensor) -> Tensor:
        if node_input.shape[1:] != (16, 128):
            raise UnsupportedFusionConfigError(
                f"Fused RMSNormSH expected [N,16,128], got {tuple(node_input.shape)}"
            )
        return _RMSNormSH.apply(
            node_input.contiguous(), self.affine_weight.contiguous(),
            self.affine_bias.contiguous(), self.eps,
        )


class FusedEdgewise(nn.Module):
    def __init__(
        self,
        original: Edgewise,
        *,
        gather: bool,
        reverse: bool,
        so2_epilogue: bool = False,
        so2_gate_bridge: bool = False,
        wigner_so2_bridge: bool = False,
        wigner_so2_hybrid: bool = False,
    ) -> None:
        super().__init__()
        self.sphere_channels = original.sphere_channels
        self.hidden_channels = original.hidden_channels
        self.lmax = original.lmax
        self.mmax = original.mmax
        self.mappingReduced = original.mappingReduced
        self.SO3_grid = original.SO3_grid
        self.edge_channels_list = original.edge_channels_list
        self.act_type = original.act_type
        self.act = original.act
        self.so2_conv_1 = original.so2_conv_1
        self.so2_conv_2 = original.so2_conv_2
        self.use_envelope = original.use_envelope
        self.cutoff = original.cutoff
        self.envelope = original.envelope
        self.out_mask = original.out_mask
        self.fuse_gather = gather
        self.fuse_reverse = reverse
        self.fuse_so2_epilogue = so2_epilogue
        self.fuse_so2_gate_bridge = so2_gate_bridge
        self.fuse_wigner_so2_bridge = wigner_so2_bridge
        self.fuse_wigner_so2_hybrid = wigner_so2_hybrid
        if (wigner_so2_bridge or wigner_so2_hybrid) and not so2_gate_bridge:
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 producer fusion requires the SO2 gate bridge"
            )
        if wigner_so2_bridge and wigner_so2_hybrid:
            raise UnsupportedFusionConfigError(
                "Wigner/SO2 bridge and hybrid are mutually exclusive"
            )
        if so2_gate_bridge:
            if not isinstance(self.act, GateActivation):
                raise UnsupportedFusionConfigError(
                    "SO2 gate bridge requires the official Edgewise GateActivation"
                )
            self.so2_gate_bridge = FusedSO2GateBridge(
                self.so2_conv_1, self.so2_conv_2
            )
        else:
            self.so2_gate_bridge = None

    def forward(self, x, x_edge, edge_distance, edge_index, wigner, wigner_inv, node_offset: int = 0):
        out_mask = self.out_mask.to(device=x.device)
        if self.fuse_so2_gate_bridge:
            if self.fuse_wigner_so2_bridge or self.fuse_wigner_so2_hybrid:
                producer = (
                    wigner_so2_hybrid
                    if self.fuse_wigner_so2_hybrid
                    else wigner_so2_prepare
                )
                prepared = producer(
                    x.contiguous(),
                    edge_index.contiguous(),
                    wigner.contiguous(),
                    out_mask,
                    self.so2_conv_1.radial_features(x_edge),
                    self.so2_conv_1.to_m_index,
                )
                m0, m1, m2 = self.so2_conv_1.linear_from_prepared(*prepared)
            else:
                if self.fuse_gather:
                    x_message = gather_cat_wigner(x, edge_index, wigner, out_mask)
                else:
                    x_message = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=2)
                    x_message = torch.bmm(wigner[:, out_mask, :], x_message)
                m0, m1, m2 = self.so2_conv_1.prepare_and_linear(x_message, x_edge)
            x0, x1, x2 = self.so2_gate_bridge(m0, m1, m2)
            x_message = self.so2_conv_2.epilogue(
                *self.so2_conv_2.linear_from_prepared(x0, x1, x2)
            )
        else:
            if self.fuse_gather:
                x_message = gather_cat_wigner(x, edge_index, wigner, out_mask)
            else:
                x_message = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=2)
                x_message = torch.bmm(wigner[:, out_mask, :], x_message)
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
            x_message = self.act(x_0_gating, x_message)
            x_message = self.so2_conv_2(x_message, x_edge)
        if self.fuse_reverse:
            return reverse_envelope_scatter(
                x_message, wigner_inv, out_mask, edge_distance,
                edge_index[1], x, self.cutoff, 1.0, node_offset, False,
            )
        if self.use_envelope:
            x_message = x_message * self.envelope(edge_distance / self.cutoff).view(-1, 1, 1)
        x_message = torch.bmm(wigner_inv[:, :, out_mask], x_message)
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:], dtype=x_message.dtype,
            device=x_message.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)
        return new_embedding


class FusedEdgeDegreeEmbedding(nn.Module):
    def __init__(self, original: EdgeDegreeEmbedding) -> None:
        super().__init__()
        self.sphere_channels = original.sphere_channels
        self.lmax = original.lmax
        self.mmax = original.mmax
        self.mappingReduced = original.mappingReduced
        self.m_0_num_coefficients = original.m_0_num_coefficients
        self.m_all_num_coefficents = original.m_all_num_coefficents
        self.rad_func = original.rad_func
        self.rescale_factor = original.rescale_factor
        self.use_envelope = original.use_envelope
        self.cutoff = original.cutoff
        self.out_mask = original.out_mask

    def forward(self, x, x_edge, edge_distance, edge_index, wigner_inv, node_offset=0):
        out_mask = self.out_mask.to(device=x.device)
        x_edge_m0 = self.rad_func(x_edge).reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        padding = x_edge_m0.new_zeros(
            x_edge_m0.shape[0],
            self.m_all_num_coefficents - self.m_0_num_coefficients,
            self.sphere_channels,
        )
        message = torch.cat((x_edge_m0, padding), dim=1)
        message = torch.einsum("nac,ab->nbc", message, self.mappingReduced.to_m)
        return reverse_envelope_scatter(
            message, wigner_inv, out_mask, edge_distance, edge_index[1], x,
            self.cutoff, 1.0 / self.rescale_factor, node_offset, True,
        )


class FusedRadialMLP(nn.Module):
    """Fused ``Linear+LayerNorm+SiLU+Linear+LayerNorm+SiLU+Linear`` row MLP.

    Replaces one ``RadialMLP`` instance (edge-degree embedding or a block's
    ``so2_conv_1.rad_func``).  Hidden widths must be powers of two.
    """

    def __init__(self, original: RadialMLP) -> None:
        super().__init__()
        expected = (
            nn.Linear, nn.LayerNorm, nn.SiLU,
            nn.Linear, nn.LayerNorm, nn.SiLU, nn.Linear,
        )
        net = original.net
        if len(net) != len(expected) or not all(
            isinstance(layer, kind) for layer, kind in zip(net, expected)
        ):
            raise UnsupportedFusionConfigError(
                "RadialMLP structure is not the supported 3x Linear + 2x "
                "LayerNorm + 2x SiLU chain"
            )
        self.l1, self.ln1, _, self.l2, self.ln2, _, self.l3 = net
        self.in_channels = self.l1.in_features
        self.h1_channels = self.l1.out_features
        self.h2_channels = self.l2.out_features
        self.out_channels = self.l3.out_features
        if self.l2.in_features != self.h1_channels:
            raise UnsupportedFusionConfigError("RadialMLP hidden widths mismatch")
        if self.l3.in_features != self.h2_channels:
            raise UnsupportedFusionConfigError("RadialMLP hidden widths mismatch")
        for width in (self.h1_channels, self.h2_channels):
            if width < 16 or width & (width - 1):
                raise UnsupportedFusionConfigError(
                    f"RadialMLP hidden width {width} is not a power of two"
                )
        self.eps = float(self.ln1.eps)
        if float(self.ln2.eps) != self.eps:
            raise UnsupportedFusionConfigError("RadialMLP LayerNorm eps mismatch")

    def forward(self, inputs: Tensor) -> Tensor:
        return _FusedRadialMLP.apply(
            inputs,
            self.l1.weight, self.l1.bias, self.ln1.weight, self.ln1.bias,
            self.l2.weight, self.l2.bias, self.ln2.weight, self.ln2.bias,
            self.l3.weight, self.l3.bias,
            self.in_channels, self.h1_channels, self.h2_channels,
            self.out_channels, self.eps,
        )


class FrozenSO3Linear(nn.Module):
    """Inference-only ``SO3_Linear`` with a configure-time weight cache.

    The official layer expands its four degree weights to all 16 spherical
    coefficients with ``index_select`` on every forward.  eSEN force inference
    freezes model parameters, so the expanded tensor is immutable and can be
    materialized once without changing the einsum/cuBLAS computation.  The
    ordinary PyTorch einsum is intentionally retained; only the repeated
    weight-gather kernel and its temporary allocation are removed.
    """

    def __init__(self, original: SO3_Linear) -> None:
        super().__init__()
        if not isinstance(original, SO3_Linear):
            raise UnsupportedFusionConfigError(
                "SO3 weight cache requires an official SO3_Linear module"
            )
        if (
            original.lmax != 3
            or original.in_features != 128
            or original.out_features != 128
        ):
            raise UnsupportedFusionConfigError(
                "SO3 weight cache only supports the 30M [16,128] layout; "
                f"got lmax={original.lmax}, in={original.in_features}, "
                f"out={original.out_features}"
            )
        if original.weight.requires_grad or original.bias.requires_grad:
            raise UnsupportedFusionConfigError(
                "SO3 weight cache requires frozen weight and bias parameters"
            )
        expected_index = torch.tensor(
            [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            dtype=torch.long,
            device=original.expand_index.device,
        )
        if not torch.equal(original.expand_index, expected_index):
            raise UnsupportedFusionConfigError(
                "SO3 weight cache received an unsupported degree expansion"
            )

        self.in_features = original.in_features
        self.out_features = original.out_features
        self.lmax = original.lmax
        # Keep the checkpoint parameters reachable under the same public names.
        # They stay frozen; forward reads only the immutable expanded buffer.
        self.weight = original.weight
        self.bias = original.bias
        expanded = original.weight.detach().index_select(
            0, original.expand_index
        ).contiguous()
        self.register_buffer("expanded_weight", expanded, persistent=False)
        self.cached_weight_bytes = expanded.numel() * expanded.element_size()

    def forward(self, input_embedding: Tensor) -> Tensor:
        if input_embedding.ndim != 3 or tuple(input_embedding.shape[1:]) != (
            16,
            self.in_features,
        ):
            raise UnsupportedFusionConfigError(
                "SO3 weight cache expected input [N,16,128], got "
                f"{tuple(input_embedding.shape)}"
            )
        if (
            input_embedding.device != self.expanded_weight.device
            or input_embedding.dtype != self.expanded_weight.dtype
        ):
            raise UnsupportedFusionConfigError(
                "SO3 weight cache input must match the cached FP32 CUDA weight"
            )
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, self.expanded_weight
        )
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"lmax={self.lmax}, cached_weight_bytes={self.cached_weight_bytes}"
        )


class FusedSpectralAtomwise(nn.Module):
    """Fused ``SO3_Linear + GateActivation + SO3_Linear`` atomwise chain.

    The per-step ``SO3_Linear`` weight ``index_select`` disappears: expanded
    weights are materialized once at configure time.
    """

    def __init__(self, original: SpectralAtomwise) -> None:
        super().__init__()
        self.sphere_channels = original.sphere_channels
        self.hidden_channels = original.hidden_channels
        self.lmax = original.lmax
        self.scalar_mlp = original.scalar_mlp
        expand = original.so3_linear_1.expand_index
        self.w1 = original.so3_linear_1.weight.index_select(0, expand).contiguous()
        self.b1 = original.so3_linear_1.bias.detach().contiguous()
        self.w2 = original.so3_linear_2.weight.index_select(0, expand).contiguous()
        self.b2 = original.so3_linear_2.bias.detach().contiguous()
        if self.lmax != 3 or tuple(self.w1.shape) != (16, 128, 128):
            raise UnsupportedFusionConfigError(
                "FusedSpectralAtomwise requires lmax=3 and 128 channels, got "
                f"lmax={self.lmax} w1={tuple(self.w1.shape)}"
            )
        if tuple(self.w2.shape) != (16, 128, 128):
            raise UnsupportedFusionConfigError(
                f"FusedSpectralAtomwise w2={tuple(self.w2.shape)}"
            )

    def forward(self, x: Tensor) -> Tensor:
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        gating = gating_scalars.reshape(-1, self.lmax, self.sphere_channels)
        return _FusedSO3MLP.apply(x, gating, self.w1, self.b1, self.w2, self.b2)


class FusedEnergyBlock(nn.Module):
    """Fused energy-head ``Linear+SiLU+Linear+SiLU+Linear(->1)`` over nodes."""

    def __init__(self, original: nn.Sequential) -> None:
        super().__init__()
        expected = (nn.Linear, nn.SiLU, nn.Linear, nn.SiLU, nn.Linear)
        if len(original) != len(expected) or not all(
            isinstance(layer, kind) for layer, kind in zip(original, expected)
        ):
            raise UnsupportedFusionConfigError(
                "energy_block is not the supported 3x Linear + 2x SiLU chain"
            )
        self.l1, _, self.l2, _, self.l3 = original
        widths = (
            self.l1.in_features,
            self.l1.out_features,
            self.l2.out_features,
            self.l3.out_features,
        )
        if widths != (128, 128, 128, 1):
            raise UnsupportedFusionConfigError(
                f"FusedEnergyBlock requires widths (128,128,128,1), got {widths}"
            )
        if self.l2.in_features != 128 or self.l3.in_features != 128:
            raise UnsupportedFusionConfigError("FusedEnergyBlock hidden widths mismatch")

    def forward(self, x: Tensor) -> Tensor:
        return _FusedEnergyMLP.apply(
            x,
            self.l1.weight, self.l1.bias,
            self.l2.weight, self.l2.bias,
            self.l3.weight, self.l3.bias,
        )


@dataclass(frozen=True)
class FusionMetadata:
    requested: tuple[str, ...]
    edgewise_replacements: int
    edge_embedding_replacements: int
    rmsnorm_replacements: int
    gate_replacements: int
    radial_mlp_replacements: int
    so3_mlp_replacements: int
    energy_head_replacements: int
    so2_convolution_replacements: int
    so2_prepare_kernel_count: int
    so2_epilogue_kernel_count: int
    so2_fusion_kernel_version: str
    so2_gate_bridge_replacements: int
    so2_gate_bridge_forward_kernel_count: int
    so2_gate_bridge_backward_kernel_count: int
    so2_gate_bridge_kernel_version: str
    wigner_so2_bridge_replacements: int
    wigner_so2_bridge_forward_kernel_count: int
    wigner_so2_bridge_backward_kernel_count: int
    wigner_so2_bridge_kernel_version: str
    wigner_so2_hybrid_replacements: int
    wigner_so2_hybrid_forward_kernel_count: int
    wigner_so2_hybrid_backward_reduce_kernel_count: int
    wigner_so2_hybrid_backward_cublas_bmm_count: int
    wigner_so2_hybrid_kernel_version: str
    so2_block_gemm_convolution_replacements: int
    so2_block_gemm_linear_replacements: int
    so2_block_gemm_version: str
    so3_weight_cache_replacements: int
    so3_weight_cache_expanded_weight_count: int
    so3_weight_cache_bytes: int
    so3_weight_cache_version: str
    so2_prepare_backward_reduce_replacements: int
    so2_prepare_backward_reduce_kernel_count: int
    so2_prepare_backward_reduce_version: str
    configure_wall_time_s: float
    torch_version: str
    triton_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "model_fusions": ",".join(self.requested),
            "model_fusion_kernel_version": FUSION_KERNEL_VERSION,
            "model_fusion_edgewise_replacements": self.edgewise_replacements,
            "model_fusion_edge_embedding_replacements": self.edge_embedding_replacements,
            "model_fusion_rmsnorm_replacements": self.rmsnorm_replacements,
            "model_fusion_gate_replacements": self.gate_replacements,
            "model_fusion_radial_mlp_replacements": self.radial_mlp_replacements,
            "model_fusion_so3_mlp_replacements": self.so3_mlp_replacements,
            "model_fusion_energy_head_replacements": self.energy_head_replacements,
            "model_fusion_so2_convolution_replacements": self.so2_convolution_replacements,
            "model_fusion_so2_prepare_kernel_count": self.so2_prepare_kernel_count,
            "model_fusion_so2_epilogue_kernel_count": self.so2_epilogue_kernel_count,
            "model_fusion_so2_kernel_version": self.so2_fusion_kernel_version,
            "model_fusion_so2_gate_bridge_replacements": (
                self.so2_gate_bridge_replacements
            ),
            "model_fusion_so2_gate_bridge_forward_kernel_count": (
                self.so2_gate_bridge_forward_kernel_count
            ),
            "model_fusion_so2_gate_bridge_backward_kernel_count": (
                self.so2_gate_bridge_backward_kernel_count
            ),
            "model_fusion_so2_gate_bridge_kernel_version": (
                self.so2_gate_bridge_kernel_version
            ),
            "model_fusion_wigner_so2_bridge_replacements": (
                self.wigner_so2_bridge_replacements
            ),
            "model_fusion_wigner_so2_bridge_forward_kernel_count": (
                self.wigner_so2_bridge_forward_kernel_count
            ),
            "model_fusion_wigner_so2_bridge_backward_kernel_count": (
                self.wigner_so2_bridge_backward_kernel_count
            ),
            "model_fusion_wigner_so2_bridge_kernel_version": (
                self.wigner_so2_bridge_kernel_version
            ),
            "model_fusion_wigner_so2_hybrid_replacements": (
                self.wigner_so2_hybrid_replacements
            ),
            "model_fusion_wigner_so2_hybrid_forward_kernel_count": (
                self.wigner_so2_hybrid_forward_kernel_count
            ),
            "model_fusion_wigner_so2_hybrid_backward_reduce_kernel_count": (
                self.wigner_so2_hybrid_backward_reduce_kernel_count
            ),
            "model_fusion_wigner_so2_hybrid_backward_cublas_bmm_count": (
                self.wigner_so2_hybrid_backward_cublas_bmm_count
            ),
            "model_fusion_wigner_so2_hybrid_kernel_version": (
                self.wigner_so2_hybrid_kernel_version
            ),
            "model_fusion_so2_block_gemm_convolution_replacements": (
                self.so2_block_gemm_convolution_replacements
            ),
            "model_fusion_so2_block_gemm_linear_replacements": (
                self.so2_block_gemm_linear_replacements
            ),
            "model_fusion_so2_block_gemm_version": self.so2_block_gemm_version,
            "model_fusion_so3_weight_cache_replacements": (
                self.so3_weight_cache_replacements
            ),
            "model_fusion_so3_weight_cache_expanded_weight_count": (
                self.so3_weight_cache_expanded_weight_count
            ),
            "model_fusion_so3_weight_cache_bytes": self.so3_weight_cache_bytes,
            "model_fusion_so3_weight_cache_version": self.so3_weight_cache_version,
            "model_fusion_so2_prepare_backward_reduce_replacements": (
                self.so2_prepare_backward_reduce_replacements
            ),
            "model_fusion_so2_prepare_backward_reduce_kernel_count": (
                self.so2_prepare_backward_reduce_kernel_count
            ),
            "model_fusion_so2_prepare_backward_reduce_version": (
                self.so2_prepare_backward_reduce_version
            ),
            "model_fusion_configure_wall_time_s": self.configure_wall_time_s,
            "model_fusion_torch_version": self.torch_version,
            "model_fusion_triton_version": self.triton_version,
        }


def _energy_head_candidates(model: nn.Module) -> list[nn.Module]:
    """Return output heads across current and legacy HydraModel layouts."""
    heads = getattr(model, "output_heads", None)
    if isinstance(heads, nn.ModuleDict):
        return list(heads.values())
    if isinstance(heads, dict):
        return list(heads.values())
    legacy = getattr(model, "head", None)
    return [legacy] if legacy is not None else []


def _validate_30m_model(model: nn.Module) -> nn.Module:
    if triton is None:
        raise UnsupportedFusionConfigError("Opt4 model fusion requires Triton")
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise UnsupportedFusionConfigError("Expected an eSEN HydraModel")
    expected = {
        "lmax": 3, "mmax": 2, "sphere_channels": 128,
        "hidden_channels": 128, "num_layers": 10,
        "act_type": "gate", "norm_type": "rms_norm_sh",
        "mlp_type": "spectral", "use_envelope": True,
    }
    mismatches = [
        f"{key}={getattr(backbone, key, None)!r} (expected {value!r})"
        for key, value in expected.items()
        if getattr(backbone, key, None) != value
    ]
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        mismatches.append(f"model dtype={parameter.dtype} (expected torch.float32)")
    if parameter.device.type != "cuda":
        mismatches.append(f"model device={parameter.device} (expected CUDA)")
    if any(parameter.requires_grad for parameter in model.parameters()):
        mismatches.append("model parameters are not frozen")
    if mismatches:
        raise UnsupportedFusionConfigError(
            "Unsupported eSEN Opt4 30M configuration: " + "; ".join(mismatches)
        )
    return backbone


def configure_esen_30m_model_fusions(
    model: nn.Module, requested: str | Iterable[str]
) -> FusionMetadata:
    """Install the requested kernels on one loaded inference model instance."""

    selected = parse_model_fusions(requested)
    existing = getattr(model, "_esen_opt4_fusion_metadata", None)
    if existing is not None:
        if tuple(existing.requested) != selected:
            raise UnsupportedFusionConfigError(
                "The model instance is already configured with a different fusion set"
            )
        return existing
    start = time.perf_counter()
    backbone = _validate_30m_model(model)
    gather = "gather-wigner" in selected
    reverse = "reverse-scatter" in selected
    so2 = "so2-epilogue" in selected
    so2_gate_bridge = "so2-gate-bridge" in selected
    wigner_so2_bridge = "wigner-so2-bridge" in selected
    wigner_so2_hybrid = "wigner-so2-hybrid" in selected
    so2_block_gemm = "so2-block-gemm" in selected
    so3_weight_cache = "so3-weight-cache" in selected
    so2_prepare_backward_reduce = "so2-prepare-backward-reduce" in selected
    edgewise_count = 0
    edge_embedding_count = 0
    rmsnorm_count = 0
    gate_count = 0
    radial_count = 0
    so3_count = 0
    energy_count = 0
    so2_count = 0
    so2_gate_bridge_count = 0
    wigner_so2_bridge_count = 0
    wigner_so2_hybrid_count = 0
    so2_block_gemm_count = 0
    so3_weight_cache_count = 0
    so3_weight_cache_bytes = 0
    so2_prepare_backward_reduce_count = 0

    # radial-mlp first: the FusedEdgewise / FusedEdgeDegreeEmbedding wrappers
    # installed below hold references to these rad_func objects.
    if "radial-mlp" in selected:
        radial_funcs = [backbone.edge_degree_embedding.rad_func]
        for block in backbone.blocks:
            radial_funcs.append(block.edge_wise.so2_conv_1.rad_func)
        for index, rad_func in enumerate(radial_funcs):
            if not isinstance(rad_func, RadialMLP):
                raise UnsupportedFusionConfigError(
                    f"RadialMLP instance {index} has an unexpected type"
                )
        backbone.edge_degree_embedding.rad_func = FusedRadialMLP(
            backbone.edge_degree_embedding.rad_func
        )
        for block in backbone.blocks:
            block.edge_wise.so2_conv_1.rad_func = FusedRadialMLP(
                block.edge_wise.so2_conv_1.rad_func
            )
        radial_count = len(radial_funcs)

    if reverse:
        if not isinstance(backbone.edge_degree_embedding, EdgeDegreeEmbedding):
            raise UnsupportedFusionConfigError("Unexpected EdgeDegreeEmbedding implementation")
        backbone.edge_degree_embedding = FusedEdgeDegreeEmbedding(
            backbone.edge_degree_embedding
        )
        edge_embedding_count = 1

    if so2:
        for block in backbone.blocks:
            if not isinstance(block.edge_wise, Edgewise):
                raise UnsupportedFusionConfigError("Unexpected Edgewise implementation")
            if not isinstance(block.edge_wise.so2_conv_1, SO2_Convolution):
                raise UnsupportedFusionConfigError("Unexpected SO2 convolution 1 implementation")
            if not isinstance(block.edge_wise.so2_conv_2, SO2_Convolution):
                raise UnsupportedFusionConfigError("Unexpected SO2 convolution 2 implementation")
            block.edge_wise.so2_conv_1 = FusedSO2Convolution(
                block.edge_wise.so2_conv_1,
                block_gemm=so2_block_gemm,
                prepare_backward_reduce=so2_prepare_backward_reduce,
            )
            block.edge_wise.so2_conv_2 = FusedSO2Convolution(
                block.edge_wise.so2_conv_2, block_gemm=so2_block_gemm
            )
            so2_count += 2
            if so2_prepare_backward_reduce:
                so2_prepare_backward_reduce_count += 1
            if so2_block_gemm:
                so2_block_gemm_count += 2

    if gather or reverse or so2:
        for block in backbone.blocks:
            if not isinstance(block.edge_wise, Edgewise):
                raise UnsupportedFusionConfigError("Unexpected Edgewise implementation")
            block.edge_wise = FusedEdgewise(
                block.edge_wise,
                gather=gather,
                reverse=reverse,
                so2_epilogue=so2,
                so2_gate_bridge=so2_gate_bridge,
                wigner_so2_bridge=wigner_so2_bridge,
                wigner_so2_hybrid=wigner_so2_hybrid,
            )
            edgewise_count += 1
            if so2_gate_bridge:
                so2_gate_bridge_count += 1
            if wigner_so2_bridge:
                wigner_so2_bridge_count += 1
            if wigner_so2_hybrid:
                wigner_so2_hybrid_count += 1
    if so2_gate_bridge and so2_gate_bridge_count != 10:
        raise UnsupportedFusionConfigError(
            "SO2 gate bridge must replace all 10 Edgewise blocks, replaced "
            f"{so2_gate_bridge_count}"
        )
    if wigner_so2_bridge and wigner_so2_bridge_count != 10:
        raise UnsupportedFusionConfigError(
            "Wigner/SO2 bridge must replace all 10 Edgewise blocks, replaced "
            f"{wigner_so2_bridge_count}"
        )
    if wigner_so2_hybrid and wigner_so2_hybrid_count != 10:
        raise UnsupportedFusionConfigError(
            "Wigner/SO2 hybrid must replace all 10 Edgewise blocks, replaced "
            f"{wigner_so2_hybrid_count}"
        )
    if so2_block_gemm and so2_block_gemm_count != 20:
        raise UnsupportedFusionConfigError(
            "SO2 block GEMM must replace all 20 Edgewise convolutions, replaced "
            f"{so2_block_gemm_count}"
        )
    if (
        so2_prepare_backward_reduce
        and so2_prepare_backward_reduce_count != 10
    ):
        raise UnsupportedFusionConfigError(
            "SO2 prepare backward reduction must replace conv1 in all 10 "
            "Edgewise blocks, replaced "
            f"{so2_prepare_backward_reduce_count}"
        )

    if "rmsnorm" in selected:
        norms = [backbone.norm]
        for block in backbone.blocks:
            norms.extend((block.norm_1, block.norm_2))
        if not all(isinstance(norm, EquivariantRMSNormArraySphericalHarmonicsV2) for norm in norms):
            raise UnsupportedFusionConfigError("Unexpected RMSNormSH implementation")
        backbone.norm = FusedRMSNormSH(backbone.norm)
        for block in backbone.blocks:
            block.norm_1 = FusedRMSNormSH(block.norm_1)
            block.norm_2 = FusedRMSNormSH(block.norm_2)
        rmsnorm_count = len(norms)

    if so3_weight_cache:
        for block in backbone.blocks:
            if not isinstance(block.atom_wise, SpectralAtomwise):
                raise UnsupportedFusionConfigError(
                    "SO3 weight cache requires the official SpectralAtomwise"
                )
            for name in ("so3_linear_1", "so3_linear_2"):
                original = getattr(block.atom_wise, name, None)
                if not isinstance(original, SO3_Linear):
                    raise UnsupportedFusionConfigError(
                        f"Unexpected {name} implementation for SO3 weight cache"
                    )
                cached = FrozenSO3Linear(original)
                setattr(block.atom_wise, name, cached)
                so3_weight_cache_count += 1
                so3_weight_cache_bytes += cached.cached_weight_bytes
        if so3_weight_cache_count != 20:
            raise UnsupportedFusionConfigError(
                "SO3 weight cache must replace all 20 SO3_Linear modules, "
                f"replaced {so3_weight_cache_count}"
            )

    if "so3-mlp" in selected:
        for block in backbone.blocks:
            if not isinstance(block.atom_wise, SpectralAtomwise):
                raise UnsupportedFusionConfigError("Unexpected Atomwise implementation")
            block.atom_wise = FusedSpectralAtomwise(block.atom_wise)
            so3_count += 1

    if "gate" in selected:
        for block in backbone.blocks:
            if not isinstance(block.edge_wise.act, GateActivation):
                raise UnsupportedFusionConfigError("Unexpected Edgewise gate implementation")
            block.edge_wise.act = FusedGateActivation(block.edge_wise.act)
            gate_count += 1
            # With so3-mlp the atomwise gate runs inside the fused chain.
            if isinstance(block.atom_wise, SpectralAtomwise):
                if not isinstance(block.atom_wise.act, GateActivation):
                    raise UnsupportedFusionConfigError("Unexpected Atomwise gate implementation")
                block.atom_wise.act = FusedGateActivation(block.atom_wise.act)
                gate_count += 1

    if "energy-head" in selected:
        # HydraModel exposes inference heads as ``nn.ModuleDict``.  The helper
        # also keeps plain-dict and legacy ``model.head`` checkpoints working.
        for head in _energy_head_candidates(model):
            energy_block = getattr(head, "energy_block", None)
            if isinstance(energy_block, nn.Sequential):
                head.energy_block = FusedEnergyBlock(energy_block)
                energy_count += 1
        if energy_count == 0:
            raise UnsupportedFusionConfigError(
                "energy-head fusion requested but no energy_block found"
            )

    metadata = FusionMetadata(
        requested=selected,
        edgewise_replacements=edgewise_count,
        edge_embedding_replacements=edge_embedding_count,
        rmsnorm_replacements=rmsnorm_count,
        gate_replacements=gate_count,
        radial_mlp_replacements=radial_count,
        so3_mlp_replacements=so3_count,
        energy_head_replacements=energy_count,
        so2_convolution_replacements=so2_count,
        # KF10 bypasses conv1 epilogue and conv2 prepare once per Edgewise.
        so2_prepare_kernel_count=(
            so2_count
            - so2_gate_bridge_count
            - wigner_so2_bridge_count
            - wigner_so2_hybrid_count
        ),
        so2_epilogue_kernel_count=so2_count - so2_gate_bridge_count,
        so2_fusion_kernel_version=(SO2_FUSION_KERNEL_VERSION if so2 else ""),
        so2_gate_bridge_replacements=so2_gate_bridge_count,
        so2_gate_bridge_forward_kernel_count=so2_gate_bridge_count,
        so2_gate_bridge_backward_kernel_count=so2_gate_bridge_count,
        so2_gate_bridge_kernel_version=(
            SO2_GATE_BRIDGE_KERNEL_VERSION if so2_gate_bridge else ""
        ),
        wigner_so2_bridge_replacements=wigner_so2_bridge_count,
        wigner_so2_bridge_forward_kernel_count=wigner_so2_bridge_count,
        # The first-order backward uses dedicated x, Wigner and radial kernels.
        wigner_so2_bridge_backward_kernel_count=3 * wigner_so2_bridge_count,
        wigner_so2_bridge_kernel_version=(
            WIGNER_SO2_BRIDGE_KERNEL_VERSION if wigner_so2_bridge else ""
        ),
        wigner_so2_hybrid_replacements=wigner_so2_hybrid_count,
        wigner_so2_hybrid_forward_kernel_count=wigner_so2_hybrid_count,
        wigner_so2_hybrid_backward_reduce_kernel_count=(
            wigner_so2_hybrid_count
        ),
        wigner_so2_hybrid_backward_cublas_bmm_count=(
            2 * wigner_so2_hybrid_count
        ),
        wigner_so2_hybrid_kernel_version=(
            WIGNER_SO2_HYBRID_KERNEL_VERSION if wigner_so2_hybrid else ""
        ),
        so2_block_gemm_convolution_replacements=so2_block_gemm_count,
        # Each convolution has one m=1 and one m=2 complex GEMM.
        so2_block_gemm_linear_replacements=2 * so2_block_gemm_count,
        so2_block_gemm_version=(
            SO2_BLOCK_GEMM_VERSION if so2_block_gemm else ""
        ),
        so3_weight_cache_replacements=so3_weight_cache_count,
        so3_weight_cache_expanded_weight_count=so3_weight_cache_count,
        so3_weight_cache_bytes=so3_weight_cache_bytes,
        so3_weight_cache_version=(
            SO3_WEIGHT_CACHE_VERSION if so3_weight_cache else ""
        ),
        so2_prepare_backward_reduce_replacements=(
            so2_prepare_backward_reduce_count
        ),
        so2_prepare_backward_reduce_kernel_count=(
            so2_prepare_backward_reduce_count
        ),
        so2_prepare_backward_reduce_version=(
            SO2_PREPARE_BACKWARD_REDUCE_VERSION
            if so2_prepare_backward_reduce
            else ""
        ),
        configure_wall_time_s=time.perf_counter() - start,
        torch_version=torch.__version__,
        triton_version=str(getattr(triton, "__version__", "unknown")),
    )
    model._esen_opt4_fusion_metadata = metadata
    return metadata
