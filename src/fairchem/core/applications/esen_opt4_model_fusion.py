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

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU-only import tests
    triton = None
    tl = None


FUSION_KERNEL_VERSION = "opt4-model-fusion-v2"
SUPPORTED_FUSIONS = (
    "gather-wigner",
    "reverse-scatter",
    "rmsnorm",
    "gate",
    "radial-mlp",
    "so3-mlp",
    "energy-head",
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
        a1 = hhat1 * tl.load(g1_ptr + j1)[None, :] + tl.load(be1_ptr + j1)[None, :]
        sig1 = 1.0 / (1.0 + tl.exp(-a1))
        a1 = a1 * sig1
        tl.store(save_hhat1_ptr + r[:, None] * h1_ch + j1[None, :], hhat1, mask=rmask[:, None])
        tl.store(save_rstd1_ptr + r, rstd1, mask=rmask)
        tl.store(save_a1_ptr + r[:, None] * h1_ch + j1[None, :], a1, mask=rmask[:, None])

        j2 = tl.arange(0, h2_ch)
        acc2 = tl.zeros((BLOCK_R, h2_ch), tl.float32)
        for k0 in range(0, h1_ch, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            a_t = tl.load(
                save_a1_ptr + r[:, None] * h1_ch + k[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            w_t = tl.load(w2_ptr + j2[:, None] * h1_ch + k[None, :], other=0.0)
            acc2 = tl.dot(a_t, tl.trans(w_t), acc2, input_precision="ieee")
        h2 = acc2 + tl.load(b2_ptr + j2)[None, :]
        mean2 = tl.sum(h2, axis=1) / h2_ch
        h2c = h2 - mean2[:, None]
        rstd2 = 1.0 / tl.sqrt(tl.sum(h2c * h2c, axis=1) / h2_ch + eps)
        hhat2 = h2c * rstd2[:, None]
        a2 = hhat2 * tl.load(g2_ptr + j2)[None, :] + tl.load(be2_ptr + j2)[None, :]
        sig2 = 1.0 / (1.0 + tl.exp(-a2))
        a2 = a2 * sig2
        tl.store(save_hhat2_ptr + r[:, None] * h2_ch + j2[None, :], hhat2, mask=rmask[:, None])
        tl.store(save_rstd2_ptr + r, rstd2, mask=rmask)
        tl.store(save_a2_ptr + r[:, None] * h2_ch + j2[None, :], a2, mask=rmask[:, None])

        for o0 in range(0, out_ch, BLOCK_O):
            o = o0 + tl.arange(0, BLOCK_O)
            omask = o < out_ch
            acc3 = tl.zeros((BLOCK_R, BLOCK_O), tl.float32)
            for k0 in range(0, h2_ch, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                a_t = tl.load(
                    save_a2_ptr + r[:, None] * h2_ch + k[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
                w_t = tl.load(
                    w3_ptr + o[:, None] * h2_ch + k[None, :],
                    mask=omask[:, None],
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
        a2 = tl.load(
            save_a2_ptr + r[:, None] * h2_ch + j2[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig2 = 1.0 / (1.0 + tl.exp(-a2))
        g_n2 = g_a2 * (sig2 * (1.0 + a2 * (1.0 - sig2)))
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
                mask=rmask[:, None],
                other=0.0,
            )
            # g_a1[r, i] = sum_j g_h2[r, j] * W2[j, i]
            w_t = tl.load(
                w2_ptr + k[:, None] * h1_ch + j1[None, :], other=0.0
            )
            g_a1 = tl.dot(gh_t, w_t, g_a1, input_precision="ieee")
        a1 = tl.load(
            save_a1_ptr + r[:, None] * h1_ch + j1[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig1 = 1.0 / (1.0 + tl.exp(-a1))
        g_n1 = g_a1 * (sig1 * (1.0 + a1 * (1.0 - sig1)))
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
                    mask=rmask[:, None],
                    other=0.0,
                )
                w_t = tl.load(
                    w1_ptr + m * channels * channels + c[:, None] * channels + k[None, :],
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
                    mask=rmask[:, None],
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
                    mask=rmask[:, None],
                    other=0.0,
                )
                w_t = tl.load(
                    w2_ptr + m * channels * channels + k[:, None] * channels + c[None, :],
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
            for k0 in range(0, channels, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                gh_t = tl.load(
                    scratch_g_h_ptr + r[:, None] * coeffs * channels + m * channels + k[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
                w_t = tl.load(
                    w1_ptr + m * channels * channels + k[:, None] * channels + c[None, :],
                    other=0.0,
                )
                acc = tl.dot(gh_t, w_t, input_precision="ieee")
                tl.store(
                    grad_x_ptr + r[:, None] * coeffs * channels + m * channels + c[None, :],
                    acc,
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
                mask=rmask[:, None],
                other=0.0,
            )
            w_t = tl.load(w1_ptr + j[:, None] * channels + k[None, :], other=0.0)
            acc1 = tl.dot(x_t, tl.trans(w_t), acc1, input_precision="ieee")
        h1 = acc1 + tl.load(b1_ptr + j)[None, :]
        sig1 = 1.0 / (1.0 + tl.exp(-h1))
        a1 = h1 * sig1
        tl.store(save_a1_ptr + r[:, None] * channels + j[None, :], a1, mask=rmask[:, None])

        acc2 = tl.zeros((BLOCK_R, channels), tl.float32)
        for k0 in range(0, channels, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            a_t = tl.load(
                save_a1_ptr + r[:, None] * channels + k[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            w_t = tl.load(w2_ptr + j[:, None] * channels + k[None, :], other=0.0)
            acc2 = tl.dot(a_t, tl.trans(w_t), acc2, input_precision="ieee")
        h2 = acc2 + tl.load(b2_ptr + j)[None, :]
        sig2 = 1.0 / (1.0 + tl.exp(-h2))
        a2 = h2 * sig2
        tl.store(save_a2_ptr + r[:, None] * channels + j[None, :], a2, mask=rmask[:, None])

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
        a2 = tl.load(
            save_a2_ptr + r[:, None] * channels + j[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig2 = 1.0 / (1.0 + tl.exp(-a2))
        g_n2 = g_a2 * (sig2 * (1.0 + a2 * (1.0 - sig2)))
        tl.store(scratch_g_a2_ptr + r[:, None] * channels + j[None, :], g_n2, mask=rmask[:, None])

        g_a1 = tl.zeros((BLOCK_R, channels), tl.float32)
        for k0 in range(0, channels, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            gn_t = tl.load(
                scratch_g_a2_ptr + r[:, None] * channels + k[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            # g_a1[r, i] = sum_j g_n2[r, j] * W2[j, i]
            w_t = tl.load(
                w2_ptr + k[:, None] * channels + j[None, :], other=0.0
            )
            g_a1 = tl.dot(gn_t, w_t, g_a1, input_precision="ieee")
        a1 = tl.load(
            save_a1_ptr + r[:, None] * channels + j[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        sig1 = 1.0 / (1.0 + tl.exp(-a1))
        g_n1 = g_a1 * (sig1 * (1.0 + a1 * (1.0 - sig1)))
        tl.store(scratch_g_a1_ptr + r[:, None] * channels + j[None, :], g_n1, mask=rmask[:, None])

        for k0 in range(0, channels, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            gn_t = tl.load(
                scratch_g_a1_ptr + r[:, None] * channels + k[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            w_t = tl.load(w1_ptr + j[:, None] * channels + k[None, :], other=0.0)
            acc = tl.dot(gn_t, tl.trans(w_t), input_precision="ieee")
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
            grid = (triton.cdiv(rows, 16),)
            _radial_mlp_backward_kernel[grid](
                grad_out,
                save_a1, save_hhat1, save_rstd1,
                save_a2, save_hhat2, save_rstd2,
                g1, g2, w1, w2, w3,
                scratch_g_h2, scratch_g_h1, grad_x,
                rows=rows, in_ch=ctx.in_ch, h1_ch=ctx.h1_ch, h2_ch=ctx.h2_ch,
                out_ch=ctx.out_ch,
                BLOCK_R=16, BLOCK_K=32, BLOCK_O=256, num_warps=4,
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


def gather_cat_wigner(x: Tensor, edge_index: Tensor, wigner: Tensor, out_mask: Tensor) -> Tensor:
    # The official model can hand us views even though the logical shapes are
    # fixed.  Normalize their layout before entering Triton.  ``contiguous`` is
    # a no-op for the common case and remains differentiable/capture-safe when
    # a real copy is required.
    return _GatherWigner.apply(
        x.contiguous(), edge_index.contiguous(), wigner.contiguous(), out_mask
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
    def __init__(self, original: Edgewise, *, gather: bool, reverse: bool) -> None:
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

    def forward(self, x, x_edge, edge_distance, edge_index, wigner, wigner_inv, node_offset: int = 0):
        out_mask = self.out_mask.to(device=x.device)
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
            "model_fusion_configure_wall_time_s": self.configure_wall_time_s,
            "model_fusion_torch_version": self.torch_version,
            "model_fusion_triton_version": self.triton_version,
        }


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
    edgewise_count = 0
    edge_embedding_count = 0
    rmsnorm_count = 0
    gate_count = 0
    radial_count = 0
    so3_count = 0
    energy_count = 0

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

    if gather or reverse:
        for block in backbone.blocks:
            if not isinstance(block.edge_wise, Edgewise):
                raise UnsupportedFusionConfigError("Unexpected Edgewise implementation")
            block.edge_wise = FusedEdgewise(
                block.edge_wise, gather=gather, reverse=reverse
            )
            edgewise_count += 1

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
        heads = getattr(model, "output_heads", None)
        if isinstance(heads, dict):
            head_candidates = list(heads.values())
        else:
            head_candidates = [
                head for head in (getattr(model, "head", None),) if head is not None
            ]
        for head in head_candidates:
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
        configure_wall_time_s=time.perf_counter() - start,
        torch_version=torch.__version__,
        triton_version=str(getattr(triton, "__version__", "unknown")),
    )
    model._esen_opt4_fusion_metadata = metadata
    return metadata
