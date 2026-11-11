"""Experimental FP64 NHC half-step. Never enabled by frozen Opt4 defaults.

One CTA owns all momenta and three thermostat variables. No host reads,
atomics, persistent cache, mixed precision, or change to the NHC splitting.
"""

from __future__ import annotations

import torch

from fairchem.core.applications.esen_matbench import MatbenchNHCIntegrator

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _nhc_half_kernel(
        P,
        M,
        Q,
        Eta,
        PEta,
        Coeff,
        Out,
        EtaOut,
        PEtaOut,
        N: tl.constexpr,
        KT: tl.constexpr,
        DT: tl.constexpr,
        B: tl.constexpr,
    ):
        i = tl.arange(0, B)
        p = tl.load(P + i, i < 3 * N, other=0)
        m = tl.load(M + i // 3, i < 3 * N, other=1)
        q0, q1, q2 = tl.load(Q), tl.load(Q + 1), tl.load(Q + 2)
        a, b, c = tl.load(PEta), tl.load(PEta + 1), tl.load(PEta + 2)
        e0, e1, e2 = tl.load(Eta), tl.load(Eta + 1), tl.load(Eta + 2)
        kt = tl.full((), KT, tl.float64)
        dt = tl.full((), DT, tl.float64)
        for j in tl.static_range(3):
            d = tl.load(Coeff + j) * dt
            d2, d4 = d / 2.0, d / 4.0
            # Reverse thermostat sweep: 2, 1, 0.
            c = c + d2 * (b * b / q1 - kt)
            f = libdevice.exp(-d4 * c / q2)
            b = (b * f + d2 * (a * a / q0 - kt)) * f
            f = libdevice.exp(-d4 * b / q1)
            a = (a * f + d2 * (tl.sum(p * p / m, 0) - 3 * N * kt)) * f
            e0, e1, e2 = e0 + d * (a / q0), e1 + d * (b / q1), e2 + d * (c / q2)
            p = p * libdevice.exp(-d * a / q0)
            # Forward thermostat sweep: 0, 1, 2; use updated momenta.
            f = libdevice.exp(-d4 * b / q1)
            a = (a * f + d2 * (tl.sum(p * p / m, 0) - 3 * N * kt)) * f
            f = libdevice.exp(-d4 * c / q2)
            b = (b * f + d2 * (a * a / q0 - kt)) * f
            c = c + d2 * (b * b / q1 - kt)
        tl.store(Out + i, p, i < 3 * N)
        tl.store(EtaOut, e0)
        tl.store(EtaOut + 1, e1)
        tl.store(EtaOut + 2, e2)
        tl.store(PEtaOut, a)
        tl.store(PEtaOut + 1, b)
        tl.store(PEtaOut + 2, c)


class FusedMatbenchNHCIntegrator(MatbenchNHCIntegrator):
    """Inference-only, 1--1024 real atoms, tchain=3 and tloop=1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.tchain != 3
            or self.tloop != 1
            or self.coefficients.numel() != 3
            or not 1 <= self.num_atoms <= 1024
        ):
            raise ValueError("NHC experiment requires tchain=3, tloop=1, 1<=N<=1024")

    def integrate_nhc(self, momenta, eta, p_eta, delta):
        if triton is None:
            raise RuntimeError("NHC experiment requires CUDA Triton")
        if not isinstance(delta, (float, int)):
            raise ValueError("NHC experiment requires a host scalar delta")
        tensors = (momenta, eta, p_eta, self.masses, self.Q, self.coefficients)
        if any(
            t.device != momenta.device
            or t.dtype != torch.float64
            or not t.is_cuda
            or t.requires_grad
            for t in tensors
        ):
            raise ValueError(
                "NHC experiment requires same-device CUDA FP64, no gradients"
            )
        if (
            momenta.shape != (self.num_atoms, 3)
            or eta.shape != (3,)
            or p_eta.shape != (3,)
        ):
            raise ValueError("Invalid momenta/thermostat shape")
        momenta, eta, p_eta = momenta.contiguous(), eta.contiguous(), p_eta.contiguous()
        p_out, e_out, pe_out = (torch.empty_like(t) for t in (momenta, eta, p_eta))
        _nhc_half_kernel[(1,)](
            momenta,
            self.masses.contiguous(),
            self.Q,
            eta,
            p_eta,
            self.coefficients,
            p_out,
            e_out,
            pe_out,
            self.num_atoms,
            self.kT,
            float(delta),
            triton.next_power_of_2(3 * self.num_atoms),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return p_out, e_out, pe_out
