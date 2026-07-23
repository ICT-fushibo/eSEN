"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math

import torch
from e3nn import o3

YTOL = 0.999999


def init_edge_rot_mat(
    edge_distance_vec,
    rot_clip=False,
    fixed_reference_vec=None,
):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # The fixed-capacity model CUDA Graph supplies a fixed auxiliary vector.
    # Besides making identical replays stable, this marks a path where all
    # input distances were already validated by the eager graph builder.  The
    # diagnostic reductions below synchronize a CUDA scalar to the host and
    # therefore must not run during graph capture.  Ordinary eager inference
    # does not supply this argument and retains the original diagnostics/RNG.
    capture_safe = fixed_reference_vec is not None

    # Make sure the atoms are far enough apart
    # assert torch.min(edge_vec_0_distance) < 0.0001
    if (
        not capture_safe
        and len(edge_vec_0_distance) > 0
        and torch.min(edge_vec_0_distance) < 0.0001
    ):
        logging.error(f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)}")

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    if rot_clip:
        if capture_safe:
            # Dotting with the Cartesian y axis is exactly the y component.
            # Avoid ``new_tensor([0, 1, 0])`` here: constructing that CUDA
            # constant from a Python list performs a host-to-device copy,
            # which is illegal while a CUDA Graph stream is being captured.
            yprod = norm_x[:, 1]
            clipped_norm_x = torch.zeros_like(norm_x)
            clipped_norm_x[:, 1] = torch.sign(yprod)
            norm_x = torch.where(
                (torch.abs(yprod) > YTOL).view(-1, 1),
                clipped_norm_x,
                norm_x,
            )
        else:
            # Keep the official eager path unchanged.  This branch is used by
            # the ASE baseline and the existing GPU-resident eager backend.
            yprod = norm_x @ norm_x.new_tensor([0.0, 1.0, 0.0])
            norm_x[yprod > YTOL] = norm_x.new_tensor([0.0, 1.0, 0.0])
            norm_x[yprod < -YTOL] = norm_x.new_tensor([0.0, -1.0, 0.0])

    if fixed_reference_vec is None:
        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    else:
        if fixed_reference_vec.shape != edge_vec_0.shape:
            raise ValueError(
                "fixed_reference_vec must match edge_distance_vec shape, got "
                f"{fixed_reference_vec.shape} and {edge_vec_0.shape}"
            )
        edge_vec_2 = fixed_reference_vec
    edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1))
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    if not capture_safe and len(vec_dot) > 0:
        assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    if rot_clip:
        return edge_rot_mat
    else:
        return edge_rot_mat.detach()


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(
    lv: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    _Jd: list[torch.Tensor],
) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor:
    M = angle.new_zeros((*angle.shape, 2 * lv + 1, 2 * lv + 1))

    # The following code needs to replaced for a for loop because
    # torch.export barfs on outer product like operations
    # ie: torch.outer(frequences, angle) (same as frequencies * angle[..., None])
    # will place a non-sense Guard on the dimensions of angle when attempting to export setting
    # angle (edge dimensions) as dynamic. This may be fixed in torch2.4.

    # inds = torch.arange(0, 2 * lv + 1, 1, device=device)
    # reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
    # frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
    # M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    # M[..., inds, inds] = torch.cos(frequencies * angle[..., None])

    inds = list(range(0, 2 * lv + 1, 1))
    reversed_inds = list(range(2 * lv, -1, -1))
    frequencies = list(range(lv, -lv - 1, -1))
    for i in range(len(frequencies)):
        M[..., inds[i], reversed_inds[i]] = torch.sin(frequencies[i] * angle)
        M[..., inds[i], inds[i]] = torch.cos(frequencies[i] * angle)
    return M


def rotation_to_wigner(
    edge_rot_mat: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    Jd: list[torch.Tensor],
    rot_clip: bool = False,
    fixed_shape: bool = False,
) -> torch.Tensor:
    """
    set <rot_clip=True> to handle gradient instability when using gradient-based force/stress prediction.
    """
    if fixed_shape:
        # Matrix-vector multiplication by the Cartesian y axis selects the
        # second matrix column.  Indexing avoids creating a CUDA tensor from a
        # Python list during graph capture.
        x = edge_rot_mat[..., 1]
    else:
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
    alpha, beta = o3.xyz_to_angles(x)
    R = (
        o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        @ edge_rot_mat
    )
    gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

    if rot_clip:
        if fixed_shape:
            yprod = x[..., 1].detach()
        else:
            yprod = (x @ x.new_tensor([0, 1, 0])).detach()
        mask = (yprod > -YTOL) & (yprod < YTOL)
        if fixed_shape:
            # Preserve the original values and gradient clipping without
            # boolean-indexing edges into data-dependent output shapes.
            beta_detach = beta.detach()
            beta_detach = torch.where(
                yprod > YTOL, torch.zeros_like(beta_detach), beta_detach
            )
            beta_detach = torch.where(
                yprod < -YTOL,
                torch.full_like(beta_detach, math.pi),
                beta_detach,
            )
            alpha = torch.where(mask, alpha, alpha.detach())
            beta = torch.where(mask, beta, beta_detach)
            gamma = torch.where(mask, gamma, gamma.detach())
        else:
            alpha_detach = alpha[~mask].clone().detach()
            gamma_detach = gamma[~mask].clone().detach()
            beta_detach = beta.clone().detach()
            beta_detach[yprod > YTOL] = 0.0
            beta_detach[yprod < -YTOL] = math.pi
            beta_detach = beta_detach[~mask]

    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    wigner = torch.zeros(
        len(alpha), size, size, device=edge_rot_mat.device, dtype=edge_rot_mat.dtype
    )
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        if rot_clip and not fixed_shape:
            block = wigner_D(lmax, alpha[mask], beta[mask], gamma[mask], Jd).to(
                wigner.dtype
            )
            block_detach = wigner_D(
                lmax, alpha_detach, beta_detach, gamma_detach, Jd
            ).to(wigner.dtype)
            end = start + block.size()[1]
            wigner[mask, start:end, start:end] = block
            wigner[~mask, start:end, start:end] = block_detach
            start = end
        else:
            block = wigner_D(lmax, alpha, beta, gamma, Jd)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

    if rot_clip:
        return wigner
    else:
        return wigner.detach()
