"""Per-force-call geometry sharing; deliberately not a persistent cache."""
from __future__ import annotations

from typing import NamedTuple

import torch


class SharedEdgeGeometry(NamedTuple):
    wigner: torch.Tensor
    wigner_inv: torch.Tensor
    envelope: torch.Tensor


def make_shared_geometry(wigner, wigner_inv, edge_distance, edgewise):
    mask = edgewise.out_mask.to(device=wigner.device)
    return SharedEdgeGeometry(
        wigner[:, mask, :],
        wigner_inv[:, :, mask],
        edgewise.envelope(edge_distance / edgewise.cutoff).view(-1, 1, 1),
    )


def configure_shared_geometry(model):
    """Validate homogeneous frozen v5 blocks before enabling the experiment."""
    from fairchem.core.applications.esen_opt4_model_fusion import FusedEdgewise

    backbone = model.backbone
    blocks = list(backbone.blocks)
    if not blocks or backbone.activation_checkpointing:
        raise ValueError(
            "Shared geometry requires blocks without activation checkpointing"
        )
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Shared geometry experiment requires frozen parameters")
    first = blocks[0].edge_wise
    for block in blocks:
        edge = block.edge_wise
        if not isinstance(edge, FusedEdgewise) or not edge.use_envelope:
            raise ValueError("Shared geometry requires fused Edgewise with envelope")
        if (
            edge.fuse_gather
            or edge.fuse_reverse
            or edge.fuse_wigner_so2_bridge
            or edge.fuse_wigner_so2_hybrid
            or edge.fuse_wigner_so2_tiled_backward
        ):
            raise ValueError(
                "Shared geometry cannot combine with Wigner producer/reverse fusions"
            )
        if (
            edge.cutoff != first.cutoff
            or type(edge.envelope) is not type(first.envelope)
            or any(
                getattr(edge.envelope, k) != getattr(first.envelope, k)
                for k in ("p", "a", "b", "c")
            )
            or not torch.equal(edge.out_mask.cpu(), first.out_mask.cpu())
        ):
            raise ValueError("Shared geometry requires identical masks and envelopes")
    backbone._opt4_shared_geometry = True
    return len(blocks)
