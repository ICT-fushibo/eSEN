from __future__ import annotations

import pytest
import torch
from torch import nn

from fairchem.core.applications.esen_cuda_graph import (
    CUDAGraphCapacityError,
    edge_capacity_from_probe,
    prepare_cuda_graph_index_tensors_,
    staticize_neighbor_graph_,
)
from fairchem.core.models.esen.esen import MLP_EFS_Head
from fairchem.core.models.esen.common.rotation import init_edge_rot_mat


def test_edge_capacity_from_probe_adds_margin_and_rounds_up():
    assert edge_capacity_from_probe(1000, margin=0.10, edge_step=256) == 1280
    assert edge_capacity_from_probe(256, margin=0.0, edge_step=256) == 512


def test_staticize_neighbor_graph_preserves_real_edges_and_uses_dummy_sinks():
    real_edge_index = torch.tensor([[0, 1], [1, 0]])
    real_cell_offsets = torch.tensor([[0, 0, 0], [1, 0, 0]])
    static_edge_index = torch.empty(2, 6, dtype=torch.long)
    static_cell_offsets = torch.empty(6, 3, dtype=torch.long)
    dummy_sinks = torch.tensor([3, 4, 3, 4, 3, 4])
    padding_offsets = torch.tensor([[8, 0, 0]]).repeat(6, 1)

    num_edges = staticize_neighbor_graph_(
        static_edge_index,
        static_cell_offsets,
        real_edge_index,
        real_cell_offsets,
        n_real=3,
        dummy_sink_template=dummy_sinks,
        padding_offset_template=padding_offsets,
    )

    assert num_edges == 2
    torch.testing.assert_close(static_edge_index[:, :2], real_edge_index)
    torch.testing.assert_close(static_cell_offsets[:2], real_cell_offsets)
    assert bool((static_edge_index[:, 2:] >= 3).all())
    torch.testing.assert_close(static_edge_index[0, 2:], static_edge_index[1, 2:])
    torch.testing.assert_close(static_cell_offsets[2:], padding_offsets[:4])


def test_staticize_neighbor_graph_rejects_capacity_overflow():
    with pytest.raises(CUDAGraphCapacityError, match="required=3, capacity=2"):
        staticize_neighbor_graph_(
            torch.empty(2, 2, dtype=torch.long),
            torch.empty(2, 3),
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            torch.zeros(3, 3),
            n_real=3,
            dummy_sink_template=torch.tensor([3, 4]),
            padding_offset_template=torch.zeros(2, 3),
        )


def test_prepare_cuda_graph_index_tensors_finds_plain_out_masks():
    model = nn.Sequential(nn.Identity())
    model[0].out_mask = torch.tensor([0, 2, 4], dtype=torch.long)
    assert dict(model[0].named_buffers()) == {}

    prepared = prepare_cuda_graph_index_tensors_(model, torch.device("cpu"))

    assert prepared == 1
    assert model[0].out_mask.device.type == "cpu"
    torch.testing.assert_close(model[0].out_mask, torch.tensor([0, 2, 4]))


def test_esen_energy_head_excludes_cuda_graph_dummy_atoms():
    head = MLP_EFS_Head.__new__(MLP_EFS_Head)
    nn.Module.__init__(head)
    head.energy_block = nn.Identity()
    head.regress_stress = False
    head.regress_forces = False
    data = {
        "natoms": torch.tensor([3]),
        "batch": torch.tensor([0, 0, 0]),
        "pos": torch.zeros(3, 3),
        "n_real": 2,
    }
    embedding = {"node_embedding": torch.tensor([[[1.0]], [[2.0]], [[99.0]]])}

    output = head(data, embedding)

    torch.testing.assert_close(output["energy"], torch.tensor([3.0]))


def test_fixed_rotation_reference_is_finite_and_does_not_consume_rng():
    edge_vectors = torch.randn(128, 3)
    fixed_reference = edge_vectors.new_tensor([0.37, -0.61, 0.71]).expand_as(
        edge_vectors
    )
    rng_state = torch.get_rng_state().clone()

    rotation = init_edge_rot_mat(
        edge_vectors,
        rot_clip=True,
        fixed_reference_vec=fixed_reference,
    )

    assert torch.equal(torch.get_rng_state(), rng_state)
    assert bool(torch.isfinite(rotation).all())
    identity = rotation @ rotation.transpose(1, 2)
    torch.testing.assert_close(
        identity,
        torch.eye(3).expand_as(identity),
        atol=2e-6,
        rtol=2e-6,
    )
