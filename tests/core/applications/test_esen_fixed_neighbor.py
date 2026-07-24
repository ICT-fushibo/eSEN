from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.applications.esen_fixed_neighbor import (
    FixedShapePBCNeighborBuilder,
    maximum_neighbors_in_graph,
    neighbor_capacity_from_probe,
)
from fairchem.core.common.utils import radius_graph_pbc


def _official_graph(positions, cell, pbc, cutoff=1.25, max_neighbors=300):
    data = SimpleNamespace(
        pos=positions,
        natoms=torch.tensor([positions.shape[0]], device=positions.device),
        cell=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )
    edge_index, cell_offsets, _ = radius_graph_pbc(
        data,
        cutoff,
        max_neighbors,
        enforce_max_neighbors_strictly=False,
    )
    return edge_index, cell_offsets


def _active_builder_edges(builder):
    active = builder.edge_index[0] < builder.num_atoms
    return builder.edge_index[:, active], builder.cell_offsets[active]


def test_neighbor_capacity_from_probe_adds_margin_and_rounds():
    assert neighbor_capacity_from_probe(300, margin=0.10, slot_step=8) == 336
    assert neighbor_capacity_from_probe(8, margin=0.0, slot_step=8) == 16
    with pytest.raises(ValueError):
        neighbor_capacity_from_probe(0)


@pytest.mark.parametrize(
    "cell",
    [
        torch.diag(torch.tensor([3.0, 3.5, 4.0])),
        torch.tensor([[3.0, 0.0, 0.0], [0.4, 3.2, 0.0], [0.2, 0.3, 3.4]]),
    ],
)
def test_fixed_builder_matches_official_edge_order(cell):
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.0, 0.2, 0.3], [2.2, 2.4, 2.7]]
    )
    pbc = torch.tensor([True, True, True])
    official_edges, official_offsets = _official_graph(
        positions, cell, pbc
    )
    maximum = max(1, maximum_neighbors_in_graph(official_edges, 3))
    builder = FixedShapePBCNeighborBuilder(
        num_atoms=3,
        cell=cell,
        pbc=pbc,
        cutoff=1.25,
        neighbors_per_atom=maximum + 1,
        dummy_atoms=2,
    )
    builder.build(positions)
    fixed_edges, fixed_offsets = _active_builder_edges(builder)

    torch.testing.assert_close(fixed_edges, official_edges)
    torch.testing.assert_close(fixed_offsets, official_offsets)
    padding = builder.edge_index[0] >= 3
    assert bool((builder.edge_index[1, padding] >= 3).all())
    torch.testing.assert_close(
        builder.edge_index[0, padding], builder.edge_index[1, padding]
    )


def test_fixed_builder_non_strict_degeneracy_and_overflow():
    # Atom zero has four equally distant neighbors.  A threshold of two must
    # retain all four under the official non-strict degeneracy rule.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    builder = FixedShapePBCNeighborBuilder(
        num_atoms=5,
        cell=torch.diag(torch.tensor([20.0, 20.0, 20.0])),
        pbc=torch.tensor([False, False, False]),
        cutoff=1.1,
        neighbors_per_atom=3,
        dummy_atoms=2,
        max_neighbors=2,
    )
    builder.build(positions, step=torch.tensor(7))
    stats = builder.stats()

    assert stats["fixed_builder_max_included_neighbors"] == 4
    assert stats["fixed_builder_capacity_misses"] == 1
    assert stats["fixed_builder_first_overflow_step"] == 7
    assert builder.edge_index.shape == (2, 15)
    assert builder.cell_offsets.shape == (15, 3)


def test_fixed_builder_padding_never_touches_real_atoms():
    builder = FixedShapePBCNeighborBuilder(
        num_atoms=2,
        cell=torch.diag(torch.tensor([10.0, 10.0, 10.0])),
        pbc=torch.tensor([True, True, True]),
        cutoff=1.0,
        neighbors_per_atom=4,
        dummy_atoms=3,
    )
    edge_address = builder.edge_index.data_ptr()
    offset_address = builder.cell_offsets.data_ptr()
    builder.build(torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]))
    assert bool((builder.edge_index >= 2).all())
    torch.testing.assert_close(builder.edge_index[0], builder.edge_index[1])
    builder.build(torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]))
    assert builder.edge_index.data_ptr() == edge_address
    assert builder.cell_offsets.data_ptr() == offset_address
