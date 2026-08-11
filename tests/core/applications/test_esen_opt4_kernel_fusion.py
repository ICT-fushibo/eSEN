from __future__ import annotations

import pytest
import torch

from fairchem.core.applications.esen_fixed_neighbor import (
    FixedShapePBCNeighborBuilder,
)
from fairchem.core.applications.esen_opt4_kernel_fusion import (
    TritonDistanceFixedShapePBCNeighborBuilder,
    triton_neighbor_fusion_available,
)


def test_opt4_module_imports_without_requiring_triton_execution():
    assert isinstance(triton_neighbor_fusion_available(), bool)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_neighbor_fusion_available(),
    reason="Opt4 KF1 correctness test requires CUDA and Triton",
)
@pytest.mark.parametrize(
    "cell",
    [
        torch.diag(torch.tensor([3.0, 3.5, 4.0])),
        torch.tensor(
            [[3.0, 0.0, 0.0], [0.4, 3.2, 0.0], [0.2, 0.3, 3.4]]
        ),
    ],
)
def test_triton_distance_builder_matches_kf0_and_is_capture_safe(cell):
    device = torch.device("cuda")
    cell = cell.to(device=device, dtype=torch.float32)
    pbc = torch.tensor([True, True, True], device=device)
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.0, 0.2, 0.3], [2.2, 2.4, 2.7]],
        device=device,
        dtype=torch.float32,
    )
    kwargs = {
        "num_atoms": 3,
        "cell": cell,
        "pbc": pbc,
        "cutoff": 1.25,
        "neighbors_per_atom": 8,
        "dummy_atoms": 2,
    }
    kf0 = FixedShapePBCNeighborBuilder(**kwargs)
    kf1 = TritonDistanceFixedShapePBCNeighborBuilder(**kwargs)
    kf0.build(positions)
    kf1.build(positions)
    torch.cuda.synchronize()

    torch.testing.assert_close(kf1.edge_index, kf0.edge_index, rtol=0, atol=0)
    torch.testing.assert_close(kf1.cell_offsets, kf0.cell_offsets, rtol=0, atol=0)
    assert kf1.stats()["kernel_fusion_stage"] == "KF1"

    edge_address = kf1.edge_index.data_ptr()
    offset_address = kf1.cell_offsets.data_ptr()
    distance_address = kf1._distance_sqr.data_ptr()
    valid_address = kf1._valid_candidates.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kf1.build(positions)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        kf1.build(positions)
    graph.replay()
    torch.cuda.synchronize()

    assert kf1.edge_index.data_ptr() == edge_address
    assert kf1.cell_offsets.data_ptr() == offset_address
    assert kf1._distance_sqr.data_ptr() == distance_address
    assert kf1._valid_candidates.data_ptr() == valid_address
    torch.testing.assert_close(kf1.edge_index, kf0.edge_index, rtol=0, atol=0)
    torch.testing.assert_close(kf1.cell_offsets, kf0.cell_offsets, rtol=0, atol=0)
    padding = kf1.edge_index[0] >= 3
    assert bool((kf1.edge_index[1, padding] >= 3).all())
