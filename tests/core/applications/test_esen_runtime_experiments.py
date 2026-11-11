from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.applications.esen_matbench import MatbenchNHCIntegrator
from fairchem.core.applications.esen_nhc_fusion import (
    FusedMatbenchNHCIntegrator,
    triton,
)
from fairchem.core.applications.esen_whole_step_cuda_graph import (
    TransactionalWholeStepCUDAGraphController,
)
from fairchem.core.models.esen.nn.radial import PolynomialEnvelope
from fairchem.core.models.esen.shared_geometry import make_shared_geometry

CUDA = torch.cuda.is_available() and triton is not None


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("misses", [0, 1, 7])
def test_rob_compact_reads_demand_only_on_failure(compact, misses):
    reads = []
    full = {
        "fixed_builder_window_capacity_misses": misses,
        "fixed_builder_window_maximum_included_neighbors_by_atom": [80, 84],
        "fixed_builder_window_overflow_dummy_only_replays": misses,
    }

    def window_stats():
        reads.append(1)
        return full

    controller = object.__new__(TransactionalWholeStepCUDAGraphController)
    controller.compact_window_status = compact
    builder = SimpleNamespace(
        window_capacity_misses=torch.tensor(misses), window_stats=window_stats
    )
    result = controller._read_transaction_status(builder)
    if compact and misses == 0:
        assert result == {"fixed_builder_window_capacity_misses": 0}
        assert not reads
    else:
        assert result is full
        assert len(reads) == 1


@pytest.mark.parametrize(
    "n,tchain,tloop", [(0, 3, 1), (1025, 3, 1), (48, 2, 1), (48, 3, 2)]
)
def test_fused_nhc_rejects_unsupported_shape(n, tchain, tloop):
    with pytest.raises(ValueError):
        FusedMatbenchNHCIntegrator(
            torch.ones(n), temperature_K=300, tchain=tchain, tloop=tloop
        )


@pytest.mark.skipif(not CUDA, reason="requires CUDA Triton FP64")
@pytest.mark.parametrize("n", [1, 48, 500, 1024])
@pytest.mark.parametrize("delta_factor", [0.0, 0.5, -0.5])
def test_fused_nhc_matches_reference_multiple_steps(n, delta_factor):
    torch.manual_seed(7)
    masses = torch.linspace(1.0, 200.0, n, dtype=torch.float64, device="cuda")
    ref = MatbenchNHCIntegrator(masses, temperature_K=900)
    fused = FusedMatbenchNHCIntegrator(masses, temperature_K=900)
    # Deliberately non-contiguous inputs, with a non-zero thermostat state.
    p = torch.randn(3, n, device="cuda", dtype=torch.float64).T
    e = torch.randn(6, device="cuda", dtype=torch.float64)[::2] * 0.001
    pe = torch.randn(6, device="cuda", dtype=torch.float64)[::2] * 0.001
    expected, actual = (p, e, pe), (p, e, pe)
    original = tuple(t.clone() for t in (p, e, pe))
    for _ in range(20):
        expected = ref.integrate_nhc(*expected, ref.dt * delta_factor)
        actual = fused.integrate_nhc(*actual, ref.dt * delta_factor)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    for a, b in zip((p, e, pe), original):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not CUDA, reason="requires CUDA Triton FP64")
def test_fused_nhc_graph_replay_changes_inputs_and_keeps_addresses():
    masses = torch.full((48,), 12.0, device="cuda", dtype=torch.float64)
    fused = FusedMatbenchNHCIntegrator(masses, temperature_K=300)
    ref = MatbenchNHCIntegrator(masses, temperature_K=300)
    p = torch.ones(48, 3, device="cuda", dtype=torch.float64)
    e, pe = fused.clone_thermostat_state()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused.integrate_nhc(p, e, pe, fused.dt / 2.0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fused.integrate_nhc(p, e, pe, fused.dt / 2.0)
    addresses = tuple(t.data_ptr() for t in out)
    for factor in (0.3, 0.7, 1.1):
        p.fill_(factor)
        e.fill_(factor / 100)
        pe.fill_(factor / 100)
        graph.replay()
        torch.cuda.synchronize()
        expected = ref.integrate_nhc(p, e, pe, ref.dt / 2.0)
        for a, b in zip(out, expected):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
        assert tuple(t.data_ptr() for t in out) == addresses


@pytest.mark.skipif(not CUDA, reason="requires CUDA Triton FP64")
def test_fused_nhc_full_harmonic_steps_and_thermostat_restore():
    from fairchem.core.applications.esen_gpu_md import GPUMDState

    torch.manual_seed(3)
    masses = torch.full((48,), 12.0, device="cuda", dtype=torch.float64)
    ref = MatbenchNHCIntegrator(masses, temperature_K=293)
    fused = FusedMatbenchNHCIntegrator(masses, temperature_K=293)
    state = GPUMDState(
        torch.randn(48, 3, device="cuda", dtype=torch.float64) * 0.1,
        torch.randn(48, 3, device="cuda", dtype=torch.float64) * 0.1,
    )
    reference = state.clone()

    def force_fn(x):
        return -0.01 * x, 0.005 * x.square().sum()

    saved_state = state.clone()
    saved_thermostat = fused.clone_thermostat_state()
    for _ in range(3):
        fused.step(state, force_fn)
    # The production ROB snapshot copies these same eta/p_eta tensors.
    state = saved_state.clone()
    fused.restore_thermostat_state_(*saved_thermostat)
    for _ in range(100):
        ref.step(reference, force_fn)
        fused.step(state, force_fn)
    for a, b in (
        (state.positions, reference.positions),
        (state.momenta, reference.momenta),
        (state.forces, reference.forces),
        (fused.eta, ref.eta),
        (fused.p_eta, ref.p_eta),
    ):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


@pytest.mark.skipif(not CUDA, reason="requires CUDA full eSEN imports")
def test_shared_geometry_fused_edgewise_plumbing_and_capture():
    from fairchem.core.applications.esen_opt4_model_fusion import FusedEdgewise

    class First(torch.nn.Module):
        def prepare_and_linear(self, x, edge):
            return x[:, :, :2], x[:, :, 2:], x[:, :, :2] * 0.1

    class Bridge(torch.nn.Module):
        def forward(self, a, b, c):
            return torch.tanh(a + b + c), b, c

    class Second(torch.nn.Module):
        def linear_from_prepared(self, a, b, c):
            return (a,)

        def epilogue(self, a):
            return a

    edge = object.__new__(FusedEdgewise)
    torch.nn.Module.__init__(edge)
    edge.out_mask = torch.arange(14, device="cuda")
    edge.fuse_so2_gate_bridge = True
    for flag in (
        "fuse_wigner_so2_bridge",
        "fuse_wigner_so2_hybrid",
        "fuse_wigner_so2_tiled_backward",
        "fuse_gather",
        "fuse_reverse",
    ):
        setattr(edge, flag, False)
    edge.so2_conv_1, edge.so2_conv_2, edge.so2_gate_bridge = First(), Second(), Bridge()
    edge.use_envelope, edge.cutoff, edge.envelope = True, 5.0, PolynomialEnvelope()
    torch.manual_seed(4)
    x = torch.randn(7, 16, 2, device="cuda", requires_grad=True)
    w = (torch.randn(9, 16, 16, device="cuda") * 0.1).requires_grad_()
    d = torch.linspace(0.1, 6.0, 9, device="cuda", requires_grad=True)
    indices = torch.randint(0, 7, (2, 9), device="cuda")

    def forward(shared):
        inv = w.transpose(1, 2).contiguous()
        common = make_shared_geometry(w, inv, d, edge) if shared else None
        out = x
        for _ in range(3):
            out = edge(out, None, d, indices, w, inv, shared_geometry=common)
        grads = torch.autograd.grad(out.sum(), (x, w, d))
        return (out, *grads)

    reference, actual = forward(False), forward(True)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward(True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = forward(True)
    addresses = tuple(t.data_ptr() for t in actual)
    for _ in range(2):
        with torch.no_grad():
            w.mul_(0.95)
            d.add_(0.01)
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(actual, forward(False)):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
        assert addresses == tuple(t.data_ptr() for t in actual)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="CUDA"))],
)
@pytest.mark.parametrize("edges", [0, 9])
def test_shared_geometry_forward_and_all_input_gradients(device, edges):
    torch.manual_seed(9)
    edge = SimpleNamespace(
        out_mask=torch.arange(14), envelope=PolynomialEnvelope(), cutoff=5.0
    )
    w = torch.randn(
        edges, 16, 16, dtype=torch.float64, device=device, requires_grad=True
    )
    d = torch.linspace(
        0.1, 6.0, edges, dtype=torch.float64, device=device, requires_grad=True
    )
    messages = [
        torch.randn(
            edges, 16, 3, dtype=torch.float64, device=device, requires_grad=True
        )
        for _ in range(10)
    ]
    weights = [
        torch.randn(edges, 16, 3, dtype=torch.float64, device=device) for _ in range(10)
    ]

    def calculate(shared):
        inverse = w.transpose(1, 2).contiguous()
        common = make_shared_geometry(w, inverse, d, edge) if shared else None
        outputs = []
        for x in messages:
            geom = (
                common
                if common is not None
                else make_shared_geometry(w, inverse, d, edge)
            )
            y = torch.bmm(geom.wigner, x)
            y = torch.tanh(y) * geom.envelope
            outputs.append(torch.bmm(geom.wigner_inv, y))
        grads = torch.autograd.grad(
            sum((o * v).sum() for o, v in zip(outputs, weights)), (w, d, *messages)
        )
        return outputs, grads

    expected, actual = calculate(False), calculate(True)
    for group_a, group_b in zip(actual, expected):
        for a, b in zip(group_a, group_b):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    # Fresh invocation must not keep geometry or autograd tensors on the module.
    before = make_shared_geometry(w, w.transpose(1, 2), d, edge)
    after = make_shared_geometry(w * 2, w.transpose(1, 2) * 2, d + 0.01, edge)
    torch.testing.assert_close(after.wigner, before.wigner * 2)
    if edges:
        assert before.wigner.data_ptr() != after.wigner.data_ptr()
        assert torch.equal(
            before.envelope[d >= 5], torch.zeros_like(before.envelope[d >= 5])
        )
