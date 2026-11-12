"""Opt-in native qualification; CUDA-unavailable hosts skip rather than fall back."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from test_plastic_backend import tiny_chain
from flylatro.fly.encoder import Stimulus
from flylatro.fly.execution import build_backend


def test_native_cpu_request_fails_explicitly(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    with pytest.raises(ValueError, match='requires a CUDA device'):
        build_backend('cuda_native', artifact, topology, device='cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA qualification requires GPU')
def test_fused_neuron_step_exact_state_and_delay_order(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    reference = build_backend('torch_csr', artifact, topology, device='cuda')
    native = build_backend('cuda_native', artifact, topology, device='cuda')
    for backend in (reference, native):
        backend.reset(2, (17, 29))
    rng = torch.Generator(device='cuda').manual_seed(472)
    # Exercise delayed excitation/inhibition, thresholds and repeated refractory transitions.
    slot = 0
    for _ in range(150):
        poisson = (torch.rand((2, 3), device='cuda', generator=rng) < .2).float() * 250
        recurrent = torch.randn((2, 3), device='cuda', generator=rng) * 30
        states = []
        for backend in (reference, native):
            g, delay, spikes, v, refractory, counts = backend._state
            g, spikes, v, refractory = backend._lif_step(g, delay, spikes, v, refractory,
                counts, poisson, recurrent, slot, 22)
            backend._state = g, delay, spikes, v, refractory, counts
            states.append(backend._state)
        for expected, actual in zip(*states):
            assert torch.equal(expected, actual)
        slot = (slot + 1) % reference._state[1].shape[0]
    assert reference._state[-1].sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA qualification requires GPU')
def test_native_full_decision_reset_rng_and_activity(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    results = []
    for execution in ('torch_csr', 'cuda_native'):
        backend = build_backend(execution, artifact, topology, device='cuda', readout_indices=[0, 1, 2])
        rates = Stimulus(np.full((2, 3), 500, dtype=np.float32), 'test', 'test')
        backend.reset(2, (17, 29))
        first = backend.simulate_tensor(rates, 20)
        backend.reset(2, (29, 17))
        reordered = backend.simulate_tensor(rates, 20)
        assert torch.equal(first.spike_counts.flip(0), reordered.spike_counts)
        assert torch.equal(first.final_voltage.flip(0), reordered.final_voltage)
        results.append(first)
    assert torch.equal(results[0].spike_counts, results[1].spike_counts)
    assert torch.equal(results[0].final_voltage, results[1].final_voltage)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA qualification requires GPU')
def test_native_buffers_and_generators_reused_across_resets(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    backend = build_backend('cuda_native', artifact, topology, device='cuda')
    backend.reset(2, (1, 2))
    addresses = [t.data_ptr() for t in backend._state]
    generators = [id(g) for g in backend._generators]
    efficacy = torch.ones((2, 1), device='cuda')
    backend.set_plastic_efficacy(efficacy)
    efficacy_address = backend._efficacy.data_ptr()
    backend.simulate_tensor(Stimulus(np.full((2, 3), 500, dtype=np.float32), 'test', 'test'), 10)
    backend.reset(2, (2, 1))
    assert addresses == [t.data_ptr() for t in backend._state]
    assert generators == [id(g) for g in backend._generators]
    for index in (0, 1, 2, 5):
        assert backend._state[index].count_nonzero() == 0
    efficacy.fill_(.5)
    backend.set_plastic_efficacy(efficacy)
    assert backend._efficacy.data_ptr() == efficacy_address
    efficacy.fill_(1.5)
    assert torch.all(backend._efficacy == .5)
    backend.reset(1, (3,))
    assert backend._state[0].shape == (1, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA qualification requires GPU')
@pytest.mark.parametrize('execution', ['cuda_native_graph', 'cuda_native_graph_blocks',
                                      'cuda_native_cusparse_graph_blocks', 'cuda_native_event_graph_blocks'])
def test_graph_replay_fresh_rng_efficacy_and_batch_changes(tmp_path, execution):
    artifact, topology = tiny_chain(tmp_path)
    eager = build_backend('cuda_native', artifact, topology, device='cuda', readout_indices=[0, 1, 2])
    graph = build_backend(execution, artifact, topology, device='cuda', readout_indices=[0, 1, 2])
    history = []
    for seeds, rate, efficacy, duration in [((17, 29), 500, .5, 20), ((71, 97), 700, 1.5, 20),
            ((29, 17), 500, .5, 20), ((13,), 600, .8, 10), ((17, 29), 500, .5, 20)]:
        stimulus = Stimulus(np.full((len(seeds), 3), rate, dtype=np.float32), 'test', 'test')
        results = []
        for backend in (eager, graph):
            backend.set_plastic_efficacy(np.full((len(seeds), 1), efficacy, dtype=np.float32))
            backend.reset(len(seeds), seeds)
            results.append(backend.simulate_tensor(stimulus, duration))
        assert torch.equal(results[0].spike_counts, results[1].spike_counts)
        assert torch.equal(results[0].final_voltage, results[1].final_voltage)
        for expected, actual in zip(eager._state, graph._state, strict=True):
            assert torch.equal(expected, actual)
        for expected, actual in zip(eager._generators, graph._generators, strict=True):
            assert torch.equal(expected.get_state(), actual.get_state())
        history.append(results[1].final_voltage)
    assert not torch.equal(history[0], history[1])
    assert torch.equal(history[0].flip(0), history[2])
    assert torch.equal(history[0], history[4])
    assert graph.capture_count == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA qualification requires GPU')
def test_cusparse_fixed_signs_plastic_separation_and_persistent_plan(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    expected = build_backend('cuda_native', artifact, topology, device='cuda')
    actual = build_backend('cuda_native_cusparse', artifact, topology, device='cuda')
    for batch in (1, 2, 4, 8, 16):
        for backend in (expected, actual):
            backend.set_plastic_efficacy(np.full((batch, 1), .75, dtype=np.float32))
            backend.reset(batch, tuple(range(batch)))
        address = actual._fixed_input.data_ptr()
        plan = actual._sparse_plan
        spikes = torch.arange(batch * 3, device='cuda', dtype=torch.float32).reshape(batch, 3)
        assert torch.equal(expected._recurrent(spikes), actual._recurrent(spikes))
        actual.reset(batch, tuple(reversed(range(batch))))
        assert actual._fixed_input.data_ptr() == address
        assert actual._sparse_plan is plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA qualification requires GPU')
def test_event_propagation_repeatability_and_independent_binary_rows(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    expected = build_backend('cuda_native', artifact, topology, device='cuda')
    actual = build_backend('cuda_native_event', artifact, topology, device='cuda')
    for batch in (1, 2, 4, 8, 16):
        for backend in (expected, actual):
            backend.set_plastic_efficacy(np.full((batch, 1), .75, dtype=np.float32))
            backend.reset(batch, tuple(range(batch)))
        address = actual._event_output.data_ptr()
        spikes = (torch.arange(batch * 3, device='cuda').reshape(batch, 3) % 2).float()
        reference = expected._recurrent(spikes)
        for _ in range(20):
            assert torch.equal(reference, actual._recurrent(spikes))
        actual.reset(batch, tuple(reversed(range(batch))))
        assert actual._event_output.data_ptr() == address
    with pytest.raises(ValueError, match='binary'):
        actual.propagate_once(np.asarray([.5, 0., 1.], dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA qualification requires GPU')
def test_native_plastic_reduction_repeatable_many_to_one(tmp_path):
    from flylatro.fly.flywire_artifact import FlyWireArtifact
    from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
    rng = np.random.default_rng(1842)
    count = 2048
    pre = np.arange(count, dtype=np.int64)
    post = np.full(count, count, dtype=np.int64)
    weights = rng.integers(1, 20, count).astype(np.float32)
    artifact = FlyWireArtifact(path=tmp_path / 'dense-post.npz', manifest={},
        root_ids=np.arange(count + 1, dtype=np.int64), pre_indices=pre, post_indices=post,
        signed_synapse_counts=weights, sensory_indices=pre[:1], descending_indices=post[:1],
        coordinates_nm=np.zeros((count + 1, 3), dtype=np.float32))
    topology = PlasticEdgeTopology.synthetic(pre, post, weights)
    backend = build_backend('cuda_native', artifact, topology, device='cuda')
    efficacy = rng.uniform(.2, 2, (2, count)).astype(np.float32)
    backend.set_plastic_efficacy(efficacy)
    spikes = torch.ones((2, count + 1), device='cuda')
    first = backend._recurrent(spikes)
    for _ in range(20):
        assert torch.equal(first, backend._recurrent(spikes))
    oracle = (efficacy.astype(np.float64) * weights).sum(axis=1)
    np.testing.assert_allclose(first[:, -1].cpu().numpy(), oracle, rtol=1e-6)
    assert first[:, :-1].count_nonzero() == 0
