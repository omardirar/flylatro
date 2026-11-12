from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from test_plastic_backend import tiny_chain
from flylatro.fly.encoder import Stimulus
from flylatro.fly.execution import build_backend, check_int32_capacity, implementation_identity


def test_event_integer_bound_is_required_without_rounding():
    from flylatro.fly.execution import check_integral_event_weights
    post = np.asarray([0, 0, 1], dtype=np.int64)
    assert check_integral_event_weights(post, np.asarray([2., -3., 5.]), 2) == 5.
    with pytest.raises(ValueError, match='integral fixed'):
        check_integral_event_weights(post, np.asarray([2., -3.1, 5.]), 2)
    with pytest.raises(ValueError, match='accumulation bound'):
        check_integral_event_weights(post, np.asarray([2**24 - 1., 1., 0.]), 2)
from flylatro.learning.config import PlasticExperimentConfig, fly_dynamics_payload


def test_csr_signed_propagation_and_plastic_separation(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    artifact.signed_synapse_counts[1] = -3
    reference = build_backend('torch_reference', artifact, topology)
    csr = build_backend('torch_csr', artifact, topology)
    assert csr.weights.crow_indices().dtype == torch.int32
    assert csr.weights.col_indices().dtype == torch.int32
    assert csr.connectivity_hash == reference.connectivity_hash
    assert csr.dynamics_hash != reference.dynamics_hash
    efficacy = np.array([[0.5], [1.5]], dtype=np.float32)
    inputs = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    for backend in (reference, csr):
        backend.set_plastic_efficacy(efficacy)
        np.testing.assert_array_equal(backend.propagate_once(inputs), [[0, 1, -3], [0, 3, -6]])


def test_csr_neural_reset_and_independent_streams(tmp_path):
    artifact, topology = tiny_chain(tmp_path)
    outputs = []
    for execution in ('torch_reference', 'torch_csr'):
        backend = build_backend(execution, artifact, topology, readout_indices=[0, 1, 2])
        stimulus = Stimulus(np.full((2, 3), 300, dtype=np.float32), 'test', 'test')
        backend.reset(2, (17, 23))
        first = backend.simulate_tensor(stimulus, 10)
        backend.reset(2, (17, 23))
        again = backend.simulate_tensor(stimulus, 10)
        assert torch.equal(first.spike_counts, again.spike_counts)
        assert torch.equal(first.final_voltage, again.final_voltage)
        backend.reset(2, (23, 17))
        reordered = backend.simulate_tensor(stimulus, 10)
        assert torch.equal(first.spike_counts.flip(0), reordered.spike_counts)
        assert torch.equal(first.final_voltage.flip(0), reordered.final_voltage)
        outputs.append(first)
    assert outputs[0].spike_counts.sum() > 0
    assert torch.equal(outputs[0].spike_counts, outputs[1].spike_counts)
    assert torch.equal(outputs[0].final_voltage, outputs[1].final_voltage)


@pytest.mark.parametrize('neurons,entries', [(2**31, 1), (1, 2**31), (0, 0), (1, -1)])
def test_csr_overflow_rejected(neurons, entries):
    with pytest.raises(ValueError, match='int32'):
        check_int32_capacity(neurons, entries)


def test_explicit_execution_identity_and_config():
    c = PlasticExperimentConfig.load(Path('configs/plastic-v783-source-complete-calibration.toml'))
    csr = replace(c, fly=replace(c.fly, execution='torch_csr'))
    csr.validate()
    assert fly_dynamics_payload(c) != fly_dynamics_payload(csr)
    assert implementation_identity('torch_csr')['index_bits'] == 32
    with pytest.raises(ValueError, match='unknown fly execution'):
        replace(c, fly=replace(c.fly, execution='missing')).validate()


def test_implementation_hash_matches_evidence_without_compiling():
    from flylatro.analysis.provenance import EvidenceIdentity
    from flylatro.fly.execution import execution_identity_hash
    assert execution_identity_hash('torch_reference') is None
    identity = EvidenceIdentity.from_components(
        {'fly_implementation': implementation_identity('torch_csr')},
        report_kind='test', report_version='test', include_git=False)
    assert identity.fly_implementation_sha256 == execution_identity_hash('torch_csr')
