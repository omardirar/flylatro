from __future__ import annotations

import numpy as np
import pytest

from flylatro.fly.backend import TinyGraphFlyBackend
from flylatro.fly.encoder import Stimulus
from flylatro.fly.features import FeatureSpec, RateFeatureExtractor


def stimulus(batch: int, neurons: int) -> Stimulus:
    rates = np.zeros((batch, neurons), dtype=np.float64)
    rates[:, :4] = 100.0
    return Stimulus(rates, "test-v1", "abc")


def test_tiny_backend_is_batched_and_reset_deterministic() -> None:
    backend = TinyGraphFlyBackend(neuron_count=12, graph_seed=7)
    backend.reset(2)
    first = backend.simulate(stimulus(2, 12), duration_ms=8)
    backend.reset(2)
    second = backend.simulate(stimulus(2, 12), duration_ms=8)

    np.testing.assert_array_equal(first.spike_counts, second.spike_counts)
    np.testing.assert_array_equal(first.final_voltage, second.final_voltage)
    assert first.spike_counts.shape == (2, 12)
    assert first.spike_counts.sum() > 0
    assert backend.connectivity_hash == TinyGraphFlyBackend(
        neuron_count=12, graph_seed=7
    ).connectivity_hash
    assert backend.connectivity_hash != TinyGraphFlyBackend(
        neuron_count=12, graph_seed=8
    ).connectivity_hash
    assert len(backend.connectivity_hash) == 64


def test_backend_requires_explicit_per_decision_reset() -> None:
    backend = TinyGraphFlyBackend(neuron_count=12)
    with pytest.raises(RuntimeError, match="reset"):
        backend.simulate(stimulus(1, 12), duration_ms=2)


def test_feature_extractor_has_fixed_small_shape() -> None:
    backend = TinyGraphFlyBackend(neuron_count=12, graph_seed=7)
    backend.reset(2)
    activity = backend.simulate(stimulus(2, 12), duration_ms=8)
    spec = FeatureSpec("readout-v1", (8, 9, 10, 11))

    features = RateFeatureExtractor(spec).extract(activity)

    assert features.shape == (2, 8)
    assert np.isfinite(features).all()
    assert len(spec.sha256) == 64


def test_fixed_graph_transforms_distinct_stimuli_into_distinct_readouts() -> None:
    backend = TinyGraphFlyBackend(neuron_count=12, graph_seed=7)
    rates = np.zeros((2, 12), dtype=np.float64)
    rates[0, 0] = 100.0
    rates[1, 1:4] = 100.0
    input_stimulus = Stimulus(rates, "test-v1", "abc")
    backend.reset(2)
    activity = backend.simulate(input_stimulus, duration_ms=20)
    features = RateFeatureExtractor(FeatureSpec("readout-v1", (8, 9, 10, 11))).extract(
        activity
    )

    assert np.any(features)
    assert not np.array_equal(features[0], features[1])
