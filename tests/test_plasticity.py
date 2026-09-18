from __future__ import annotations

import numpy as np
import pytest

from flylatro.fly.mushroom_body.plasticity import (
    PlasticityConfig,
    ThreeFactorPlasticity,
)
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


def rule(
    *,
    learning_rate: float = 0.1,
    eligibility_decay: float = 0.5,
    min_efficacy: float = 0.5,
    max_efficacy: float = 1.5,
) -> ThreeFactorPlasticity:
    topology = PlasticEdgeTopology.synthetic(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
        np.asarray([4.0, 8.0], dtype=np.float32),
    )
    state = PlasticEdgeState.initialize(2)
    return ThreeFactorPlasticity(
        topology,
        state,
        PlasticityConfig(
            learning_rate=learning_rate,
            eligibility_decay=eligibility_decay,
            dopamine_decay=0.0,
            min_efficacy=min_efficacy,
            max_efficacy=max_efficacy,
        ),
    )


def test_no_dopamine_or_zero_eligibility_means_no_update() -> None:
    plasticity = rule()
    initial = plasticity.state.efficacy.copy()
    plasticity.record_activity(
        np.asarray([[1.0, 0.5]], dtype=np.float32),
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    event = plasticity.apply_dopamine(0.0, 0.0)[0]
    np.testing.assert_array_equal(plasticity.state.efficacy, initial)
    assert event.changed_synapses == 0

    empty = rule()
    empty.apply_dopamine(1.0, 0.0)
    np.testing.assert_array_equal(empty.state.efficacy, initial)


def test_positive_and_negative_dopamine_change_eligible_edges() -> None:
    positive = rule()
    positive.record_activity(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    event = positive.apply_dopamine(0.5, 0.0)[0]
    assert positive.state.efficacy[0, 0] > 1.0
    assert positive.state.efficacy[0, 1] == 1.0
    assert event.changed_synapses == 1

    negative = rule()
    negative.record_activity(
        np.ones((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
    )
    negative.apply_dopamine(0.0, 0.5)
    assert np.all(negative.state.efficacy < 1.0)


def test_bounds_and_eligibility_decay_are_exact() -> None:
    plasticity = rule(learning_rate=2.0)
    ones = np.ones((1, 2), dtype=np.float32)
    plasticity.record_activity(ones, ones)
    np.testing.assert_allclose(plasticity.state.eligibility, 1.0)
    plasticity.record_activity(np.zeros_like(ones), ones)
    np.testing.assert_allclose(plasticity.state.eligibility, 0.5)
    plasticity.apply_dopamine(10.0, 0.0)
    np.testing.assert_allclose(plasticity.state.efficacy, 1.5)
    assert plasticity.state.upper_bound_hits[0] == 2

    plasticity.apply_dopamine(0.0, 10.0)
    np.testing.assert_allclose(plasticity.state.efficacy, 0.5)
    assert plasticity.state.lower_bound_hits[0] == 2


def test_plastic_weights_survive_fast_reset_and_evaluation_freezes() -> None:
    plasticity = rule()
    ones = np.ones((1, 2), dtype=np.float32)
    plasticity.record_activity(ones, ones)
    plasticity.apply_dopamine(1.0, 0.0)
    learned = plasticity.state.efficacy.copy()
    eligibility = plasticity.state.eligibility.copy()
    plasticity.state.reset_fast_traces()
    np.testing.assert_array_equal(plasticity.state.efficacy, learned)
    np.testing.assert_array_equal(plasticity.state.eligibility, eligibility)
    plasticity.apply_dopamine(0.0, 1.0, plasticity_enabled=False)
    np.testing.assert_array_equal(plasticity.state.efficacy, learned)


def test_state_round_trip_reproduces_future_plasticity() -> None:
    first = rule()
    pre = np.asarray([[0.2, 0.8]], dtype=np.float32)
    post = np.asarray([[0.5, 0.5]], dtype=np.float32)
    first.record_activity(pre, post)
    restored_state = PlasticEdgeState.from_state_dict(first.state.state_dict())
    second = ThreeFactorPlasticity(first.topology, restored_state, first.config)

    first.apply_dopamine(0.7, 0.0)
    second.apply_dopamine(0.7, 0.0)

    np.testing.assert_array_equal(first.state.efficacy, second.state.efficacy)
    assert first.state.weight_sha256 == second.state.weight_sha256
    assert len(first.config.sha256) == 64
    assert len(first.topology.sha256) == 64


def test_independent_learners_never_share_plastic_state() -> None:
    topology = rule().topology
    state = PlasticEdgeState.initialize(2, learners=2)
    plasticity = ThreeFactorPlasticity(topology, state, PlasticityConfig())
    plasticity.record_activity(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
    )
    plasticity.apply_dopamine(
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])
    )
    assert state.efficacy[0, 0] > 1.0
    assert state.efficacy[1, 1] < 1.0
    assert state.efficacy[0, 1] == pytest.approx(1.0)
    assert state.efficacy[1, 0] == pytest.approx(1.0)
