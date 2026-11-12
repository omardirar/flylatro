"""Post-hoc candidates must not silently change the scientific protocol."""

from dataclasses import replace

import numpy as np
import pytest

from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.learning.reinforcement import (
    ReinforcementConfig,
    ReinforcementMapper,
    exploratory_condition,
    sensitivity_condition,
)


def test_exploratory_identity_preserves_original_protocol_and_rejects_retuning():
    original = ReinforcementConfig()
    assert original.sha256 == "1f94e00bec69c10212e2002234d3f21e4ab6f1c03830899b3d823f4eaae8270e"
    candidate = exploratory_condition("exploratory-weak-failure-v1")
    assert candidate.sha256 != original.sha256
    with pytest.raises(ValueError, match="fixes"):
        replace(candidate, run_failure_pulse=0.1)
    with pytest.raises(ValueError, match="fixes"):
        replace(candidate, blind_clear_pulse=0.8)
    with pytest.raises(ValueError, match="unknown predeclared"):
        sensitivity_condition(candidate.condition)


def test_candidate_retains_failure_and_progress_on_the_same_terminal_action():
    components = {"blind_progress": 0.2}
    info = {"episode": {"won": False, "ante": 1}}
    original = ReinforcementMapper().map(components, info)
    candidate = ReinforcementMapper(exploratory_condition("exploratory-weak-failure-v1")).map(components, info)
    assert candidate.events == original.events == ("progress", "run_failure")
    assert candidate.appetitive == original.appetitive == pytest.approx(0.01)
    assert original.aversive == 0.6
    assert candidate.aversive == 0.01


def test_candidate_preserves_positive_milestone_pulses_and_clipping():
    components = {"blind_progress": 1, "blind_clear": 1, "ante_clear": 1, "win": 1}
    info = {"won": True, "ante": 8}
    original = ReinforcementMapper().map(components, info)
    candidate = ReinforcementMapper(exploratory_condition("exploratory-weak-failure-v1")).map(components, info)
    assert candidate == original
    assert candidate.appetitive == 2.0
    assert candidate.aversive == 0.0


def test_weaker_failure_reduces_depression_without_touching_ineligible_edges():
    topology = PlasticEdgeTopology.synthetic(
        np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4., 8.], dtype=np.float32)
    )
    rules = []
    for reinforcement in [ReinforcementConfig(), exploratory_condition("exploratory-weak-failure-v1")]:
        rule = ThreeFactorPlasticity(topology, PlasticEdgeState.initialize(2), PlasticityConfig(learning_rate=0.002))
        rule.record_activity(np.asarray([[1., 0.]], dtype=np.float32), np.ones((1, 2), dtype=np.float32))
        pulse = ReinforcementMapper(reinforcement).map({}, {"run_failed": True})
        rule.apply_dopamine(pulse.appetitive, pulse.aversive)
        rules.append(rule)
    original, candidate = rules
    assert original.state.efficacy[0, 0] < candidate.state.efficacy[0, 0] < 1.0
    assert original.state.efficacy[0, 1] == candidate.state.efficacy[0, 1] == 1.0
    ratio = (1 - original.state.efficacy[0, 0]) / (1 - candidate.state.efficacy[0, 0])
    assert ratio == pytest.approx(60, rel=0.005)
