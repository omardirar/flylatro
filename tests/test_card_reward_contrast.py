"""Native outcome centering must not become pair supervision or leak state."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import compare_card_reward_contrast as experiment
from test_card_credit_direction_audit import corpus, toy
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.mushroom_body.state import TorchPlasticEdgeState


def test_feedback_centers_native_progress_and_matches_mean_absolute_pulse():
    feedback, report = experiment.fit_feedback([.001, .002, .003], max_pulse=2)
    assert feedback.center == pytest.approx(.002)
    assert feedback.scale == pytest.approx(3)
    np.testing.assert_allclose(feedback.signed([.001, .002, .003]), [-.003, 0, .003])
    assert report['mean_centered_signed_pulse'] == pytest.approx(0, abs=1e-15)
    assert report['mean_centered_absolute_pulse'] == pytest.approx(.002)
    assert report['relative_absolute_pulse_matching_error'] < 1e-14
    # Out-of-calibration extremes respect the original pulse cap.
    assert feedback.signed(100) == 2


@pytest.mark.parametrize('values', [[1], [1, 1], [0, 1], [-1, 1], [np.nan, 1], [np.inf, 1], [[1, 2], [3, 4]]])
def test_feedback_calibration_rejects_invalid_or_constant_pulses(values):
    with pytest.raises(ValueError):
        experiment.fit_feedback(values, max_pulse=2)


def test_calibration_refuses_unmatched_capped_signal():
    with pytest.raises(ValueError, match='cap prevents matching'):
        experiment.fit_feedback([.001, .002, .003], max_pulse=.001)


def test_transform_rejects_nonfinite_or_negative_native_pulses():
    feedback = experiment.CenteredFeedback(.002, 3, 2)
    for value in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            feedback.signed(value)


def environment(corpus):
    env = BalatroSimAdapter(3, win_ante=1, allow_skip_blinds=False)
    env.reset([0]*3)
    return env


@pytest.mark.parametrize('torch_state', [False, True])
def test_comparison_changes_only_feedback_with_exact_source_and_frozen_controls(corpus, torch_state):
    stack = toy(); agent = stack.agent
    if torch_state:
        pytest.importorskip('torch')
        agent.plasticity.state = TorchPlasticEdgeState.from_state_dict(agent.plasticity.state.state_dict(), device='cpu')
    env = environment(corpus)
    before = plastic_hashes(agent.plasticity.state)
    env_before = [env.state_hash(env.observe()[0], row) for row in range(3)]
    report = experiment.compare(agent, env, corpus, steps=16, evaluation_repeats=4, calibration_trials=128)
    assert report['lane_order'] == ['native', 'centered', 'frozen']
    assert report['unchanged_plasticity_config'] == asdict(agent.plasticity.config)
    assert report['calibration']['relative_absolute_pulse_matching_error'] < 1e-14
    assert report['feedback_calibration_weights_unchanged']
    assert report['actual_update_l1'][0] > 0 and report['actual_update_l1'][1] > 0
    assert report['max_absolute_drift'][2] == report['actual_update_l1'][2] == 0
    assert report['applied_negative_pulse_counts'][0] == 0
    assert report['applied_negative_pulse_counts'][1] > 0
    assert report['decreased_edge_fraction'][1] > 0
    assert report['frozen_weights_and_individual_choices_unchanged']
    assert plastic_hashes(agent.plasticity.state) == before
    assert [env.state_hash(env.observe()[0], row) for row in range(3)] == env_before


def test_feedback_calibration_never_queries_target_pair_labels(corpus, monkeypatch):
    stack = toy(); env = environment(corpus)
    index = corpus.indices('training')[0]
    for row in range(3):
        corpus.restore(env, row, index)
    obs, masks = env.observe()
    local = experiment.local_rule(stack.agent, 0)
    before = plastic_hashes(local.state)
    def forbidden(*args):
        raise AssertionError('pair label accessed during feedback calibration')
    monkeypatch.setattr(experiment, 'pair_slots', forbidden)
    _, report = experiment.calibrate_feedback(stack.agent, env, corpus, index,
        obs, masks, np.arange(10), local.state.efficacy, trials=128)
    assert report['labels_used'] is False and report['reward_free'] is False
    assert report['unchosen_actions_scored'] is False
    assert plastic_hashes(local.state) == before


def test_changing_diagnostic_pair_labels_cannot_change_feedback_or_learning(corpus, monkeypatch):
    first = experiment.compare(toy().agent, environment(corpus), corpus,
        steps=8, evaluation_repeats=2, calibration_trials=128)
    monkeypatch.setattr(experiment, 'pair_slots', lambda obs, row=0: (0, 1))
    second = experiment.compare(toy().agent, environment(corpus), corpus,
        steps=8, evaluation_repeats=2, calibration_trials=128)
    for key in ('calibration', 'applied_signed_pulse_sum', 'actual_update_l1',
                'mean_signed_drift', 'mean_absolute_drift', 'max_absolute_drift'):
        assert first[key] == second[key]
    assert first['final_evaluation']['choices'] == second['final_evaluation']['choices']


def test_failed_calibration_restores_caller_state_and_environment(corpus):
    stack = toy(); agent = stack.agent; env = environment(corpus)
    before = plastic_hashes(agent.plasticity.state)
    env_before = [env.state_hash(env.observe()[0], row) for row in range(3)]
    process = agent.processor.process
    def broken(*args, **kwargs):
        result = process(*args, **kwargs)
        return replace(result, output_activity=np.full_like(result.output_activity, np.nan))
    agent.processor.process = broken
    with pytest.raises(ValueError, match='nonfinite calibration'):
        experiment.compare(agent, env, corpus, steps=1, evaluation_repeats=1, calibration_trials=2)
    assert plastic_hashes(agent.plasticity.state) == before
    assert [env.state_hash(env.observe()[0], row) for row in range(3)] == env_before


def test_validation_and_invalid_budgets_are_refused(corpus):
    agent = toy().agent; env = environment(corpus)
    for kwargs in ({'steps': 0}, {'calibration_trials': 1}, {'learner': 50},
                   {'case_index': corpus.indices('validation')[0]}):
        with pytest.raises(ValueError):
            experiment.compare(agent, env, corpus, **kwargs)
