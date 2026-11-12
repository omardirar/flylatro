"""Calibration must stay reward-free and control update magnitude explicitly."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from compare_card_reference_scaling import calibrate, compare, fit_references, uniform_pair_edge_probability
from audit_card_credit_direction import isolated_rule, array
from test_card_credit_direction_audit import corpus, toy, stimulus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import TorchPlasticEdgeState
from flylatro.learning.credit import SelectedMotorCredit


def data():
    return [{'pre': np.asarray([0, 20, 100, 1000.]),
             'post': np.asarray([100, 200, 400, 800.]),
             'probability': np.asarray([.25, .25, .5, .5])}]


def test_matching_is_exact_on_expected_coactivity_and_changes_only_three_parameters():
    config = PlasticityConfig(kc_reference_hz=20, mbon_reference_hz=20, learning_rate=.002)
    scaled, report = fit_references(data(), np.arange(4), np.arange(4), config)
    assert report['relative_matching_error'] < 1e-14
    assert scaled.kc_reference_hz == pytest.approx(910)
    assert scaled.mbon_reference_hz == pytest.approx(740)
    assert report['rescaled']['pre_clipped_fraction'] < report['baseline']['pre_clipped_fraction']
    assert report['rescaled']['post_clipped_fraction'] < report['baseline']['post_clipped_fraction']
    assert scaled.learning_rate > config.learning_rate
    changed = {k for k, v in asdict(config).items() if v != asdict(scaled)[k]}
    assert changed == {'kc_reference_hz', 'mbon_reference_hz', 'learning_rate'}


def test_reference_percentile_counts_neurons_once_despite_connection_degree():
    config = PlasticityConfig(kc_reference_hz=20, mbon_reference_hz=20)
    single, _ = fit_references(data(), np.arange(4), np.arange(4), config)
    # Repeating one anatomical connection must not overweight its neuron rate.
    index = np.r_[np.arange(4), np.full(99, 3)]
    repeated = [{k: v[index] for k, v in data()[0].items()}]
    repeated_config, _ = fit_references(repeated, index, index, config)
    assert repeated_config.kc_reference_hz == single.kc_reference_hz
    assert repeated_config.mbon_reference_hz == single.mbon_reference_hz


@pytest.mark.parametrize('bad', ['zero', 'nan', 'negative', 'probability', 'alignment'])
def test_invalid_or_silent_calibration_fails_closed(bad):
    samples = data()
    if bad == 'zero':
        samples[0]['pre'].fill(0)
    elif bad == 'nan':
        samples[0]['pre'][0] = np.nan
    elif bad == 'negative':
        samples[0]['post'][0] = -1
    elif bad == 'probability':
        samples[0]['probability'][0] = 1.1
    else:
        samples[0]['pre'] = samples[0]['pre'][:2]
    with pytest.raises(ValueError):
        fit_references(samples, np.arange(4), np.arange(4), PlasticityConfig())


def test_expected_gate_matches_actual_pair_union_and_has_no_forced_head_credit(corpus):
    stack = toy(); agent = stack.agent
    _, _, masks, _ = stimulus(corpus)
    local = isolated_rule(agent, agent.plasticity.state.initial_efficacy[:1])
    credit = SelectedMotorCredit(agent.motor, local)
    probability = uniform_pair_edge_probability(credit, masks)
    assert np.any(probability > 0)
    # Toy card pools are disjoint: each legal card is included in 2 of 8 slots.
    np.testing.assert_array_equal(np.unique(probability[probability > 0]), [.25])
    unselectable = ~credit.pool_posts[credit.card_pool_ids[:8]].any(axis=0)[credit.edge_posts]
    assert not np.any(probability[unselectable])


def test_calibration_only_reads_frozen_observations_masks_and_preserves_source(corpus):
    stack = toy(); agent = stack.agent
    _, obs, masks, _ = stimulus(corpus)
    class RewardFreeCorpus:
        sha256 = 'test-frozen-inputs'
        def __init__(self):
            self.masks = masks
        def row(self, index):
            assert index == 0
            return obs, masks
        @property
        def rewards(self):
            raise AssertionError('calibration accessed outcomes')
        @property
        def cases(self):
            raise AssertionError('calibration accessed pair labels')
    before = plastic_hashes(agent.plasticity.state)
    scaled, report = calibrate(agent, RewardFreeCorpus(), repeats=2)
    assert report['state_indices'] == [0] and report['samples'] == 2
    assert report['relative_matching_error'] < 1e-14
    assert scaled.kc_reference_hz >= agent.plasticity.config.kc_reference_hz
    assert plastic_hashes(agent.plasticity.state) == before


def test_matched_coactivity_predicts_actual_three_factor_update_l1(corpus):
    stack = toy(); agent = stack.agent
    _, obs, masks, _ = stimulus(corpus)
    initial = agent.plasticity.state.initial_efficacy[:1]
    base = isolated_rule(agent, initial)
    neural = agent.processor.process(obs, fly_seeds=[1], efficacy=initial)
    pre, post = array(neural.edge_pre_activity).ravel(), array(neural.edge_post_activity).ravel()
    credit = SelectedMotorCredit(agent.motor, base)
    probability = uniform_pair_edge_probability(credit, masks)
    scaled, report = fit_references([dict(pre=pre, post=post, probability=probability)],
        base.topology.pre_root_ids, base.topology.post_root_ids, base.config)
    other = isolated_rule(agent, initial)
    other = ThreeFactorPlasticity(base.topology, other.state, scaled)
    # All candidate card edges, then average their actual delta using exact
    # inclusion probabilities. Unit reward and reset traces match the fit.
    gate = (probability > 0)[None, :]
    values = []
    for rule in (base, other):
        rule.record_activity(pre[None, :], post[None, :], edge_credit=gate)
        rule.apply_dopamine(1., 0.)
        values.append(float(np.sum((array(rule.state.efficacy)-array(initial)).ravel()*probability)))
    assert values[0] == pytest.approx(values[1], rel=2e-4, abs=1e-6)


@pytest.mark.parametrize('torch_state', [False, True])
def test_comparison_keeps_source_and_frozen_control_exact_and_restores_environment(corpus, torch_state):
    stack = toy(); agent = stack.agent
    if torch_state:
        pytest.importorskip('torch')
        agent.plasticity.state = TorchPlasticEdgeState.from_state_dict(agent.plasticity.state.state_dict(), device='cpu')
    env = BalatroSimAdapter(3, win_ante=1, allow_skip_blinds=False); env.reset([0]*3)
    env_before = [env.state_hash(env.observe()[0], row) for row in range(3)]
    before = plastic_hashes(agent.plasticity.state)
    scaled = replace(agent.plasticity.config, kc_reference_hz=100, mbon_reference_hz=100)
    report = compare(agent, env, corpus, scaled, steps=12, evaluation_repeats=4)
    assert report['lane_order'] == ['baseline', 'rescaled', 'frozen']
    assert report['actual_update_l1'][0] > 0 and report['actual_update_l1'][1] > 0
    assert report['max_absolute_drift'][2] == report['actual_update_l1'][2] == 0
    assert report['frozen_weights_and_individual_choices_unchanged']
    assert all(a[2] == b[2] for a, b in zip(report['initial_evaluation']['choices'], report['final_evaluation']['choices']))
    assert plastic_hashes(agent.plasticity.state) == before
    assert [env.state_hash(env.observe()[0], row) for row in range(3)] == env_before
    assert all(pulse > 0 for pulse in report['pulse_sum'])


def test_validation_or_unrelated_rule_changes_are_refused(corpus):
    stack = toy(); agent = stack.agent
    env = BalatroSimAdapter(3, win_ante=1, allow_skip_blinds=False); env.reset([0]*3)
    config = agent.plasticity.config
    with pytest.raises(ValueError, match='training-corpus only'):
        compare(agent, env, corpus, config, case_index=corpus.indices('validation')[0], steps=1)
    with pytest.raises(ValueError, match='only references'):
        compare(agent, env, corpus, replace(config, eligibility_decay=.5), steps=1)


def test_native_progress_mapper_is_unchanged_for_all_three_lanes(corpus):
    stack = toy(); agent = stack.agent
    env = BalatroSimAdapter(3, win_ante=1, allow_skip_blinds=False); env.reset([0]*3)
    mapper = agent.reinforcement.map
    pulses = []
    def observed(components, info):
        pulse = mapper(components, info)
        pulses.append(pulse.appetitive)
        return pulse
    agent.reinforcement.map = observed
    report = compare(agent, env, corpus, agent.plasticity.config, steps=4, evaluation_repeats=2)
    np.testing.assert_allclose(report['pulse_sum'], np.asarray(pulses).reshape(4, 3).sum(axis=0))
