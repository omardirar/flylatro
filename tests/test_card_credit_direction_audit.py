"""Causal probes must leave canonical learners untouched and detect wrong signs."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from audit_card_credit_direction import audit, clipping_summary, metrics, one_update
from probe_fixed_pair_learning import probe
from card_learning.motor import CardExploration, install_motor
from card_learning.pairs import PairCorpus, build_corpus, pair_slots
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig
from flylatro.fly.mushroom_body.state import TorchPlasticEdgeState
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    pytest.importorskip('balatro_sim')
    path = tmp_path_factory.mktemp('direction-pairs') / 'pairs.json'
    build_corpus(path, train_per_rank=1, validation_per_rank=1, max_attempts=5000)
    return PairCorpus(path)


def toy():
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    return install_motor(build_plastic_stack(config), CardExploration(.05), pair_task=True)


def stimulus(corpus):
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    env.reset([0]); corpus.restore(env, 0, corpus.indices('training')[0])
    obs, masks = env.observe()
    masks['action_type_mask'][:] = False; masks['action_type_mask'][:, 0] = True
    action = empty_action_batch(1); action['action_type'][:] = 0; action['n_cards'][:] = 2
    action['cards'][0, :2] = pair_slots(obs)
    return env, obs, masks, action


def test_clipping_uses_selected_edges_and_reports_zero_and_unit_coincidences():
    config = PlasticityConfig(kc_reference_hz=20, mbon_reference_hz=20)
    pre = np.asarray([[0, 10, 20, 100, 100]], dtype=np.float32)
    post = np.asarray([[100, 20, 10, 100, 0]], dtype=np.float32)
    gate = np.asarray([[True, True, True, True, False]])
    result = clipping_summary(pre, post, gate, config)
    assert result['selected_edges'] == 4
    assert result['pre_clipped_fraction'] == .5
    assert result['post_clipped_fraction'] == .75
    assert result['both_clipped_fraction'] == .25
    assert result['zero_coincidence_fraction'] == .25
    assert result['unit_coincidence_fraction'] == .25
    assert result['mean_coincidence'] == .5
    with pytest.raises(ValueError, match='aligned'):
        clipping_summary(pre, post, gate & False, config)


@pytest.mark.parametrize('torch_state', [False, True])
def test_isolated_actual_update_is_selected_sign_correct_and_source_unchanged(corpus, torch_state):
    stack = toy(); agent = stack.agent
    if torch_state:
        pytest.importorskip('torch')
        agent.plasticity.state = TorchPlasticEdgeState.from_state_dict(agent.plasticity.state.state_dict(), device='cpu')
    _, obs, masks, action = stimulus(corpus)
    source = agent.plasticity.state
    before = plastic_hashes(source)
    efficacy = source.efficacy[:1]
    neural = agent.processor.process(obs, fly_seeds=[1], efficacy=efficacy)
    plus, stats = one_update(agent, neural, action, masks, efficacy, .01, 1)
    minus, _ = one_update(agent, neural, action, masks, efficacy, .01, -1)
    no_op, _ = one_update(agent, neural, action, masks, efficacy, 0, 1)
    assert np.any(plus > 0) and np.all(plus >= 0)
    assert np.any(minus < 0) and np.all(minus <= 0)
    assert not np.any(no_op)
    assert stats['selected_edges'] > 0
    np.testing.assert_allclose(plus, -minus, atol=1.3e-7)
    assert plastic_hashes(source) == before


def test_full_direction_audit_distinguishes_signs_with_exact_controls(corpus):
    stack = toy()
    env, *_ = stimulus(corpus)
    before_env = env.state_hash(env.observe()[0], 0)
    before = plastic_hashes(stack.agent.plasticity.state)
    report = audit(stack.agent, env, corpus, cases=2, repeats=1, multipliers=(1, 100), weights=('initial',))
    assert report['source_state_unchanged']
    assert report['exact_repeat_checks'] == report['restoration_checks'] == 2
    assert report['no_update_checks'] == 6
    assert plastic_hashes(stack.agent.plasticity.state) == before
    assert env.state_hash(env.observe()[0], 0) == before_env
    rows = [r for r in report['trials'] if r['probe'] == 'same_seed' and r['multiplier'] == 100]
    assert all(r['delta_mean_score_gap'] > 0 for r in rows if r['sign'] == 1)
    assert all(r['delta_mean_score_gap'] < 0 for r in rows if r['sign'] == -1)
    assert all(r['delta_mean_score_gap'] == 0 for r in report['trials'] if r['multiplier'] == 0)
    assert all(corpus.cases[i]['split'] == 'training' for i in report['case_indices'])


def test_audit_fails_closed_on_nondeterminism_and_restores_caller_environment(corpus):
    stack = toy(); env, *_ = stimulus(corpus)
    state_hash = env.state_hash(env.observe()[0], 0)
    process = stack.agent.processor.process
    count = 0
    def noisy(*args, **kwargs):
        nonlocal count
        count += 1
        result = process(*args, **kwargs)
        return replace(result, output_activity=result.output_activity + count)
    stack.agent.processor.process = noisy
    with pytest.raises(ValueError, match='exact neural repeat'):
        audit(stack.agent, env, corpus, cases=1, repeats=1)
    assert env.state_hash(env.observe()[0], 0) == state_hash


def test_permutation_probe_targets_follow_physical_cards(corpus):
    stack = toy(); env, *_ = stimulus(corpus)
    report = audit(stack.agent, env, corpus, cases=1, repeats=1, multipliers=(1,), weights=('initial',))
    original = next(r for r in report['trials'] if r['action'] == 'correct_pair' and r['probe'] == 'same_seed')
    changed = next(r for r in report['trials'] if r['action'] == 'correct_pair' and r['probe'] == 'permuted_cards')
    assert original['source_cards'] == changed['source_cards']
    assert set(changed['target_cards']) != set(original['target_cards'])


def test_bad_counts_or_update_sizes_are_refused_without_mutation(corpus):
    stack = toy(); env, *_ = stimulus(corpus)
    before = plastic_hashes(stack.agent.plasticity.state)
    for kwargs in [{'cases': 0}, {'learner': 4}, {'multipliers': (0,)}, {'multipliers': (1, 1)}, {'weights': ('invalid',)}]:
        with pytest.raises(ValueError):
            audit(stack.agent, env, corpus, **kwargs)
    assert plastic_hashes(stack.agent.plasticity.state) == before


def test_fixed_hand_probe_uses_independent_disposable_plastic_and_frozen_lanes(corpus):
    stack = toy()
    env = BalatroSimAdapter(2, win_ante=1, allow_skip_blinds=False); env.reset([0, 0])
    before = plastic_hashes(stack.agent.plasticity.state)
    report = probe(stack.agent, env, corpus, steps=12, evaluation_repeats=4)
    assert report['lane_order'] == ['plastic', 'frozen']
    assert report['steps_per_lane'] == 12 and report['source_state_unchanged']
    assert report['per_lane_max_absolute_drift'][0] > 0
    assert report['per_lane_max_absolute_drift'][1] == 0
    assert report['initial_evaluation']['pair_fraction'][0] == report['initial_evaluation']['pair_fraction'][1]
    assert report['initial_evaluation']['pair_fraction'][1] == report['final_evaluation']['pair_fraction'][1]
    assert report['blocks'][-1]['through_step'] == 12
    assert report['blocks'][-1]['trials'] == 12
    assert plastic_hashes(stack.agent.plasticity.state) == before
    for row in range(2):
        assert env.run_info(row)['round'] == 0


def test_fixed_hand_probe_refuses_validation_examples(corpus):
    stack = toy()
    env = BalatroSimAdapter(2, win_ante=1, allow_skip_blinds=False); env.reset([0, 0])
    with pytest.raises(ValueError, match='training-corpus only'):
        probe(stack.agent, env, corpus, case_index=corpus.indices('validation')[0], steps=1)
