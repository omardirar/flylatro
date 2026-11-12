"""Frozen sampling and delayed-discard probes preserve source experiments."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from evaluate_motor_sampling import evaluate_policy
from train_discard_play_curriculum import probe
from test_card_credit_direction_audit import toy
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.array_mock import MockArrayBalatroEnv


def conditioned_toy(tmp_path):
    from dataclasses import replace
    from card_learning.motor import install_motor, CardExploration
    from flylatro.interface.action_conditioned_motor import structural_mapping
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    source = build_plastic_stack(config)
    path = structural_mapping(source.agent.processor.output_root_ids, pool_width=1).save(tmp_path/'motor.json')
    config = replace(config, fly=replace(config.fly, motor_pool_width=1, motor_mapping_path=str(path)))
    return install_motor(build_plastic_stack(config), CardExploration(.05), pair_task=False)


@pytest.fixture(scope='module')
def native_corpus(tmp_path_factory):
    from card_learning.pairs import build_corpus, PairCorpus
    path = tmp_path_factory.mktemp('discard-pairs')/'pairs.json'
    build_corpus(path, train_per_rank=1, validation_per_rank=1, max_attempts=5000)
    return PairCorpus(path)


def test_frozen_policies_are_reproducible_and_do_not_update_source_state():
    agent = toy().agent
    agent.motor.pair_task = False
    before = plastic_hashes(agent.plasticity.state)
    factory = lambda n: MockArrayBalatroEnv(n, blind_target=18, initial_hands=3, initial_discards=2)
    weights = agent.plasticity.state.initial_efficacy[:1]
    for policy in ('greedy', 'training-sampled'):
        left = evaluate_policy(agent, weights, [20_180_000,20_180_001], policy=policy,
                               batch_size=2, max_steps=32, env_factory=factory)
        right = evaluate_policy(agent, weights, [20_180_000,20_180_001], policy=policy,
                                batch_size=2, max_steps=32, env_factory=factory)
        assert left == right
        assert left['summary']['episodes'] == 2
        assert left['source_state_unchanged']
    assert plastic_hashes(agent.plasticity.state) == before


def test_evaluation_passes_explicit_sampling_and_seed_keys():
    agent = toy().agent; calls = []
    agent.motor.pair_task = False
    decode = agent.motor.decode
    def observed(activity, masks, **kw):
        calls.append(kw.copy())
        return decode(activity, masks, **kw)
    agent.motor.decode = observed
    factory = lambda n: MockArrayBalatroEnv(n, blind_target=18, initial_hands=3, initial_discards=2)
    for policy, expected in [('greedy',True),('training-sampled',False)]:
        calls.clear()
        report = evaluate_policy(agent, agent.plasticity.state.initial_efficacy[:1], [20180000], policy=policy,
                                 max_steps=1, env_factory=factory)
        assert calls and all(c['deterministic'] is expected for c in calls)
        assert calls[0]['learner_ids'] == [20180000]
        assert calls[0]['decision_ids'] == [0]
        assert report['summary']['truncations'] == 1


def test_frozen_evaluation_refuses_invalid_policy_or_seed_lists():
    agent = toy().agent
    for kwargs in ({'policy':'unknown'}, {'policy':'greedy','seeds':[1,1]}, {'policy':'greedy','batch_size':0}):
        args={'policy':'greedy','seeds':[1]};args.update(kwargs)
        with pytest.raises(ValueError):
            evaluate_policy(agent, agent.plasticity.state.initial_efficacy[:1], **args)


@pytest.mark.parametrize('torch_state', [False, True])
def test_two_step_credit_changes_only_discard_weights_and_preserves_source(native_corpus, tmp_path, torch_state):
    stack = conditioned_toy(tmp_path)
    if torch_state:
        from flylatro.fly.mushroom_body.state import TorchPlasticEdgeState
        stack.agent.plasticity.state = TorchPlasticEdgeState.from_state_dict(stack.agent.plasticity.state.state_dict(), device='cpu')
    before = plastic_hashes(stack.agent.plasticity.state)
    report, state = probe(stack.agent, native_corpus, trials=12, replicates=1, evaluation_repeats=1)
    assert report['source_state_unchanged'] and report['frozen_choices_and_weights_exact']
    assert report['non_discard_efficacy_unchanged']
    assert report['actual_update_l1'][0] > 0 and report['actual_update_l1'][1] == 0
    assert report['max_absolute_drift'][0] > 0 and report['max_absolute_drift'][1] == 0
    assert not state['eligibility'].any() and not state['dopamine'].any()
    assert state['decision_count'].tolist() == [24,24]
    assert plastic_hashes(stack.agent.plasticity.state) == before
    for policy in ('greedy','training-sampled'):
        for row in report['after'][policy]['trials']:
            assert row['fixed_play_uses_initial_weights']
            assert row['random_discard_counts'] == row['discard_counts']


def test_clearing_trace_after_discard_removes_all_learning(native_corpus, tmp_path):
    stack = conditioned_toy(tmp_path)
    report, state = probe(stack.agent, native_corpus, trials=8, replicates=1, evaluation_repeats=1, trace_cut=True)
    assert report['actual_update_l1'] == [0,0]
    np.testing.assert_array_equal(state['efficacy'],state['initial_efficacy'])


def test_pair_labels_cannot_change_native_feedback_or_curriculum_weights(native_corpus, tmp_path, monkeypatch):
    import train_discard_play_curriculum as study
    first, state = probe(conditioned_toy(tmp_path).agent,native_corpus,trials=4,replicates=1,evaluation_repeats=1)
    monkeypatch.setattr(study,'pair_slots',lambda obs,row=0:(0,1))
    second, changed = probe(conditioned_toy(tmp_path).agent,native_corpus,trials=4,replicates=1,evaluation_repeats=1)
    np.testing.assert_array_equal(state['efficacy'],changed['efficacy'])
    assert first['actual_update_l1'] == second['actual_update_l1']
    assert first['after']['greedy']['summary']['progress'] == second['after']['greedy']['summary']['progress']


def test_transfer_accepts_only_isolated_discard_state():
    from evaluate_discard_curriculum_transfer import validate_state
    initial = np.ones((2,4),dtype=np.float32); efficacy=initial.copy();efficacy[0,0]=1.1
    options=dict(replicates=1,edges=4,minimum=.2,maximum=2,discard_edges=[True,False,False,False])
    validate_state(efficacy,initial,**options)
    invalid=efficacy.copy();invalid[1,0]=1.1
    with pytest.raises(ValueError,match='frozen'):
        validate_state(invalid,initial,**options)
    invalid=efficacy.copy();invalid[0,1]=1.1
    with pytest.raises(ValueError,match='non-DISCARD'):
        validate_state(invalid,initial,**options)
