from itertools import combinations
import json
from types import SimpleNamespace

import numpy as np
import pytest

from flylatro.analysis.card_selection import (
    diagnose_learner, permutations, permute_hand, plastic_hashes, score_plays,
)
from flylatro.analysis.card_selection_cli import main
from flylatro.env.upstream_contract import HAND_MAX, MASK_SPEC, OBS_SPEC
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.interface.motor import FixedMotorInterface, MotorMapping, MOTOR_POOL_COUNT


def case():
    obs = {key: np.zeros((1, *shape), dtype=dtype) for key, (shape, dtype) in OBS_SPEC.items()}
    masks = {key: np.zeros((1, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}
    obs['hand_len'][0] = 4
    obs['hand'][0, :4, 0] = [1, 4, 2, 3]
    masks['card_select_mask'][0, :4] = True
    masks['action_type_mask'][0, :2] = True
    scores = {cards: float(sum(obs['hand'][0, card, 0] for card in cards))
              for n in range(1, 5) for cards in combinations(range(4), n)}
    return {'index': 7, 'observations': obs, 'masks': masks, 'scores': scores}


def agent(mode='content', count=1):
    motor = FixedMotorInterface(MotorMapping.contiguous_pools(
        np.arange(MOTOR_POOL_COUNT), mode='mbon_direct', pool_width=1))
    state = PlasticEdgeState.initialize(2, learners=2)
    state.efficacy[1] = 1.5
    seeds = []

    def process(observations, *, fly_seeds, efficacy):
        seeds.append(fly_seeds)
        values = np.zeros((1, MOTOR_POOL_COUNT))
        routes = motor.mapping.routing.head_routes
        values[0, routes['card_count'][count]] = 10
        # Unconstrained decoder chooses DISCARD: card diagnostic must still run.
        values[0, routes['action_type'][1]] = 10
        if mode == 'content' or (mode == 'learned-content' and efficacy[0, 0] > 1):
            values[0, list(routes['card'])] = observations['hand'][0, :, 0]
        else:
            values[0, list(routes['card'])] = np.arange(HAND_MAX, 0, -1)
        return SimpleNamespace(output_activity=values)
    return SimpleNamespace(motor=motor, processor=SimpleNamespace(process=process),
                           plasticity=SimpleNamespace(state=state), seeds=seeds)


def test_permutation_moves_every_feature_and_legal_bit_without_mutating_inputs():
    item = case()
    item['masks']['card_select_mask'][0, 1] = False
    obs, masks = item['observations'], item['masks']
    order = permutations(4, 3, 123)[0]
    altered, legal = permute_hand(obs, masks, order)
    np.testing.assert_array_equal(altered['hand'], obs['hand'][:, order])
    np.testing.assert_array_equal(legal['card_select_mask'], masks['card_select_mask'][:, order])
    np.testing.assert_array_equal(obs['hand'][0, :4, 0], [1, 4, 2, 3])
    for key in obs.keys() - {'hand'}:
        np.testing.assert_array_equal(altered[key], obs[key])
    with pytest.raises(ValueError, match='occupied'):
        permute_hand(obs, masks, np.arange(HAND_MAX)[::-1])


def test_derangements_are_seeded_unique_and_report_exhaustion_for_two_cards():
    first = permutations(4, 4, 90)
    assert [p.tolist() for p in first] == [p.tolist() for p in permutations(4, 4, 90)]
    assert len({tuple(p) for p in first}) == 4
    assert all(np.all(p[:4] != np.arange(4)) for p in first)
    assert len(permutations(2, 4, 90)) == 1


@pytest.mark.parametrize('mode,card_match,slot_match', [('content', 1, 0), ('slot', 0, 1)])
def test_diagnostic_distinguishes_content_tracking_from_slot_bias(mode, card_match, slot_match):
    fly = agent(mode)
    before = plastic_hashes(fly.plasticity.state)
    report = diagnose_learner(fly, 1, [case()], repeats=2, permutation_count=3, seed=123)
    assert report['plastic_state_hashes_after'] == before
    assert report['exact_repeat_checks'] == 4
    for summary in report['summary'].values():
        assert summary['same_card_set_fraction'] == card_match
        assert summary['same_slot_set_fraction'] == slot_match
        assert summary['natural_play_fraction'] == 0
    if mode == 'content':
        assert report['summary']['trained']['mean_card_aligned_score_mae'] == 0
    else:
        assert report['summary']['trained']['mean_slot_aligned_score_mae'] == 0
    # Original, exact duplicate and each intervention reuse the same neural seed.
    assert all(seed == fly.seeds[0] for seed in fly.seeds[:10])


def test_quality_is_paired_with_initial_weights_and_explicit_learner():
    fly = agent('learned-content')
    changed = diagnose_learner(fly, 1, [case()], repeats=2, permutation_count=1, seed=10)
    assert changed['summary']['initial']['mean_immediate_reward'] == 1
    assert changed['summary']['trained']['mean_immediate_reward'] == 4
    assert changed['mean_paired_reward_change'] == 3
    unchanged = diagnose_learner(fly, 0, [case()], repeats=1, permutation_count=1, seed=10)
    assert unchanged['mean_paired_reward_change'] == 0
    with pytest.raises(ValueError, match='outside'):
        diagnose_learner(fly, 2, [case()], repeats=1, permutation_count=1, seed=10)


def test_forced_membership_and_zero_reward_states_are_not_positive_evidence():
    fly = agent(count=4)
    item = case()
    item['scores'] = {key: 0 for key in item['scores']}
    report = diagnose_learner(fly, 0, [item], repeats=1, permutation_count=2, seed=10)
    for summary in report['summary'].values():
        assert summary['membership_trials'] == 0
        assert summary['same_card_set_fraction'] is None
        assert summary['mean_normalized_regret_informative'] is None
        assert summary['informative_quality_trials'] == 0


def test_exact_repeat_and_frozen_state_checks_fail_closed():
    fly = agent()
    original = fly.processor.process
    def noisy(*args, **kwargs):
        result = original(*args, **kwargs)
        result.output_activity += len(fly.seeds)
        return result
    fly.processor.process = noisy
    with pytest.raises(ValueError, match='exact neural'):
        diagnose_learner(fly, 0, [case()], repeats=1, permutation_count=1, seed=10)
    fly = agent()
    original = fly.processor.process
    def mutating(*args, **kwargs):
        result = original(*args, **kwargs)
        fly.plasticity.state.eligibility[0, 0] += 1
        return result
    fly.processor.process = mutating
    with pytest.raises(ValueError, match='plastic state'):
        diagnose_learner(fly, 0, [case()], repeats=1, permutation_count=1, seed=10)


def test_simulator_scoring_restores_every_subset_and_caller_state_even_on_error():
    item = case()
    class Env:
        state = b'caller'
        calls = 0
        fail = False
        def snapshot(self, index): return self.state
        def restore(self, index, value): self.state = value
        def observe(self): return item['observations'], item['masks']
        def step(self, action):
            assert self.state == b'corpus'
            self.state = b'mutated'
            self.calls += 1
            if self.fail: raise RuntimeError('scoring failure')
            cards = tuple(action['cards'][0, :action['n_cards'][0]])
            return SimpleNamespace(rewards=[item['scores'][cards]])
    env = Env()
    assert score_plays(env, b'corpus', item['observations'], item['masks']) == item['scores']
    assert env.calls == 15 and env.state == b'caller'
    env.fail = True
    with pytest.raises(RuntimeError, match='scoring failure'):
        score_plays(env, b'corpus', item['observations'], item['masks'])
    assert env.state == b'caller'


def test_real_simulator_cli_checkpoint_is_frozen_and_output_is_not_overwritten(tmp_path):
    pytest.importorskip('balatro_sim')
    from flylatro.analysis.corpus import build_calibration_corpus
    from flylatro.env.balatro_sim import BalatroSimAdapter
    from flylatro.learning.cli import main as train
    from flylatro.fly.flywire_artifact import sha256_file
    from helpers import write_config

    corpus = build_calibration_corpus(BalatroSimAdapter(1, allow_skip_blinds=False),
        environment_seeds=(5,), states_per_seed=10, navigation_seed=10, store_snapshots=True)
    path = tmp_path/'corpus.npz'
    corpus.save(path)
    config = write_config(tmp_path, corpus_path=path, overrides={
        'environment': {'backend':'balatro_sim', 'num_envs':2, 'allow_skip_blinds':False},
        'training': {'max_environment_decisions':8, 'checkpoint_every_decisions':4}})
    run = tmp_path/'run'
    assert train(['--config',str(config),'--run-dir',str(run),'--no-dashboard','--heavy']) == 0
    checkpoint = run/'plastic-checkpoint-final.pkl'
    digest = sha256_file(checkpoint)
    output = tmp_path/'diagnostic'
    args = ['--config',str(config),'--checkpoint',str(checkpoint),'--output-dir',str(output),
            '--states','2','--repeats','1','--permutations','1','--heavy']
    assert main(args) == 0
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'completed' and report['sources_unchanged']
    assert report['learning_disabled']
    learners = report['arms']['checkpoint']['learners']
    assert [r['learner_index'] for r in learners] == [0, 1]
    assert all(r['plastic_state_unchanged'] and r['exact_repeat_checks'] == 4 for r in learners)
    assert sha256_file(checkpoint) == digest
    with pytest.raises(FileExistsError): main(args)
