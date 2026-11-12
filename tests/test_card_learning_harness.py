"""Research harness tests run on CPU while production training stays unchanged."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from card_learning.audit import audit_files, score
from card_learning.common import identity
from card_learning.experiment import build_condition_stack, load_condition, prepare, worker
from card_learning.motor import CardExploration, CardSubsetMotor, install_motor
from card_learning.pairs import PairCorpus, PairTaskEnvironment, build_corpus, pair_slots
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import MASK_SPEC, empty_action_batch, validate_strict_action_batch
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.interface.motor import FixedMotorInterface, MotorMapping, MOTOR_POOL_COUNT
from flylatro.learning.checkpoints import load_plastic_checkpoint, save_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.protocol_materialize import render_toml
from flylatro.learning.reinforcement import ReinforcementMapper, exploratory_condition


def base_config(native=False):
    base = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    return replace(base,
        environment=replace(base.environment, backend='balatro_sim' if native else 'mock',
                            allow_skip_blinds=False, num_envs=1),
        plasticity=replace(base.plasticity, version='three-factor-choice-gated-v1'),
        reinforcement=exploratory_condition('exploratory-weak-failure-v1'),
        curriculum=replace(base.curriculum, enabled=True, ladder=(1,)),
        training=replace(base.training, max_environment_decisions=12, checkpoint_every_decisions=4))


def write_config(path, config):
    document = asdict(config); document.pop('source_path')
    path.write_text(render_toml(document))
    return path


def motor_case(batch=1):
    mapping = MotorMapping.contiguous_pools(np.arange(MOTOR_POOL_COUNT), mode='mbon_direct', pool_width=1)
    base = FixedMotorInterface(mapping)
    masks = {key: np.zeros((batch, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}
    masks['action_type_mask'][:, 0] = True
    masks['card_select_mask'][:, [0, 2, 4, 6, 7]] = True
    activity = np.zeros((batch, MOTOR_POOL_COUNT))
    activity[:, list(mapping.routing.head_routes['card'])] = np.arange(10)
    activity[:, mapping.routing.head_routes['card_count'][2]] = 10
    return base, activity, masks


@pytest.mark.parametrize('ranks,suits,points,kind', [
    ([14, 3, 5], [0, 1, 2], 16, 'high_card'),
    ([11, 11, 14], [0, 1, 2], 60, 'pair'),
    ([2, 2, 3, 3, 14], [0, 1, 2, 3, 0], 60, 'two_pair'),
    ([10, 10, 10, 14], [0, 1, 2, 3], 180, 'three_kind'),
    ([14, 2, 3, 4, 5], [0, 1, 2, 3, 0], 220, 'straight'),
    ([14, 12, 10, 6, 3], [3]*5, 300, 'flush'),
    ([14, 14, 11, 11, 11], [0, 1, 0, 1, 2], 368, 'full_house'),
    ([2, 2, 2, 2, 14], [0, 1, 2, 3, 0], 476, 'four_kind'),
    ([10, 11, 12, 13, 14], [0]*5, 1208, 'straight_flush'),
    ([13, 14, 2, 3, 4], [0, 1, 2, 3, 0], 16, 'high_card'),
])
def test_plain_score_uses_only_scoring_cards_and_correct_ace_rules(ranks, suits, points, kind):
    assert score(list(zip(ranks, suits)), tuple(range(len(ranks)))) == (points, kind)


def test_sampling_preserves_greedy_and_zero_probability_baseline():
    base, activity, masks = motor_case()
    for deterministic in (False, True):
        kwargs = dict(deterministic=deterministic, learner_ids=[4], decision_ids=[9])
        expected = base.decode(activity, masks, **kwargs)
        actual = CardSubsetMotor(base, CardExploration(0)).decode(activity, masks, **kwargs)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        greedy = CardSubsetMotor(base, CardExploration(1)).decode(activity, masks, deterministic=True)
        for key in greedy:
            np.testing.assert_array_equal(greedy[key], base.decode(activity, masks, deterministic=True)[key])


def test_sampling_explores_non_prefix_subsets_legally_and_keeps_type_count_rng():
    base, activity, masks = motor_case()
    candidate = CardSubsetMotor(base, CardExploration(1))
    seen = set()
    for decision in range(100):
        kwargs = dict(deterministic=False, learner_ids=[7], decision_ids=[decision])
        expected = base.decode(activity, masks, **kwargs)
        actual = candidate.decode(activity, masks, **kwargs)
        for key in actual.keys() - {'cards'}:
            np.testing.assert_array_equal(actual[key], expected[key])
        validate_strict_action_batch(actual, masks)
        seen.add(tuple(sorted(actual['cards'][0, :int(actual['n_cards'][0])])))
    assert len(seen) > 5
    assert (0, 2) in seen
    with pytest.raises(ValueError, match='explicit learner'):
        candidate.decode(activity, masks, deterministic=False)


def test_sampling_is_independent_of_batch_row_and_resume_checks_its_identity():
    base, activity, masks = motor_case(batch=2)
    motor = CardSubsetMotor(base, CardExploration(1))
    batched = motor.decode(activity, masks, deterministic=False, learner_ids=[42, 99], decision_ids=[81, 81])
    for row in range(2):
        single = motor.decode(activity[row:row+1], {k: v[row:row+1] for k, v in masks.items()},
            deterministic=False, learner_ids=[[42, 99][row]], decision_ids=[81])
        for key in single:
            np.testing.assert_array_equal(single[key][0], batched[key][row])
    motor.load_rng_state(motor.rng_state())
    with pytest.raises(ValueError, match='exploration or task'):
        CardSubsetMotor(base, CardExploration(.05)).load_rng_state(motor.rng_state())


def test_sampler_does_not_change_consumable_targets_or_membership():
    base, activity, masks = motor_case()
    masks['action_type_mask'][:] = False; masks['action_type_mask'][:, 8] = True
    masks['consumable_target_mask'][:, 0] = True
    kwargs = dict(deterministic=False, learner_ids=[42], decision_ids=[8])
    expected = base.decode(activity, masks, **kwargs)
    actual = CardSubsetMotor(base, CardExploration(1)).decode(activity, masks, **kwargs)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_pair_task_credit_is_only_for_the_executed_card_choices():
    base, activity, masks = motor_case()
    motor = CardSubsetMotor(base, CardExploration(1), pair_task=True)
    seen = set()
    for decision in range(100):
        action = motor.decode(activity, masks, deterministic=False, learner_ids=[2], decision_ids=[decision])
        cards = tuple(action['cards'][0, :2]); seen.add(cards)
        gate = motor.selected_pool_mask(action, masks)
        expected = {base.mapping.routing.head_routes['card'][int(c)] for c in cards}
        assert set(np.flatnonzero(gate[0])) == expected
        assert motor.last_choices[0]['cards'] == cards
    assert len(seen) == 10  # every choose(5, 2) subset is reachable


def test_normal_exploration_credit_follows_sampled_cards_not_the_greedy_prefix():
    from types import SimpleNamespace
    from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
    from flylatro.learning.credit import SelectedMotorCredit
    base, activity, masks = motor_case()
    motor = CardSubsetMotor(base, CardExploration(1))
    topology = PlasticEdgeTopology.synthetic(np.arange(MOTOR_POOL_COUNT), np.arange(MOTOR_POOL_COUNT),
        np.ones(MOTOR_POOL_COUNT, dtype=np.float32), post_root_ids=base.mapping.output_root_ids)
    credit = SelectedMotorCredit(motor, SimpleNamespace(topology=topology, is_torch=False))
    for decision in range(20):
        kwargs = dict(deterministic=False, learner_ids=[2], decision_ids=[decision])
        greedy = base.decode(activity, masks, **kwargs)
        executed = motor.decode(activity, masks, **kwargs)
        n = int(executed['n_cards'][0])
        if set(executed['cards'][0, :n]) != set(greedy['cards'][0, :n]):
            break
    else:
        pytest.fail('fixture never explored a different subset')
    admitted = set(np.flatnonzero(credit.mask(executed, masks)[0]))
    card_routes = base.mapping.routing.head_routes['card']
    actual_card_posts = admitted & set(card_routes)
    assert actual_card_posts == {card_routes[int(c)] for c in executed['cards'][0, :n]}
    assert actual_card_posts != {card_routes[int(c)] for c in greedy['cards'][0, :n]}


def test_checkpoint_resume_reproduces_executed_choices_and_plastic_state(tmp_path):
    config = base_config()
    stack = install_motor(build_plastic_stack(config), CardExploration(1))
    for _ in range(5):
        stack.trainer.step()
    checkpoint = tmp_path / 'card.pkl'
    save_plastic_checkpoint(checkpoint, stack.trainer, experiment_config=config.to_dict(), component_metadata=stack.components)
    restored = install_motor(build_plastic_stack(config), CardExploration(1))
    load_plastic_checkpoint(checkpoint, restored.trainer, expected_components=restored.components)
    for _ in range(5):
        stack.trainer.step(); restored.trainer.step()
        for key in stack.trainer.last_executed_actions:
            np.testing.assert_array_equal(stack.trainer.last_executed_actions[key], restored.trainer.last_executed_actions[key])
        assert stack.trainer.last_state_hashes_after == restored.trainer.last_state_hashes_after
        for key, value in stack.agent.plasticity.state.state_dict().items():
            np.testing.assert_array_equal(value, restored.agent.plasticity.state.state_dict()[key])
    different = install_motor(build_plastic_stack(config), CardExploration(.05))
    with pytest.raises(ValueError, match='components differ'):
        load_plastic_checkpoint(checkpoint, different.trainer, expected_components=different.components)


def test_declaration_is_a_matched_four_arm_factorial_and_rejects_mutation(tmp_path):
    path = write_config(tmp_path / 'base.toml', base_config())
    result = prepare(path, tmp_path / 'pilot', steps=4, replicates=2, episodes=2)
    assert len(result['arms']) == 4 and not result['final_test_seeds_used']
    assert result['training_seeds'] == [0, 1]
    configs = []
    for arm in result['arms'].values():
        condition, corpus = load_condition(Path(arm['condition']))
        assert corpus is None
        config = PlasticExperimentConfig.load(Path(condition['config']))
        assert config.training.max_environment_decisions == 8
        doc = config.to_dict(); doc.pop('name')
        doc['training'].pop('condition'); doc['training'].pop('plasticity_enabled')
        configs.append(doc)
    assert all(c == configs[0] for c in configs)
    condition_path = Path(next(iter(result['arms'].values()))['condition'])
    with pytest.raises(ValueError, match='overriding'):
        worker(condition_path, 'train', ['--max-environment-decisions=100'])
    with pytest.raises(ValueError, match='validation only'):
        worker(condition_path, 'evaluate', ['--seed-stream=final_test'])
    condition = json.loads(condition_path.read_text())
    Path(condition['config']).write_text('mutated')
    with pytest.raises(ValueError, match='config changed'):
        load_condition(condition_path)


@pytest.fixture(scope='module')
def native_corpus(tmp_path_factory):
    pytest.importorskip('balatro_sim')
    path = tmp_path_factory.mktemp('one-pair') / 'corpus.json'
    build_corpus(path, train_per_rank=1, validation_per_rank=1, max_attempts=5000)
    return PairCorpus(path)


def test_native_pair_corpus_splits_are_disjoint_and_rank_balanced(native_corpus):
    assert len(native_corpus.cases) == 26
    split_seeds = []
    for split in ('training', 'validation'):
        cases = [native_corpus.cases[i] for i in native_corpus.indices(split)]
        assert Counter(c['pair_rank'] for c in cases) == Counter(range(2, 15))
        split_seeds.append({c['seed'] for c in cases})
    assert not split_seeds[0] & split_seeds[1]


def test_pair_task_native_scoring_boundaries_do_not_become_failures(native_corpus):
    env = PairTaskEnvironment(BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False), native_corpus)
    obs, masks = env.reset([100])
    snapshot = env.snapshot_all()
    correct = empty_action_batch(1); correct['action_type'][:] = 0; correct['n_cards'][:] = 2
    correct['cards'][0, :2] = pair_slots(obs)
    result = env.step(correct)
    assert result.dones[0] and result.infos[0]['task']['pair_selected']
    pulse = ReinforcementMapper(exploratory_condition('exploratory-weak-failure-v1')).map(
        result.infos[0]['reward_components'], result.infos[0])
    assert pulse.appetitive > 0 and pulse.aversive == 0
    assert pulse.events == ('progress',)
    env.restore_all(snapshot)
    restored, _ = env.observe()
    for key in obs:
        np.testing.assert_array_equal(obs[key], restored[key])
    ranks = obs['hand'][0, :8, :13].argmax(axis=1)
    wrong = {k: v.copy() for k, v in correct.items()}
    wrong['cards'][0, :2] = next((a, b) for a in range(8) for b in range(a+1, 8) if ranks[a] != ranks[b])
    inferior = env.step(wrong)
    assert inferior.rewards[0] < result.rewards[0]
    assert not inferior.infos[0]['task']['pair_selected']


def test_pair_resume_rejects_another_corpus_identity(native_corpus, tmp_path):
    env = PairTaskEnvironment(BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False), native_corpus)
    env.reset([100])
    snapshot = env.snapshot_all()
    other_path = tmp_path / 'other.json'
    other_path.write_bytes(native_corpus.path.read_bytes() + b'\n')
    other = PairTaskEnvironment(BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False), PairCorpus(other_path))
    other.reset([100])
    with pytest.raises(ValueError, match='corpus differs'):
        other.restore_all(snapshot)


def test_pair_task_trainer_resume_preserves_trials_and_clears_eligibility(native_corpus, tmp_path):
    path = write_config(tmp_path / 'base.toml', base_config(native=True))
    declaration = prepare(path, tmp_path / 'pair-plan', steps=8, replicates=1, episodes=2, pair_corpus=native_corpus.path)
    condition, corpus = load_condition(Path(declaration['arms']['subset-plastic']['condition']))
    config = PlasticExperimentConfig.load(Path(condition['config']))
    original = build_condition_stack(config, condition, corpus)
    for _ in range(3):
        original.trainer.step()
    assert not np.any(state_numpy(original.agent.plasticity.state.eligibility))
    checkpoint = tmp_path / 'pair.pkl'
    save_plastic_checkpoint(checkpoint, original.trainer, experiment_config=config.to_dict(), component_metadata=original.components)
    resumed = build_condition_stack(config, condition, corpus)
    load_plastic_checkpoint(checkpoint, resumed.trainer, expected_components=resumed.components)
    for _ in range(3):
        original.trainer.step(); resumed.trainer.step()
        assert original.env.trials == resumed.env.trials
        assert original.trainer.last_state_hashes_after == resumed.trainer.last_state_hashes_after
        np.testing.assert_array_equal(original.agent.plasticity.state.efficacy, resumed.agent.plasticity.state.efficacy)
        np.testing.assert_array_equal(original.trainer.last_executed_actions['cards'], resumed.trainer.last_executed_actions['cards'])


def test_audit_replays_native_hashes_and_refuses_tampered_logs(tmp_path):
    pytest.importorskip('balatro_sim')
    from flylatro.evaluation.state_hash import terminal_hash
    from flylatro.env.balatro_sim import action_batch_row_to_composite
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    obs, masks = env.reset([20070000])
    rows = []
    for decision in range(10):
        action = empty_action_batch(1); action['action_type'][:] = 2 if decision == 0 else 0
        action['n_cards'][:] = 0 if decision == 0 else 1
        if decision:
            action['cards'][0, 0] = 0
        before = env.state_hash(obs, 0)
        result = env.step(action)
        rows.append(dict(seed=20070000, decision_id=decision, action=action_batch_row_to_composite(action, 0).to_payload(),
            state_hash_before=before, state_hash_after=terminal_hash(result.infos[0]) if result.dones[0] else env.state_hash(result.observations, 0),
            reward=float(result.rewards[0]), done=bool(result.dones[0]), reward_components=result.infos[0]['reward_components'],
            synthetic_appetitive_reinforcement=0, synthetic_aversive_reinforcement=0))
        obs, masks = result.observations, result.masks
        if result.dones[0]:
            break
    source = tmp_path / 'transitions.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = audit_files([source], tmp_path / 'audit', native_check_states=1)
    assert report['status'] == 'completed'
    assert report['arms'][0]['counts']['verified_before_hashes'] == len(rows)
    assert report['arms'][0]['counts']['first_blind_failures_with_unused_discards'] == 1
    original_hash = rows[1]['state_hash_before']
    rows[1]['state_hash_before'] = 'invalid'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='before hash differs'):
        audit_files([source], tmp_path / 'tampered')
    assert json.loads((tmp_path / 'tampered/report.json').read_text())['status'] == 'failed'
    rows[1]['state_hash_before'] = original_hash
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows[:-1]))
    with pytest.raises(ValueError, match='incomplete episode'):
        audit_files([source], tmp_path / 'incomplete')
    source.write_text('{}')
    with pytest.raises(ValueError, match='newline-terminated'):
        audit_files([source], tmp_path / 'partial')


@pytest.mark.parametrize('label', ['subset-plastic', 'subset-frozen'])
def test_worker_cli_resume_and_independent_frozen_evaluation(tmp_path, label):
    path = write_config(tmp_path / 'base.toml', base_config())
    declaration = prepare(path, tmp_path / 'pilot', steps=6, replicates=1, episodes=2, batch=2)
    condition = Path(declaration['arms'][label]['condition'])
    run = tmp_path / 'training'
    worker(condition, 'train', ['--run-dir', str(run), '--stop-after-decisions', '3', '--no-dashboard', '--no-tensorboard'])
    summary = json.loads((run / 'run-summary.json').read_text())
    assert summary['status'] == 'stage_complete'
    worker(condition, 'train', ['--run-dir', str(run), '--resume', summary['checkpoint'], '--no-dashboard', '--no-tensorboard'])
    summary = json.loads((run / 'run-summary.json').read_text())
    assert summary['status'] == 'completed' and summary['environment_decisions'] == 6
    output = tmp_path / 'evaluation'
    worker(condition, 'evaluate', ['--checkpoint', summary['checkpoint'], '--learner-index', '0',
        '--episodes', '2', '--batch-size', '2', '--win-ante', '1', '--output-dir', str(output)])
    report = json.loads((output / 'summary.json').read_text())
    assert report['learning_disabled'] and not report['training_action_schedule_replayed']
    assert report['components']['card_learning_motor']['card_exploration']['probability'] == .05
    if label.endswith('frozen'):
        import pickle
        with open(summary['checkpoint'], 'rb') as stream:
            checkpoint = pickle.load(stream)
        state = checkpoint['trainer']['plastic_state']
        np.testing.assert_array_equal(state['efficacy'], state['initial_efficacy'])


def test_pair_benchmark_and_full_game_transfer_evaluation_use_frozen_weights(native_corpus, tmp_path):
    from card_learning.pair_evaluation import evaluate
    path = write_config(tmp_path / 'base.toml', base_config(native=True))
    declaration = prepare(path, tmp_path / 'pilot', steps=3, replicates=1, episodes=2, batch=2, pair_corpus=native_corpus.path)
    condition = Path(declaration['arms']['subset-plastic']['condition'])
    run = tmp_path / 'training'
    worker(condition, 'train', ['--run-dir', str(run), '--no-dashboard', '--no-tensorboard', '--heavy'])
    summary = json.loads((run / 'run-summary.json').read_text())
    assert summary['status'] == 'completed'
    report = evaluate(condition, Path(summary['checkpoint']), tmp_path / 'pairs.json', repeats=1, heavy=True)
    assert report['plastic_state_unchanged'] and report['independent_validation_hands'] == 13
    assert {row['seed'] for row in report['trials']} == {c['seed'] for c in native_corpus.cases if c['split'] == 'validation'}
    output = tmp_path / 'transfer'
    worker(condition, 'evaluate', ['--checkpoint', summary['checkpoint'], '--learner-index', '0',
        '--episodes', '2', '--batch-size', '2', '--win-ante', '1', '--output-dir', str(output), '--heavy'])
    result = json.loads((output / 'summary.json').read_text())
    assert result['learning_disabled']
    assert all(row['decisions'] > 1 for row in result['episodes'])  # actual games, not one-step task boundaries
