"""Executed-choice credit: gating, delayed outcomes, isolation and exact resume."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flylatro.env.upstream_contract import MASK_SPEC, UpstreamActionType as A, empty_action_batch
from flylatro.fly.mushroom_body.plasticity import (
    CHOICE_GATED_PLASTICITY_VERSION as CHOICE, PlasticityConfig, ThreeFactorPlasticity,
)
from flylatro.fly.mushroom_body.state import PlasticEdgeState, TorchPlasticEdgeState, state_numpy
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.interface.motor import FixedMotorInterface, MotorMapping, MOTOR_POOL_COUNT
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.credit import SelectedMotorCredit


def choice_config(**training):
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    return replace(config, plasticity=replace(config.plasticity, version=CHOICE),
        training=replace(config.training, **training))


def motor():
    return FixedMotorInterface(MotorMapping.contiguous_pools(
        np.arange(1000, 1000 + 2 * MOTOR_POOL_COUNT), mode='mbon_direct', pool_width=2))


def masks(batch=1):
    return {key: np.zeros((batch, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}


def pool_ids(interface, **heads):
    return {int(interface.mapping.routing.head_routes[head][option])
        for head, options in heads.items() for option in options}


def test_credit_marks_only_selected_action_count_and_card_slots_per_learner():
    interface = motor()
    legal = masks(2)
    legal['action_type_mask'][:, [int(A.PLAY_HAND), int(A.DISCARD)]] = True
    legal['card_select_mask'][:, :3] = True
    # Inactive contexts deliberately look available; they must not gain credit.
    legal['shop_target_mask'][:] = True
    actions = empty_action_batch(2)
    actions['action_type'][:] = [int(A.PLAY_HAND), int(A.DISCARD)]
    actions['n_cards'][:] = [2, 1]
    actions['cards'][0, :2] = [0, 2]
    actions['cards'][1, 0] = 1
    selected = interface.selected_pool_mask(actions, legal)
    assert set(np.flatnonzero(selected[0])) == pool_ids(interface,
        action_type=[int(A.PLAY_HAND)], card_count=[2], card=[0, 2])
    assert set(np.flatnonzero(selected[1])) == pool_ids(interface,
        action_type=[int(A.DISCARD)], card_count=[1], card=[1])


def test_forced_actions_and_all_selected_cards_do_not_invent_choices():
    interface = motor()
    legal = masks()
    actions = empty_action_batch(1)
    actions['n_cards'][:] = 0
    actions['action_type'][:] = int(A.CASH_OUT)
    legal['action_type_mask'][:, int(A.CASH_OUT)] = True
    assert not interface.selected_pool_mask(actions, legal).any()
    legal['action_type_mask'][:] = False
    legal['action_type_mask'][:, int(A.PLAY_HAND)] = True
    legal['card_select_mask'][:, :3] = True
    actions['action_type'][:] = int(A.PLAY_HAND)
    actions['n_cards'][:] = 3
    actions['cards'][0, :3] = [0, 1, 2]
    assert set(np.flatnonzero(interface.selected_pool_mask(actions, legal)[0])) == pool_ids(interface, card_count=[3])


@pytest.mark.parametrize('action,field,head,mask_name', [
    (A.BUY_SHOP, 'shop_target', 'shop', 'shop_target_mask'),
    (A.SELL_JOKER, 'joker_target', 'joker', 'joker_target_mask'),
    (A.USE_CONSUMABLE, 'consumable_target', 'consumable', 'consumable_target_mask'),
    (A.PICK_PACK, 'pack_target', 'pack', 'pack_target_mask'),
])
def test_only_the_active_contextual_target_receives_credit(action, field, head, mask_name):
    interface = motor()
    legal = masks()
    legal['action_type_mask'][:, int(action)] = True
    legal[mask_name][:, :2] = True
    actions = empty_action_batch(1)
    actions['action_type'][:] = int(action)
    actions['n_cards'][:] = 0
    actions[field][:] = 1
    assert set(np.flatnonzero(interface.selected_pool_mask(actions, legal)[0])) == pool_ids(interface, **{head: [1]})
    legal[mask_name][:, 0] = False
    assert not interface.selected_pool_mask(actions, legal).any()


def small_rule(backend='numpy'):
    topology = PlasticEdgeTopology.synthetic(np.arange(4), np.arange(4) + 4, np.ones(4, dtype=np.float32))
    state = PlasticEdgeState.initialize(4, learners=2)
    if backend == 'torch':
        pytest.importorskip('torch')
        state = TorchPlasticEdgeState.from_state_dict(state.state_dict(), device='cpu')
    return ThreeFactorPlasticity(topology, state, PlasticityConfig(version=CHOICE,
        learning_rate=.1, eligibility_decay=.5, kc_reference_hz=1, mbon_reference_hz=1))


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_only_gated_coactivity_is_added_but_previous_selected_traces_decay(backend):
    rule = small_rule(backend)
    active = np.ones((2, 4), dtype=np.float32)
    gate = np.asarray([[True, False, False, False], [False, True, False, False]])
    rule.record_activity(active, active, edge_credit=gate)
    rule.apply_dopamine([.2, .3], [0, 0])
    np.testing.assert_allclose(state_numpy(rule.state.efficacy), [[1.02, 1, 1, 1], [1, 1.03, 1, 1]])
    # No current choice: a delayed terminal-like pulse still reaches old choices.
    rule.record_activity(active, active, edge_credit=np.zeros_like(gate))
    np.testing.assert_array_equal(state_numpy(rule.state.eligibility), gate * .5)
    rule.apply_dopamine([0, 0], [.1, .1])
    np.testing.assert_allclose(state_numpy(rule.state.efficacy), [[1.015, 1, 1, 1], [1, 1.025, 1, 1]])
    rule.state.reset_fast_traces(learners=(0,), eligibility=True)
    assert not state_numpy(rule.state.eligibility)[0].any()
    assert state_numpy(rule.state.eligibility)[1, 1] == .5


def test_numpy_and_torch_choice_updates_agree_over_a_reinforcement_sequence():
    left, right = small_rule(), small_rule('torch')
    rng = np.random.default_rng(1620)
    for _ in range(12):
        pre = rng.uniform(0, 2, (2, 4)).astype(np.float32)
        post = rng.uniform(0, 2, (2, 4)).astype(np.float32)
        gate = rng.random((2, 4)) > .5
        for rule in (left, right):
            rule.record_activity(pre, post, edge_credit=gate)
            rule.apply_dopamine([.02, .01], [.01, .03])
        for key, value in left.state.state_dict().items():
            np.testing.assert_array_equal(value, right.state.state_dict()[key])


@pytest.mark.parametrize('bad', [None, np.ones((2, 4)), np.ones((1, 4), dtype=bool)])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_bad_credit_masks_fail_before_changing_state(bad, backend):
    rule = small_rule(backend)
    initial = rule.state.state_dict()
    with pytest.raises(ValueError):
        rule.record_activity(np.ones((2, 4)), np.ones((2, 4)), edge_credit=bad)
    for key, value in initial.items():
        np.testing.assert_array_equal(value, rule.state.state_dict()[key])


def test_shared_neurons_are_unioned_and_torch_mask_storage_is_reused():
    import torch
    interface = motor()
    pools = list(interface.mapping.pool_indices)
    action_pool = int(interface.mapping.routing.head_routes['action_type'][int(A.PLAY_HAND)])
    count_pool = int(interface.mapping.routing.head_routes['card_count'][2])
    pools[count_pool] = pools[action_pool]
    interface = FixedMotorInterface(replace(interface.mapping, pool_indices=tuple(pools)))
    roots = interface.mapping.output_root_ids
    topology = PlasticEdgeTopology.synthetic(np.arange(len(roots)), np.arange(len(roots)),
        np.ones(len(roots), dtype=np.float32), post_root_ids=roots)
    state = TorchPlasticEdgeState.initialize(len(roots), learners=1, device='cpu')
    rule = ThreeFactorPlasticity(topology, state, PlasticityConfig(version=CHOICE))
    credit = SelectedMotorCredit(interface, rule)
    legal = masks()
    legal['action_type_mask'][:, :2] = True
    legal['card_select_mask'][:, :2] = True
    actions = empty_action_batch(1)
    actions['action_type'][:] = int(A.PLAY_HAND)
    actions['n_cards'][:] = 2
    actions['cards'][0, :2] = [0, 1]
    gate = credit.mask(actions, legal)
    assert gate.sum() == len(pools[action_pool])
    pointer = gate.data_ptr()
    assert credit.mask(actions, legal).data_ptr() == pointer
    assert credit.manifest['external_trainable_parameter_count'] == 0


def test_matched_schedule_credits_executed_actions_instead_of_proposed_actions(monkeypatch):
    from flylatro.learning.reward_schedule import MatchedActionStep
    stack = build_plastic_stack(choice_config())
    proposed = empty_action_batch(1)
    proposed['action_type'][:] = int(A.PLAY_HAND)
    proposed['n_cards'][:] = 1
    proposed['cards'][0, 0] = 0
    executed = {key: value.copy() for key, value in proposed.items()}
    executed['action_type'][:] = int(A.DISCARD)
    monkeypatch.setattr(stack.agent.motor, 'decode', lambda *args, **kwargs: proposed)
    stack.trainer.action_schedule = [MatchedActionStep(executed)]
    before_masks = {key: value.copy() for key, value in stack.trainer.masks.items()}
    stack.trainer.step()
    neural = stack.trainer.last_decision.neural
    gate = stack.agent.credit.mask(executed, before_masks)
    expected = (np.clip(neural.edge_pre_activity / stack.agent.plasticity.config.kc_reference_hz, 0, 1)
        * np.clip(neural.edge_post_activity / stack.agent.plasticity.config.mbon_reference_hz, 0, 1) * gate)
    np.testing.assert_array_equal(stack.agent.plasticity.state.eligibility, expected)
    assert stack.agent.plasticity.state.decision_count[0] == 1


def test_choice_rule_resume_is_exact_and_cross_rule_restore_is_rejected(tmp_path):
    from flylatro.analysis.resume_cli import audit_resume
    from flylatro.learning.checkpoints import load_plastic_checkpoint
    config = choice_config()
    path = tmp_path / 'choice.pkl'
    report = audit_resume(config, path, prefix=4, future=4)
    assert report['gates']['status'] == 'PASS'
    original = build_plastic_stack(replace(config, plasticity=replace(config.plasticity, version='three-factor-global-v1')))
    with pytest.raises(ValueError, match='plasticity rule differs'):
        load_plastic_checkpoint(path, original.trainer, require_matching_exposure_budget=False)


def test_frozen_choice_rule_does_not_modify_any_learning_state():
    from flylatro.evaluation.plastic import evaluate_plastic_fly
    stack = build_plastic_stack(choice_config())
    stack.trainer.train(stop_after_environment_decisions=4)
    before = stack.agent.plasticity.state.state_dict()
    evaluate_plastic_fly(stack.env, stack.agent, [31], max_vector_steps=100)
    for key, value in before.items():
        np.testing.assert_array_equal(value, stack.agent.plasticity.state.state_dict()[key])


def test_choice_terminal_outcome_updates_weights_before_clearing_only_its_traces():
    from flylatro.learning.reinforcement import ReinforcementPulse
    config = choice_config()
    config = replace(config, environment=replace(config.environment, num_envs=2))
    stack = build_plastic_stack(config)
    stack.agent.act(stack.trainer.observations, stack.trainer.masks,
        fly_seeds=(17, 29), deterministic_motor=False)
    before = stack.agent.plasticity.state.state_dict()
    assert before['eligibility'][0].any()
    stack.agent.learn([{'episode': {'won': False}}, {}], plasticity_enabled=True,
        override_pulses=[ReinforcementPulse(.1, 0), ReinforcementPulse(0, 0)])
    after = stack.agent.plasticity.state.state_dict()
    assert np.any(after['efficacy'][0] > before['efficacy'][0])
    assert not after['eligibility'][0].any()
    np.testing.assert_array_equal(after['eligibility'][1], before['eligibility'][1])
    np.testing.assert_array_equal(after['efficacy'][1], before['efficacy'][1])
    assert after['episode_count'].tolist() == [1, 0]


def test_choice_rule_refuses_whole_brain_motor_mode():
    config = choice_config()
    with pytest.raises(ValueError, match='mbon_direct'):
        replace(config, fly=replace(config.fly, mode='whole_brain')).validate()
