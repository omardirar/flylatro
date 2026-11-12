"""PLAY and DISCARD must express and reinforce different card/count choices."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flylatro.env.upstream_contract import MASK_SPEC, UpstreamActionType as A, empty_action_batch
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import PlasticEdgeState, TorchPlasticEdgeState, state_numpy
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.interface.action_conditioned_motor import (
    structural_mapping, calibrate_action_conditioned, separation_summary,
)
from flylatro.interface.motor import FixedMotorInterface, MotorMapping, MotorRouting
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.interface.motor_calibration import MotorCalibrationError
from flylatro.learning.credit import SelectedMotorCredit


def masks(batch=2):
    result = {key: np.zeros((batch, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}
    result['action_type_mask'][:, [int(A.PLAY_HAND), int(A.DISCARD)]] = True
    result['card_select_mask'][:, :8] = True
    return result


def fixture():
    mapping = structural_mapping(np.arange(96)+1000)
    motor = FixedMotorInterface(mapping)
    activity = np.zeros((2, 96))
    def drive(head, option, value):
        pool = mapping.routing.head_routes[head][option]
        activity[:, list(mapping.pool_indices[pool])] = value
    drive('card_count', 2, 50); drive('discard_card_count', 5, 50)
    for card in (1, 2):
        drive('card', card, 50)
    for card in (0, 3, 4, 5, 6):
        drive('discard_card', card, 50)
    legal = masks()
    legal['action_type_mask'][:] = False
    legal['action_type_mask'][0, int(A.PLAY_HAND)] = True
    legal['action_type_mask'][1, int(A.DISCARD)] = True
    return mapping, motor, activity, legal


def test_same_activity_and_hand_support_opposite_card_choices_and_independent_counts():
    mapping, motor, activity, legal = fixture()
    assert np.array_equal(activity[0], activity[1])
    actions = motor.decode(activity, legal, deterministic=True)
    assert actions['n_cards'].tolist() == [2, 5]
    assert set(actions['cards'][0, :2]) == {1, 2}
    assert set(actions['cards'][1, :5]) == {0, 3, 4, 5, 6}
    assert not set(actions['cards'][0, :2]) & set(actions['cards'][1, :5])
    assert set(motor.head_scores(activity)) >= {'card', 'discard_card', 'card_count', 'discard_card_count'}
    # Both actions also choose independently when their count is held at two.
    for pool in mapping.routing.head_routes['discard_card_count']:
        activity[:, list(mapping.pool_indices[pool])] = 0
    activity[:, list(mapping.pool_indices[mapping.routing.head_routes['discard_card_count'][2]])] = 50
    assert motor.decode(activity, legal)['n_cards'].tolist() == [2, 2]
    assert set(motor.decode(activity, legal)['cards'][1, :2]) == {0, 3}


@pytest.mark.parametrize('torch_state', [False, True])
def test_credit_and_actual_updates_do_not_touch_other_action_card_or_count_pools(torch_state):
    mapping, motor, activity, legal = fixture()
    topology = PlasticEdgeTopology.synthetic(np.arange(96), np.arange(96), np.ones(96, dtype=np.float32),
        post_root_ids=mapping.output_root_ids)
    state = PlasticEdgeState.initialize(96, learners=2)
    if torch_state:
        pytest.importorskip('torch')
        state = TorchPlasticEdgeState.from_state_dict(state.state_dict(), device='cpu')
    rule = ThreeFactorPlasticity(topology, state, PlasticityConfig(version='three-factor-choice-gated-v1', learning_rate=.1))
    credit = SelectedMotorCredit(motor, rule)
    actions = motor.decode(activity, legal)
    gate = credit.mask(actions, legal)
    npgate = state_numpy(gate)
    for row, own_heads, other_heads in ((0, ('card', 'card_count'), ('discard_card', 'discard_card_count')),
                                      (1, ('discard_card', 'discard_card_count'), ('card', 'card_count'))):
        own = {i for head in own_heads for pool in mapping.routing.head_routes[head] for i in mapping.pool_indices[pool]}
        other = {i for head in other_heads for pool in mapping.routing.head_routes[head] for i in mapping.pool_indices[pool]}
        assert npgate[row, list(own)].any()
        assert not npgate[row, list(other)].any()
    rule.record_activity(np.ones((2, 96)), np.ones((2, 96)), edge_credit=gate)
    rule.apply_dopamine(.1, 0)
    delta = state_numpy(state.efficacy)-1
    assert np.all(delta[npgate] > 0) and not np.any(delta[~npgate])
    assert credit.metrics['credit/card_alias_edge_fraction_mean'] == 0


def test_mapping_cannot_alias_play_discard_neurons_or_routes():
    mapping = structural_mapping(np.arange(96))
    pools = list(mapping.pool_indices)
    pools[mapping.routing.head_routes['discard_card'][0]] = pools[mapping.routing.head_routes['card'][0]]
    with pytest.raises(ValueError, match='disjoint'):
        replace(mapping, pool_indices=tuple(pools))
    routes = dict(mapping.routing.head_routes)
    routes['discard_card'] = routes['card']
    with pytest.raises(ValueError, match='must own'):
        replace(mapping.routing, head_routes=routes)
    with pytest.raises(MotorCalibrationError, match='distinct outputs'):
        structural_mapping(np.arange(89))


def test_action_conditioned_serialization_and_legacy_identity(tmp_path):
    legacy_path = Path('artefacts/separated-motor-20261024-pilot-v2/separated-motor.json')
    if legacy_path.exists():
        import json
        original = json.loads(legacy_path.read_text())
        legacy = MotorMapping.load(legacy_path)
        assert legacy.sha256 == original['sha256']
        assert not legacy.routing.action_conditioned
        assert legacy.routing.card_heads(int(A.DISCARD)) == ('card', 'card_count')
    mapping = structural_mapping(np.arange(96))
    path = mapping.save(tmp_path/'motor.json')
    loaded = MotorMapping.load(path)
    assert loaded.sha256 == mapping.sha256
    assert loaded.routing.card_heads(int(A.PLAY_HAND)) == ('card', 'card_count')
    assert loaded.routing.card_heads(int(A.DISCARD)) == ('discard_card', 'discard_card_count')
    assert loaded.routing.card_heads(int(A.USE_CONSUMABLE)) == ('card', 'card_count')
    assert separation_summary(loaded)['unique_core_neurons'] == 90


def test_contexts_only_qualify_each_action_where_it_is_legal():
    legal = masks(3)
    legal['action_type_mask'][:] = False
    legal['action_type_mask'][0, int(A.PLAY_HAND)] = True
    legal['action_type_mask'][1, int(A.DISCARD)] = True
    legal['action_type_mask'][2, int(A.USE_CONSUMABLE)] = True
    windows = motor_context_windows(legal, routing=MotorRouting.action_conditioned_v2())
    assert windows['card_slot'].relevant.tolist() == [True, False, True]
    assert windows['discard_card_slot'].relevant.tolist() == [False, True, False]
    assert not windows['discard_card_count'].option_legal[:, 0].any()


@pytest.mark.parametrize('width', [1, 2])
def test_reward_free_calibration_passes_declared_gates_and_preserves_exclusivity(width):
    rng = np.random.default_rng(293)
    values = np.maximum(40+rng.normal(0, 10, (128, 96)), 0)
    legal = {key: np.ones((128, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}
    mapping, report = calibrate_action_conditioned(np.arange(96), values, masks=legal,
        attempts=2, refinement_rounds=1, annealing_steps=100, pool_width=width)
    assert report['qualification']['gates']['status'] == 'PASS'
    assert not report['reward_used'] and not report['outcome_information_used']
    assert all(v == 0 for v in separation_summary(mapping)['core_overlap_neuron_counts'].values())
    assert separation_summary(mapping)['unique_core_neurons'] == 45*width
    assert report['thresholds']['minimum_pool_width'] == width
    from dataclasses import asdict
    from flylatro.interface.motor_calibration import MotorCalibrationThresholds
    for name, value in asdict(MotorCalibrationThresholds()).items():
        if name != 'minimum_pool_width':
            assert report['thresholds'][name] == value
    with pytest.raises(MotorCalibrationError):
        calibrate_action_conditioned(np.arange(96), np.ones((128, 96)), masks=legal, attempts=1)


def test_single_neuron_layout_is_explicit_and_legacy_config_cannot_silently_use_it(tmp_path):
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    original = build_plastic_stack(config)
    legacy_path = original.agent.motor.mapping.save(tmp_path/'legacy.json')
    with pytest.raises(ValueError, match='explicit action-conditioned'):
        replace(config, fly=replace(config.fly, motor_pool_width=1, motor_mapping_path=str(legacy_path))).validate()
    mapping = structural_mapping(original.agent.processor.output_root_ids, pool_width=1)
    path = mapping.save(tmp_path/'conditioned.json')
    candidate = replace(config, fly=replace(config.fly, motor_pool_width=1, motor_mapping_path=str(path)))
    candidate.validate()
    stack = build_plastic_stack(candidate)
    assert stack.agent.motor.mapping.routing.action_conditioned
    assert stack.components['motor_routing_sha256'] == mapping.routing.sha256
    with pytest.raises(ValueError, match='artifact width differs'):
        build_plastic_stack(replace(candidate, fly=replace(candidate.fly, motor_pool_width=2)))


def test_legacy_checkpoint_is_rejected_under_the_new_card_interpretation(tmp_path):
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
    from flylatro.learning.checkpoints import save_plastic_checkpoint, load_plastic_checkpoint
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    original = build_plastic_stack(config)
    checkpoint = tmp_path/'legacy.pkl'
    save_plastic_checkpoint(checkpoint, original.trainer, experiment_config=config.to_dict(), component_metadata=original.components)
    mapping = structural_mapping(original.agent.processor.output_root_ids, pool_width=1)
    path = mapping.save(tmp_path/'conditioned.json')
    candidate = build_plastic_stack(replace(config, fly=replace(config.fly, motor_pool_width=1, motor_mapping_path=str(path))))
    with pytest.raises(ValueError, match='motor'):
        load_plastic_checkpoint(checkpoint, candidate.trainer, expected_components=candidate.components)


def test_action_conditioned_training_checkpoint_resumes_exactly(tmp_path):
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
    from flylatro.analysis.resume_cli import audit_resume
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    original = build_plastic_stack(config)
    mapping = structural_mapping(original.agent.processor.output_root_ids, pool_width=1)
    path = mapping.save(tmp_path/'conditioned.json')
    candidate = replace(config, fly=replace(config.fly, motor_pool_width=1, motor_mapping_path=str(path)))
    report = audit_resume(candidate, tmp_path/'conditioned.pkl', prefix=4, future=4)
    assert report['gates']['status'] == 'PASS'
