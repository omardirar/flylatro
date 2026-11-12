from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from flylatro.evaluation.compare import MATCHED_FIELDS
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, CHOICE_GATED_PLASTICITY_VERSION as CHOICE
from flylatro.learning.choice_credit_pilot import prepare, validate_pair, compare_replicate
from flylatro.learning.config import PlasticExperimentConfig


def base_config():
    return PlasticExperimentConfig.load(Path('configs/exploratory-event-no-skip-weak-failure-v1.toml'))


def test_prepared_pilot_changes_only_rule_and_preserves_per_learner_exposure(tmp_path):
    report = prepare(base_config(), tmp_path/'pilot', steps=10000, replicates=3,
        evaluation_episodes=256, evaluation_batch_size=4)
    baseline, candidate = [PlasticExperimentConfig.load(Path(report['arms'][label]['config']))
        for label in ('baseline','choice')]
    validate_pair(baseline, candidate)
    assert baseline.environment.num_envs == 3
    assert baseline.training.max_environment_decisions == 30000
    assert baseline.training.checkpoint_every_decisions == 3000
    assert baseline.reinforcement.run_failure_pulse == .01
    assert baseline.plasticity.learning_rate == .002
    assert baseline.motor.exploration_temperature == 1
    assert not baseline.environment.allow_skip_blinds
    assert Path(baseline.fly.artifact_path).is_absolute()
    assert report['training_seeds'][0] != report['training_seeds'][1]
    assert report['final_test_seeds_used'] is False
    with pytest.raises(FileExistsError):
        prepare(base_config(), tmp_path/'pilot', steps=1, replicates=1,
            evaluation_episodes=1, evaluation_batch_size=1)


def test_pilot_pair_refuses_uncontrolled_reinforcement_or_temperature_changes():
    base = base_config()
    choice = replace(base, plasticity=replace(base.plasticity, version=CHOICE))
    validate_pair(base, choice)
    with pytest.raises(ValueError, match='only'):
        validate_pair(base, replace(choice, motor=replace(choice.motor, exploration_temperature=.5)))
    with pytest.raises(ValueError, match='only'):
        validate_pair(base, replace(choice, plasticity=replace(choice.plasticity, learning_rate=.001)))


def summary(version):
    rule = PlasticityConfig(version=version, learning_rate=.002)
    components = {key:'same' for key in MATCHED_FIELDS}
    components.update(plasticity_rule=asdict(rule), plasticity_rule_sha256=rule.sha256,
        action_constraints={'allow_skip_blinds':False}, motor_exploration={'epsilon':.05},
        training_seeds=[11,12,13], exposure_budget_decisions=30000)
    if version == CHOICE:
        components['credit_assignment']={'version':CHOICE}
    return {'learning_disabled':True,'training_action_schedule_replayed':False,
        'evaluation_weight_source':'trained-checkpoint',
        'action_policy':'independent-fixed-motor-choices-no-training-action-replay',
        'components':components,'selected_learner_index':0,'seeds':[20000000,20000001],
        'seed_stream':'validation','win_ante':1,'evaluation_batch_size':4,'source_checkpoint_learners':3,
        'metrics':{'evaluation/mean_skipped_blinds':0},
        'episodes':[{'seed':seed,'balatro_seed':str(seed),'episode_return':.1,'blinds_beaten':0,'won':False}
            for seed in [20000000,20000001]]}


def test_experimental_comparator_permits_only_the_declared_rule_difference():
    baseline, candidate = summary('three-factor-global-v1'), summary(CHOICE)
    candidate['episodes'][0].update(blinds_beaten=1,episode_return=.3)
    result = compare_replicate(baseline,candidate,0)
    assert result['choice']['blinds_beaten'] == 1
    assert result['mean_paired_return_delta'] == pytest.approx(.1)
    for field,value in [('motor_mapping_sha256','changed'),('action_constraints',{}),
                        ('motor_exploration',{'epsilon':.2}),('training_seeds',[14,15,16])]:
        changed = deepcopy(candidate)
        changed['components'][field]=value
        with pytest.raises(ValueError):compare_replicate(baseline,changed,0)
    changed = deepcopy(candidate)
    changed['components']['plasticity_rule']['learning_rate']=.03
    with pytest.raises(ValueError):compare_replicate(baseline,changed,0)
    changed = deepcopy(candidate)
    changed['selected_learner_index']=1
    with pytest.raises(ValueError):compare_replicate(baseline,changed,0)
