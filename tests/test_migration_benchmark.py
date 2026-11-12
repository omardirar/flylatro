from dataclasses import replace
from pathlib import Path

import pytest

from flylatro.benchmarks.migration import measure
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.safety import TrainingSafetyStop


def stack():
    return build_plastic_stack(PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml')))


def test_repetitions_restore_learning_rng_and_exposure():
    training = stack()
    result = measure(training, warmup=2, steps=4, repetitions=3, device='cpu')
    assert result['repeat_weight_agreement']
    assert training.trainer.state.environment_decisions == 6
    assert len(result['trials']) == 3
    assert all(t['total_decisions'] == 4 for t in result['trials'])
    assert all(len(t['vector_step_latency_seconds']['samples']) == 4 for t in result['trials'])
    assert result['trials'][0]['last_metrics'] == result['trials'][1]['last_metrics']


def test_benchmark_preserves_production_safety_stop():
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    training = build_plastic_stack(config)
    training.agent.plasticity.state.efficacy[:] = float('nan')
    with pytest.raises(TrainingSafetyStop):
        measure(training, warmup=0, steps=2, repetitions=1, device='cpu')
    assert training.trainer.state.environment_decisions == 0


def test_benchmark_rejects_over_budget_before_training():
    training = stack()
    with pytest.raises(ValueError, match='exposure budget'):
        measure(training, warmup=2, steps=16, repetitions=1, device='cpu')
    assert training.trainer.state.environment_decisions == 0


def test_aggregate_and_per_learner_accounting():
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, environment=replace(config.environment, num_envs=2))
    result = measure(build_plastic_stack(config), warmup=1, steps=2, repetitions=2, device='cpu')
    for trial in result['trials']:
        assert trial['total_decisions'] == 4
        assert trial['decisions_per_second'] == 2 * trial['per_learner_decisions_per_second']
