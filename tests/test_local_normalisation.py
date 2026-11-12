"""Budget conservation, controls, backend parity, and exact trainer resume."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_normalisation import NormalisedThreeFactorPlasticity
from normalisation_stack import install_normalisation
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import PlasticEdgeState, TorchPlasticEdgeState, state_numpy
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


def rule(backend='numpy', learners=3, config=None, **kwargs):
    topology = PlasticEdgeTopology.synthetic(np.arange(6), np.array([9, 8, 9, 8, 9, 8]),
        np.array([1, 3, 7, 2, 5, 11], dtype=np.float32))
    state = (PlasticEdgeState.initialize(6, learners=learners) if backend == 'numpy' else
        TorchPlasticEdgeState.initialize(6, learners=learners, device=backend))
    return NormalisedThreeFactorPlasticity(topology, state, config, **kwargs)


@pytest.mark.parametrize('backend', ['numpy', 'cpu', 'cuda'])
def test_local_competition_and_exact_no_update_controls(backend):
    if backend == 'cuda':
        import torch
        if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    r = rule(backend, normalised_learners=np.array([True, False, True]))
    before = state_numpy(r.state.efficacy).copy()
    r.state.eligibility[:, 0] = 1
    events = r.apply_dopamine([1, 1, 0], 0, detail=True)
    after = state_numpy(r.state.efficacy)
    assert after[0, 0] > 1 and after[0, 2] < 1 and after[0, 4] < 1
    np.testing.assert_array_equal(after[:, [1, 3, 5]], before[:, [1, 3, 5]])
    np.testing.assert_array_equal(after[2], before[2])
    np.testing.assert_allclose(state_numpy(r.incoming_strengths())[0], state_numpy(r.budgets)[0], rtol=2e-7)
    base = ThreeFactorPlasticity(r.topology, PlasticEdgeState.initialize(6), PlasticityConfig())
    base.state.eligibility[0, 0] = 1
    base.apply_dopamine(1, 0)
    np.testing.assert_array_equal(after[1], base.state.efficacy[0])
    for i, event in enumerate(events):
        delta = after[i] - before[i]
        assert event.changed_synapses == np.count_nonzero(delta)
        assert event.absolute_change == pytest.approx(abs(delta).sum(), abs=1e-7)
        assert event.detail is not None
        assert set(event.changed_edge_indices) == set(np.flatnonzero(delta))


@pytest.mark.parametrize('backend', ['numpy', 'cpu', 'cuda'])
def test_bounds_projection_and_punishment(backend):
    if backend == 'cuda':
        import torch
        if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    r = rule(backend, config=PlasticityConfig(learning_rate=1))
    # Establish nonuniform, valid initial budgets before constructing the rule.
    r.state.initial_efficacy[:, 0] = .2
    r.state.efficacy[:, 0] = .2
    r = NormalisedThreeFactorPlasticity(r.topology, r.state, r.config)
    r.state.eligibility[:, 2] = 1
    r.apply_dopamine(2, 0)
    np.testing.assert_allclose(state_numpy(r.incoming_strengths()), state_numpy(r.budgets), rtol=3e-7)
    assert np.all(state_numpy(r.state.efficacy) >= .2)
    assert np.all(state_numpy(r.state.efficacy) <= 2)
    previous = state_numpy(r.state.efficacy).copy()
    r.apply_dopamine(0, .5)
    assert np.any(state_numpy(r.state.efficacy) < previous)
    np.testing.assert_allclose(state_numpy(r.incoming_strengths()), state_numpy(r.budgets), rtol=3e-7)
    frozen = state_numpy(r.state.efficacy).copy()
    r.apply_dopamine(1, 0, plasticity_enabled=False)
    np.testing.assert_array_equal(state_numpy(r.state.efficacy), frozen)


def test_numpy_torch_parity_and_uniform_gain_stress():
    a, b = rule('numpy'), rule('cpu')
    rng = np.random.default_rng(42)
    for step in range(100):
        eligibility = rng.uniform(0, 4, size=(3, 6)).astype(np.float32)
        a.state.eligibility[:] = eligibility
        import torch
        b.state.eligibility.copy_(torch.from_numpy(eligibility))
        positive, negative = rng.uniform(0, 2, size=(2, 3)).astype(np.float32)
        a.apply_dopamine(positive, negative)
        b.apply_dopamine(positive, negative)
    np.testing.assert_allclose(a.state.efficacy, state_numpy(b.state.efficacy), atol=2e-6)
    r = rule(config=PlasticityConfig(learning_rate=.1))
    base = ThreeFactorPlasticity(r.topology, PlasticEdgeState.initialize(6, learners=3), PlasticityConfig(learning_rate=.1))
    r.state.eligibility[:] = 1; base.state.eligibility[:] = 1
    for _ in range(1000):
        r.apply_dopamine(2, 0); base.apply_dopamine(2, 0)
    np.testing.assert_array_equal(r.state.efficacy, r.state.initial_efficacy)
    np.testing.assert_array_equal(base.state.efficacy, np.full((3, 6), 2, dtype=np.float32))
    assert r.config.sha256 != base.config.sha256


def test_original_treatment_matches_original_rule_over_many_signed_updates():
    config = PlasticityConfig(version='three-factor-choice-gated-v1',
        learning_rate=.2, dopamine_decay=.4)
    treatment = rule(config=config, normalised_learners=np.zeros(3, dtype=bool))
    original = ThreeFactorPlasticity(treatment.topology, PlasticEdgeState.initialize(6, learners=3), config)
    rng = np.random.default_rng(73)
    for step in range(150):
        pre, post = rng.uniform(0, 50, size=(2, 3, 6)).astype(np.float32)
        gate = rng.random((3, 6)) < .5
        for model in (original, treatment):
            model.record_activity(pre, post, edge_credit=gate)
        positive, negative = rng.uniform(0, 2, size=(2, 3)).astype(np.float32)
        original.apply_dopamine(positive, negative, plasticity_enabled=step % 11 != 0)
        treatment.apply_dopamine(positive, negative, plasticity_enabled=step % 11 != 0)
        for name, value in original.state.state_dict().items():
            np.testing.assert_array_equal(value, treatment.state.state_dict()[name])


def test_projection_and_learning_do_not_use_other_posts_budget_or_activity():
    a, b = rule(), rule()
    # Same local inputs at root 9; root 8 has different initial values and an
    # update that forces a bounded projection. Root 9 must stay bit-identical.
    b.state.initial_efficacy[:, 1] = .2
    b.state.efficacy[:, 1] = .2
    b = NormalisedThreeFactorPlasticity(b.topology, b.state, b.config)
    for model in (a, b):
        model.state.eligibility[:, 0] = 1
        model.state.eligibility[:, 3] = 4
        model.apply_dopamine(2, 0)
    np.testing.assert_array_equal(a.state.efficacy[:, [0, 2, 4]], b.state.efficacy[:, [0, 2, 4]])


def test_rule_can_learn_conditional_choices_without_increasing_neuron_budgets():
    # Capacity control on a tiny transparent circuit, not the FlyWire model.
    topology = PlasticEdgeTopology.synthetic(np.array([0, 1, 0, 1]), np.array([2, 2, 3, 3]),
        np.ones(4, dtype=np.float32))
    model = NormalisedThreeFactorPlasticity(topology, PlasticEdgeState.initialize(4, learners=2),
        PlasticityConfig(version='three-factor-choice-gated-v1', learning_rate=.002,
            kc_reference_hz=20, mbon_reference_hz=20))
    contexts = [np.array([40, 0, 40, 0], dtype=np.float32), np.array([0, 40, 0, 40], dtype=np.float32)]
    for step in range(100):
        context = step % 2
        pre = np.tile(contexts[context], (2, 1))
        currents = (model.state.efficacy * pre).reshape(2, 2, 2).sum(axis=2)
        chosen = currents.argmax(axis=1)
        gate = np.repeat(np.arange(2)[None, :] == chosen[:, None], 2, axis=1)
        post = np.repeat(np.maximum(currents, 1), 2, axis=1)
        model.record_activity(pre, post, edge_credit=gate)
        positive = np.array([.05 if chosen[0] == context else 0, 0], dtype=np.float32)
        negative = np.array([.01 if chosen[0] != context else 0, 0], dtype=np.float32)
        model.apply_dopamine(positive, negative)
        model.state.reset_fast_traces(eligibility=True)
    choices = [(model.state.efficacy * x).reshape(2, 2, 2).sum(axis=2).argmax(axis=1) for x in contexts]
    assert [int(x[0]) for x in choices] == [0, 1]
    assert [int(x[1]) for x in choices] == [0, 0]
    np.testing.assert_allclose(model.incoming_strengths(), model.budgets, rtol=2e-7)
    np.testing.assert_array_equal(model.state.efficacy[1], 1)


def test_real_native_study_exact_resume_and_torn_tail(tmp_path):
    from dataclasses import asdict
    from card_learning.common import write_json
    from card_learning.motor import CardExploration, install_motor
    from card_learning.pairs import build_corpus, PairCorpus
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
    from flylatro.learning.reinforcement import exploratory_condition
    from run_local_normalisation import Study
    corpus_path = tmp_path / 'pairs.json'
    build_corpus(corpus_path, train_per_rank=1, validation_per_rank=1,
        training_offset=27000, validation_offset=940000, max_attempts=5000)
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, environment=replace(config.environment, allow_skip_blinds=False),
        reinforcement=exploratory_condition('exploratory-weak-failure-v1'),
        plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    def study():
        return Study(install_motor(build_plastic_stack(config), CardExploration(.05)), PairCorpus(corpus_path), replicates=1)
    a = study()
    write_json(tmp_path/'declaration.json', {'test': 'immutable native exact-resume study'})
    log = tmp_path / 'training.jsonl'
    with log.open('wb') as stream:
        for _ in range(3):
            import json
            stream.write((json.dumps(a.train_step())+'\n').encode())
    a.save(tmp_path, log.stat().st_size)
    expected = [a.train_step() for _ in range(3)]
    with log.open('ab') as stream: stream.write(b'{"torn')
    b = study(); b.restore(tmp_path)
    actual = [b.train_step() for _ in range(3)]
    assert actual == expected
    assert len(log.read_text().splitlines()) == 3
    np.testing.assert_array_equal(a.local.state.efficacy, b.local.state.efficacy)
    assert b.health()['frozen_exact'] and b.health()['unrelated_exact']


def test_study_and_independent_native_replay_reject_corruption(tmp_path):
    import json
    from dataclasses import asdict
    from flylatro.learning.config import PlasticExperimentConfig
    from flylatro.learning.protocol_materialize import render_toml
    from flylatro.learning.reinforcement import exploratory_condition
    from run_local_normalisation import prepare, execute
    from assess_local_normalisation import assess
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, environment=replace(config.environment, allow_skip_blinds=False),
        reinforcement=exploratory_condition('exploratory-weak-failure-v1'),
        plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    document = asdict(config); document.pop('source_path')
    source = tmp_path / 'config.toml'; source.write_text(render_toml(document))
    output = tmp_path / 'study'
    prepare(source, output, steps=4, replicates=1, repeats=1, training_per_rank=1, validation_per_rank=1)
    result = execute(output)
    assert result['health']['frozen_exact'] and result['health']['unrelated_exact']
    assessment = assess(output)
    assert assessment['record_counts']['training_records'] == 12
    assert assessment['initial_matched_choices_exact']
    assert assessment['native_rewards_and_pulses_verified']
    log = output / 'training.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[0]['pair_selected'][0] = not rows[0]['pair_selected'][0]
    log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='pair mismatch'):
        assess(output)


def test_normalised_trainer_resume_and_old_checkpoint_rejection(tmp_path):
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
    from flylatro.learning.checkpoints import save_plastic_checkpoint, load_plastic_checkpoint
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, environment=replace(config.environment, allow_skip_blinds=False),
        plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    first = install_normalisation(build_plastic_stack(config))
    for _ in range(3): first.trainer.step()
    path = tmp_path / 'normalised.pkl'
    save_plastic_checkpoint(path, first.trainer, experiment_config=config.to_dict(), component_metadata=first.components)
    restored = install_normalisation(build_plastic_stack(config))
    load_plastic_checkpoint(path, restored.trainer, expected_components=restored.components)
    for _ in range(3):
        first.trainer.step(); restored.trainer.step()
        np.testing.assert_array_equal(first.agent.plasticity.state.efficacy, restored.agent.plasticity.state.efficacy)
        np.testing.assert_array_equal(first.trainer.last_executed_actions['cards'], restored.trainer.last_executed_actions['cards'])
    base = build_plastic_stack(config)
    with pytest.raises(ValueError, match='plasticity rule differs'):
        load_plastic_checkpoint(path, base.trainer)
    old = tmp_path / 'old.pkl'
    save_plastic_checkpoint(old, base.trainer, experiment_config=config.to_dict(), component_metadata=base.components)
    with pytest.raises(ValueError, match='plasticity rule differs'):
        load_plastic_checkpoint(old, restored.trainer)
    changed_factory = install_normalisation(build_plastic_stack(config))
    changed_factory.components['normalisation_research_sources']['local_normalisation.py'] = 'changed'
    with pytest.raises(ValueError, match='checkpoint components differ'):
        load_plastic_checkpoint(path, changed_factory.trainer, expected_components=changed_factory.components)
