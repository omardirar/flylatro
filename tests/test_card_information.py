from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from flylatro.analysis.card_information import (
    change_card, contrast_scores, measure_learner, prepare_cases, response_metrics,
    sensory_effects, summarize,
    verify_erased_input_controls,
)
from flylatro.analysis.corpus import CalibrationCorpus, CALIBRATION_CORPUS_VERSION
from flylatro.env.upstream_contract import MASK_SPEC, OBS_SPEC
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.plastic_features import feature_names
from flylatro.interface.motor import FixedMotorInterface, MotorMapping, MOTOR_POOL_COUNT
from flylatro.interface.sensory import FixedPlasticSensoryEncoder, SensoryMapping, SENSORY_MAPPING_VERSION


def corpus():
    observations = {name: np.zeros((2, *shape), dtype=dtype)
                    for name, (shape, dtype) in OBS_SPEC.items()}
    masks = {name: np.zeros((2, *shape), dtype=dtype)
             for name, (shape, dtype) in MASK_SPEC.items()}
    observations['hand_len'][:] = 3
    observations['hand'][:, 0, 0] = 1
    observations['hand'][:, 1, 2] = 1
    observations['hand'][:, 2, 4] = 1
    observations['hand'][:, :3, 13] = 1
    observations['global'][:, 14] = 1
    masks['card_select_mask'][:, :3] = True
    masks['action_type_mask'][:, :2] = True
    return CalibrationCorpus(version=CALIBRATION_CORPUS_VERSION, simulator_version='fixture',
        environment_backend='fixture', generation_method='fixture', navigation_rule='fixture',
        navigation_seed=0, environment_seeds=(0,), sample_every=1, observations=observations,
        masks=masks, state_hashes=('a', 'b'), phase_labels=(1, 1), source_seed=(0, 0), source_decision=(0, 1))


def encoder(saturated=False):
    names = feature_names()
    indices = np.zeros((len(names), 1), dtype=np.int64)
    for name, index in [('hand:0:0', 1), ('hand:0:1', 2), ('hand:0:13', 3), ('hand:0:14', 4)]:
        indices[names.index(name)] = index
    if saturated:
        indices[names.index('global:13')] = 1
        indices[names.index('global:14')] = 2
    return FixedPlasticSensoryEncoder(SensoryMapping(version=SENSORY_MAPPING_VERSION,
        mapping_seed=0, neuron_count=10, feature_names=names, population_indices=indices,
        population_root_ids=100 + indices, available_alpn_indices=np.arange(10),
        available_alpn_root_ids=100 + np.arange(10)))


def fly(mode='readout'):
    motor = FixedMotorInterface(MotorMapping.contiguous_pools(np.arange(MOTOR_POOL_COUNT) + 1000,
                                                            mode='mbon_direct', pool_width=1))
    state = PlasticEdgeState.initialize(3, learners=2)
    state.efficacy[1] = 1.1
    sensory = encoder()
    roots = np.arange(MOTOR_POOL_COUNT + 2) + 1000
    seeds = []

    def process(observations, *, fly_seeds, efficacy):
        seeds.append(fly_seeds)
        rates = sensory.encode(observations).rates_hz[0]
        signal = float((rates[1] + rates[3] / 2) / 150) * efficacy[0, 0]
        rng = np.random.default_rng(fly_seeds[0])
        kc = rng.normal(0, .1, (1, 4)) + signal
        mbon = rng.normal(0, .1, (1, len(roots)))
        if mode == 'readout':
            mbon[0, motor.mapping.routing.head_routes['card'][0]] += signal
        else:
            mbon[0, -1] += signal
        return SimpleNamespace(kc_activity=kc, mbon_activity=mbon,
                               output_activity=mbon[:, :MOTOR_POOL_COUNT])
    processor = SimpleNamespace(process=process, encoder=sensory, kc_root_ids=np.arange(4),
        mbon_root_ids=roots, output_root_ids=motor.mapping.output_root_ids)
    return SimpleNamespace(motor=motor, processor=processor, plasticity=SimpleNamespace(state=state), seeds=seeds)


def test_single_card_edit_preserves_every_other_field_and_only_changes_one_category():
    obs, masks = corpus().row(0)
    for kind, columns in [('rank', [0, 1]), ('suit', [13, 14])]:
        altered, record = change_card(obs, 0, kind)
        assert record['changed_hand_columns'] == columns
        changed = np.argwhere(obs['hand'] != altered['hand']).tolist()
        assert changed == [[0, 0, columns[0]], [0, 0, columns[1]]]
        for field in obs.keys() - {'hand'}:
            np.testing.assert_array_equal(obs[field], altered[field])
        assert altered['hand'][0, 0, :13].sum() == 1
        assert altered['hand'][0, 0, 13:17].sum() == 1
    assert obs['hand'][0, 0, 0] == 1
    for flag in [23, 37]:
        hidden = {key: value.copy() for key, value in obs.items()}
        hidden['hand'][0, 0, flag] = 1
        with pytest.raises(ValueError, match='visible'):
            change_card(hidden, 0, 'rank')


def test_case_selection_is_spaced_visible_legal_and_not_outcome_based():
    source = corpus()
    source.observations['hand'][0, 0, 37] = 1
    source.masks['card_select_mask'][1, 2] = False
    cases = prepare_cases(source, states=2, cards_per_state=2)
    assert [(case['state_index'], case['slot']) for case in cases] == [(0, 1), (0, 2), (1, 0), (1, 1)]
    assert all(np.array_equal(case['masks']['action_type_mask'], source.row(case['state_index'])[1]['action_type_mask']) for case in cases)


def test_encoder_audit_identifies_complete_clipping_loss_separately_from_projection_alias():
    source = corpus()
    source.observations['global'][:, 13] = 1
    cases = prepare_cases(source, states=1, cards_per_state=1)
    clear, rates = sensory_effects(encoder(), cases)
    assert clear[0]['changes'][0]['retained_l1_fraction'] == 1
    assert clear[0]['changes'][0]['changed_input_neurons'] == 2
    masked, clipped = sensory_effects(encoder(saturated=True), cases)
    rank = masked[0]['changes'][0]
    assert rank['unclipped_l1_change_hz'] == 300
    assert rank['clipped_l1_change_hz'] == 0
    assert rank['clipping_erased_nonzero_change'] and not rank['projection_alias']
    np.testing.assert_array_equal(clipped[0, 0], clipped[0, 1])
    assert masked[0]['changes'][1]['changed_input_neurons'] > 0
    alias = encoder()
    names = feature_names()
    alias.mapping.population_indices[names.index('hand:0:1')] = 1
    aliased, _ = sensory_effects(alias, cases)
    assert aliased[0]['changes'][0]['projection_alias']
    assert not aliased[0]['changes'][0]['clipping_erased_nonzero_change']


def test_noise_metrics_are_signed_and_match_known_response_and_variance():
    baseline = np.array([[0., 0.], [2., 2.]])
    metric = response_metrics(baseline, baseline + 3)
    assert metric['paired_rms'] == 3
    assert metric['baseline_repeat_rms'] == 2
    assert metric['cross_repeat_effect_power'] == 9
    assert metric['trial_noise_power'] == 2
    assert metric['effect_to_trial_noise_power_ratio'] == 4.5
    opposing = response_metrics(np.zeros((2, 1)), np.array([[1.], [-1.]]))
    assert opposing['cross_repeat_effect_power'] == -1
    empty = response_metrics(np.empty((2, 0)), np.empty((2, 0)))
    assert empty['population_size'] == 0 and empty['paired_rms'] is None
    assert response_metrics(np.zeros((2, 1)), np.ones((2, 1)))['effect_to_repeat_rms_ratio'] is None
    with pytest.raises(ValueError, match='nonfinite'):
        response_metrics(baseline, baseline * np.nan)


def test_common_motor_offset_disappears_in_contrasts_and_illegal_cards_are_excluded():
    scores = np.array([1., 2., 3., 999.])
    legal = np.array([True, True, True, False])
    np.testing.assert_allclose(contrast_scores(scores, legal), [-1, 0, 1, 0])
    np.testing.assert_allclose(contrast_scores(scores + 7, legal), [-1, 0, 1, 0])


@pytest.mark.parametrize('mode', ['readout', 'unused'])
def test_pipeline_localizes_fixture_signal_and_preserves_checkpoint_state(mode, tmp_path):
    agent = fly(mode)
    cases = prepare_cases(corpus(), states=1, cards_per_state=1)
    result = measure_learner(agent, 1, cases, repeats=3, seed=20, output_dir=tmp_path, label='fixture')
    assert result['plastic_state_hashes_before'] == result['plastic_state_hashes_after']
    assert result['exact_repeat_checks'] == 6
    assert all(value == agent.seeds[0] for value in agent.seeds[:2])
    for weights in result['weights'].values():
        rows = weights['rows']
        assert rows[0]['neural_seeds'] == rows[1]['neural_seeds']
        assert all(row['populations']['kc']['paired_rms'] > 0 for row in rows)
        for row in rows:
            if mode == 'readout':
                assert row['populations']['card_contrasts']['cross_repeat_effect_power'] > 0
                assert row['populations']['mbon_unused_by_motor']['paired_rms'] == 0
            else:
                assert row['populations']['mbon_unused_by_motor']['paired_rms'] > 0
                assert row['populations']['card_contrasts']['paired_rms'] == 0
        with np.load(tmp_path/weights['activity_file']) as arrays:
            assert arrays['kc'].shape == (1, 3, 3, 4)
            if mode == 'unused':
                np.testing.assert_array_equal(arrays['card_contrasts'][0, 0], arrays['card_contrasts'][0, 1])
        summary = summarize(rows)
        assert summary['rank']['states'] == 1 and summary['rank']['cases'] == 1


def test_erased_input_control_rejects_downstream_responses_to_identical_drive():
    zero = response_metrics(np.zeros((2, 1)), np.zeros((2, 1)))
    result = {'weights':{'initial':{'rows':[{'state_index':0, 'slot':1, 'condition':'rank',
                                            'populations':{'kc':zero}}]}}}
    sensory = [{'state_index':0, 'slot':1, 'changes':[{'kind':'rank', 'identical_encoded_input':True}]}]
    assert verify_erased_input_controls(result, sensory) == 1
    zero['same_seed_identical_fraction'] = 0
    with pytest.raises(ValueError, match='identical encoded input'):
        verify_erased_input_controls(result, sensory)


def test_exact_repeat_and_frozen_state_failures_are_rejected(tmp_path):
    agent = fly()
    original = agent.processor.process
    def noisy(*args, **kwargs):
        result = original(*args, **kwargs)
        result.kc_activity += len(agent.seeds)
        return result
    agent.processor.process = noisy
    cases = prepare_cases(corpus(), states=1, cards_per_state=1)
    with pytest.raises(ValueError, match='exact neural'):
        measure_learner(agent, 0, cases, repeats=2, seed=10, output_dir=tmp_path, label='bad')
    agent = fly()
    original = agent.processor.process
    def mutating(*args, **kwargs):
        result = original(*args, **kwargs)
        agent.plasticity.state.eligibility[0, 0] += 1
        return result
    agent.processor.process = mutating
    with pytest.raises(ValueError, match='plastic state'):
        measure_learner(agent, 0, cases, repeats=2, seed=10, output_dir=tmp_path, label='bad')


def test_summary_weights_states_equally_when_they_have_different_slot_counts():
    metric = response_metrics(np.zeros((2, 1)), np.ones((2, 1)))
    rows = []
    for state, value in [(0, 1.), (0, 1.), (1, 4.)]:
        rows.append({'condition':'rank', 'state_index':state,
                     'populations':{stage:{**metric, 'paired_rms':value} for stage in
                                    ('kc','mbon','card_pool_neurons','mbon_outside_card_pools',
                                     'mbon_unused_by_motor','card_pool_rates','card_scores','card_contrasts')}})
    assert summarize(rows)['rank']['populations']['kc']['paired_rms'] == 2.5


@pytest.mark.parametrize('duration', [None, 250])
def test_cli_writes_auditable_arrays_and_rejects_overwrite(monkeypatch, tmp_path, duration):
    from flylatro.analysis import card_information_cli as cli
    from flylatro.fly.flywire_artifact import sha256_file
    from helpers import write_config
    source = corpus()
    corpus_path = tmp_path/'corpus.npz'
    source.save(corpus_path)
    config = write_config(tmp_path, corpus_path=corpus_path, overrides={
        'environment':{'num_envs':2}, 'fly':{'backend':'flywire', 'artifact_path':'fixture.npz', 'duration_ms':100}})
    checkpoint = tmp_path/'fixture.pkl'
    checkpoint.write_bytes(b'fixture')
    processors=[]
    def stack(config):
        agent=fly()
        agent.processor.duration_ms=100
        agent.processor.backend=SimpleNamespace(parameters=SimpleNamespace(dt_ms=.1))
        processors.append(agent.processor)
        original=agent.processor.process
        def process(*args,**kwargs):
            assert agent.processor.duration_ms==(duration or 100)
            return original(*args,**kwargs)
        agent.processor.process=process
        return SimpleNamespace(agent=agent, trainer=SimpleNamespace(agent=agent), components={'fixture':True})
    monkeypatch.setattr(cli, 'build_plastic_stack', stack)
    def load(path, trainer, **kwargs):
        assert kwargs == {'expected_components':{'fixture':True}, 'require_matching_exposure_budget':False}
        assert trainer.agent.processor.duration_ms==100
    monkeypatch.setattr(cli, 'load_plastic_checkpoint', load)
    monkeypatch.setattr(cli, 'load_plastic_manifest', lambda path: {'checkpoint_sha256':sha256_file(path)})
    output = tmp_path/'diagnostic'
    args = ['--config',str(config),'--checkpoint',str(checkpoint),'--output-dir',str(output),
            '--states','1','--cards-per-state','1','--repeats','2','--heavy']
    if duration:args.extend(['--analysis-duration-ms',str(duration)])
    assert cli.main(args) == 0
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'completed' and report['sources_unchanged']
    assert report['arms']['checkpoint']['analysis_window']['duration_ms']==(duration or 100)
    assert all(processor.duration_ms==100 for processor in processors)
    learners = report['arms']['checkpoint']['learners']
    assert [item['learner_index'] for item in learners] == [0, 1]
    for learner in learners:
        assert learner['plastic_state_unchanged']
        for weights in learner['weights'].values():
            assert sha256_file(output/weights['activity_file']) == weights['activity_sha256']
    with pytest.raises(FileExistsError): cli.main(args)
