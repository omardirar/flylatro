from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest

from flylatro.analysis.card_window import integration_window, validate_pair, compare_windows
from flylatro.learning.config import payload_sha256, PlasticExperimentConfig


def test_window_changes_only_duration_and_restores_it_even_on_error():
    processor = SimpleNamespace(duration_ms=100, backend=SimpleNamespace(parameters=SimpleNamespace(dt_ms=.1)))
    with integration_window(processor, 250):
        assert processor.duration_ms == 250
    assert processor.duration_ms == 100
    with pytest.raises(RuntimeError):
        with integration_window(processor, 250):
            raise RuntimeError('fixture')
    assert processor.duration_ms == 100
    for duration in (0, -1, float('nan'), float('inf'), .001, 100.01):
        with pytest.raises(ValueError, match='simulation steps'):
            with integration_window(processor, duration): pass
    assert processor.duration_ms == 100


def report(duration):
    dynamics = {'decision_duration_ms':duration, 'backend_dynamics_sha256':'same', 'reset_fast_state_each_decision':True}
    metrics = {'kc':{'baseline_repeat_rms':2 if duration==100 else 1},
               'card_contrasts':{'baseline_repeat_rms':2 if duration==100 else 1,
                                'ratio_of_mean_effect_to_noise_power':.1 if duration==100 else .3}}
    weights = {'activity_file':'fixture.npz', 'activity_sha256':'replace',
               'summary':{kind:{'populations':deepcopy(metrics)} for kind in ('rank','suit')},
               'rows':[{'case_index':0, 'state_index':0, 'slot':0, 'condition':kind, 'neural_seeds':[1,2]}
                       for kind in ('rank','suit')]}
    learner = {'learner_index':0,'plastic_state_unchanged':True, 'plastic_state_hashes_before':{'efficacy':'same'},
               'plastic_state_hashes_after':{'efficacy':'same'}, 'population_root_ids':{'kc':[1,2]},
               'exact_repeat_checks':6, 'weights':{kind:deepcopy(weights) for kind in ('initial','trained')}}
    arm = {'config':'same', 'checkpoint':'same', 'manifest':{'experiment_config':{'safety':{
        'minimum_kc_active_fraction':.01,'minimum_motor_active_fraction':.01}}},'components':{'same':True},
        'analysis_window':{'duration_ms':duration, 'training_duration_ms':100, 'dynamics':dynamics,
            'dynamics_sha256':payload_sha256(dynamics), 'normalization_policy':'fixed','checkpoint_policy':'strict'},
        'learners':[learner]}
    return {'version':'fixture', 'status':'completed','learning_disabled':True,'sources_unchanged':True,
        'source_hashes':{'checkpoint':'same'}, 'corpus_sha256':'same', 'seed':1, 'repeats':2,
        'conditions':['original','rank','suit'], 'population_units':{'kc':'Hz'},
        'cases':[{'state_index':0,'slot':0,'legal_card_mask':[True,True]}],
        'sensory':{'mapping_sha256':'same','records':[1],'array_axes':['case','condition','alpn'],
                   'activity_file':'sensory.npz','activity_sha256':'replace'},
        'arms':{label:deepcopy(arm) for label in ('baseline','choice')}}


def test_comparison_allows_only_declared_duration_and_preserves_trial_pairing():
    first, second = report(100), report(250)
    validate_pair(first, second)
    for mutate in (
        lambda r:r.update(seed=2),
        lambda r:r.update(learning_disabled=False),
        lambda r:r['arms']['choice']['components'].update(motor_mapping='changed'),
        lambda r:r['arms']['choice']['analysis_window']['dynamics'].update(backend_dynamics_sha256='changed'),
        lambda r:r['arms']['choice']['learners'][0]['plastic_state_hashes_after'].update(efficacy='changed'),
        lambda r:r['arms']['choice']['learners'][0]['weights']['trained']['rows'][0].update(neural_seeds=[3,4]),
    ):
        changed = deepcopy(second)
        mutate(changed)
        with pytest.raises(ValueError): validate_pair(first, changed)


def materialize(path, value):
    from flylatro.fly.flywire_artifact import sha256_file
    path.mkdir()
    np.savez_compressed(path/'sensory.npz', drive=np.ones((1,3,2)))
    value['sensory']['activity_sha256'] = sha256_file(path/'sensory.npz')
    scores = np.array([[[[2.,1.],[2.,1.]], [[2.,1.],[2.,1.]], [[2.,1.],[2.,1.]]]])
    np.savez_compressed(path/'fixture.npz', kc=np.ones((1,3,2,2)),
                        card_pool_neurons=np.ones((1,3,2,2)), card_scores=scores)
    digest = sha256_file(path/'fixture.npz')
    for arm in value['arms'].values():
        for learner in arm['learners']:
            for weights in learner['weights'].values():weights['activity_sha256']=digest
    (path/'report.json').write_text(json.dumps(value))


def test_screen_requires_signal_improvement_not_just_reduced_noise(tmp_path):
    first, second = report(100), report(250)
    materialize(tmp_path/'first', first)
    materialize(tmp_path/'second', second)
    result = compare_windows(tmp_path/'first', tmp_path/'second')
    assert result['comparability']=='PASS' and result['predeclared_follow_up_signal']
    # Quieter responses with no stronger repeatable content signal must fail.
    for arm in second['arms'].values():
        for kind in ('rank','suit'):
            arm['learners'][0]['weights']['trained']['summary'][kind]['populations']['card_contrasts']['ratio_of_mean_effect_to_noise_power']=.01
    (tmp_path/'second'/'report.json').write_text(json.dumps(second))
    assert not compare_windows(tmp_path/'first',tmp_path/'second')['predeclared_follow_up_signal']
    (tmp_path/'second'/'fixture.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='activity hash'):
        compare_windows(tmp_path/'first',tmp_path/'second')


def test_preparation_pins_sources_commands_and_criterion_before_any_gpu_work(tmp_path, monkeypatch):
    from flylatro.analysis import card_window_pilot as study
    from flylatro.learning.choice_credit_pilot import prepare
    from test_card_information import corpus
    source = tmp_path/'source'
    prepare(PlasticExperimentConfig.load(Path('configs/exploratory-event-no-skip-weak-failure-v1.toml')),
        source, steps=10000, replicates=3, evaluation_episodes=256, evaluation_batch_size=4)
    (source/'assessment.json').write_text(json.dumps({'comparability':'PASS'}))
    for label in ('baseline','choice'):
        (source/label).mkdir()
        (source/label/'plastic-checkpoint-final.pkl').write_bytes(b'fixture')
    monkeypatch.setattr(study, 'load_corpus', lambda config:corpus())
    out=tmp_path/'study'
    declaration=study.prepare(source,out,durations=[100,250],states=2,cards_per_state=1,repeats=2)
    assert (out/'declaration.json').is_file() and not (out/'suite.json').exists()
    assert declaration['primary_arm']=='choice'
    assert '--analysis-duration-ms' in declaration['windows'][1]['command']
    assert declaration['sources']['choice']['checkpoint_sha256']
    study.check_sources(declaration)
    with pytest.raises(FileExistsError):study.prepare(source,out,durations=[100,250])
    with pytest.raises(ValueError, match='durations'):
        study.prepare(source,tmp_path/'bad',durations=[250,500])
    (source/'choice'/'plastic-checkpoint-final.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'):study.check_sources(declaration)


def test_reference_audit_compares_arrays_and_rejects_changed_original_response(tmp_path):
    from flylatro.analysis.card_window import audit_reference
    first, second=report(100),report(100)
    materialize(tmp_path/'first',first)
    materialize(tmp_path/'second',second)
    result=audit_reference(tmp_path/'first',tmp_path/'second')
    assert result['status']=='PASS' and result['neural_files_compared']==4
    second['seed']=99
    (tmp_path/'second'/'report.json').write_text(json.dumps(second))
    with pytest.raises(ValueError,match='seed'):audit_reference(tmp_path/'first',tmp_path/'second')
