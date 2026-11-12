from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.motor_normalization import (
    assert_only_normalization_changed, assess, evaluate_cached, fit_fixed_normalization,
    heldout_cases, membership_hash, rescore,
)
from flylatro.analysis.corpus import CalibrationCorpus, CALIBRATION_CORPUS_VERSION
from flylatro.env.upstream_contract import OBS_SPEC, MASK_SPEC
from flylatro.interface.motor import FixedMotorInterface, MotorMapping, MOTOR_POOL_COUNT


def mapping():
    return MotorMapping.contiguous_pools(np.arange(MOTOR_POOL_COUNT),mode='mbon_direct',pool_width=1)


def corpus(prefix, count=4, seeds=(1,)):
    obs={k:np.zeros((count,*shape),dtype=dtype) for k,(shape,dtype) in OBS_SPEC.items()}
    masks={k:np.zeros((count,*shape),dtype=dtype) for k,(shape,dtype) in MASK_SPEC.items()}
    obs['hand_len'][:]=8
    for slot in range(8):obs['hand'][:,slot,slot]=1
    obs['hand'][:,:8,13]=1
    obs['global'][:,14]=1
    masks['action_type_mask'][:,:2]=True
    masks['card_select_mask'][:,:8]=True
    return CalibrationCorpus(version=CALIBRATION_CORPUS_VERSION,simulator_version='fixture',
        environment_backend='fixture',generation_method='fixture',navigation_rule='fixture',navigation_seed=1,
        environment_seeds=seeds,sample_every=1,observations=obs,masks=masks,
        state_hashes=tuple(f'{prefix}-state-{i}' for i in range(count)),phase_labels=(1,)*count,
        source_seed=(seeds[0],)*count,source_decision=tuple(range(count)),
        source_run_seed=tuple(f'{prefix}-run-{i}' for i in range(count)),
        source_episode=(0,)*count,source_collection_index=tuple(range(count)))


def test_fit_preserves_every_pool_and_uses_existing_fixed_median_iqr_estimator():
    source=mapping()
    rates=np.arange(8*MOTOR_POOL_COUNT,dtype=float).reshape(8,MOTOR_POOL_COUNT)
    candidate=fit_fixed_normalization(source,rates,provenance={'corpus':'fit-only'})
    assert membership_hash(source)==membership_hash(candidate)
    assert source.sha256!=candidate.sha256
    np.testing.assert_array_equal(candidate.normalization.baseline,np.median(rates,axis=0))
    expected=np.maximum((np.quantile(rates,.75,axis=0)-np.quantile(rates,.25,axis=0))/1.349,.5)
    np.testing.assert_array_equal(candidate.normalization.scale,expected)
    assert candidate.calibration_stats['reward_used'] is False
    assert candidate.calibration_stats['unit_initial_weights_only'] is True
    assert_only_normalization_changed(source,candidate)
    changed=replace(candidate,pool_indices=tuple(reversed(candidate.pool_indices)))
    with pytest.raises(ValueError,match='membership'):assert_only_normalization_changed(source,changed)
    with pytest.raises(ValueError,match='nonnegative'):
        fit_fixed_normalization(source,rates*np.nan,provenance={})


def test_reserved_cases_exclude_calibration_hashes_runs_seeds_and_probe_sixth_slot():
    fit=corpus('fit',2)
    test=corpus('test',5,seeds=(2,))
    # These rows must be excluded even when the other identifier differs.
    test=replace(test,state_hashes=(fit.state_hashes[0],*test.state_hashes[1:]),
                 source_run_seed=(test.source_run_seed[0],fit.source_run_seed[0],*test.source_run_seed[2:]))
    cases=heldout_cases(fit,test,states=2)
    assert {case['state_index'] for case in cases}=={2,4}
    assert {case['slot'] for case in cases}=={0,5,7}
    assert not {case['state_hash'] for case in cases}&set(fit.state_hashes)
    with pytest.raises(ValueError,match='root seeds overlap'):
        heldout_cases(fit,replace(test,environment_seeds=(1,)),states=2)
    with pytest.raises(ValueError,match='found'):
        heldout_cases(fit,test,states=4)


def test_cached_rescoring_exactly_matches_existing_decoder_and_never_needs_neural_calls():
    source=mapping()
    generator=np.random.default_rng(102)
    output=generator.uniform(0,100,(24,len(source.output_root_ids)))
    interface=FixedMotorInterface(source)
    rates=interface.pool_activity(output)
    candidate=fit_fixed_normalization(source,rates,provenance={})
    card_rates=rates[:,list(source.routing.head_routes['card'])]
    np.testing.assert_array_equal(rescore(card_rates,source),interface.head_scores(output)['card'])
    np.testing.assert_array_equal(rescore(card_rates,candidate),FixedMotorInterface(candidate).head_scores(output)['card'])


def result(old_bias=.95,new_bias=.3,new_rank=.3,new_suit=.3):
    def side(bias,rank,suit):
        return {'preference':{'dominant_slot_fraction':bias},'summary':{
            kind:{'populations':{'card_contrasts':{'ratio_of_mean_effect_to_noise_power':value}}}
            for kind,value in [('rank',rank),('suit',suit)]}}
    comparison={'old':side(old_bias,.1,.1),'fresh':side(new_bias,new_rank,new_suit),
                'health':{stage:{'minimum_active_fraction':.5,'maximum_high_rate_fraction':0}
                          for stage in ('kc','card_pool_neurons')}}
    return {'learner_index':0,'plastic_state_unchanged':True,'plastic_state_hashes_before':{'efficacy':'same'},
            'plastic_state_hashes_after':{'efficacy':'same'},
            'normalization_comparison':{'initial':deepcopy(comparison),'trained':comparison}}


def test_bias_relief_alone_does_not_pass_content_signal_screen():
    assert assess([result()])['predeclared_follow_up_signal']
    assert not assess([result(new_suit=-.1)])['predeclared_follow_up_signal']
    assert not assess([result(new_bias=.8)])['predeclared_follow_up_signal']
    unhealthy=result()
    unhealthy['normalization_comparison']['trained']['health']['kc']['minimum_active_fraction']=0
    assert not assess([unhealthy])['predeclared_follow_up_signal']


def test_cli_fits_only_initial_weights_and_holds_neural_samples_fixed(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from flylatro.analysis import motor_normalization_cli as cli
    from flylatro.fly.flywire_artifact import sha256_file
    from flylatro.learning.config import PlasticExperimentConfig
    from test_card_information import fly
    from helpers import write_config
    from flylatro.seeds import SeedPlan
    fit=corpus('fit',2)
    test=corpus('test',4,seeds=SeedPlan().seeds('validation',8,offset=10000))
    fit_path=tmp_path/'fit.npz';fit.save(fit_path)
    agent=fly()
    source_motor=tmp_path/'source-motor.json';agent.motor.mapping.save(source_motor)
    path=write_config(tmp_path,corpus_path=fit_path,overrides={
        'environment':{'backend':'balatro_sim','num_envs':2,'allow_skip_blinds':False},
        'reinforcement':{'run_failure_pulse':.01},'plasticity':{'version':'three-factor-choice-gated-v1'},
        'fly':{'backend':'flywire','duration_ms':100,'artifact_path':'fixture.npz','motor_mapping_path':str(source_motor)}})
    checkpoint=tmp_path/'checkpoint.pkl';checkpoint.write_bytes(b'fixture')
    agent.processor.duration_ms=100
    agent.processor.backend=SimpleNamespace(parameters=SimpleNamespace(dt_ms=.1))
    original=agent.processor.process
    calls=[]
    def process(*args,**kwargs):
        assert agent.processor.duration_ms==250
        calls.append(np.asarray(kwargs['efficacy']).copy())
        values=original(*args,**kwargs)
        values.kc_activity=np.maximum(values.kc_activity,0)
        values.mbon_activity=np.maximum(values.mbon_activity,0)
        values.output_activity=values.mbon_activity[:,:MOTOR_POOL_COUNT]
        return values
    agent.processor.process=process
    monkeypatch.setattr(cli,'build_plastic_environment',lambda config:object())
    monkeypatch.setattr(cli,'build_calibration_corpus',lambda *args,**kwargs:test)
    monkeypatch.setattr(cli,'build_plastic_stack',lambda config:SimpleNamespace(agent=agent,trainer=object(),components={'fixture':True}))
    def load(*args,**kwargs):
        assert agent.processor.duration_ms==100
        assert kwargs['expected_components']=={'fixture':True}
    monkeypatch.setattr(cli,'load_plastic_checkpoint',load)
    monkeypatch.setattr(cli,'load_plastic_manifest',lambda path:{'fixture':True})
    output=tmp_path/'study'
    assert cli.main(['--config',str(path),'--checkpoint',str(checkpoint),'--output-dir',str(output),
                     '--states','2','--repeats','2','--fit-repeats','2','--heavy'])==0
    r=json.loads((output/'report.json').read_text())
    d=json.loads((output/'declaration.json').read_text())
    assert r['status']=='completed' and r['pool_memberships_unchanged'] and r['plastic_state_unchanged']
    assert agent.processor.duration_ms==100
    # Fit runs 2 states x 2 repeats, plus an exact duplicate at the first seed.
    assert all(np.all(call==1) for call in calls[:6])
    assert not set(d['fit_neural_seeds'])&set(d['test_neural_seeds'])
    assert not d['final_test_seeds_used']
    assert len(r['learners'])==2
    for learner in r['learners']:
        for entry in learner['normalization_comparison'].values():
            assert sha256_file(output/entry['derived_file'])==entry['derived_sha256']
    assert MotorMapping.load(output/'fresh-motor.json').pool_indices==agent.motor.mapping.pool_indices
    # Cached-score corruption is rejected before any reinterpretation.
    target=r['learners'][0]['weights']['initial']
    (output/target['activity_file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='activity hash'):
        evaluate_cached(output,r['learners'][0],agent.motor.mapping,MotorMapping.load(output/'fresh-motor.json'),
                        heldout_cases(fit,test,states=2))
