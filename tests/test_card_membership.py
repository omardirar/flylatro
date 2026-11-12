"""Fixed-count experiments must isolate membership and preserve native feedback."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_card_membership import binary_probe,fixed_motor,matched_gain,native_layout,native_probe,native_trial
from test_discard_studies import conditioned_toy,native_corpus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.learning.credit import SelectedMotorCredit
from assess_training_bottlenecks import verify_state
import train_discard_play_curriculum as delayed


@pytest.mark.parametrize('torch_state',[False,True])
@pytest.mark.parametrize('study',[binary_probe,native_probe])
def test_membership_updates_exclude_counts_and_source(native_corpus,tmp_path,study,torch_state):
    agent=conditioned_toy(tmp_path).agent
    if torch_state:
        from flylatro.fly.mushroom_body.state import TorchPlasticEdgeState
        agent.plasticity.state=TorchPlasticEdgeState.from_state_dict(agent.plasticity.state.state_dict(),device='cpu')
    source=plastic_hashes(agent.plasticity.state)
    extra={'calibration_trials':4} if study is native_probe else {}
    r,state=study(agent,native_corpus,steps=6,replicates=1,repeats=2,**extra)
    assert r['frozen_weights_exact'] and r['unrelated_weights_exact'] and r['efficacy_bound_hits']==0
    assert sum(r['actual_update_l1'])>0
    assert plastic_hashes(agent.plasticity.state)==source
    assert not state['eligibility'].any() and not state['dopamine'].any()
    path=tmp_path/'state.npz';np.savez(path,**state)
    proxy={**r,'study':'followup'} if study is native_probe else r
    verify_state(proxy,path)
    if study is native_probe:
        assert r['actual_update_l1'][3]==0 and r['calibration']['gain']<=1
        assert r['calibration']['matched_expected_update_l1']==pytest.approx(r['calibration']['weak_expected_update_l1'])
        for row in r['after']['greedy']:
            assert row['discard_counts']==[2]*4
    else:
        p=agent.motor.exploration.probability
        for row in r['after']:
            assert all(x==pytest.approx(p/2) or x==pytest.approx(1-p/2) for x in row['rewarded_probability'])


def test_matching_uses_selected_eligibility_and_rejects_degenerate_feedback():
    row={'eligibility_mass':[2.]*4,'weak_pulse':[.1]*4,'reference_pulse':[.2]*4}
    result=matched_gain([row],.01)
    assert result['gain']==pytest.approx(.5)
    assert result['weak_expected_update_l1']==pytest.approx(.002)
    assert result['matched_expected_update_l1']==pytest.approx(.002)
    for invalid in ({**row,'reference_pulse':[0.]*4},{**row,'weak_pulse':[-1.]*4},
                    {**row,'eligibility_mass':[float('nan')]*4}):
        with pytest.raises(ValueError):matched_gain([invalid],.01)
    with pytest.raises(ValueError):matched_gain([],.01)


def test_fixed_motor_preserves_source_subset_sampler_and_forced_counts(native_corpus,tmp_path):
    agent=conditioned_toy(tmp_path).agent;motor=fixed_motor(agent,1,2)
    env=BalatroSimAdapter(2,win_ante=1,allow_skip_blinds=False);env.reset([0]*2)
    for lane in range(2):native_corpus.restore(env,lane,native_corpus.indices('training')[0])
    obs,masks=env.observe();masks=delayed.forced(masks,1)
    neural=agent.processor.process(obs,fly_seeds=[4,4],efficacy=np.repeat(agent.plasticity.state.initial_efficacy[:1],2,axis=0))
    action=motor.decode(neural.output_activity,masks,deterministic=False,learner_ids=[0,0],decision_ids=[7,7])
    assert action['n_cards'].tolist()==[2,2] and action['action_type'].tolist()==[1,1]
    np.testing.assert_array_equal(action['cards'][0],action['cards'][1])
    selected=motor.selected_pool_mask(action,masks)
    pools=set(agent.motor.mapping.routing.head_routes['discard_card'])
    assert set(np.flatnonzero(selected[0])).issubset(pools)
    assert selected.sum(axis=1).tolist()==[2,2]


def test_trace_cut_eliminates_updates_and_gain_cannot_leak_between_lanes(native_corpus,tmp_path):
    agent=conditioned_toy(tmp_path).agent;ids,role,_=native_layout(1);motor=fixed_motor(agent,1,2)
    def run(gain,cut=False):
        local=delayed.local_rule(agent,2);credit=SelectedMotorCredit(motor,local)
        env=BalatroSimAdapter(4,win_ante=1,allow_skip_blinds=False);env.reset([0]*4)
        alt=BalatroSimAdapter(4,win_ante=1,allow_skip_blinds=False);alt.reset([0]*4)
        row=native_trial(agent,native_corpus,env,alt,local,motor,credit,ids,role,
            [native_corpus.indices('training')[0]]*4,('test',0),sampled=True,learn=True,matched_gain=gain,trace_cut=cut)
        return row,local.state.state_dict()
    a,first=run(.1);b,second=run(.9);cut,zero=run(.1,True)
    assert sum(a['update_l1'])>0 and cut['update_l1']==[0]*4
    np.testing.assert_array_equal(zero['efficacy'],zero['initial_efficacy'])
    np.testing.assert_array_equal(first['efficacy'][[0,1,3]],second['efficacy'][[0,1,3]])
    assert np.any(first['efficacy'][2]!=second['efficacy'][2])
    assert a['weak_progress']==b['weak_progress'] and a['reference_progress']==b['reference_progress']


def test_native_assessment_verifies_shared_control_and_both_play_branches(native_corpus,tmp_path):
    from assess_card_membership import replay,summarize
    r,_=native_probe(conditioned_toy(tmp_path).agent,native_corpus,steps=2,replicates=1,repeats=1,calibration_trials=2)
    r.update(corpus=str(native_corpus.path),corpus_sha256=native_corpus.sha256)
    summary=summarize(r)
    assert set(summary['greedy'])=={'neural-native','reference-native','reference-matched'}
    assert replay(r)['native_actions']>0
    r['after']['greedy'][0]['reference_progress'][0]+=.1
    with pytest.raises(ValueError,match='PLAY replay differs'):replay(r)


def test_confirmation_keeps_every_state_field_frozen_and_replays(native_corpus,tmp_path):
    from confirm_card_membership import evaluate
    from assess_card_membership import replay
    agent=conditioned_toy(tmp_path).agent
    _,state=native_probe(agent,native_corpus,steps=2,replicates=1,repeats=1,calibration_trials=2)
    source=plastic_hashes(agent.plasticity.state);saved={k:v.copy() for k,v in state.items()}
    a=evaluate(agent,native_corpus,state,replicates=1,repeats=1)
    b=evaluate(agent,native_corpus,state,replicates=1,repeats=1)
    assert a==b and a['source_state_unchanged']
    assert replay(a)['native_actions']==len(native_corpus.indices('validation'))*4*3
    assert plastic_hashes(agent.plasticity.state)==source
    for key in state:np.testing.assert_array_equal(state[key],saved[key])


def test_activity_audit_identifies_silent_post_without_changing_source(native_corpus,tmp_path,monkeypatch):
    from dataclasses import replace
    from audit_card_membership_activity import audit
    agent=conditioned_toy(tmp_path).agent
    r,_=binary_probe(agent,native_corpus,steps=2,replicates=1,repeats=1)
    process=agent.processor.process
    def quiet(*args,**kwargs):
        neural=process(*args,**kwargs);post=neural.edge_post_activity.copy()
        output=neural.output_activity.copy();post[2]=0;output[2]=0
        return replace(neural,edge_post_activity=post,output_activity=output)
    monkeypatch.setattr(agent.processor,'process',quiet)
    before=plastic_hashes(agent.plasticity.state)
    result=audit(agent,native_corpus,r,repeats=2)
    assert result['summary']['zero_eligibility_fraction'][2]==1
    assert result['summary']['update_l1'][2]==0
    assert result['summary']['selected_post_active_fraction'][2]==0
    assert plastic_hashes(agent.plasticity.state)==before


def test_coactivity_coverage_matches_actual_fresh_rule_and_preserves_state(tmp_path):
    from audit_motor_learning_opportunities import aggregate,audit
    from flylatro.env.array_mock import MockArrayBalatroEnv
    agent=conditioned_toy(tmp_path).agent
    obs,masks=MockArrayBalatroEnv(4).reset([1,2,3,4])
    source=plastic_hashes(agent.plasticity.state)
    r,arrays=audit(agent,obs,masks,repeats=1,batch_size=4)
    assert r['source_state_unchanged'] and r['roles']
    assert arrays['coactivity'].shape[:2]==(4,1)
    assert plastic_hashes(agent.plasticity.state)==source
    local=delayed.local_rule(agent,2)
    neural=agent.processor.process(obs,fly_seeds=[7]*4,efficacy=local.state.efficacy)
    inverse=np.unique(local.topology.post_root_ids,return_inverse=True)[1]
    mass,_,_=aggregate(neural.edge_pre_activity,neural.edge_post_activity,inverse,len(np.unique(inverse)),local.config)
    local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=np.ones_like(neural.edge_pre_activity,dtype=bool))
    np.testing.assert_allclose(mass.sum(axis=1),local.state.eligibility.sum(axis=1))
