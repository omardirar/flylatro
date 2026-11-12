"""Native diagnostic interventions must preserve feedback, controls and sources."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_training_bottlenecks import (
    best_visible_play,binary_probabilities,binary_probe,followup_probe,layout,
    mixed_pair_action,pair_probe,reference_followup,
)
from test_discard_studies import conditioned_toy,native_corpus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
import train_discard_play_curriculum as delayed


def test_layout_balances_reward_targets_and_frozen_pairs():
    ids,treatment,plastic=layout(3)
    assert ids==[0]*4+[1]*4+[2]*4
    assert treatment.tolist()==[0,0,1,1]*3
    assert plastic.tolist()==[True,False]*6
    with pytest.raises(ValueError):layout(0)


def test_binary_probability_matches_epsilon_softmax_and_is_shift_invariant(tmp_path):
    mapping=conditioned_toy(tmp_path).agent.motor.mapping
    scores=np.asarray([[0.,2.],[0.,0.]])
    p=binary_probabilities(scores,mapping)
    np.testing.assert_allclose(p.sum(axis=1),1)
    np.testing.assert_allclose(p,binary_probabilities(scores+100,mapping))
    assert p[0,1]>p[0,0] and p[1,0]==.5


@pytest.mark.parametrize('study',[binary_probe,pair_probe,followup_probe])
@pytest.mark.parametrize('torch_state',[False,True])
def test_studies_preserve_frozen_and_source_and_restrict_updates(native_corpus,tmp_path,study,torch_state):
    agent=conditioned_toy(tmp_path).agent
    if torch_state:
        from flylatro.fly.mushroom_body.state import TorchPlasticEdgeState
        agent.plasticity.state=TorchPlasticEdgeState.from_state_dict(agent.plasticity.state.state_dict(),device='cpu')
    source=plastic_hashes(agent.plasticity.state)
    original=delayed.followup
    report,state=study(agent,native_corpus,steps=8,replicates=1,repeats=1)
    assert delayed.followup is original
    assert report['source_state_unchanged'] and report['frozen_weights_exact'] and report['unrelated_weights_exact']
    assert report['efficacy_bound_hits']==0
    assert sum(report['actual_update_l1'])>0
    assert report['actual_update_l1'][1]==report['actual_update_l1'][3]==0
    np.testing.assert_array_equal(state['efficacy'][1::2],state['initial_efficacy'][1::2])
    assert not state['eligibility'].any() and not state['dopamine'].any()
    assert plastic_hashes(agent.plasticity.state)==source
    from assess_training_bottlenecks import verify_state
    path=tmp_path/'state.npz';np.savez(path,**state)
    assert verify_state(report,path)
    corrupted={k:v.copy() for k,v in state.items()};corrupted['efficacy'][1,0]+=.1
    np.savez(path,**corrupted)
    with pytest.raises(AssertionError):verify_state(report,path)
    if study is followup_probe:
        assert report['verified_native_followup_scores']>0
        assert report['training_pulse_sum'][2]>=0


def test_reference_uses_visible_plain_cards_and_restores_helper_on_error(native_corpus,tmp_path):
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([0])
    native_corpus.restore(env,0,native_corpus.indices('validation')[0])
    obs,_=env.observe();hand=obs['hand'][0,:8]
    chips,cards=best_visible_play(hand)
    assert chips>0 and 1<=len(cards)<=5
    reverse=np.arange(7,-1,-1)
    changed,selected=best_visible_play(hand[reverse])
    assert changed==chips
    from card_learning.audit import score
    plain=[(int(c[:13].argmax())+2,int(c[13:17].argmax())) for c in hand]
    assert score(plain,reverse[list(selected)])[0]==chips
    modified=hand.copy();modified[0,17]=0;modified[0,18]=1
    with pytest.raises(ValueError,match='plain'):best_visible_play(modified)
    original=delayed.followup
    with pytest.raises(RuntimeError):
        with reference_followup(conditioned_toy(tmp_path).agent,[True]):
            raise RuntimeError('controlled failure')
    assert delayed.followup is original


def test_diagnostic_pair_labels_cannot_change_exploration_training(native_corpus,tmp_path,monkeypatch):
    import run_training_bottlenecks as studies
    agent=conditioned_toy(tmp_path).agent
    report,state=pair_probe(agent,native_corpus,steps=4,replicates=1,repeats=1)
    monkeypatch.setattr(studies,'pair_slots',lambda obs,row=0:(0,1))
    other,changed=pair_probe(agent,native_corpus,steps=4,replicates=1,repeats=1)
    np.testing.assert_array_equal(state['efficacy'],changed['efficacy'])
    assert report['actual_update_l1']==other['actual_update_l1']


def test_uniform_sampler_factor_changes_membership_only(native_corpus,tmp_path):
    agent=conditioned_toy(tmp_path).agent
    env=BalatroSimAdapter(4,win_ante=1,allow_skip_blinds=False);env.reset([0]*4)
    for row in range(4):native_corpus.restore(env,row,native_corpus.indices('training')[0])
    obs,masks=env.observe();masks=delayed.forced(masks,0)
    neural=agent.processor.process(obs,fly_seeds=[5]*4,efficacy=np.repeat(agent.plasticity.state.initial_efficacy[:1],4,axis=0))
    for sampled in (False,True):
        actions=mixed_pair_action(agent.motor.base,241024,np.asarray([0.,0.,1.,1.]),neural,masks,[0]*4,[10]*4,sampled)
        assert np.all(actions['action_type']==0) and np.all(actions['n_cards']==2)
        for cards in actions['cards']:assert len(set(cards[:2]))==2 and np.all(cards[:2]<8)
        if not sampled:np.testing.assert_array_equal(actions['cards'][0],actions['cards'][2])


@pytest.mark.parametrize('study',[pair_probe,followup_probe])
def test_assessment_independently_replays_recorded_actions(native_corpus,tmp_path,study):
    from assess_training_bottlenecks import summarize,replay
    report,_=study(conditioned_toy(tmp_path).agent,native_corpus,steps=2,replicates=1,repeats=1)
    report.update(corpus=str(native_corpus.path),corpus_sha256=native_corpus.sha256)
    result=replay(report)
    assert result['native_actions']>0 and result['recorded_action_progress_and_nonterminal_verified']
    summary=summarize(report)
    assert 'greedy' in summary and 'training-sampled' in summary
    if study is followup_probe:
        assert summary['greedy']['1']['progress']['mean_before']>=summary['greedy']['0']['progress']['mean_before']
    broken=report['after']['greedy'][0]['progress'][0]
    report['after']['greedy'][0]['progress'][0]=broken+.1
    with pytest.raises(ValueError,match='replay differs'):replay(report)


def test_fresh_confirmation_is_frozen_reproducible_and_replayable(native_corpus,tmp_path):
    from confirm_subset_exploration import evaluate
    from assess_training_bottlenecks import replay
    agent=conditioned_toy(tmp_path).agent
    _,state=pair_probe(agent,native_corpus,steps=2,replicates=1,repeats=1)
    source=plastic_hashes(agent.plasticity.state)
    first=evaluate(agent,native_corpus,state['efficacy'],replicates=1,repeats=1)
    second=evaluate(agent,native_corpus,state['efficacy'],replicates=1,repeats=1)
    assert first==second and first['source_state_unchanged']
    assert replay(first)['native_actions']==4*len(native_corpus.indices('validation'))
    assert plastic_hashes(agent.plasticity.state)==source
    with pytest.raises(ValueError,match='shape'):evaluate(agent,native_corpus,state['efficacy'][:2],replicates=1)
