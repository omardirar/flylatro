"""Native puzzle provenance, hidden futures, credit, controls and transfer."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from strategy_puzzles import build, PuzzleCorpus, Reference, action, resample_future, resources
from discard_strategy import StrategyStudy, StrategyMotor, decay_helper_trace
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.mushroom_body.state import state_numpy
from test_discard_studies import conditioned_toy


@pytest.fixture(scope='module')
def puzzles(tmp_path_factory):
    return build(tmp_path_factory.mktemp('strategy-data')/'puzzles.json',per_suit=1)


def test_puzzles_use_real_prefixes_balanced_splits_and_equal_counts(puzzles):
    assert len(puzzles.cases)==24
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False)
    for index,c in enumerate(puzzles.cases):
        env.reset([c['seed']])
        for entry in c['prefix']:env.step(action(entry['kind'],entry['cards']))
        obs,_=env.observe()
        assert env.state_hash(obs,0)==c['state_hash']
        assert resources(obs)==(1,1)
        assert all(len(set(s))==4 for s in c['candidates'])
        assert (c['visible_best_chips']>=c['remaining_target']-.01)==(c['family']=='play-now')
    assert set(puzzles.indices('training')).isdisjoint(puzzles.indices('validation'))
    assert set(puzzles.indices('fresh')).isdisjoint(puzzles.indices('validation'))


def test_hidden_resampling_keeps_every_observation_and_mask_but_changes_future(puzzles):
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([0])
    seen=set()
    for seed in range(12):
        puzzles.restore(env,0,0);before,masks=env.observe()
        resample_future(env,0,seed)
        after,new_masks=env.observe()
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        for key in masks:np.testing.assert_array_equal(masks[key],new_masks[key])
        result=env.step(action(1,puzzles.cases[0]['candidates'][0]))
        seen.add(result.observations['hand'].tobytes())
    assert len(seen)>1


def test_reference_evaluates_future_outcomes_and_has_strategy_headroom(puzzles):
    reference=Reference(puzzles);best=[]
    for index,c in enumerate(puzzles.cases):
        assert reference.outcome(index,None,('test',0))['cleared']==(c['family']=='play-now')
        q=[float(c['family']=='play-now')]
        for candidate in c['candidates']:
            q.append(np.mean([reference.outcome(index,candidate,('test',i))['cleared'] for i in range(32)]))
        best.append(max(q))
    assert np.mean(best)>.65


def test_resampled_flush_draws_agree_with_without_replacement_odds(puzzles):
    from math import comb
    index=next(i for i,c in enumerate(puzzles.cases) if c['family']=='draw-needed')
    case=puzzles.cases[index];env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([0])
    puzzles.restore(env,0,index);obs,_=env.observe()
    suited=np.flatnonzero(obs['hand'][0,:8,13:17].argmax(axis=1)==case['suit'])
    discard=[i for i in range(8) if i not in suited]
    total=round(float(obs['drawpile_counts'].sum()))
    outs=round(float(obs['drawpile_counts'][0].reshape(13,4)[:,case['suit']].sum()))
    expected=1-comb(total-outs,4)/comb(total,4)
    hit=0
    for seed in range(512):
        puzzles.restore(env,0,index);resample_future(env,0,seed)
        result=env.step(action(1,discard))
        hit+=np.count_nonzero(result.observations['hand'][0,:8,13:17].argmax(axis=1)==case['suit'])>=5
    assert abs(hit/512-expected)<.07


@pytest.mark.parametrize('torch_state',[False,True])
def test_helper_trace_advances_discard_only(torch_state):
    from local_normalisation import NormalisedThreeFactorPlasticity
    from flylatro.fly.mushroom_body.state import PlasticEdgeState,TorchPlasticEdgeState
    from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
    from flylatro.fly.mushroom_body.plasticity import PlasticityConfig
    topology=PlasticEdgeTopology.synthetic(np.array([0,1]),np.array([2,2]),np.ones(2,dtype=np.float32))
    state=(TorchPlasticEdgeState.initialize(2,learners=2) if torch_state else PlasticEdgeState.initialize(2,learners=2))
    local=NormalisedThreeFactorPlasticity(topology,state,PlasticityConfig(eligibility_decay=.9))
    local.state.eligibility[:]=1
    decay_helper_trace(local,[False,True])
    np.testing.assert_array_equal(state_numpy(local.state.eligibility)[0],[1,1])
    np.testing.assert_allclose(state_numpy(local.state.eligibility)[1],[.9,.9])
    assert state_numpy(local.state.decision_count).tolist()==[0,1]


def test_credit_excludes_helper_play_cards_and_fixed_counts(puzzles,tmp_path):
    stack=conditioned_toy(tmp_path);study=StrategyStudy(stack,puzzles,replicates=1)
    obs,masks=study.env.observe()
    for row in range(3):puzzles.restore(study.env,row,0)
    _,masks=study.env.observe()
    selected=action(0,batch=3);selected['action_type'][1]=1
    selected['n_cards'][1]=4;selected['cards'][1,:4]=[0,1,2,3]
    pools=study.motor.selected_pool_mask(selected,masks)
    mapping=study.motor.mapping
    assert pools[0].sum()==1 and pools[1].sum()==5
    forbidden=list(mapping.routing.head_routes['card'])+list(mapping.routing.head_routes['card_count'])+list(mapping.routing.head_routes['discard_card_count'])
    assert not pools[:,[p for p in forbidden if p>=0]].any()


def test_choices_cannot_see_future_labels_or_padding(puzzles,tmp_path):
    stack=conditioned_toy(tmp_path);study=StrategyStudy(stack,puzzles,replicates=1)
    a=study.choose([0]*3,('choice',0),free_cards=True)
    for row in range(3):resample_future(study.env,row,123+row)
    for c in puzzles.cases:c['family']='diagnostic-label-changed'
    try:b=study.choose([0]*3,('choice',0),free_cards=True)
    finally:
        for c in puzzles.cases:c['family']='play-now' if c['visible_best_chips']>=c['remaining_target']-.01 else 'draw-needed'
    for key in a[0]:np.testing.assert_array_equal(a[0][key],b[0][key])
    for choice in (a,b):
        for lane in range(3):
            if choice[0]['action_type'][lane]==1:assert max(choice[0]['cards'][lane,:4])<8


def test_strategy_resume_preserves_outcomes_and_removes_torn_tail(puzzles,tmp_path):
    a=StrategyStudy(conditioned_toy(tmp_path),puzzles,replicates=1)
    (tmp_path/'declaration.json').write_text('{"test":"strategy-resume"}')
    log=tmp_path/'training.jsonl'
    with log.open('wb') as f:
        for _ in range(5):f.write((json.dumps(a.train_step())+'\n').encode())
    a.save(tmp_path,log.stat().st_size)
    expected=[a.train_step() for _ in range(5)]
    with log.open('ab') as f:f.write(b'{"torn')
    second=tmp_path/'second';second.mkdir()
    b=StrategyStudy(conditioned_toy(second),puzzles,replicates=1);b.restore(tmp_path)
    actual=[b.train_step() for _ in range(5)]
    assert actual==expected
    for name,value in a.local.state.state_dict().items():np.testing.assert_array_equal(value,b.local.state.state_dict()[name])
    assert len(log.read_text().splitlines())==5
    assert b.health()['frozen_exact'] and b.health()['unrelated_exact']


def test_mixed_play_discard_branches_and_own_play_transfer(puzzles,tmp_path,monkeypatch):
    from strategy_transfer import fly_transfer
    study=StrategyStudy(conditioned_toy(tmp_path),puzzles,replicates=1)
    original=study.motor.decode_strategy
    def mixed(activity,masks,candidates,**kwargs):
        actions,choices,forced=original(activity,masks,candidates,**kwargs)
        actions['action_type'][:]=[0,1,1];actions['n_cards'][:]=[0,4,4]
        actions['cards'][:]=-1
        actions['cards'][1,:4]=candidates[1][0]
        actions['cards'][2,:4]=candidates[2][1]
        return actions,[0,1,2],forced
    monkeypatch.setattr(study.motor,'decode_strategy',mixed)
    row=study.train_step()
    assert row['choices']==[0,1,2]
    np.testing.assert_array_equal(state_numpy(study.local.state.decision_count),[1,2,2])
    result=fly_transfer(study,draws=2)
    assert result['learning_disabled']
    assert all(r['discarded']==[False,True,True] for r in result['trained']['rows'])
    assert study.health()['frozen_exact']


def test_full_native_cpu_pilot_audit_and_corruption_rejection(tmp_path):
    from flylatro.learning.reinforcement import exploratory_condition
    from flylatro.learning.protocol_materialize import render_toml
    from flylatro.learning.config import PlasticExperimentConfig
    from run_discard_strategy import prepare,execute
    from assess_discard_strategy import assess
    model=conditioned_toy(tmp_path)
    base=PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config=replace(base,environment=replace(base.environment,allow_skip_blinds=False),
        fly=replace(base.fly,motor_mapping_path=str(tmp_path/'motor.json'),motor_pool_width=1),
        plasticity=replace(base.plasticity,version='three-factor-choice-gated-v1'),
        reinforcement=exploratory_condition('exploratory-weak-failure-v1'))
    document=asdict(config);document.pop('source_path')
    path=tmp_path/'config.toml';path.write_text(render_toml(document))
    output=tmp_path/'pilot';prepare(path,output,steps=5,replicates=1,per_suit=1,draws=4,repeats=1)
    result=execute(output)
    assert result['assisted'] and result['health']['frozen_exact']
    report=assess(output)
    assert report['counts']['training_records']==15 and report['native_rewards_verified']
    log=output/'training.jsonl';rows=[json.loads(l) for l in log.read_text().splitlines()]
    rows[0]['cleared'][0]=not rows[0]['cleared'][0]
    log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='outcome/reinforcement mismatch'):assess(output)
    rows[0]['cleared'][0]=not rows[0]['cleared'][0]
    rows[0]['helper_play_cards_native'][0]=[(rows[0]['helper_play_cards_native'][0][0]+1)%8]
    log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='reference PLAY differs'):assess(output)
