"""Card-output learning controls and fixed-count native DISCARD capacity.

Independent research entry point; source model/checkpoints remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import array
from card_learning.audit import score
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack,load_condition
from card_learning.motor import CardSubsetMotor
from card_learning.pairs import PairCorpus,build_corpus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed
from run_training_bottlenecks import (
    best_visible_play,check_frozen,check_result,head_edges,layout,
)
import train_discard_play_curriculum as delayed


VERSION='fixed-card-membership-v1'


class FixedCountBase:
    """Force type/count, retaining actual neural membership scores and credit."""
    def __init__(self,base,action_type,count):
        if action_type not in (0,1) or not 1<=count<=5:raise ValueError('valid card type/count required')
        self.base,self.action_type,self.count=base,action_type,count

    def __getattr__(self,name):return getattr(self.base,name)

    def decode(self,activity,masks,**kwargs):
        head,_=self.mapping.routing.card_heads(self.action_type)
        scores=self.base.head_scores(activity)[head].copy()
        if not np.isfinite(scores[masks['card_select_mask']]).all():raise ValueError('nonfinite legal card scores')
        scores[~masks['card_select_mask']]=-np.inf
        if np.any(masks['card_select_mask'].sum(axis=1)<self.count):raise ValueError('insufficient legal cards')
        action=empty_action_batch(len(scores));action['action_type'][:]=self.action_type;action['n_cards'][:]=self.count
        action['cards'][:,:self.count]=np.argsort(-scores,axis=1,kind='stable')[:,:self.count]
        return action

    def selected_pool_mask(self,actions,masks):
        if np.any(actions['action_type']!=self.action_type) or np.any(actions['n_cards']!=self.count):
            raise ValueError('forced action/count differs')
        selected=np.zeros((len(actions['action_type']),self.mapping.routing.pool_count),dtype=bool)
        head,_=self.mapping.routing.card_heads(self.action_type)
        for row in range(len(selected)):
            cards=actions['cards'][row,:self.count]
            selected[row,np.asarray(self.mapping.routing.head_routes[head])[cards]]=True
        return selected


def fixed_motor(agent,action_type,count):
    return CardSubsetMotor(FixedCountBase(agent.motor.base,action_type,count),agent.motor.exploration)


def binary_probe(agent,corpus,*,steps=3000,replicates=3,repeats=128,
                 action_type=1,reward_pulse=.05,progress=None,log=None):
    if min(steps,repeats)<1 or not 0<reward_pulse<=2:raise ValueError('positive budgets and bounded pulse required')
    source=plastic_hashes(agent.plasticity.state)
    ids,targets,plastic=layout(replicates);n=len(ids)
    local=delayed.local_rule(agent,2*replicates);motor=fixed_motor(agent,action_type,1)
    credit=SelectedMotorCredit(motor,local)
    head,_=agent.motor.mapping.routing.card_heads(action_type);allowed=head_edges(agent,[head])
    env=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);env.reset([0]*n)
    training=corpus.indices('training');cases=[training[r*len(training)//replicates] for r in range(replicates)]
    # Different fixed slot pairs expose more than one anatomical output pair.
    slot_pairs=[(2*(r%4),2*(r%4)+1) for r in range(replicates)]
    for lane,r in enumerate(ids):corpus.restore(env,lane,cases[r])
    obs,masks=env.observe();masks=delayed.forced(masks,action_type)
    masks['card_select_mask'][:]=False
    for lane,r in enumerate(ids):masks['card_select_mask'][lane,list(slot_pairs[r])]=True
    target_slots=np.asarray([slot_pairs[r][targets[lane]] for lane,r in enumerate(ids)])
    exploration=motor.exploration.probability
    def evaluate():
        before=plastic_hashes(local.state);rows=[]
        for repeat in range(repeats):
            seeds=[derive_seed(VERSION,'binary',action_type,'evaluation',r,repeat) for r in ids]
            neural=agent.processor.process(obs,fly_seeds=seeds,efficacy=local.state.efficacy)
            greedy=motor.decode(neural.output_activity,masks,deterministic=True)
            sampled=motor.decode(neural.output_activity,masks,deterministic=False,
                learner_ids=ids,decision_ids=[derive_seed(VERSION,'binary-choice',action_type,repeat)]*n)
            correct=greedy['cards'][:,0]==target_slots
            # Source membership is greedy + uniform subset, not softmax.
            probability=(1-exploration)*correct+exploration/2
            rows.append({'rewarded_probability':probability.tolist(),'greedy_correct':correct.tolist(),
                         'sampled_correct':(sampled['cards'][:,0]==target_slots).tolist()})
        if plastic_hashes(local.state)!=before:raise ValueError('card evaluation changed state')
        return rows
    before=evaluate();hits=np.zeros(n);updates=np.zeros(n)
    for step in range(steps):
        seeds=[derive_seed(VERSION,'binary',action_type,'training',r,step) for r in ids]
        neural=agent.processor.process(obs,fly_seeds=seeds,efficacy=local.state.efficacy)
        action=motor.decode(neural.output_activity,masks,deterministic=False,learner_ids=ids,decision_ids=[step]*n)
        hit=action['cards'][:,0]==target_slots
        local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=credit.mask(action,masks))
        events=local.apply_dopamine((hit&plastic).astype(np.float32)*reward_pulse,0)
        updates += [e.absolute_change for e in events];hits+=hit
        local.state.reset_fast_traces(eligibility=True)
        if log:log({'step':step,'chosen_slots':action['cards'][:,0].tolist(),'rewarded':hit.tolist()})
        if progress and ((step+1)%250==0 or step+1==steps):
            progress({'phase':'training','step':step+1,'reward_frequency':(hits/(step+1)).tolist()})
    after=evaluate();check_frozen(before,after,plastic)
    return {'version':VERSION,'study':'binary','subtype':'card-membership','head':head,
        'steps':steps,'plastic':plastic.tolist(),'replicate_ids':ids,'targets':targets.tolist(),
        'slot_pairs':slot_pairs,'target_slots':target_slots.tolist(),'training_cases':cases,
        'before':before,'after':after,'actual_update_l1':updates.tolist(),'reward_pulse':reward_pulse,
        'learning_rule':asdict(local.config),'sampler':motor.experiment,
        'feedback':'diagnostic reward for a fixed card output; counterbalanced targets; no native actions or poker labels',
        **check_result(agent,local,source,plastic,allowed)},local.state.state_dict()


def native_layout(replicates):
    if replicates<1:raise ValueError('positive replicate count required')
    ids=np.repeat(np.arange(replicates),4).tolist()
    role=np.tile(np.arange(4),replicates)
    return ids,role,role!=3


def native_trial(agent,corpus,env,alternate,local,motor,credit,ids,role,indices,key,
                 *,sampled,learn=False,calibrate=False,matched_gain=1.,uniform=False,trace_cut=False):
    n=len(ids);orders=[];next_orders=[]
    for lane,(r,index) in enumerate(zip(ids,indices,strict=True)):
        corpus.restore(env,lane,index)
        rng=np.random.default_rng(derive_seed(VERSION,'native',key,r,index,'display'))
        orders.append(np.r_[rng.permutation(8),8,9]);next_orders.append(np.r_[rng.permutation(8),8,9])
    obs,masks=delayed.permute_rows(*env.observe(),orders);masks=delayed.forced(masks,1)
    neural=agent.processor.process(obs,fly_seeds=[derive_seed(VERSION,'native',key,r,index,'discard') for r,index in zip(ids,indices,strict=True)],efficacy=local.state.efficacy)
    action=motor.decode(neural.output_activity,masks,deterministic=not sampled,learner_ids=ids,
                        decision_ids=[derive_seed(VERSION,'native',key,index,'exploration') for index in indices])
    if learn or calibrate:
        local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=credit.mask(action,masks))
    discarded=env.step(delayed.translate(action,orders))
    if discarded.dones.any() or discarded.rewards.any():raise ValueError('discard must be nonterminal and zero reward')
    if trace_cut and (learn or calibrate):local.state.reset_fast_traces(eligibility=True)
    for lane in range(n):alternate.restore(lane,env.snapshot(lane))
    visible,legal=delayed.permute_rows(*env.observe(),next_orders);legal=delayed.forced(legal,0)
    play_neural=agent.processor.process(visible,fly_seeds=[derive_seed(VERSION,'native',key,r,index,'play') for r,index in zip(ids,indices,strict=True)],efficacy=local.state.initial_efficacy)
    if not np.isfinite(play_neural.output_activity).all():raise ValueError('nonfinite PLAY output')
    weak=agent.motor.decode(play_neural.output_activity,legal,deterministic=True)
    reference={k:v.copy() for k,v in weak.items()}
    expected_weak=[];expected_reference=[]
    for lane in range(n):
        features=visible['hand'][lane,:int(visible['hand_len'][lane])]
        chips,cards=best_visible_play(features)
        hand=[(int(c[:13].argmax())+2,int(c[13:17].argmax())) for c in features]
        chosen=weak['cards'][lane,:int(weak['n_cards'][lane])]
        expected_weak.append(min(score(hand,chosen)[0],300)/300)
        expected_reference.append(min(chips,300)/300)
        reference['n_cards'][lane]=len(cards);reference['cards'][lane]=-1
        reference['cards'][lane,:len(cards)]=cards
    weak_result=env.step(delayed.translate(weak,next_orders))
    ref_result=alternate.step(delayed.translate(reference,next_orders))
    if weak_result.dones.any() or ref_result.dones.any():raise ValueError('opening PLAY unexpectedly ended run')
    np.testing.assert_allclose([info['reward_components']['blind_progress'] for info in weak_result.infos],expected_weak,atol=2e-6,rtol=0)
    np.testing.assert_allclose([info['reward_components']['blind_progress'] for info in ref_result.infos],expected_reference,atol=2e-6,rtol=0)
    results={'weak':weak_result,'reference':ref_result};native_pulses={};out={}
    for name,result in results.items():
        pulses=[agent.reinforcement.map(info['reward_components'],info) for info in result.infos]
        if any(p.aversive for p in pulses):raise ValueError('no artificial failure punishment')
        native_pulses[name]=np.asarray([p.appetitive for p in pulses],dtype=np.float32)
        out[name+'_progress']=[info['reward_components']['blind_progress'] for info in result.infos]
        out[name+'_clears']=[bool(info['reward_components'].get('blind_clear',0)) for info in result.infos]
        out[name+'_pulse']=native_pulses[name].tolist()
    eligibility_mass=np.zeros(n);updates=np.zeros(n);applied=np.zeros(n,dtype=np.float32)
    if learn or calibrate:
        local.record_activity(play_neural.edge_pre_activity,play_neural.edge_post_activity,
                              edge_credit=delayed.zero_gate(play_neural.edge_pre_activity))
        eligibility_mass=array(local.state.eligibility.sum(axis=1))
        if learn:
            applied=np.where(role==0,native_pulses['weak'],native_pulses['reference'])
            applied[role==2]*=matched_gain;applied[role==3]=0
            events=local.apply_dopamine(applied,0)
            if any(e.lower_bound_hits or e.upper_bound_hits for e in events):raise ValueError('membership efficacy bound reached')
            updates=np.asarray([e.absolute_change for e in events])
        local.state.reset_fast_traces(eligibility=True)
    row={'key':key,'case_indices':list(indices),'discard_cards':action['cards'].tolist(),
        'discard_counts':action['n_cards'].tolist(),'weak_play_cards':weak['cards'].tolist(),
        'reference_play_cards':reference['cards'].tolist(),'eligibility_mass':eligibility_mass.tolist(),
        'applied_pulse':applied.tolist(),'update_l1':updates.tolist(),**out}
    if uniform:
        # A reference-only random baseline needs no extra neural forward call.
        random=empty_action_batch(n);random['action_type'][:]=1;random['n_cards'][:]=motor.base.count
        for lane,(r,index) in enumerate(zip(ids,indices,strict=True)):
            corpus.restore(alternate,lane,index)
            rng=np.random.default_rng(derive_seed(VERSION,'native',key,r,index,'uniform-discard'))
            random['cards'][lane,:motor.base.count]=np.sort(rng.choice(8,motor.base.count,replace=False))
        alternate.step(delayed.translate(random,orders))
        random_obs,_=alternate.observe();random_play=empty_action_batch(n);random_play['action_type'][:]=0
        for lane in range(n):
            _,cards=best_visible_play(random_obs['hand'][lane,:int(random_obs['hand_len'][lane])])
            random_play['n_cards'][lane]=len(cards);random_play['cards'][lane,:len(cards)]=cards
        result=alternate.step(random_play)
        row['uniform_reference_progress']=[info['reward_components']['blind_progress'] for info in result.infos]
    return row


def matched_gain(rows,learning_rate):
    if not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('positive finite learning rate required')
    weak=reference=0.
    for row in rows:
        mass=np.asarray(row['eligibility_mass'])[::4]
        a,b=np.asarray(row['weak_pulse'])[::4],np.asarray(row['reference_pulse'])[::4]
        if mass.shape!=a.shape or a.shape!=b.shape or not np.isfinite(np.r_[mass,a,b]).all() or np.any(np.r_[mass,a,b]<0):
            raise ValueError('invalid calibration activity/pulses')
        weak+=float(np.sum(mass*a)*learning_rate)
        reference+=float(np.sum(mass*b)*learning_rate)
    if not np.isfinite(weak+reference) or weak<=0 or reference<weak:raise ValueError('invalid update-matching calibration')
    gain=weak/reference
    return {'gain':gain,'weak_expected_update_l1':weak,'reference_expected_update_l1':reference,
        'matched_expected_update_l1':reference*gain,'scope':'training-only, initial-policy selected-eligibility weighted scalar match; frozen during learning'}


def native_probe(agent,corpus,*,steps=2000,replicates=3,repeats=2,count=2,
                 calibration_trials=128,progress=None,log=None):
    if min(steps,repeats,calibration_trials)<1:raise ValueError('positive budgets required')
    source=plastic_hashes(agent.plasticity.state);ids,role,plastic=native_layout(replicates);n=len(ids)
    motor=fixed_motor(agent,1,count);allowed=head_edges(agent,['discard_card'])
    env=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);env.reset([0]*n)
    alternate=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);alternate.reset([0]*n)
    training=corpus.indices('training')
    calibration_local=delayed.local_rule(agent,2*replicates);calibration_credit=SelectedMotorCredit(motor,calibration_local)
    schedules=[np.random.default_rng(derive_seed(VERSION,'calibration-order',r)).permutation(training).tolist() for r in range(replicates)]
    calibration=[]
    if progress:progress({'phase':'calibration'})
    for step in range(calibration_trials):
        indices=[schedules[r][step%len(training)] for r in ids]
        calibration.append(native_trial(agent,corpus,env,alternate,calibration_local,motor,calibration_credit,
            ids,role,indices,('calibration',step),sampled=True,calibrate=True))
    np.testing.assert_array_equal(array(calibration_local.state.efficacy),array(calibration_local.state.initial_efficacy))
    match=matched_gain(calibration,calibration_local.config.learning_rate)
    local=delayed.local_rule(agent,2*replicates);credit=SelectedMotorCredit(motor,local)
    def evaluate():
        before=plastic_hashes(local.state);out={}
        for policy in ('greedy','training-sampled'):
            out[policy]=[native_trial(agent,corpus,env,alternate,local,motor,credit,ids,role,[index]*n,
                ('evaluation',repeat,policy),sampled=policy!='greedy',uniform=True)
                for index in corpus.indices('validation') for repeat in range(repeats)]
        if plastic_hashes(local.state)!=before:raise ValueError('native evaluation changed state')
        return out
    if progress:progress({'phase':'initial-evaluation','matched_gain':match['gain']})
    before=evaluate();updates=np.zeros(n);pulses=np.zeros(n)
    schedules=[np.random.default_rng(derive_seed(VERSION,'training-order',r)).permutation(training).tolist() for r in range(replicates)]
    for step in range(steps):
        indices=[schedules[r][step%len(training)] for r in ids]
        row=native_trial(agent,corpus,env,alternate,local,motor,credit,ids,role,indices,('training',step),
            sampled=True,learn=True,matched_gain=match['gain'])
        updates+=row['update_l1'];pulses+=row['applied_pulse']
        if log:log({'step':step,**row})
        if progress and ((step+1)%250==0 or step+1==steps):
            progress({'phase':'training','step':step+1,'actual_update_l1':updates.tolist()})
    if progress:progress({'phase':'final-evaluation'})
    after=evaluate()
    for policy in before:check_frozen(before[policy],after[policy],plastic)
    return {'version':VERSION,'study':'fixed-membership','steps':steps,'count':count,
        'replicate_ids':ids,'roles':role.tolist(),'plastic':plastic.tolist(),'calibration':match,
        'calibration_rows':calibration,'before':before,'after':after,
        'actual_update_l1':updates.tolist(),'training_applied_pulse_sum':pulses.tolist(),
        'learning_rule':asdict(local.config),'feedback':'neural native / reference native / reference attenuated by frozen initial update match; shared frozen DISCARD actor evaluated with both PLAY policies',
        **check_result(agent,local,source,plastic,allowed)},local.state.state_dict()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=('binary','native'),required=True)
    for name in ('condition','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--corpus',type=Path);p.add_argument('--steps',type=int);p.add_argument('--replicates',type=int,default=3)
    p.add_argument('--repeats',type=int);p.add_argument('--calibration-trials',type=int,default=128)
    p.add_argument('--heavy',action='store_true');args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    steps=args.steps if args.steps is not None else (3000 if args.study=='binary' else 2000)
    repeats=args.repeats if args.repeats is not None else (128 if args.study=='binary' else 2)
    if min(steps,repeats,args.replicates,args.calibration_trials)<1:raise ValueError('positive budgets required')
    condition,_=load_condition(args.condition);config=PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy)
    sources={str(path.resolve()):sha256_file(path) for path in (args.condition,args.checkpoint,Path(__file__),
        Path(__file__).with_name('run_training_bottlenecks.py'),Path(__file__).with_name('train_discard_play_curriculum.py'))}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'design.json',{'version':VERSION,'study':args.study,'steps':steps,'replicates':args.replicates,
        'repeats':repeats,'source_hashes':sources,'fresh_unit_weights':True,'final_test_used':False,
        'training_offset':14500,'validation_offset':280000,
        'binary':'actual DISCARD output, fixed count1/two legal slots, balanced target slots, source 5% subset sampler; .05 diagnostic pulse, no native actions',
        'native':'fixed count2, only DISCARD membership credit; neural-native/reference-native/reference-initial-update-matched/shared-frozen per replicate',
        'matching':'128 training-only calibration trials by default; eligibility-weighted initial L1 ratio, fixed scalar thereafter; realized budgets may diverge'})
    started=time.perf_counter()
    def progress(row):
        write_json(args.output_dir/'status.json',{'status':'running',**row});print(json.dumps(row),flush=True)
    try:
        if args.corpus is None:
            path=args.output_dir/'pairs.json';build_corpus(path,train_per_rank=8,validation_per_rank=4,
                training_offset=14500,validation_offset=280000,max_attempts=5000)
        else:path=args.corpus
        corpus=PairCorpus(path);sources[str(path.resolve())]=sha256_file(path)
        stack=build_condition_stack(config,condition)
        load_plastic_checkpoint(args.checkpoint,stack.trainer,expected_components=stack.components)
        if not stack.agent.motor.mapping.routing.action_conditioned or stack.agent.plasticity.config.dopamine_decay!=0:
            raise ValueError('action-conditioned interface and decision-local dopamine required')
        with (args.output_dir/'training-trials.jsonl').open('w') as stream:
            def log(row):stream.write(json.dumps(row,sort_keys=True)+'\n')
            fn=binary_probe if args.study=='binary' else native_probe
            extra={} if args.study=='binary' else {'calibration_trials':args.calibration_trials}
            report,state=fn(stack.agent,corpus,steps=steps,replicates=args.replicates,repeats=repeats,
                            progress=progress,log=log,**extra)
        if any(sha256_file(Path(x))!=h for x,h in sources.items()):raise ValueError('source changed')
        np.savez_compressed(args.output_dir/'diagnostic-final-state.npz',**state)
        report.update(status='completed',source_hashes=sources,corpus=str(corpus.path),corpus_sha256=corpus.sha256,
                      wall_seconds=time.perf_counter()-started)
        write_json(args.output_dir/'report.json',report);write_json(args.output_dir/'status.json',{'status':'completed'})
    except BaseException as error:
        write_json(args.output_dir/'status.json',{'status':'failed','error':f'{type(error).__name__}: {error}'});raise
    return 0


if __name__=='__main__':raise SystemExit(main())
