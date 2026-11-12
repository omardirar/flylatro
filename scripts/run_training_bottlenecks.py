"""Isolated learning-capacity, subset-exploration and follow-up-policy screens.

No production source, source checkpoint, or existing experiment is modified.
The binary reward and visible-hand reference player are diagnostic interventions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import array, efficacy_like
from card_learning.audit import score
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from card_learning.motor import CardExploration, CardSubsetMotor
from card_learning.pairs import PairCorpus, build_corpus, pair_slots
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed
import train_discard_play_curriculum as delayed


VERSION = 'training-bottleneck-screens-v1'


def layout(replicates):
    """Four paired lanes per seed: two treatments, each plastic/frozen."""
    if replicates < 1:
        raise ValueError('positive replicate count required')
    ids = np.repeat(np.arange(replicates), 4)
    treatment = np.tile([0, 0, 1, 1], replicates)
    plastic = np.tile([True, False, True, False], replicates)
    return ids.tolist(), treatment, plastic


def head_edges(agent, heads):
    m = agent.motor.mapping
    pools = [p for h in heads for p in m.routing.head_routes[h] if p >= 0]
    roots = [m.output_root_ids[i] for p in pools for i in m.pool_indices[p]]
    return np.isin(agent.plasticity.topology.post_root_ids, roots)


def restricted_gate(gate, allowed, state):
    if getattr(state, 'is_torch', False):
        import torch
        return gate & torch.as_tensor(allowed, dtype=torch.bool, device=state.device)[None, :]
    return gate & allowed[None, :]


def binary_probabilities(scores, mapping):
    """Exact marginal of the existing epsilon/softmax action selector."""
    logits = np.asarray(scores)[:, :2] / mapping.exploration_temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    return (1-mapping.exploration_epsilon)*p + mapping.exploration_epsilon/2


def check_result(agent, local, source_hash, plastic, allowed):
    delta = array(local.state.efficacy)-array(local.state.initial_efficacy)
    if np.any(delta[~plastic]) or np.any(delta[:, ~allowed]):
        raise ValueError('frozen or unrelated efficacy changed')
    if np.any(array(local.state.lower_bound_hits)) or np.any(array(local.state.upper_bound_hits)):
        raise ValueError('efficacy bound reached')
    if array(local.state.eligibility).any() or array(local.state.dopamine).any():
        raise ValueError('fast traces must be clear at task boundaries')
    if plastic_hashes(agent.plasticity.state) != source_hash:
        raise ValueError('source plastic state changed')
    return {'source_state_unchanged': True, 'frozen_weights_exact': True,
            'unrelated_weights_exact': True, 'efficacy_bound_hits': 0,
            'mean_absolute_drift': np.abs(delta).mean(axis=1).tolist(),
            'max_absolute_drift': np.abs(delta).max(axis=1).tolist()}


def check_frozen(before, after, plastic):
    for a, b in zip(before, after, strict=True):
        for key in a:
            if key in ('key', 'case_indices'):
                continue
            x, y = np.asarray(a[key]), np.asarray(b[key])
            if x.ndim and x.shape[0] == len(plastic):
                np.testing.assert_array_equal(x[~plastic], y[~plastic], err_msg=f'frozen {key}')


def binary_probe(agent, corpus, *, steps=3000, replicates=3, repeats=128,
                 reward_pulse=.05, progress=None, log=None):
    if min(steps, repeats) < 1 or not 0 < reward_pulse <= 2:
        raise ValueError('positive budgets and bounded diagnostic pulse required')
    source = plastic_hashes(agent.plasticity.state)
    ids, targets, plastic = layout(replicates); n = len(ids)
    local = delayed.local_rule(agent, 2*replicates)
    credit = SelectedMotorCredit(agent.motor, local)
    allowed = head_edges(agent, ['action_type'])
    env = BalatroSimAdapter(n, win_ante=1, allow_skip_blinds=False); env.reset([0]*n)
    cases = corpus.indices('training')
    selected = [cases[(i*len(cases))//replicates] for i in range(replicates)]
    for lane, replicate in enumerate(ids):
        corpus.restore(env, lane, selected[replicate])
    obs, masks = env.observe()
    masks['action_type_mask'][:] = False; masks['action_type_mask'][:, :2] = True
    base = agent.motor.base

    def evaluate():
        state_hash = plastic_hashes(local.state); rows=[]
        for repeat in range(repeats):
            seeds=[derive_seed(VERSION,'binary-evaluation',r,repeat) for r in ids]
            neural=agent.processor.process(obs,fly_seeds=seeds,efficacy=local.state.efficacy)
            scores=base.head_scores(neural.output_activity)['action_type']
            p=binary_probabilities(scores,base.mapping)
            greedy=base.decode(neural.output_activity,masks,deterministic=True)
            sampled=base.decode(neural.output_activity,masks,deterministic=False,
                                learner_ids=ids,decision_ids=[derive_seed(VERSION,'binary-eval-choice',repeat)]*n)
            rows.append({'rewarded_probability':p[np.arange(n),targets].tolist(),
                         'greedy_correct':(greedy['action_type']==targets).tolist(),
                         'sampled_correct':(sampled['action_type']==targets).tolist()})
        if plastic_hashes(local.state)!=state_hash:
            raise ValueError('binary evaluation updated state')
        return rows

    before=evaluate(); totals=np.zeros(n); updates=np.zeros(n)
    for step in range(steps):
        seeds=[derive_seed(VERSION,'binary-training',r,step) for r in ids]
        neural=agent.processor.process(obs,fly_seeds=seeds,efficacy=local.state.efficacy)
        action=base.decode(neural.output_activity,masks,deterministic=False,
                           learner_ids=ids,decision_ids=[step]*n)
        hit=action['action_type']==targets
        gate=restricted_gate(credit.mask(action,masks),allowed,local.state)
        local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=gate)
        events=local.apply_dopamine((hit & plastic).astype(np.float32)*reward_pulse,0)
        updates += [e.absolute_change for e in events]
        local.state.reset_fast_traces(eligibility=True)
        totals+=hit
        if log:
            log({'step':step,'action_types':action['action_type'].tolist(),'rewarded':hit.tolist()})
        if progress and ((step+1)%250==0 or step+1==steps):
            progress({'phase':'training','step':step+1,'reward_frequency':(totals/(step+1)).tolist()})
    after=evaluate();check_frozen(before,after,plastic)
    return {'version':VERSION,'study':'binary','before':before,'after':after,
            'targets':targets.tolist(),'replicate_ids':ids,'plastic':plastic.tolist(),
            'training_cases':selected,'steps':steps,'reward_pulse':reward_pulse,
            'actual_update_l1':updates.tolist(), 'learning_rule':asdict(local.config),
            'feedback':'diagnostic reward for PLAY or DISCARD, balanced targets; no native action executed; only action-type posts learn',
            **check_result(agent,local,source,plastic,allowed)},local.state.state_dict()


def mixed_pair_action(base, exploration_seed, probabilities, neural, masks, ids, decisions, sampled):
    actions=empty_action_batch(len(ids))
    for probability in sorted(set(probabilities.tolist())):
        motor=CardSubsetMotor(base,CardExploration(float(probability),exploration_seed),pair_task=True)
        candidate=motor.decode(neural.output_activity,masks,deterministic=not sampled,
                               learner_ids=ids,decision_ids=decisions)
        selected=probabilities==probability
        for key in actions:
            actions[key][selected]=candidate[key][selected]
    return actions


def pair_probe(agent, corpus, *, steps=5000, replicates=3, repeats=2,
               probabilities=(.05,.25), progress=None, log=None):
    if min(steps,repeats)<1 or len(probabilities)!=2 or any(not 0<=p<=1 for p in probabilities):
        raise ValueError('positive budgets and two exploration probabilities required')
    source=plastic_hashes(agent.plasticity.state)
    ids,treatment,plastic=layout(replicates);n=len(ids)
    probability=np.asarray(probabilities)[treatment]
    local=delayed.local_rule(agent,2*replicates)
    motor=CardSubsetMotor(agent.motor.base,agent.motor.exploration,pair_task=True)
    credit=SelectedMotorCredit(motor,local);allowed=head_edges(agent,['card'])
    env=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);env.reset([0]*n)
    def trial(indices,key,sampled,learn):
        for row,index in enumerate(indices):corpus.restore(env,row,index)
        obs,masks=env.observe();orders=[]
        for r,index in zip(ids,indices,strict=True):
            rng=np.random.default_rng(derive_seed(VERSION,'pair',key,r,index,'display'))
            orders.append(np.r_[rng.permutation(8),8,9])
        obs,masks=delayed.permute_rows(obs,masks,orders);masks=delayed.forced(masks,0)
        seeds=[derive_seed(VERSION,'pair',key,r,index,'neural') for r,index in zip(ids,indices,strict=True)]
        neural=agent.processor.process(obs,fly_seeds=seeds,efficacy=local.state.efficacy)
        decisions=[derive_seed(VERSION,'pair',key,index) for index in indices]
        actions=mixed_pair_action(agent.motor.base,agent.motor.exploration.seed,probability,
                                  neural,masks,ids,decisions,sampled)
        result=env.step(delayed.translate(actions,orders))
        pulses=[agent.reinforcement.map(info['reward_components'],info) for info in result.infos]
        if np.any(result.dones) or any(p.aversive for p in pulses):
            raise ValueError('pair task unexpectedly emitted a terminal/failure pulse')
        hit=[set(actions['cards'][row,:2])==set(pair_slots(obs,row)) for row in range(n)]
        updates=np.zeros(n)
        if learn:
            gate=credit.mask(actions,masks)
            local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=gate)
            events=local.apply_dopamine(np.asarray([p.appetitive for p in pulses],dtype=np.float32)*plastic,0)
            updates=np.asarray([e.absolute_change for e in events]);local.state.reset_fast_traces(eligibility=True)
        return {'key':key,'case_indices':indices,'pair_correct':hit,'cards':actions['cards'].tolist(),
                'progress':[info['reward_components']['blind_progress'] for info in result.infos],
                'pulse':[p.appetitive for p in pulses],'update_l1':updates.tolist()}
    def evaluate():
        state_hash=plastic_hashes(local.state);rows={}
        for policy in ('greedy','training-sampled'):
            rows[policy]=[trial([index]*n,('evaluation',repeat,policy),policy!='greedy',False)
                          for index in corpus.indices('validation') for repeat in range(repeats)]
        if plastic_hashes(local.state)!=state_hash:raise ValueError('pair evaluation updated state')
        return rows
    before=evaluate();training=corpus.indices('training')
    schedules=[np.random.default_rng(derive_seed(VERSION,'pair-order',r)).permutation(training).tolist() for r in range(replicates)]
    discoveries=np.zeros(n);updates=np.zeros(n)
    for step in range(steps):
        indices=[schedules[r][step%len(training)] for r in ids]
        row=trial(indices,('training',step),True,True)
        discoveries+=row['pair_correct'];updates+=row['update_l1']
        if log:log({'step':step,**row})
        if progress and ((step+1)%250==0 or step+1==steps):
            progress({'phase':'training','step':step+1,'discovered_pairs':discoveries.tolist()})
    after=evaluate()
    for policy in before:check_frozen(before[policy],after[policy],plastic)
    return {'version':VERSION,'study':'exploration','before':before,'after':after,
            'probabilities':probability.tolist(),'replicate_ids':ids,'plastic':plastic.tolist(),
            'training_pair_discoveries':discoveries.tolist(),'steps':steps,
            'actual_update_l1':updates.tolist(),'learning_rule':asdict(local.config),
            'feedback':'native progress; forced PLAY/two cards; only PLAY membership posts learn; labels diagnostic only',
            **check_result(agent,local,source,plastic,allowed)},local.state.state_dict()


def best_visible_play(features):
    """Highest immediate plain-hand chips; ties prefer fewer cards, then slots.

    Called only after DISCARD, on the current visible hand. Never sees draw order.
    """
    n=len(features)
    if n<1 or n>10:raise ValueError('nonempty visible hand required')
    normal=np.zeros((n,21),dtype=np.float32);normal[:,[0,9,14]]=1
    if not np.array_equal(features[:,17:],normal):raise ValueError('reference requires plain cards')
    hand=[(int(c[:13].argmax())+2,int(c[13:17].argmax())) for c in features]
    candidates=[(score(hand,subset)[0],subset) for count in range(1,min(5,n)+1)
                for subset in combinations(range(n),count)]
    chips=max(x[0] for x in candidates)
    cards=next(subset for value,subset in candidates if value==chips)
    return chips,cards


@contextmanager
def reference_followup(agent, reference_rows):
    """Scoped substitution in the isolated research helper; always restored."""
    original=delayed.followup;cache={};verified=[0]
    def hybrid(agent,env,initial_efficacy,orders,seeds):
        obs,masks=delayed.permute_rows(*env.observe(),orders);masks=delayed.forced(masks,0)
        neural=agent.processor.process(obs,fly_seeds=seeds,efficacy=initial_efficacy)
        action=agent.motor.decode(neural.output_activity,masks,deterministic=True)
        expected=[]
        for row in range(len(reference_rows)):
            features=obs['hand'][row,:int(obs['hand_len'][row])]
            if reference_rows[row]:
                key=features.tobytes()
                if key not in cache:cache[key]=best_visible_play(features)
                _,cards=cache[key];action['n_cards'][row]=len(cards);action['cards'][row]=-1
                action['cards'][row,:len(cards)]=cards
            hand=[(int(c[:13].argmax())+2,int(c[13:17].argmax())) for c in features]
            chosen=action['cards'][row,:int(action['n_cards'][row])]
            expected.append(min(score(hand,chosen)[0],300)/300)
        result=env.step(delayed.translate(action,orders))
        actual=[info['reward_components']['blind_progress'] for info in result.infos]
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)
        verified[0]+=len(reference_rows)
        pulses=[agent.reinforcement.map(info['reward_components'],info) for info in result.infos]
        if np.any(result.dones) or any(p.aversive for p in pulses):
            raise ValueError('follow-up unexpectedly emitted terminal/failure reward')
        return neural,action,result,pulses
    delayed.followup=hybrid
    try:yield verified
    finally:delayed.followup=original


def followup_probe(agent, corpus, *, steps=2500, replicates=3, repeats=2,
                   progress=None,log=None):
    if min(steps,repeats)<1:raise ValueError('positive budgets required')
    source=plastic_hashes(agent.plasticity.state)
    ids,treatment,plastic=layout(replicates);n=len(ids)
    local=delayed.local_rule(agent,2*replicates);credit=SelectedMotorCredit(agent.motor,local)
    allowed=head_edges(agent,['discard_card','discard_card_count'])
    env=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);env.reset([0]*n)
    def trial(indices,key,sampled,learn,random=False):
        row=delayed.trial(agent,env,corpus,indices,efficacy=local.state.efficacy,
            initial_efficacy=local.state.initial_efficacy,replicate_ids=ids,
            key=(VERSION,'followup',key),sampled=sampled,
            learning=local if learn else None,credit=credit,plastic_rows=plastic,
            random_control=random)
        return {'key':key,**row}
    def evaluate():
        state_hash=plastic_hashes(local.state)
        rows={p:[trial([index]*n,('evaluation',repeat,p),p!='greedy',False,True)
                 for index in corpus.indices('validation') for repeat in range(repeats)]
              for p in ('greedy','training-sampled')}
        if plastic_hashes(local.state)!=state_hash:raise ValueError('follow-up evaluation updated state')
        return rows
    with reference_followup(agent,treatment==1) as verified:
        before=evaluate();training=corpus.indices('training')
        schedules=[np.random.default_rng(derive_seed(VERSION,'followup-order',r)).permutation(training).tolist() for r in range(replicates)]
        updates=np.zeros(n);pulse=np.zeros(n)
        for step in range(steps):
            row=trial([schedules[r][step%len(training)] for r in ids],('training',step),True,True)
            updates+=row['update_l1'];pulse+=row['pulse']
            if log:log({'step':step,**row})
            if progress and ((step+1)%250==0 or step+1==steps):
                progress({'phase':'training','step':step+1,'actual_update_l1':updates.tolist()})
        after=evaluate()
        for policy in before:check_frozen(before[policy],after[policy],plastic)
        verified_count=verified[0]
    return {'version':VERSION,'study':'followup','before':before,'after':after,
            'reference_rows':(treatment==1).tolist(),'replicate_ids':ids,'plastic':plastic.tolist(),
            'steps':steps,'actual_update_l1':updates.tolist(),'training_pulse_sum':pulse.tolist(),
            'verified_native_followup_scores':verified_count,'learning_rule':asdict(local.config),
            'feedback':'native progress/clear; fixed initial neural PLAY versus visible-hand immediate-score reference; only delayed DISCARD card/count learns',
            **check_result(agent,local,source,plastic,allowed)},local.state.state_dict()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=('binary','exploration','followup'),required=True)
    parser.add_argument('--condition',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--replicates',type=int,default=3)
    parser.add_argument('--repeats',type=int)
    parser.add_argument('--heavy',action='store_true')
    args=parser.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    steps=args.steps if args.steps is not None else {'binary':3000,'exploration':5000,'followup':2500}[args.study]
    repeats=args.repeats if args.repeats is not None else (128 if args.study=='binary' else 2)
    if min(steps,repeats,args.replicates)<1:raise ValueError('positive budgets required')
    condition,_=load_condition(args.condition)
    config=PlasticExperimentConfig.load(Path(condition['config']));config.require_heavy_opt_in(args.heavy)
    sources={str(p.resolve()):sha256_file(p) for p in (args.condition,args.checkpoint,Path(__file__),
        Path(delayed.__file__),Path(__file__).with_name('audit_card_credit_direction.py'))}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'design.json',{'version':VERSION,'study':args.study,'steps':steps,
        'replicates':args.replicates,'evaluation_repeats':repeats,'source_hashes':sources,
        'initial_weights':'fresh unit efficacy','lane_order':'per replicate: treatment0 plastic/frozen, treatment1 plastic/frozen',
        'treatments':{'binary':'reward PLAY / reward DISCARD; fixed .05 pulse, action-type-only credit',
            'exploration':'5% / 25% uniform subset exploration; PLAY/two cards; native reward',
            'followup':'frozen initial neural PLAY / visible-hand immediate-score reference PLAY; native reward'}[args.study],
        'scope':'bounded exploratory diagnostic; no primary model changes; no final-test seeds',
        'training_offset':10500,'validation_offset':240000,
        'shared_inputs':'paired replicate neural seeds, displays, native snapshots, exploration keys; divergent choices are not forced to match',
        'feedback_budget':'native rewards unchanged in exploration/followup; stronger reference may change realized pulse/update magnitude'})
    started=time.perf_counter()
    def progress(row):
        value={'status':'running','study':args.study,**row}
        write_json(args.output_dir/'status.json',value);print(json.dumps(value),flush=True)
    try:
        if args.corpus is None:
            path=args.output_dir/'pairs.json'
            build_corpus(path,train_per_rank=8,validation_per_rank=4,training_offset=10500,
                         validation_offset=240000,max_attempts=5000)
        else:path=args.corpus
        corpus=PairCorpus(path)
        sources[str(path.resolve())]=sha256_file(path)
        stack=build_condition_stack(config,condition)
        load_plastic_checkpoint(args.checkpoint,stack.trainer,expected_components=stack.components)
        if not stack.agent.motor.mapping.routing.action_conditioned:
            raise ValueError('current action-conditioned mapping required')
        if stack.agent.plasticity.config.dopamine_decay!=0:raise ValueError('decision-local dopamine required')
        fn={'binary':binary_probe,'exploration':pair_probe,'followup':followup_probe}[args.study]
        with (args.output_dir/'training-trials.jsonl').open('w') as stream:
            def log(row):stream.write(json.dumps(row,sort_keys=True)+'\n')
            report,state=fn(stack.agent,corpus,steps=steps,replicates=args.replicates,repeats=repeats,
                            progress=progress,log=log)
        if any(sha256_file(Path(p))!=h for p,h in sources.items()):raise ValueError('source identity changed')
        np.savez_compressed(args.output_dir/'diagnostic-final-state.npz',**state)
        report.update(status='completed',source_hashes=sources,corpus=str(path.resolve()),corpus_sha256=corpus.sha256,
                      wall_seconds=time.perf_counter()-started)
        write_json(args.output_dir/'report.json',report);write_json(args.output_dir/'status.json',{'status':'completed'})
    except BaseException as error:
        write_json(args.output_dir/'status.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
