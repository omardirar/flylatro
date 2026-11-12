"""CPU native replay of strategy puzzles, outcomes, controls and saved state."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from card_learning.common import write_json
from discard_strategy import VERSION
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.reinforcement import ReinforcementMapper
from flylatro.seeds import derive_seed
from run_local_normalisation import verify
from run_training_bottlenecks import best_visible_play
from strategy_puzzles import PuzzleCorpus, Reference, action, resample_future, resources


def read(path):return json.loads(Path(path).read_text())


def assess(output):
    output=output.resolve();declaration=read(output/'declaration.json');verify(declaration)
    report=read(output/'report.json');cp=read(output/'checkpoint.json')
    if (report['status']!='completed' or report['steps']!=declaration['steps']
            or sha256_file(Path(cp['path']))!=cp['sha256']
            or cp['declaration_sha256']!=sha256_file(output/'declaration.json')):
        raise ValueError('completed checkpoint/declaration differs')
    corpus=PuzzleCorpus(output/'puzzles.json');n=3*declaration['replicates']
    ids=np.repeat(np.arange(declaration['replicates']),3);roles=np.tile(np.arange(3),declaration['replicates'])
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([0])
    config=PlasticExperimentConfig.load(Path(declaration['config']));mapper=ReinforcementMapper(config.reinforcement)
    counts=Counter();cache={}
    for index,case in enumerate(corpus.cases):
        env.reset([case['seed']])
        for a in case['prefix']:env.step(action(a['kind'],a['cards']))
        obs,_=env.observe()
        if env.state_hash(obs,0)!=case['state_hash'] or resources(obs)!=(1,1):raise ValueError('native prefix differs')
        counts['prefix_actions']+=len(case['prefix'])
    def replay(index,discard,play,seed,require_best=False):
        key=(index,None if discard is None else tuple(discard),tuple(play),int(seed))
        if key not in cache:
            corpus.restore(env,0,index);resample_future(env,0,seed)
            if discard is not None:
                r=env.step(action(1,discard));counts['unique_training_or_transfer_native_actions']+=1
                if r.dones[0] or r.rewards[0]!=0:raise ValueError('invalid discard replay')
            if require_best:
                obs,_=env.observe()
                _,best=best_visible_play(obs['hand'][0,:int(obs['hand_len'][0])])
                if list(best)!=play:raise ValueError('reference PLAY differs from visible best play')
            r=env.step(action(0,play));counts['unique_training_or_transfer_native_actions']+=1
            pulse=mapper.map(r.infos[0]['reward_components'],r.infos[0])
            cache[key]=(r.infos[0]['reward_components'].get('blind_clear',0)>0,pulse.appetitive,pulse.aversive)
        return cache[key]
    updates=np.zeros(n);discard_count=np.zeros(n,dtype=int)
    with (output/'training.jsonl').open() as stream:
        for step,line in enumerate(stream):
            row=json.loads(line)
            if row['step']!=step:raise ValueError('training cursor differs')
            cases=corpus.indices('training')
            expected=[cases[derive_seed(VERSION,step,int(rep),'case')%len(cases)] for rep in ids]
            if row['case_indices']!=expected:raise ValueError('training case schedule differs')
            for lane,index in enumerate(expected):
                choice=row['choices'][lane];discard=row['discard_sets_native'][lane]
                if choice not in (0,1,2) or ((discard is None)!=(choice==0)):
                    raise ValueError('choice/discard mismatch')
                if choice and discard!=corpus.cases[index]['candidates'][choice-1]:raise ValueError('candidate choice differs')
                seed=derive_seed(VERSION,('training',step),int(ids[lane]),index,'hidden-future')
                clear,pos,neg=replay(index,discard,row['helper_play_cards_native'][lane],seed,require_best=True)
                if (clear!=row['cleared'][lane] or abs(pos-row['appetitive'][lane])>2e-7
                        or abs(neg-row['aversive'][lane])>2e-7):raise ValueError('native outcome/reinforcement mismatch')
                discard_count[lane]+=choice!=0
                counts['training_records']+=1
            updates+=row['update_l1']
    if counts['training_records']!=declaration['steps']*n:raise ValueError('training budget differs')
    benchmark=read(output/'benchmark.json');receipt=read(output/'benchmark-manifest.json')
    if receipt['sha256']!=sha256_file(output/'benchmark.json') or benchmark['corpus_sha256']!=corpus.sha256:
        raise ValueError('benchmark identity differs')
    reference=Reference(corpus)
    for row in benchmark['rows']:
        index=row['case_index']
        for choice in range(3):
            discard=None if choice==0 else corpus.cases[index]['candidates'][choice-1]
            outcomes=[reference.outcome(index,discard,('benchmark',draw))['cleared'] for draw in range(declaration['reference_draws'])]
            if outcomes!=row['outcomes'][choice]:raise ValueError('reference draw outcomes differ')
            if abs(float(np.mean(outcomes))-row['clear_probability'][choice])>1e-12:raise ValueError('reference expectation differs')
            counts['benchmark_outcome_records']+=len(outcomes)
    counts['benchmark_native_actions']=reference.native_actions
    initial=read(output/'evaluation-0000000.json')
    for path in sorted(output.glob('evaluation-*.json')):
        evaluated=read(path)
        for split,rows in evaluated['rows'].items():
            if len(rows)!=len(corpus.indices(split))*declaration['repeats']:raise ValueError('evaluation coverage differs')
            for i,row in enumerate(rows):
                for lane in range(n):
                    choice=row['choices'][lane]
                    expected=None if choice==0 else corpus.cases[row['case_index']]['candidates'][choice-1]
                    if row['discard_sets_native'][lane]!=expected:raise ValueError('evaluation discard set differs')
                if any(row['choices'][lane]!=initial['rows'][split][i]['choices'][lane] for lane in np.flatnonzero(roles==2)):
                    raise ValueError('frozen evaluation choices changed')
    fresh=read(output/'fresh-evaluation.json')
    q={r['case_index']:np.asarray(r['clear_probability']) for r in benchmark['rows']}
    changes=[];advantages=[]
    for before,after in zip(fresh['rows']['initial'],fresh['rows']['trained'],strict=True):
        index=before['case_index']
        if index!=after['case_index'] or corpus.cases[index]['split']!='fresh':raise ValueError('fresh coverage differs')
        a=q[index][before['choices']];b=q[index][after['choices']]
        if np.any(np.asarray(before['choices'])[roles==2]!=np.asarray(after['choices'])[roles==2]):raise ValueError('fresh frozen choices changed')
        changes.append((index,float(np.mean((b-a)[roles==1]))))
        advantages.append((index,float(np.mean(b[roles==1]-b[roles==0]))))
    free=read(output/'free-membership-transfer.json')
    for label in ('initial','trained'):
        data=free[label]
        for i,row in enumerate(data['decisions']):
            for lane,discard in enumerate(row['discard_sets_native']):
                value=np.mean([reference.outcome(row['case_index'],discard,('free-transfer',draw))['cleared'] for draw in range(data['draws'])])
                if abs(value-data['values'][i][lane])>1e-12:raise ValueError('free-membership transfer differs')
                counts['free_membership_policy_records']+=1
    transfer=read(output/'fly-play-transfer.json')
    for label in ('initial','trained'):
        for row in transfer[label]['rows']:
            for lane,discard in enumerate(row['discard_sets_native']):
                seed=derive_seed(VERSION,row['case_index'],row['draw'],'fly-transfer-future')
                clear,_,_=replay(row['case_index'],discard,row['play_cards_native'][lane],seed)
                if clear!=row['cleared'][lane]:raise ValueError('fly-only transfer outcome differs')
                counts['fly_only_transfer_records']+=1
    for a,b in zip(transfer['initial']['rows'],transfer['trained']['rows'],strict=True):
        for lane in np.flatnonzero(roles==2):
            if (a['discard_sets_native'][lane]!=b['discard_sets_native'][lane]
                    or a['play_cards_native'][lane]!=b['play_cards_native'][lane]):raise ValueError('frozen fly-only decisions changed')
    with np.load(cp['path'],allow_pickle=False) as z:
        w,initial_weights=z['efficacy'],z['initial_efficacy']
        np.testing.assert_array_equal(z['decision_count'],declaration['steps']+discard_count)
        np.testing.assert_allclose(z['update_l1'],updates,atol=1e-9,rtol=1e-9)
        if np.any(w[roles==2]!=initial_weights[roles==2]) or np.any(z['eligibility']) or np.any(z['dopamine']):raise ValueError('frozen state or task-boundary traces differ')
    if config.fly.backend=='flywire':
        with np.load(resolve_path(config,config.fly.artifact_path),allow_pickle=False) as z:
            edges=z['kc_mbon_edge_indices'];a=z['signed_synapse_counts'][edges];roots=z['root_ids'][z['post_indices'][edges]]
        keep=a>=config.fly.minimum_synapse_count;a=a[keep].astype(np.float64);roots=roots[keep]
        error=np.zeros(n)
        for root in np.unique(roots):
            mask=roots==root;budget=(a[mask]*initial_weights[:,mask]).sum(axis=1)
            error=np.maximum(error,abs((a[mask]*w[:,mask]).sum(axis=1)-budget)/budget)
        if np.any(error[roles==1]>1e-6):raise ValueError('normalised strength budgets differ')
        mapping=MotorMapping.load(resolve_path(config,config.fly.motor_mapping_path))
        pools=list(mapping.routing.head_routes['discard_card'])+[mapping.routing.head_routes['action_type'][k] for k in (0,1)]
        allowed=[mapping.output_root_ids[i] for p in pools if p>=0 for i in mapping.pool_indices[p]]
        if np.any(w[:,~np.isin(roots,allowed)]!=initial_weights[:,~np.isin(roots,allowed)]):raise ValueError('unrelated efficacy changed')
    groups=sorted(set(i for i,_ in changes));matrix=np.asarray([[np.mean([v for k,v in records if k==index]) for records in (changes,advantages)] for index in groups])
    rng=np.random.default_rng(20261101);samples=matrix[rng.integers(0,len(groups),size=(10000,len(groups)))].mean(axis=1)
    result={'status':'completed','report_sha256':sha256_file(output/'report.json'),'checkpoint_sha256':cp['sha256'],
        'assessor_sha256':sha256_file(Path(__file__)),'counts':dict(counts),
        'conditional_case_intervals':{'normalised_gain_95ci':np.quantile(samples[:,0],[.025,.975]).tolist(),
                                     'normalised_advantage_95ci':np.quantile(samples[:,1],[.025,.975]).tolist()},
        'interval_scope':'shared fresh visible cases, conditional on learners and finite reference draw bank',
        'frozen_and_unrelated_exact':True,'delayed_credit_clock_verified':True,'native_rewards_verified':True,
        'assistance_explicit':True,'final_test_used':False}
    write_json(output/'assessment.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(assess(p.parse_args().output_dir),indent=2))
