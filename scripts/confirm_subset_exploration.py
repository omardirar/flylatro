"""Fresh frozen pair-task confirmation of every exploration-screen learner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from assess_training_bottlenecks import replay,verify_state
from audit_card_credit_direction import efficacy_like
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack,load_condition
from card_learning.motor import CardSubsetMotor
from card_learning.pairs import PairCorpus,build_corpus,pair_slots
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.seeds import derive_seed
from run_training_bottlenecks import VERSION,layout
import train_discard_play_curriculum as delayed


def evaluate(agent,corpus,efficacy,*,replicates,repeats=4,progress=None):
    if repeats<1:raise ValueError('positive repeat count required')
    ids,_,_=layout(replicates);n=len(ids)
    if np.shape(efficacy)!=(n,agent.plasticity.topology.edge_count):raise ValueError('confirmation efficacy shape differs')
    source=plastic_hashes(agent.plasticity.state)
    weights=efficacy_like(agent.plasticity.state,efficacy)
    motor=CardSubsetMotor(agent.motor.base,agent.motor.exploration,pair_task=True)
    env=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);env.reset([0]*n)
    rows=[]
    for case,index in enumerate(corpus.indices('validation')):
        for repeat in range(repeats):
            key=('fresh-confirmation',repeat);orders=[]
            for lane,r in enumerate(ids):
                corpus.restore(env,lane,index)
                rng=np.random.default_rng(derive_seed(VERSION,'pair',key,r,index,'display'))
                orders.append(np.r_[rng.permutation(8),8,9])
            obs,masks=delayed.permute_rows(*env.observe(),orders);masks=delayed.forced(masks,0)
            seeds=[derive_seed(VERSION,'pair',key,r,index,'neural') for r in ids]
            neural=agent.processor.process(obs,fly_seeds=seeds,efficacy=weights)
            action=motor.decode(neural.output_activity,masks,deterministic=True)
            result=env.step(delayed.translate(action,orders))
            if result.dones.any():raise ValueError('confirmation opening PLAY unexpectedly terminated')
            rows.append({'key':key,'case_indices':[index]*n,'cards':action['cards'].tolist(),
                'pair_correct':[set(action['cards'][lane,:2])==set(pair_slots(obs,lane)) for lane in range(n)],
                'progress':[info['reward_components']['blind_progress'] for info in result.infos]})
        if progress and ((case+1)%13==0):progress({'completed_cases':case+1})
    if plastic_hashes(agent.plasticity.state)!=source:raise ValueError('confirmation changed source state')
    summary={}
    for metric in ('pair_correct','progress'):
        values=np.asarray([r[metric] for r in rows],dtype=float)
        np.testing.assert_array_equal(values[:,1::4],values[:,3::4])
        summary[metric]={'initial_per_learner':values[:,1::4].mean(axis=0).tolist(),
            'five_percent_per_learner':values[:,0::4].mean(axis=0).tolist(),
            'twentyfive_percent_per_learner':values[:,2::4].mean(axis=0).tolist()}
        for name,difference in [('five_minus_initial',values[:,0::4]-values[:,1::4]),
                ('twentyfive_minus_initial',values[:,2::4]-values[:,3::4]),
                ('twentyfive_minus_five',values[:,2::4]-values[:,0::4])]:
            cases=len(corpus.indices('validation'))
            per_case=difference.reshape(cases,repeats,replicates).mean(axis=(1,2))
            rng=np.random.default_rng(derive_seed(VERSION,'confirmation-bootstrap',metric,name))
            boot=per_case[rng.integers(0,cases,size=(2000,cases))].mean(axis=1)
            summary[metric][name]={'mean':float(difference.mean()),'per_learner':difference.mean(axis=0).tolist(),
                'case_bootstrap_95_interval_conditional_on_these_learners':np.quantile(boot,[.025,.975]).tolist()}
    return {'status':'completed','study':'exploration','scope':'fresh greedy confirmation, no learning',
        'replicate_ids':ids,'before':{'greedy':[]},'after':{'greedy':rows},'summary':summary,
        'source_state_unchanged':True,'corpus':str(corpus.path),'corpus_sha256':corpus.sha256}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','condition','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=4);p.add_argument('--heavy',action='store_true');args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    source_report=args.run_dir/'report.json';state=args.run_dir/'diagnostic-final-state.npz'
    original=json.loads(source_report.read_text());assessment=json.loads((args.run_dir/'assessment.json').read_text())
    if original['status']!='completed' or original['study']!='exploration':raise ValueError('completed exploration screen required')
    if assessment['source_report_sha256']!=sha256_file(source_report) or assessment['final_state_sha256']!=sha256_file(state):
        raise ValueError('screen assessment/state identities differ')
    for path,digest in original['source_hashes'].items():
        if sha256_file(Path(path))!=digest:raise ValueError(f'screen source changed: {path}')
    verify_state(original,state)
    condition,_=load_condition(args.condition);config=PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy)
    sources={str(x.resolve()):sha256_file(x) for x in (source_report,state,args.condition,args.checkpoint,Path(__file__),
        args.run_dir/'assessment.json',Path(__file__).with_name('run_training_bottlenecks.py'),
        Path(__file__).with_name('assess_training_bottlenecks.py'))}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'design.json',{'source_hashes':sources,'selection':'all six trained policies and paired frozen controls',
        'reason':'small positive greedy pair-selection screen; fresh confirmation before promotion',
        'validation_offset':260000,'validation_per_rank':8,'repeats':args.repeats,
        'learning_disabled':True,'final_test_used':False,'primary':'greedy pair fraction; compare each learner with its own initial control and 25% with 5%',
        'uncertainty':'2000 whole-case bootstrap draws conditional on three trained learners; no population-level claim'})
    started=time.perf_counter()
    build_corpus(args.output_dir/'pairs.json',train_per_rank=1,validation_per_rank=8,
        training_offset=12500,validation_offset=260000,max_attempts=5000)
    corpus=PairCorpus(args.output_dir/'pairs.json')
    source_corpus=PairCorpus(Path(original['corpus']))
    if {c['seed'] for c in corpus.cases}&{c['seed'] for c in source_corpus.cases}:raise ValueError('confirmation roots overlap screen')
    stack=build_condition_stack(config,condition)
    load_plastic_checkpoint(args.checkpoint,stack.trainer,expected_components=stack.components)
    with np.load(state,allow_pickle=False) as data:weights=data['efficacy'].copy()
    def progress(row):
        write_json(args.output_dir/'status.json',{'status':'running',**row});print(json.dumps(row),flush=True)
    result=evaluate(stack.agent,corpus,weights,replicates=len(original['plastic'])//4,repeats=args.repeats,progress=progress)
    result['replay']=replay(result)
    if any(sha256_file(Path(x))!=h for x,h in sources.items()):raise ValueError('confirmation sources changed')
    result.update(source_hashes=sources,wall_seconds=time.perf_counter()-started)
    write_json(args.output_dir/'report.json',result);write_json(args.output_dir/'status.json',{'status':'completed'})
    print(json.dumps(result['summary'],indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
