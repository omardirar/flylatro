"""Fresh greedy validation of all fixed-count DISCARD actors, without learning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from assess_card_membership import replay
from assess_training_bottlenecks import verify_state
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack,load_condition
from card_learning.pairs import PairCorpus,build_corpus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.plasticity import ThreeFactorPlasticity
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed
from run_card_membership import VERSION,fixed_motor,native_layout,native_trial


def evaluate(agent,corpus,values,*,replicates,count=2,repeats=2,progress=None):
    if repeats<1:raise ValueError('positive repeats required')
    ids,role,plastic=native_layout(replicates);n=len(ids)
    source=plastic_hashes(agent.plasticity.state)
    factory=type(agent.plasticity.state)
    if values['efficacy'].shape!=(n,agent.plasticity.topology.edge_count):raise ValueError('confirmation state shape differs')
    state=(factory.from_state_dict(values,device=str(agent.plasticity.state.device))
           if getattr(agent.plasticity.state,'is_torch',False) else factory.from_state_dict(values))
    local=ThreeFactorPlasticity(agent.plasticity.topology,state,agent.plasticity.config)
    local_before=plastic_hashes(local.state)
    motor=fixed_motor(agent,1,count);credit=SelectedMotorCredit(motor,local)
    env=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);env.reset([0]*n)
    alternate=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);alternate.reset([0]*n)
    rows=[];indices=corpus.indices('validation')
    for number,index in enumerate(indices):
        for repeat in range(repeats):
            rows.append(native_trial(agent,corpus,env,alternate,local,motor,credit,ids,role,[index]*n,
                ('fresh-confirmation',repeat,'greedy'),sampled=False,uniform=True))
        if progress and (number+1)%13==0:progress({'completed_cases':number+1})
    if plastic_hashes(local.state)!=local_before or plastic_hashes(agent.plasticity.state)!=source:
        raise ValueError('frozen confirmation changed state')
    summary={}
    for arm,label in enumerate(('neural-native','reference-native','reference-matched')):
        lanes=np.flatnonzero(role==arm);result={}
        for followup in ('weak','reference'):
            values=np.asarray([row[followup+'_progress'] for row in rows])
            model,control=values[:,lanes],values[:,3::4];difference=model-control
            per_case=difference.reshape(len(indices),repeats,replicates).mean(axis=(1,2))
            rng=np.random.default_rng(derive_seed(VERSION,'fresh-bootstrap',arm,followup))
            boot=per_case[rng.integers(0,len(indices),size=(2000,len(indices)))].mean(axis=1)
            result[followup]={'model_mean':float(model.mean()),'initial_mean':float(control.mean()),
                'model_per_learner':model.mean(axis=0).tolist(),'initial_per_learner':control.mean(axis=0).tolist(),
                'change_per_learner':difference.mean(axis=0).tolist(),'mean_change':float(difference.mean()),
                'case_bootstrap_95_interval_conditional_on_these_learners':np.quantile(boot,[.025,.975]).tolist()}
        uniform=np.asarray([row['uniform_reference_progress'] for row in rows])[:,lanes]
        actual=np.asarray([row['reference_progress'] for row in rows])[:,lanes]
        result['reference_minus_uniform_per_learner']=(actual-uniform).mean(axis=0).tolist()
        summary[label]=result
    return {'study':'fixed-membership','status':'completed','scope':'fresh greedy validation; no learning; one shared frozen actor per replicate',
        'roles':role.tolist(),'plastic':plastic.tolist(),'replicate_ids':ids,'count':count,
        'calibration_rows':[],'before':{'greedy':[]},'after':{'greedy':rows},'summary':summary,
        'corpus':str(corpus.path),'corpus_sha256':corpus.sha256,'source_state_unchanged':True}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','condition','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--heavy',action='store_true');args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    source_report=args.run_dir/'report.json';state_path=args.run_dir/'diagnostic-final-state.npz'
    report=json.loads(source_report.read_text());assessment_path=args.run_dir/'assessment.json'
    assessment=json.loads(assessment_path.read_text())
    if report['status']!='completed' or report['study']!='fixed-membership':raise ValueError('completed native screen required')
    if assessment['source_report_sha256']!=sha256_file(source_report) or assessment['final_state_sha256']!=sha256_file(state_path):
        raise ValueError('screen assessment/state differs')
    for path,digest in report['source_hashes'].items():
        if sha256_file(Path(path))!=digest:raise ValueError(f'screen source changed: {path}')
    verify_state({**report,'study':'followup'},state_path)
    condition,_=load_condition(args.condition);config=PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy)
    paths=(source_report,state_path,assessment_path,args.condition,args.checkpoint,Path(__file__),
        Path(__file__).with_name('run_card_membership.py'),Path(__file__).with_name('assess_card_membership.py'))
    sources={str(x.resolve()):sha256_file(x) for x in paths}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'design.json',{'source_hashes':sources,'learning_disabled':True,
        'validation_offset':300000,'validation_per_rank':8,'repeats':args.repeats,'final_test_used':False,
        'selection':'all three treatment actors plus shared initial control; greedy membership; both frozen PLAY policies',
        'scope':'fresh cases, no reward/weight selection; case-bootstrap intervals conditional on original learner seeds'})
    started=time.perf_counter()
    build_corpus(args.output_dir/'pairs.json',train_per_rank=1,validation_per_rank=8,
        training_offset=16500,validation_offset=300000,max_attempts=5000)
    corpus=PairCorpus(args.output_dir/'pairs.json');old=PairCorpus(Path(report['corpus']))
    if {c['seed'] for c in corpus.cases}&{c['seed'] for c in old.cases}:raise ValueError('confirmation cases overlap screen')
    stack=build_condition_stack(config,condition)
    load_plastic_checkpoint(args.checkpoint,stack.trainer,expected_components=stack.components)
    with np.load(state_path,allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
    def progress(row):
        write_json(args.output_dir/'status.json',{'status':'running',**row});print(json.dumps(row),flush=True)
    result=evaluate(stack.agent,corpus,values,replicates=len(report['plastic'])//4,count=report['count'],repeats=args.repeats,progress=progress)
    result['replay']=replay(result)
    if any(sha256_file(Path(x))!=h for x,h in sources.items()):raise ValueError('confirmation source changed')
    result.update(source_hashes=sources,wall_seconds=time.perf_counter()-started)
    write_json(args.output_dir/'report.json',result);write_json(args.output_dir/'status.json',{'status':'completed'})
    print(json.dumps(result['summary'],indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
