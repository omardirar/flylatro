"""Post-hoc activity/eligibility audit of rewarded card-output capacity targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_card_credit_direction import array
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack,load_condition
from card_learning.pairs import PairCorpus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed
from run_card_membership import VERSION,fixed_motor
import train_discard_play_curriculum as delayed


def audit(agent,corpus,report,*,repeats=16):
    if repeats<1 or report['head']!='discard_card':raise ValueError('positive repeats and DISCARD card-output report required')
    source=plastic_hashes(agent.plasticity.state);ids=report['replicate_ids'];n=len(ids)
    local=delayed.local_rule(agent,n//2);motor=fixed_motor(agent,1,1);credit=SelectedMotorCredit(motor,local)
    env=BalatroSimAdapter(n,win_ante=1,allow_skip_blinds=False);env.reset([0]*n)
    for lane,r in enumerate(ids):corpus.restore(env,lane,report['training_cases'][r])
    obs,masks=env.observe();masks=delayed.forced(masks,1);masks['card_select_mask'][:]=False
    for lane,r in enumerate(ids):masks['card_select_mask'][lane,report['slot_pairs'][r]]=True
    target=empty_action_batch(n);target['action_type'][:]=1;target['n_cards'][:]=1
    target['cards'][:,0]=report['target_slots']
    gate=credit.mask(target,masks);g=array(gate).astype(bool)
    pools=[motor.mapping.routing.head_routes['discard_card'][slot] for slot in report['target_slots']]
    rows=[]
    for repeat in range(repeats):
        neural=agent.processor.process(obs,fly_seeds=[derive_seed(VERSION,'binary',1,'evaluation',r,repeat) for r in ids],
                                       efficacy=local.state.initial_efficacy)
        pre,post=array(neural.edge_pre_activity),array(neural.edge_post_activity)
        rates=motor.pool_activity(neural.output_activity)[np.arange(n),pools]
        local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=gate)
        mass=array(local.state.eligibility.sum(axis=1))
        events=local.apply_dopamine(np.full(n,report['reward_pulse'],dtype=np.float32),0)
        rows.append({'target_pool_hz':rates.tolist(),'eligibility_mass':mass.tolist(),
            'update_l1':[e.absolute_change for e in events],
            'selected_pre_active_fraction':[float(np.mean(pre[lane,g[lane]]>0)) for lane in range(n)],
            'selected_post_active_fraction':[float(np.mean(post[lane,g[lane]]>0)) for lane in range(n)]})
        local.state.efficacy[:]=local.state.initial_efficacy
        local.state.reset_fast_traces(eligibility=True)
    if plastic_hashes(agent.plasticity.state)!=source:raise ValueError('audit changed source state')
    summary={key:np.mean([row[key] for row in rows],axis=0).tolist() for key in rows[0]}
    summary['zero_eligibility_fraction']=np.mean(np.asarray([r['eligibility_mass'] for r in rows])==0,axis=0).tolist()
    roots=[[int(motor.mapping.output_root_ids[i]) for i in motor.mapping.pool_indices[p]] for p in pools]
    return {'status':'completed','scope':'post-hoc forced-target positive-pulse probes on initial-weight copies; no native actions; not runtime teaching',
        'repeats':repeats,'target_post_root_ids':roots,'summary':summary,'trials':rows,'source_state_unchanged':True}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','condition','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=16);p.add_argument('--heavy',action='store_true');args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    path=args.run_dir/'report.json';r=json.loads(path.read_text())
    if r['status']!='completed' or r.get('subtype')!='card-membership':raise ValueError('completed card binary required')
    for source,digest in r['source_hashes'].items():
        if sha256_file(Path(source))!=digest:raise ValueError('source changed: '+source)
    condition,_=load_condition(args.condition);config=PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy)
    sources={str(x.resolve()):sha256_file(x) for x in (path,args.condition,args.checkpoint,Path(__file__),Path(__file__).with_name('run_card_membership.py'))}
    corpus=PairCorpus(Path(r['corpus']));stack=build_condition_stack(config,condition)
    load_plastic_checkpoint(args.checkpoint,stack.trainer,expected_components=stack.components)
    result=audit(stack.agent,corpus,r,repeats=args.repeats)
    rewards=np.zeros(len(r['plastic']),dtype=int)
    for line in (args.run_dir/'training-trials.jsonl').open():rewards+=json.loads(line)['rewarded']
    result['training_rewarded_choices']=rewards.tolist();result['training_update_l1']=r['actual_update_l1']
    if any(sha256_file(Path(x))!=h for x,h in sources.items()):raise ValueError('audit source changed')
    result['source_hashes']=sources
    args.output_dir.mkdir(parents=True);write_json(args.output_dir/'report.json',result)
    print(json.dumps(result['summary'],indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
