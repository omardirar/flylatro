"""Reward-free fresh KC-to-MBON eligibility coverage of fixed motor options."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_card_credit_direction import array
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack,load_condition
from evaluate_motor_sampling import repeat_efficacy
from flylatro.analysis.corpus import CalibrationCorpus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.seeds import derive_seed


def aggregate(pre,post,inverse,posts,config):
    pre,post=array(pre),array(post)
    if pre.shape!=post.shape or pre.ndim!=2 or pre.shape[1]!=len(inverse):raise ValueError('aligned edge activity required')
    a=np.clip(pre/config.kc_reference_hz,0,1);b=np.clip(post/config.mbon_reference_hz,0,1)
    fresh=np.clip(a*b,0,config.max_eligibility)
    mass=np.asarray([np.bincount(inverse,weights=row,minlength=posts) for row in fresh])
    presynaptic=np.asarray([np.bincount(inverse,weights=row,minlength=posts) for row in a])
    first=[np.flatnonzero(inverse==i)[0] for i in range(posts)]
    return mass,presynaptic,post[:,first]


def audit(agent,observations,masks,*,repeats=2,batch_size=4):
    if min(repeats,batch_size)<1:raise ValueError('positive budgets required')
    before=plastic_hashes(agent.plasticity.state);topology=agent.plasticity.topology
    roots,inverse=np.unique(topology.post_root_ids,return_inverse=True)
    cases=len(next(iter(observations.values())))
    coactivity=np.zeros((cases,repeats,len(roots)));pre_mass=coactivity.copy();post_rate=coactivity.copy()
    for repeat in range(repeats):
        for start in range(0,cases,batch_size):
            indices=np.arange(start,min(start+batch_size,cases))
            obs={k:v[indices].copy() for k,v in observations.items()}
            neural=agent.processor.process(obs,fly_seeds=[derive_seed('motor-learning-opportunities-v1',int(i),repeat) for i in indices],
                efficacy=repeat_efficacy(agent.plasticity.state.initial_efficacy[:1],len(indices)))
            mass,pre,post=aggregate(neural.edge_pre_activity,neural.edge_post_activity,inverse,len(roots),agent.plasticity.config)
            coactivity[indices,repeat]=mass;pre_mass[indices,repeat]=pre;post_rate[indices,repeat]=post
    if plastic_hashes(agent.plasticity.state)!=before:raise ValueError('read-only coverage audit changed source state')
    lookup={int(root):i for i,root in enumerate(roots)};mapping=agent.motor.mapping
    roles=[]
    for window in motor_context_windows(masks,routing=mapping.routing).values():
        head=window.spec.head
        for option,pool in enumerate(mapping.routing.head_routes[head]):
            if pool<0:continue
            selected=window.relevant & (window.option_legal.sum(axis=1)>1) & window.option_legal[:,option]
            ids=[lookup[int(mapping.output_root_ids[i])] for i in mapping.pool_indices[pool]]
            values=coactivity[selected][:,:,ids].sum(axis=2)
            roles.append({'head':head,'option':option,'post_root_ids':roots[ids].tolist(),
                'competing_cases':int(selected.sum()),
                'never_coactive_in_tested_contexts':bool(len(values) and not np.any(values>0)),
                'coactive_sample_fraction':float(np.mean(values>0)) if len(values) else None,
                'mean_fresh_eligibility_mass':float(values.mean()) if len(values) else None})
    root_summary=[]
    for i,root in enumerate(roots):
        root_summary.append({'root_id':int(root),'kc_inputs':int(np.count_nonzero(inverse==i)),
            'coactive_sample_fraction':float(np.mean(coactivity[:,:,i]>0)),
            'presynaptically_active_sample_fraction':float(np.mean(pre_mass[:,:,i]>0)),
            'postsynaptically_active_sample_fraction':float(np.mean(post_rate[:,:,i]>0))})
    return {'status':'completed','scope':'read-only reward-free calibration activity; no environment scores, learning or mapping changes',
        'cases':cases,'repeats':repeats,'source_state_unchanged':True,'roles':roles,'roots':root_summary}, {
            'coactivity':coactivity,'presynaptic_mass':pre_mass,'post_rate_hz':post_rate,'post_root_ids':roots}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('condition','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--heavy',action='store_true');args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    condition,_=load_condition(args.condition);config=PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy);corpus_path=Path(config.calibration.corpus_path)
    corpus=CalibrationCorpus.load(corpus_path)
    sources={str(x.resolve()):sha256_file(x) for x in (args.condition,args.checkpoint,corpus_path,Path(__file__))}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'design.json',{'source_hashes':sources,'repeats':args.repeats,
        'selection':'every state in the existing reward-free calibration corpus; all current motor roles with competing legal options',
        'criterion':'report exactly zero eligible coactivity; no threshold tuning or mapping replacement',
        'scope':'post-hoc learning-opportunity diagnosis; no game-quality or generalization claim'})
    stack=build_condition_stack(config,condition)
    load_plastic_checkpoint(args.checkpoint,stack.trainer,expected_components=stack.components)
    result,arrays=audit(stack.agent,corpus.observations,corpus.masks,repeats=args.repeats)
    if any(sha256_file(Path(x))!=h for x,h in sources.items()):raise ValueError('audit source changed')
    np.savez_compressed(args.output_dir/'activity.npz',**arrays)
    result.update(source_hashes=sources,corpus_sha256=corpus.sha256)
    write_json(args.output_dir/'report.json',result)
    print(json.dumps({'cases':result['cases'],'zero_opportunity_roles':[x for x in result['roles'] if x['never_coactive_in_tested_contexts']]},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
