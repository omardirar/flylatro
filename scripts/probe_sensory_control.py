"""Anatomy-selected ALPN cues and a labelled local association capacity test."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import numpy as np
import torch
from card_learning.common import write_json,identity
from card_learning.motor import install_motor,CardExploration
from flylatro.fly.encoder import Stimulus
from flylatro.learning.config import PlasticExperimentConfig,build_plastic_stack
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.state import state_numpy
from local_normalisation import NormalisedThreeFactorPlasticity
from run_card_membership import fixed_motor
from strategy_puzzles import action
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.seeds import derive_seed


def route_scores(artifact,topology,alpn,target_roots,reference_roots):
    """Signed, normalised two-hop contact score; no neural outcomes used."""
    roots=artifact.root_ids
    delta=np.zeros(len(roots),dtype=np.float64)
    for selected,sign in [(target_roots,1),(reference_roots,-1)]:
        keep=np.isin(topology.post_root_ids,selected)
        weights=topology.anatomical_weights[keep].astype(float)
        np.add.at(delta,topology.pre_indices[keep],sign*weights/weights.sum())
    keep=np.isin(artifact.pre_indices,alpn)&np.isin(artifact.post_indices,topology.pre_indices)&(artifact.signed_synapse_counts>0)
    pre=artifact.pre_indices[keep];post=artifact.post_indices[keep];weights=artifact.signed_synapse_counts[keep]
    mass=np.bincount(pre,weights=weights,minlength=len(roots))
    score=np.bincount(pre,weights=weights*delta[post],minlength=len(roots))
    return np.divide(score[alpn],mass[alpn],out=np.zeros(len(alpn)),where=mass[alpn]>0)


class CueEncoder:
    def __init__(self,base,routes):
        self.base=base;self.routes=routes;self.cues=None
        self.reserved=np.unique(np.concatenate(list(routes.values())))
    def encode(self,obs):
        old=self.base.encode(obs);rates=old.rates_hz.copy()
        rates[:,self.reserved]*=.5
        for row,cue in enumerate(self.cues):
            if cue!='off':rates[row,self.routes[cue]]+=75
        assert rates.max()<=150
        return Stimulus(rates,'anatomical-cue-v1','diagnostic')


def run(output,steps=1000):
    output.mkdir(parents=True,exist_ok=False)
    config=PlasticExperimentConfig.load(Path('artefacts/learnable-motor-20261028-v2/config.toml'))
    config=replace(config,environment=replace(config.environment,num_envs=12),
        training=replace(config.training,max_environment_decisions=steps*12))
    stack=install_motor(build_plastic_stack(config),CardExploration(.05));agent=stack.agent
    base=agent.processor.encoder;topology=agent.plasticity.topology
    artifact=agent.processor.backend.artifact;mapping=agent.motor.mapping
    def output_roots(slot):
        pool=mapping.routing.head_routes['card'][slot]
        return np.array(mapping.output_root_ids)[list(mapping.pool_indices[pool])]
    alpn=base.mapping.available_alpn_indices
    scores=route_scores(artifact,topology,alpn,output_roots(7),output_roots(5))
    order=np.argsort(scores,kind='stable')
    routes={'target':alpn[order[-3:]],'reference':alpn[order[:3]]}
    other=np.setdiff1d(alpn,np.concatenate(list(routes.values())))
    routes['random']=np.random.default_rng(77117).choice(other,3,replace=False)
    encoder=CueEncoder(base,routes);agent.processor.encoder=encoder
    motor=fixed_motor(agent,0,1)
    local=NormalisedThreeFactorPlasticity(topology,agent.plasticity.state,agent.plasticity.config)
    credit=SelectedMotorCredit(motor,local)
    env=BalatroSimAdapter(12,win_ante=1,allow_skip_blinds=False)
    env.reset([4000]*12);env.step(action(2,batch=12));obs,masks=env.observe()
    # Two legal outputs; no simulator step is used as a training reward here.
    masks['card_select_mask'][:]=False;masks['card_select_mask'][:,[5,7]]=True
    roles=np.tile(np.arange(4),3);ids=np.repeat(np.arange(3),4)
    plastic=roles<2;reversed_mapping=(roles%2)==1
    declaration={'version':'anatomical-sensory-control-v1','implementation':identity(),
        'script_sha256':sha256_file(Path(__file__)),'steps':steps,'replicates':3,
        'roles':['aligned-plastic','reversed-plastic','aligned-frozen','reversed-frozen'],
        'routes':{k:artifact.root_ids[v].tolist() for k,v in routes.items()},
        'route_scores':{'target':scores[order[-3:]].tolist(),'reference':scores[order[:3]].tolist()},
        'limits':'synthetic binary cue association, fixed background; reward is diagnostic correctness +/-0.05, not native Balatro progress; route chosen from anatomy only; not a gameplay improvement',
        'intervention':'shared 50% headroom on reserved ALPNs, +75 Hz on three cue neurons'}
    write_json(output/'declaration.json',declaration)
    def forward(cues,key,learn=False):
        encoder.cues=cues
        neural=agent.processor.process(obs,fly_seeds=[derive_seed('sensory-control',key,int(rep)) for rep in ids],efficacy=local.state.efficacy)
        a=motor.decode(neural.output_activity,masks,deterministic=not learn,learner_ids=ids,decision_ids=[derive_seed('sensory-control',key,'action')]*12)
        return neural,a
    def evaluate():
        rows=[]
        for cue in ['off','target','reference','random']:
            for repeat in range(16):
                neural,a=forward([cue]*12,('eval',cue,repeat))
                scores=motor.head_scores(neural.output_activity)['card']
                rows.append({'cue':cue,'repeat':repeat,'cards':a['cards'][:,0].tolist(),
                    'target_gap':(scores[:,7]-scores[:,5]).tolist()})
        return rows
    initial=evaluate()
    with (output/'training.jsonl').open('w') as stream:
        for step in range(steps):
            cue='target' if step%2==0 else 'reference'
            desired=np.where(reversed_mapping,5 if cue=='target' else 7,7 if cue=='target' else 5)
            neural,a=forward([cue]*12,('train',step),True)
            correct=a['cards'][:,0]==desired
            local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=credit.mask(a,masks))
            positive=(correct&plastic)*.05;negative=(~correct&plastic)*.05
            local.apply_dopamine(positive,negative);local.state.reset_fast_traces(eligibility=True)
            stream.write(json.dumps({'step':step,'cue':cue,'cards':a['cards'][:,0].tolist(),'desired':desired.tolist(),'correct':correct.tolist()})+'\n')
            if (step+1)%250==0:
                w=state_numpy(local.state.efficacy)
                if np.max(np.mean((w<=.2)|(w>=2),axis=1))>.1 or np.max(np.mean(abs(w-1),axis=1))>.5:raise ValueError('safety stop')
                print(f'{step+1}/{steps}',flush=True)
    final=evaluate();state=local.state.state_dict()
    np.testing.assert_array_equal(state['efficacy'][~plastic],state['initial_efficacy'][~plastic])
    for a,b in zip(initial,final):np.testing.assert_array_equal(np.array(a['cards'])[~plastic],np.array(b['cards'])[~plastic])
    errors=np.max(abs(state_numpy(local.incoming_strengths())-state_numpy(local.budgets))/state_numpy(local.budgets),axis=1)
    assert max(errors)<1e-6
    np.savez_compressed(output/'checkpoint.npz',**state)
    summary={}
    for role,name in enumerate(declaration['roles']):
        summary[name]={}
        for label,rows in [('initial',initial),('final',final)]:
            correct=[]
            for r in rows:
                if r['cue'] not in ('target','reference'):continue
                desired=7 if (r['cue']=='target')!=(role%2==1) else 5
                correct.append(np.array(r['cards'])[roles==role]==desired)
            summary[name][label]=np.mean(correct,axis=0).tolist()
    write_json(output/'report.json',{'status':'completed','initial':initial,'final':final,'summary':summary,
        'roles':roles.tolist(),'budget_error':errors.tolist(),'checkpoint_sha256':sha256_file(output/'checkpoint.npz')})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--heavy',action='store_true');a=p.parse_args()
    if not a.heavy or a.steps<1:p.error('positive steps and --heavy required')
    run(a.output_dir,a.steps)
