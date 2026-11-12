"""Native shuffled pair learning: relationship/off/wrong, plastic/frozen."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import time
import numpy as np

from card_learning.common import write_json,identity
from card_learning.pairs import PairCorpus,pair_slots
from card_learning.motor import install_motor,CardExploration
from flylatro.learning.config import PlasticExperimentConfig,build_plastic_stack
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.seeds import derive_seed
from local_normalisation import NormalisedThreeFactorPlasticity
from run_card_membership import fixed_motor
from train_discard_play_curriculum import permute_rows,translate,forced
from relational_sensory import RelationalEncoder

ARMS=['off-plastic','true-plastic','wrong-plastic','off-frozen','true-frozen','wrong-frozen']
VERSION='relational-sensory-pilot-v1'


def run(output,steps=1000):
    output.mkdir(parents=True,exist_ok=False)
    corpus_path=Path('artefacts/local-normalisation-20261031-v1/pairs.json')
    corpus=PairCorpus(corpus_path)
    config_path=Path('artefacts/learnable-motor-20261028-v2/config.toml')
    config=PlasticExperimentConfig.load(config_path)
    config=replace(config,environment=replace(config.environment,num_envs=18),
        training=replace(config.training,max_environment_decisions=18*steps))
    roles=np.tile(np.arange(6),3);ids=np.repeat(np.arange(3),6)
    modes=[ARMS[r].split('-')[0] for r in roles];plastic=roles<3
    declaration={'version':VERSION,'steps':steps,'replicates':3,'arms':ARMS,
        'input_sha256':{str(p):sha256_file(p) for p in [config_path,corpus_path,Path(__file__),Path(__file__).with_name('relational_sensory.py'),Path(__file__).with_name('local_normalisation.py')]},
        'implementation':identity(),'fixed_count':2,'exploration':.05,
        'training_cases':'104 one-pair hands; random display each trial; native progress feedback',
        'evaluation':'first two hands per rank in each split, two neural/display repeats; same cases before/after',
        'limits':'feature comparison, no full games; existing diagnostic corpus reused; no best-hand label in reward; all modes share reduced headroom on reserved sensory roots'}
    write_json(output/'declaration.json',declaration)
    stack=install_motor(build_plastic_stack(config),CardExploration(.05))
    agent=stack.agent;encoder=RelationalEncoder(agent.processor.encoder,modes,ids)
    agent.processor.encoder=encoder
    write_json(output/'sensory-map.json',{str(rep):indices.tolist() for rep,indices in encoder.indices.items()})
    local=NormalisedThreeFactorPlasticity(agent.plasticity.topology,agent.plasticity.state,agent.plasticity.config)
    motor=fixed_motor(agent,0,2);credit=SelectedMotorCredit(motor,local)
    env=BalatroSimAdapter(18,win_ante=1,allow_skip_blinds=False);env.reset([0]*18)
    def trial(indices,key,learn=False):
        # Native restore in one batch; refresh observations/reward cache once.
        for row,index in enumerate(indices):env._env.restore(row,corpus.cases[index]['_snapshot'])
        original,masks=env.observe()
        orders=[np.r_[np.random.default_rng(derive_seed(VERSION,key,int(rep),'display')).permutation(8),8,9] for rep in ids]
        obs,masks=permute_rows(original,forced(masks,0),orders)
        neural=agent.processor.process(obs,fly_seeds=[derive_seed(VERSION,key,int(rep),'neural') for rep in ids],efficacy=local.state.efficacy)
        actions=motor.decode(neural.output_activity,masks,deterministic=not learn,learner_ids=ids,decision_ids=[derive_seed(VERSION,key,'motor')]*18)
        result=env.step(translate(actions,orders))
        if np.any(result.dones):raise ValueError('unexpected terminal')
        pulse=np.array([agent.reinforcement.map(i['reward_components'],i).appetitive for i in result.infos])
        if learn:
            local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=credit.mask(actions,masks))
            pulse[~plastic]=0;local.apply_dopamine(pulse,0)
            local.state.reset_fast_traces(eligibility=True)
        return {'case_indices':list(map(int,indices)),'orders':[o.tolist() for o in orders],
            'cards':actions['cards'][:,:2].tolist(),'pair':[set(actions['cards'][i,:2])==set(pair_slots(obs,i)) for i in range(18)],
            'native_progress':[i['reward_components']['blind_progress'] for i in result.infos]}
    def evaluate():
        result={}
        for split in ['training','validation']:
            counts={};selected=[]
            for index in corpus.indices(split):
                rank=corpus.cases[index]['pair_rank']
                if counts.get(rank,0)<2:selected.append(index);counts[rank]=counts.get(rank,0)+1
            result[split]=[trial([i]*18,('eval',split,i,repeat)) for i in selected for repeat in range(2)]
        return result
    initial=evaluate();started=time.monotonic()
    with (output/'training.jsonl').open('w') as stream:
        for step in range(steps):
            choices=corpus.indices('training')
            indices=[choices[derive_seed(VERSION,step,int(rep),'case')%len(choices)] for rep in ids]
            row=trial(indices,('training',step),True);row['step']=step
            stream.write(json.dumps(row)+'\n')
            if (step+1)%250==0:
                w=state_numpy(local.state.efficacy)
                if np.max(np.mean((w<=.2)|(w>=2),axis=1))>.1 or np.max(np.mean(abs(w-1),axis=1))>.5:raise ValueError('safety stop')
                print(f'{step+1}/{steps}',flush=True)
    final=evaluate();state=local.state.state_dict()
    np.testing.assert_array_equal(state['efficacy'][~plastic],state['initial_efficacy'][~plastic])
    for split in initial:
        for a,b in zip(initial[split],final[split]):
            np.testing.assert_array_equal(np.array(a['cards'])[~plastic],np.array(b['cards'])[~plastic])
    error=np.max(abs(state_numpy(local.incoming_strengths())-state_numpy(local.budgets))/state_numpy(local.budgets),axis=1)
    assert np.max(error)<1e-6
    np.savez_compressed(output/'checkpoint.npz',**state)
    summary={}
    for role,arm in enumerate(ARMS):
        summary[arm]={}
        for split in initial:
            a=np.array([r['pair'] for r in initial[split]])[:,roles==role].mean(axis=0)
            b=np.array([r['pair'] for r in final[split]])[:,roles==role].mean(axis=0)
            summary[arm][split]={'initial':a.tolist(),'final':b.tolist(),'delta_mean':float((b-a).mean())}
    write_json(output/'report.json',{'status':'completed','summary':summary,'initial':initial,'final':final,
        'roles':roles.tolist(),'replicates':ids.tolist(),'budget_error':error.tolist(),
        'checkpoint_sha256':sha256_file(output/'checkpoint.npz'),'seconds':time.monotonic()-started})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--heavy',action='store_true');p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    if not a.heavy:p.error('requires --heavy')
    if a.steps<1:p.error('positive steps required')
    run(a.output_dir,a.steps)
