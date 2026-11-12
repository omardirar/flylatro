"""Frozen rank-matching factorial contrasts; no labels enter neural inputs."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import numpy as np
import torch
from card_learning.common import write_json
from card_learning.motor import install_motor, CardExploration
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.analysis.card_selection import permute_hand
from run_card_membership import fixed_motor
from strategy_puzzles import action


def rank_case(obs, target_rank, partner_rank):
    result={k:v.copy() for k,v in obs.items()}
    for slot,rank in [(7,target_rank),(6,partner_rank)]:
        result['hand'][0,slot,:13]=0
        result['hand'][0,slot,rank-2]=1
    return result


def contrast(values):
    # [condition AA, AB, BA, BB; neural repeat; population]
    x=np.asarray(values,dtype=float)
    d=(x[0]+x[3]-x[1]-x[2])/2
    pairs=[float(np.mean(d[i]*d[j])) for i in range(len(d)) for j in range(i)]
    noise=float(np.var(x,axis=1,ddof=1).mean())
    return {'matching_minus_nonmatching_mean':float(d.mean()),
        'mean_contrast_rms':float(np.sqrt(np.mean(d.mean(axis=0)**2))),
        'repeat_noise_rms':float(np.sqrt(noise)),
        'cross_repeat_contrast_power':float(np.mean(pairs)),
        'positive_contrast_repeats':int(np.sum(d.mean(axis=1)>1e-9)),
        'negative_contrast_repeats':int(np.sum(d.mean(axis=1)<-1e-9))}


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    source=Path('artefacts/exchange-credit-20261106-v1/checkpoint.npz')
    digest=sha256_file(source)
    config=PlasticExperimentConfig.load(Path('artefacts/learnable-motor-20261028-v2/config.toml'))
    config=replace(config,environment=replace(config.environment,num_envs=1))
    stack=install_motor(build_plastic_stack(config),CardExploration(0));agent=stack.agent
    motor=fixed_motor(agent,0,5)
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False)
    env.reset([4000]);env.step(action(2));obs,masks=env.observe()
    with np.load(source) as z:trained=z['efficacy'][3:4].copy()
    records=[];arrays={};controls=0
    for name,w in [('initial',np.ones_like(trained)),('trained',trained)]:
        efficacy=torch.as_tensor(w,device=agent.plasticity.state.device)
        for ranks in [(2,3),(7,8)]:
            cases=[rank_case(obs,a,b) for a,b in [(ranks[0],ranks[0]),ranks,(ranks[1],ranks[0]),(ranks[1],ranks[1])]]
            for display in ['original','reversed']:
                order=np.arange(10) if display=='original' else np.r_[np.arange(7,-1,-1),8,9]
                target=int(np.flatnonzero(order==7)[0]);reference=int(np.flatnonzero(order==5)[0])
                pool=motor.mapping.routing.head_routes['card'][target]
                roots=np.array(motor.mapping.output_root_ids)[list(motor.mapping.pool_indices[pool])]
                topology=agent.plasticity.topology
                edge_mask=np.isin(topology.post_root_ids,roots)
                input_roots=np.unique(topology.pre_root_ids[edge_mask])
                kc_mask=np.isin(agent.processor.kc_root_ids,input_roots)
                populations={k:[] for k in ['kc','target_afferents','target_coactivity','target_rate','card_gap','sensory']}
                for index,case in enumerate(cases):
                    o,m=permute_hand(case,masks,order)
                    samples={k:[] for k in populations}
                    for repeat in range(8):
                        neural=agent.processor.process(o,fly_seeds=[82000+repeat],efficacy=efficacy)
                        kc=np.asarray(neural.kc_activity)[0]
                        samples['kc'].append(kc)
                        samples['target_afferents'].append(kc[kc_mask])
                        pre=neural.edge_pre_activity[0,edge_mask].detach().cpu().numpy()
                        post=neural.edge_post_activity[0,edge_mask].detach().cpu().numpy()
                        samples['target_coactivity'].append(np.clip(pre/20,0,1)*np.clip(post/20,0,1))
                        pools=motor.pool_activity(neural.output_activity)[0]
                        samples['target_rate'].append([pools[pool]])
                        scores=motor.head_scores(neural.output_activity)['card'][0]
                        samples['card_gap'].append([scores[target]-scores[reference]])
                        encoded=np.asarray(agent.processor.encoder.encode(o).rates_hz)[0]
                        samples['sensory'].append(encoded[agent.processor.encoder.mapping.available_alpn_indices])
                        if repeat==0:
                            exact=agent.processor.process(o,fly_seeds=[82000],efficacy=efficacy)
                            np.testing.assert_array_equal(neural.output_activity,exact.output_activity);controls+=1
                    for k in populations:populations[k].append(samples[k])
                key=f'{name}-{ranks[0]}-{ranks[1]}-{display}'
                summaries={}
                for k,v in populations.items():
                    value=np.array(v);arrays[f'{key}-{k}']=value
                    summaries[k]=contrast(value)
                records.append({'weights':name,'ranks':ranks,'display':display,
                    'target_unique_kc_inputs':int(kc_mask.sum()),'populations':summaries})
                print(key,json.dumps(summaries['card_gap']),flush=True)
    assert sha256_file(source)==digest
    np.savez_compressed(output/'responses.npz',**arrays)
    report={'status':'completed','source_sha256':digest,'script_sha256':sha256_file(Path(__file__)),
        'responses_sha256':sha256_file(output/'responses.npz'),'exact_repeat_checks':controls,'records':records,
        'scope':'rank matching interaction after cancelling additive rank effects; two rank pairs, one background, two displays, eight neural seeds; no learning',
        'limits':'synthetic visible input edits leave deck summaries unchanged; not simulator worlds, general decoding tests or proof of absent information; cross-repeat power may be negative at finite sample size'}
    write_json(output/'report.json',report)


if __name__=='__main__':run(Path(sys.argv[1]))
