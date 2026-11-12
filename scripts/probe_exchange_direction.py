"""Disposable causal efficacy interventions; source checkpoints stay frozen."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import numpy as np
import torch

from card_learning.common import write_json
from card_learning.motor import install_motor, CardExploration
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.analysis.card_selection import permute_hand
from local_normalisation import NormalisedThreeFactorPlasticity
from run_card_membership import fixed_motor
from strategy_puzzles import action
from run_exchange_credit import signed_tags
from flylatro.env.balatro_sim import BalatroSimAdapter


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    source=Path('artefacts/exchange-credit-20261106-v1/checkpoint.npz')
    source_hash=sha256_file(source)
    config=PlasticExperimentConfig.load(Path('artefacts/learnable-motor-20261028-v2/config.toml'))
    config=replace(config,environment=replace(config.environment,num_envs=1))
    stack=install_motor(build_plastic_stack(config),CardExploration(0));agent=stack.agent
    motor=fixed_motor(agent,0,5)
    local=NormalisedThreeFactorPlasticity(agent.plasticity.topology,agent.plasticity.state,agent.plasticity.config)
    credit=SelectedMotorCredit(motor,local)
    routes=motor.mapping.routing.head_routes['card']
    slot_edges=np.stack([credit.pool_posts[p][credit.edge_posts] for p in routes])
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False)
    env.reset([4000]);env.step(action(2));obs,masks=env.observe();snapshot=env.snapshot(0)
    bad=np.array([[0,1,4,5,6]]);good=np.array([[0,1,4,6,7]])
    scores=[]
    for cards in (bad[0],good[0]):
        env.restore(0,snapshot);result=env.step(action(0,cards.tolist()))
        scores.append(float(result.rewards[0])*300)
    assert abs(scores[0]-60)<1e-4 and abs(scores[1]-92)<1e-4
    pulse=.05*(scores[1]-scores[0])/300
    with np.load(source) as z:trained=z['efficacy'][3:4].copy()
    report={'status':'running','source_checkpoint_sha256':source_hash,
        'script_sha256':sha256_file(Path(__file__)),'native_scores':scores,
        'intervention':'forced beneficial exchange; native paired score advantage, not the pilot EMA; diagnostic only',
        'limits':'one hand, one trained lane, eight shared neural seeds; scaled one-update directions are sensitivity probes, not repeated training',
        'rows':[],'directions':[]}
    for weight_name,base_np in [('initial',np.ones_like(trained)),('trained',trained)]:
        base=torch.as_tensor(base_np,device=local.state.device).clone()
        local.state.efficacy.copy_(base)
        # Budgets always derive from unit initial weights, as in the pilot.
        neural=agent.processor.process(obs,fly_seeds=[71000],efficacy=base)
        pre=torch.clamp(neural.edge_pre_activity/local.config.kc_reference_hz,0,1)
        post=torch.clamp(neural.edge_post_activity/local.config.mbon_reference_hz,0,1)
        for kind in ('signed','selected'):
            local.state.efficacy.copy_(base);local.state.reset_fast_traces(eligibility=True)
            if kind=='signed':
                tags=torch.as_tensor(signed_tags(good,bad)@slot_edges.astype(np.float32),device=base.device)
                local.state.eligibility.copy_(pre*post*tags)
            else:
                actual=action(0,good[0].tolist())
                local.record_activity(neural.edge_pre_activity,neural.edge_post_activity,edge_credit=credit.mask(actual,masks))
            local.apply_dopamine(pulse,0)
            direction=local.state.efficacy-base
            norm=float(direction.abs().sum())
            assert norm>0
            report['directions'].append({'weights':weight_name,'kind':kind,'actual_update_l1':norm})
            for target in (norm,.1,1.,10.):
                for sign in (1,-1):
                    local.state.efficacy.copy_((base+sign*direction*(target/norm)).clamp(.2,2))
                    local._project(base)
                    candidate=local.state.efficacy.clone()
                    magnitude=float((candidate-base).abs().sum())
                    error=float(((local.incoming_strengths()-local.budgets).abs()/local.budgets).max())
                    assert error<1e-6
                    for context in ('original','permuted'):
                        order=np.arange(10) if context=='original' else np.r_[np.arange(7,-1,-1),8,9]
                        o,m=permute_hand(obs,masks,order)
                        enter=int(np.flatnonzero(order==7)[0]);leave=int(np.flatnonzero(order==5)[0])
                        for repeat in range(8):
                            seed=71000+repeat
                            b=agent.processor.process(o,fly_seeds=[seed],efficacy=base)
                            after=agent.processor.process(o,fly_seeds=[seed],efficacy=candidate)
                            restored=agent.processor.process(o,fly_seeds=[seed],efficacy=base)
                            np.testing.assert_array_equal(b.output_activity,restored.output_activity)
                            def gap(neural):
                                s=motor.head_scores(neural.output_activity)['card'][0]
                                return float(s[enter]-s[leave])
                            report['rows'].append({'weights':weight_name,'kind':kind,'target_l1':target,
                                'actual_l1':magnitude,'sign':sign,'context':context,'seed':seed,
                                'gap_before':gap(b),'gap_after':gap(after),'delta_gap':gap(after)-gap(b)})
        local.state.efficacy.copy_(base)
    summary=[]
    for key in sorted(set((r['weights'],r['kind'],r['target_l1'],r['sign'],r['context']) for r in report['rows'])):
        rows=[r for r in report['rows'] if (r['weights'],r['kind'],r['target_l1'],r['sign'],r['context'])==key]
        changes=np.array([r['delta_gap'] for r in rows])
        summary.append(dict(zip(['weights','kind','target_l1','sign','context'],key),
            actual_l1=rows[0]['actual_l1'],mean_delta_gap=float(changes.mean()),
            improves=int((changes>1e-9).sum()),worsens=int((changes< -1e-9).sum()),unchanged=int((abs(changes)<=1e-9).sum())))
    assert sha256_file(source)==source_hash
    report.update(status='completed',summary=summary,exact_restoration_checks=len(report['rows']))
    write_json(output/'report.json',report)
    print(json.dumps(summary))


if __name__=='__main__':run(Path(sys.argv[1]))
