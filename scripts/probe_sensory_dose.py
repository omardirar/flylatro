"""Frozen dose response of anatomy-routed cues; no training or route fitting."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import numpy as np
from card_learning.common import write_json
from card_learning.motor import install_motor,CardExploration
from flylatro.learning.config import PlasticExperimentConfig,build_plastic_stack
from flylatro.fly.encoder import Stimulus
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.env.balatro_sim import BalatroSimAdapter
from probe_sensory_control import route_scores
from strategy_puzzles import action


class DoseEncoder:
    def __init__(self,base,routes):
        self.base=base;self.routes=routes;self.width=3;self.amplitude=75;self.cue='off'
        self.reserved=np.unique(np.concatenate(list(routes.values())))
    def encode(self,obs):
        original=self.base.encode(obs);rates=original.rates_hz.copy()
        rates[:,self.reserved]*=.5
        if self.cue!='off':rates[:,self.routes[self.cue][:self.width]]+=self.amplitude
        assert rates.max()<=150
        return Stimulus(rates,'frozen-sensory-dose-v1','diagnostic')


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    config=PlasticExperimentConfig.load(Path('artefacts/learnable-motor-20261028-v2/config.toml'))
    config=replace(config,environment=replace(config.environment,num_envs=1))
    stack=install_motor(build_plastic_stack(config),CardExploration(0));agent=stack.agent
    base=agent.processor.encoder;artifact=agent.processor.backend.artifact;mapping=agent.motor.mapping
    def roots(slot):return np.array(mapping.output_root_ids)[list(mapping.pool_indices[mapping.routing.head_routes['card'][slot]])]
    alpn=base.mapping.available_alpn_indices
    score=route_scores(artifact,agent.plasticity.topology,alpn,roots(7),roots(5))
    order=np.argsort(score,kind='stable')
    routes={'target':alpn[order[::-1][:24]],'reference':alpn[order[:24]]}
    routes['random']=np.random.default_rng(77117).choice(np.setdiff1d(alpn,np.concatenate(list(routes.values()))),24,replace=False)
    encoder=DoseEncoder(base,routes);agent.processor.encoder=encoder
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([4000]);env.step(action(2));obs,_=env.observe()
    before=agent.plasticity.state.weight_sha256;rows=[]
    for width in [3,12,24]:
        for amplitude in [25,75]:
            encoder.width=width;encoder.amplitude=amplitude
            gaps={cue:[] for cue in ['off','target','reference','random']}
            for cue in gaps:
                encoder.cue=cue
                for repeat in range(8):
                    neural=agent.processor.process(obs,fly_seeds=[93000+repeat],efficacy=agent.plasticity.state.efficacy)
                    scores=agent.motor.head_scores(neural.output_activity)['card'][0]
                    gaps[cue].append(float(scores[7]-scores[5]))
            target=np.array(gaps['target']);reference=np.array(gaps['reference']);off=np.array(gaps['off'])
            rows.append({'width':width,'added_hz_per_neuron':amplitude,'gaps':gaps,
                'target_minus_reference_mean_gap':float((target-reference).mean()),
                'target_minus_off_mean_gap':float((target-off).mean()),
                'reference_minus_off_mean_gap':float((reference-off).mean()),
                'aligned_accuracy':float(((target>0).sum()+(reference<=0).sum())/16)})
    assert agent.plasticity.state.weight_sha256==before
    write_json(output/'report.json',{'status':'completed','rows':rows,'script_sha256':sha256_file(Path(__file__)),
        'routes':{k:artifact.root_ids[v].tolist() for k,v in routes.items()},
        'limits':'one background, initial weights, eight shared neural seeds; broad reserved headroom differs from three-neuron learning control; grid is exploratory and cannot select a validated training setting'})
    print(json.dumps(rows))


if __name__=='__main__':run(Path(sys.argv[1]))
