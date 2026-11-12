"""Quantify same-input repeat differences at a learned source checkpoint."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import numpy as np
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
with a.checkpoint.open('rb') as f: saved=pickle.load(f)
config=PlasticExperimentConfig.load(a.config)
result={}
for execution in ['torch_reference','cuda_native']:
    stack=build_plastic_stack(replace(config,fly=replace(config.fly,execution=execution)))
    stack.trainer.load_state_dict(saved['trainer'])
    b=stack.agent.processor.backend
    b.set_plastic_efficacy(stack.agent.plasticity.state.efficacy)
    stimulus=stack.agent.processor.encoder.encode(stack.trainer.observations)
    original=None
    differences=[]
    for repeat in range(3):
        b.reset(1,(92871,))
        b.simulate_tensor(stimulus,config.fly.duration_ms)
        state=[t.detach().cpu().numpy().copy() for t in b._state]
        if original is None:original=state
        differences.append({str(i):{'unequal':int(np.count_nonzero(x!=y)),
            'maximum_absolute_difference':float(np.max(np.abs(x.astype(float)-y)))}
            for i,(x,y) in enumerate(zip(original,state))})
    result[execution]=differences
    del stack,b
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
