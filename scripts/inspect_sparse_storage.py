"""Measure whether sparse values views retain discarded index layouts."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--heavy', action='store_true')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
config = PlasticExperimentConfig.load(a.config)
config.require_heavy_opt_in(a.heavy)
stack = build_plastic_stack(replace(config, fly=replace(config.fly, execution='cuda_native_event')))
b = stack.agent.processor.backend
import torch
rows = []
def sample(stage):
    gc.collect()
    torch.cuda.synchronize()
    rows.append({'stage': stage, 'allocated_bytes': torch.cuda.memory_allocated(),
        'incoming_payload_bytes': sum(t.numel() * t.element_size() for t in
            (b.weights.crow_indices(), b.weights.col_indices(), b.weights.values())),
        'outgoing_payload_bytes': sum(t.numel() * t.element_size() for t in
            (b._outgoing_crow, b._outgoing_posts, b._outgoing_values))})
sample('constructed')
b.weights = torch.sparse_csr_tensor(b.weights.crow_indices(), b.weights.col_indices(),
    b.weights.values().clone(), size=b.weights.shape, device=b.device)
sample('owned_incoming_values')
b._outgoing_values = b._outgoing_values.clone()
sample('owned_outgoing_values')
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('x') as stream:
    json.dump({'measurements': rows, 'scope': 'startup allocated memory; explicit storage ownership experiment'}, stream, indent=2)
print(a.output)
