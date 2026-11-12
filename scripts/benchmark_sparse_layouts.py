"""Controlled fixed-propagation layout probe; not a training speedup claim.

Uses PyTorch CSR/COO primitives and both dense operand layouts. The native
cuSPARSE algorithm/workspace probe remains a separate implementation task.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.benchmarks.migration import source_identity

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--heavy',action='store_true')
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
config=PlasticExperimentConfig.load(a.config);config.require_heavy_opt_in(a.heavy)
import torch
results=[]
for execution in ['torch_reference','torch_csr']:
    stack=build_plastic_stack(replace(config,fly=replace(config.fly,execution=execution)))
    backend=stack.agent.processor.backend
    weights=backend.weights
    for batch in [1,2,4,8,16]:
        generator=torch.Generator(device=backend.device).manual_seed(18931)
        spikes=(torch.rand((batch,backend.neuron_count),device=backend.device,generator=generator)<.01).float()
        for layout in ['column-major','row-major']:
            dense=spikes.T if layout=='column-major' else spikes.T.contiguous()
            for _ in range(10):out=torch.sparse.mm(weights,dense)
            torch.cuda.synchronize()
            start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();start_event.record()
            for _ in range(a.iterations):out=torch.sparse.mm(weights,dense)
            end_event.record();torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            results.append({'execution':execution,'batch':batch,'dense_layout':layout,
                'wall_seconds':seconds,'cuda_ms':start_event.elapsed_time(end_event),
                'iterations':a.iterations,'matrix_payload_bytes':sum(t.numel()*t.element_size() for t in
                    ([weights.indices(),weights.values()] if execution=='torch_reference' else
                     [weights.crow_indices(),weights.col_indices(),weights.values()])),
                'finite':bool(torch.isfinite(out).all())})
    del stack,backend,weights,out,spikes,dense
    torch.cuda.empty_cache()
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump({'source':source_identity(),'results':results,
    'scope':'fixed propagation only, synthetic 1% spikes, no plastic contribution; timings include dispatch'},f,indent=2)
