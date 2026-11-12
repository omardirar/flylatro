"""Optional native cuSPARSE algorithm/preprocessing probe, separate from training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from flylatro.fly.sparse_build import load_sparse
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--build-only',action='store_true')
p.add_argument('--heavy',action='store_true')
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
config=PlasticExperimentConfig.load(a.config);config.require_heavy_opt_in(a.heavy)
import torch
module,identity=load_sparse(verbose=True)
if a.build_only:
    print('build complete');raise SystemExit(0)
stack=build_plastic_stack(replace(config,fly=replace(config.fly,execution='torch_csr')))
b=stack.agent.processor.backend;w=b.weights
rows=[]
for batch in [1,2,4,8,16]:
    gen=torch.Generator(device=b.device).manual_seed(10112)
    spikes=(torch.rand((batch,b.neuron_count),device=b.device,generator=gen)<.01).float()
    oracle=torch.sparse.mm(w,spikes.T)
    for vector,algorithms in ([(True,[0,2,3])] if batch==1 else [])+[(False,[0,4,6,12])]:
        for algorithm in algorithms:
            for row_major in [False,True]:
                dense=spikes.T.contiguous() if row_major else spikes.T
                for preprocess in [False,True]:
                    row={'batch':batch,'operation':'SpMV' if vector else 'SpMM',
                        'algorithm':algorithm,'row_major':row_major,'preprocess':preprocess}
                    plan=None
                    try:
                        torch.cuda.synchronize();start=time.perf_counter()
                        plan=module.SparsePlan(w.crow_indices(),w.col_indices(),w.values(),dense,vector,algorithm,row_major,preprocess)
                        torch.cuda.synchronize();row['prepare_seconds']=time.perf_counter()-start
                        for _ in range(10):output=plan.run()
                        torch.cuda.synchronize()
                        start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start=time.perf_counter();start_event.record()
                        for _ in range(a.iterations):output=plan.run()
                        end_event.record();torch.cuda.synchronize()
                        row.update(status='measured',wall_seconds=time.perf_counter()-start,
                            cuda_ms=start_event.elapsed_time(end_event),iterations=a.iterations,
                            workspace_bytes=plan.workspace_bytes(),exact_oracle=torch.equal(output,oracle),
                            max_absolute_difference=float((output-oracle).abs().max()))
                    except RuntimeError as error:
                        row.update(status='unsupported_or_failed',error=str(error))
                    finally:
                        del plan
                    rows.append(row)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump({'build':identity,'results':rows,
    'scope':'controlled fixed recurrence at 1% activity; excludes plasticity and training'},f,indent=2)
print(a.output)
