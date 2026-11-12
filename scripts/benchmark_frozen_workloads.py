"""Repeated frozen-evaluation and reward-free corpus activity measurements."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.analysis.evidence import load_corpus, record_corpus_activity
from flylatro.evaluation.plastic_cli import _evaluate_seed_batch
from flylatro.seeds import SeedPlan
from flylatro.benchmarks.cli import _metrics
from flylatro.benchmarks.migration import source_identity
from flylatro.learning.checkpoints import software_metadata

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--execution',required=True)
p.add_argument('--workload',choices=['evaluation','calibration'],required=True)
p.add_argument('--batch-size',type=int,default=1)
p.add_argument('--repetitions',type=int,default=3)
p.add_argument('--episodes',type=int,default=8)
p.add_argument('--heavy',action='store_true')
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
if min(a.batch_size,a.repetitions,a.episodes)<1:raise ValueError('positive counts required')
config=PlasticExperimentConfig.load(a.config)
config=replace(config,fly=replace(config.fly,execution=a.execution),
               environment=replace(config.environment,num_envs=1))
config.require_heavy_opt_in(a.heavy)
source=source_identity()
start=time.perf_counter();stack=build_plastic_stack(config);startup=time.perf_counter()-start
import torch
seeds=SeedPlan().seeds('validation',a.episodes)
corpus=load_corpus(config) if a.workload=='calibration' else None
frozen_hash=stack.agent.plasticity.state.weight_sha256

def run(warmup=False):
    if a.workload=='evaluation':
        episodes=[];transitions=[];sample_latencies=[];simulated_lanes=0
        selected=seeds[:a.batch_size] if warmup else seeds
        for begin in range(0,len(selected),a.batch_size):
            group=selected[begin:begin+a.batch_size]
            start=time.perf_counter()
            result=_evaluate_seed_batch(stack,config,group,win_ante=1,max_vector_steps=1000,neural_recorder=None)
            sample_latencies.append(time.perf_counter()-start)
            episodes.extend(asdict(e) for e in result.episodes)
            transitions.extend(asdict(t) for t in result.transitions)
            simulated_lanes+=len(group)*max(e.decisions for e in result.episodes)
        decisions=sum(e['decisions'] for e in episodes)
        transitions.sort(key=lambda row: (row['seed'],row['decision_id']))
        episode_digest=hashlib.sha256(json.dumps(episodes,sort_keys=True).encode()).hexdigest()
        transition_digest=hashlib.sha256(json.dumps(transitions,sort_keys=True).encode()).hexdigest()
        detail={'episodes':episodes,'chunk_latencies_seconds':sample_latencies,
                'simulated_lane_decisions':simulated_lanes,
                'episodes_sha256':episode_digest,'transitions_sha256':transition_digest,
                'transition_count':len(transitions)}
        digest=hashlib.sha256((episode_digest+transition_digest).encode()).hexdigest()
    else:
        activity=record_corpus_activity(stack,corpus,seed=12_000_000,repeats=1,
            batch_size=a.batch_size,limit=a.batch_size if warmup else None)
        decisions=len(activity.state_labels)
        digest=hashlib.sha256(b''.join(np.asarray(getattr(activity,key)).tobytes()
            for key in ('kc','mbon','descending','output'))).hexdigest()
        detail={'corpus_sha256':corpus.sha256,'states':decisions}
    if stack.agent.plasticity.state.weight_sha256!=frozen_hash:raise RuntimeError('frozen weights changed')
    return decisions,digest,detail

warm=time.perf_counter();run(True);warmup=time.perf_counter()-warm
rows=[]
for repetition in range(a.repetitions):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begin_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin_event.record()
    start=time.perf_counter();decisions,digest,detail=run();torch.cuda.synchronize()
    end_event.record();torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    rows.append({**_metrics(decisions,elapsed),'decisions':decisions,'output_sha256':digest,
        'cuda_trial_elapsed_ms':begin_event.elapsed_time(end_event),
        'gpu_reserved_bytes':torch.cuda.memory_reserved(),
        'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'per_lane_decisions_per_second':decisions/(elapsed*a.batch_size),**detail})
report={'schema':'performance-migration-frozen-v2','workload':a.workload,'batch_size':a.batch_size,
    'execution':a.execution,'source':source,'components':stack.components,'trials':rows,
    'startup_seconds':startup,'warmup_seconds':warmup,'frozen_weight_sha256':frozen_hash,
    'effective_config':config.to_dict(),'hardware_software':software_metadata(),
    'source_changed_during_measurement':source_identity()['python_source_sha256']!=source['python_source_sha256'],
    'modes':{'recording':False,'logging':False,'dashboard':False,'checkpoints':False,'profiling':False},
    'repeat_output_agreement':len({row['output_sha256'] for row in rows})==1}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(report,f,indent=2,default=str)
print(a.output)
