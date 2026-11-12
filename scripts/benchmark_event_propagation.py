"""Experimental outgoing propagation at measured and high spike densities.

The integral-weight and absolute-sum checks prove order-independent FP32 atomic
accumulation for binary spikes. No fractional anatomical weights are rounded.
Compaction, zeroing and graph replay are included; this is not training timing.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.fly.native_build import describe_native
from flylatro.fly.sparse_build import load_sparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--iterations', type=int, default=100)
p.add_argument('--heavy', action='store_true')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
config = PlasticExperimentConfig.load(a.config)
config.require_heavy_opt_in(a.heavy)
import torch
from torch.utils import cpp_extension
toolkit, identity = describe_native()
cpp_extension.CUDA_HOME = toolkit
source = Path('src/flylatro/fly/native/event_probe.cu').resolve()
identity = {**identity, 'event_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
directory = Path('artefacts/native-event-probe') / digest
directory.mkdir(parents=True, exist_ok=True)
if not (directory / 'libcudart.so').exists():
    (directory / 'libcudart.so').symlink_to(Path(toolkit) / 'lib/libcudart.so.13')
module = cpp_extension.load(name='flylatro_event_' + digest[:12], sources=[str(source)],
    extra_cuda_cflags=['-O3', '--fmad=false'], extra_cflags=['-O3'],
    extra_ldflags=['-L' + str(directory.resolve())],
    build_directory=str(directory.resolve()), verbose=True)
sparse, sparse_identity = load_sparse()
stack = build_plastic_stack(replace(config, fly=replace(config.fly, execution='cuda_native')))
b = stack.agent.processor.backend
w = b.weights
coo = w.to_sparse_coo().coalesce()
post, pre = coo.indices().cpu().numpy()
values = coo.values().cpu().numpy()
if not np.isfinite(values).all() or not np.array_equal(values, np.round(values)):
    raise ValueError('event prototype requires exact integral anatomical weights; no rounding permitted')
maximum_sum = np.bincount(post, weights=np.abs(values), minlength=b.neuron_count).max()
if maximum_sum >= 2**24:
    raise ValueError('FP32 exact integral accumulation bound exceeded')
outgoing = coo.transpose(0, 1).coalesce().to_sparse_csr()
crow = outgoing.crow_indices().to(torch.int32)
posts = outgoing.col_indices().to(torch.int32)
weights = outgoing.values()
del coo, outgoing, values, pre, post

# Collect real source-complete activity without modifying the learning path.
samples = []
lif = b._lif_step
step_number = 0
def sampled(*args, **kwargs):
    global step_number
    result = lif(*args, **kwargs)
    if step_number % 31 == 0:
        samples.append(result[1].clone())
    step_number += 1
    return result
b._lif_step = sampled
stack.trainer.train(stop_after_environment_decisions=3)
b._lif_step = lif

def measure(function):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        function()
    durations = []
    for _ in range(3):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        for _ in range(a.iterations):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        durations.append({'cuda_ms': begin.elapsed_time(end), 'wall_seconds': time.perf_counter() - started})
    return durations

rows = []
for batch in (1, 2, 4, 8, 16):
    incoming = torch.empty((batch, b.neuron_count), device=b.device)
    active = torch.empty_like(incoming, dtype=torch.int32)
    counts = torch.empty((batch,), dtype=torch.int32, device=b.device)
    dense = torch.empty((b.neuron_count, batch), device=b.device)
    plan = sparse.SparsePlan(w.crow_indices(), w.col_indices(), w.values(), dense,
        batch == 1, 3 if batch == 1 else 4, True, True)
    for density in ('actual', .00024, .001, .01, .1, .5):
        gen = torch.Generator(device=b.device).manual_seed(61429)
        spikes = (samples[len(samples) // 2].expand(batch, -1).contiguous() if density == 'actual'
            else (torch.rand(incoming.shape, generator=gen, device=b.device) < density).float())
        def event():
            module.event_step(crow, posts, weights, spikes, active, counts, incoming)
        def csr():
            dense.copy_(spikes.T)
            return plan.run()
        event()
        reference = csr().T
        torch.cuda.synchronize()
        exact = torch.equal(incoming, reference)
        if not exact:
            raise AssertionError('event prototype changed fixed recurrence')
        rows.append({'batch': batch, 'density': density, 'active_fraction': float(spikes.mean()),
            'exact_csr': exact, 'event': measure(event), 'csr': measure(csr)})
    # Match every collected real timestep, rather than only the timed sample.
    real_exact = True
    for sample in samples:
        spikes = sample.expand(batch, -1).contiguous()
        event()
        real_exact &= torch.equal(incoming, csr().T)
    if not real_exact:
        raise AssertionError('event prototype changed actual timestep recurrence')
    del plan
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('x') as stream:
    json.dump({'build': identity, 'sparse_build': sparse_identity, 'iterations': a.iterations,
        'maximum_absolute_incoming_sum': float(maximum_sum), 'actual_timestep_samples': len(samples),
        'results': rows, 'scope': 'fixed propagation only; actual batched samples duplicate one fly; no plasticity or training speedup claim'}, stream, indent=2)
print(a.output)
