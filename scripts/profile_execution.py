"""Intrusive timeline, checkpoint-stage and neural sparsity diagnostics.

These measurements are explicitly separate from sustained throughput trials.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
import flylatro.learning.checkpoints as checkpoints
from flylatro.benchmarks.migration import source_identity

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--execution', default='cuda_native')
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--mode', choices=['timeline','checkpoint','sparsity'], required=True)
p.add_argument('--heavy', action='store_true')
p.add_argument('--reuse-provenance', action='store_true', help='profile the production CLI provenance reuse path')
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
config = PlasticExperimentConfig.load(a.config)
config = replace(config, fly=replace(config.fly, execution=a.execution))
config.require_heavy_opt_in(a.heavy)
stack = build_plastic_stack(config)
import torch
backend = stack.agent.processor.backend
stack.trainer.train(stop_after_environment_decisions=3 * stack.env.num_envs)
report = {'mode': a.mode, 'source': source_identity(), 'components': stack.components,
          'timing_warning': 'intrusive diagnostic; not sustained throughput'}
if a.mode == 'timeline':
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], profile_memory=True,
            record_shapes=True) as profiler:
        stack.trainer.train(stop_after_environment_decisions=4 * stack.env.num_envs)
    profiler.export_chrome_trace(str(a.output_dir / 'trace.json'))
    events = profiler.key_averages()
    (a.output_dir / 'operators.txt').write_text(events.table(sort_by='self_device_time_total', row_limit=50))
    report['operators'] = [{ 'name': e.key, 'count': e.count,
        'self_cpu_us': e.self_cpu_time_total, 'self_device_us': e.self_device_time_total}
        for e in events]
elif a.mode == 'sparsity':
    if 'graph' in a.execution:
        raise ValueError('sparsity instrumentation requires eager backend')
    samples = []
    step = backend._lif_step
    def sampled(*args, **kwargs):
        result = step(*args, **kwargs)
        samples.append(result[1].sum(dim=1))
        return result
    backend._lif_step = sampled
    stack.trainer.train(stop_after_environment_decisions=4 * stack.env.num_envs)
    counts = torch.stack(samples).cpu().numpy()
    fraction = counts / backend.neuron_count
    report.update(neurons=backend.neuron_count, timestep_spike_counts=counts.tolist(),
        mean_active_fraction=float(fraction.mean()),
        p50_active_fraction=float(np.percentile(fraction, 50)),
        p95_active_fraction=float(np.percentile(fraction, 95)),
        maximum_active_fraction=float(fraction.max()))
else:
    timings = {}
    provenance = checkpoints.capture_checkpoint_provenance() if a.reuse_provenance else None
    report['reused_checkpoint_provenance'] = a.reuse_provenance
    def timed(name, function):
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timings.setdefault(name, []).append(time.perf_counter() - started)
        return wrapped
    state_cls = type(stack.agent.plasticity.state)
    with patch.object(state_cls, 'state_dict', timed('plastic_capture_including_device_to_host', state_cls.state_dict)), \
            patch.object(stack.trainer, 'state_dict', timed('full_snapshot_capture', stack.trainer.state_dict)), \
            patch.object(checkpoints.pickle, 'dump', timed('serialization', checkpoints.pickle.dump)), \
            patch.object(checkpoints, 'sha256_file', timed('file_hash', checkpoints.sha256_file)):
        for index in range(5):
            start = time.perf_counter()
            checkpoints.save_plastic_checkpoint(a.output_dir / f'checkpoint-{index}.pkl', stack.trainer,
                experiment_config=config.to_dict(), component_metadata=stack.components, provenance=provenance)
            timings.setdefault('complete_save', []).append(time.perf_counter() - start)
    report['timings_seconds'] = timings
(a.output_dir / 'profile.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print(a.output_dir)
