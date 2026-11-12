"""Workload A: controlled-input neural reset/simulation, excluding encoding/learning."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np

from flylatro.benchmarks.cli import _metrics
from flylatro.benchmarks.migration import source_identity
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.checkpoints import software_metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--execution', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--heavy', action='store_true')
    args = p.parse_args(argv)
    if min(args.steps, args.repetitions, args.batch_size) < 1 or args.warmup < 0:
        p.error('invalid measurement counts')
    if args.output.exists():
        raise FileExistsError(args.output)
    config = PlasticExperimentConfig.load(args.config)
    per_learner_budget = config.training.max_environment_decisions // config.environment.num_envs
    config = replace(config, fly=replace(config.fly, execution=args.execution),
        environment=replace(config.environment, num_envs=args.batch_size),
        training=replace(config.training, max_environment_decisions=per_learner_budget * args.batch_size))
    config.validate()
    config.require_heavy_opt_in(args.heavy)
    source = source_identity()
    start = time.perf_counter()
    stack = build_plastic_stack(config)
    startup = time.perf_counter() - start
    processor = stack.agent.processor
    backend = processor.backend
    backend.set_plastic_efficacy(stack.agent.plasticity.state.efficacy)
    stimulus = processor.encoder.encode(stack.trainer.observations)
    import torch
    cuda = config.fly.device.startswith('cuda')
    def sync():
        if cuda:
            torch.cuda.synchronize(config.fly.device)
    def decision(index):
        backend.reset(args.batch_size, tuple(900000 + index * 100 + row for row in range(args.batch_size)))
        return backend.simulate_tensor(stimulus, processor.duration_ms)
    first = time.perf_counter()
    for index in range(args.warmup):
        decision(index)
    sync()
    warmup_seconds = time.perf_counter() - first
    rows = []
    for repetition in range(args.repetitions):
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(config.fly.device)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        latencies = []
        started = previous = time.perf_counter()
        for index in range(args.steps):
            activity = decision(index)
            now = time.perf_counter()
            latencies.append(now - previous)
            previous = now
        if cuda:
            end_event.record()
        sync()
        seconds = time.perf_counter() - started
        # Validation outside timing; every backend still preserves simulation checks.
        if not bool(torch.isfinite(activity.final_voltage).all()):
            raise ValueError('nonfinite neural voltage')
        readout_hash = hashlib.sha256(activity.spike_counts.cpu().numpy().tobytes()
            + activity.final_voltage.cpu().numpy().tobytes()).hexdigest()
        rows.append({**_metrics(args.steps * args.batch_size, seconds),
            'last_readout_sha256': readout_hash,
            'per_learner_decisions_per_second': args.steps / seconds,
            'latency_seconds': {'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)), 'samples': latencies},
            'cuda_elapsed_ms': start_event.elapsed_time(end_event) if cuda else None,
            'reserved_bytes': torch.cuda.memory_reserved() if cuda else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if cuda else None})
    report = {'schema': 'performance-migration-neural-v1', 'workload': 'A-controlled-neural',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'execution': args.execution, 'batch_size': args.batch_size, 'source': source,
        'source_changed_during_measurement': source_identity()['python_source_sha256'] != source['python_source_sha256'],
        'effective_config': config.to_dict(), 'hardware_software': software_metadata(),
        'repeat_readout_agreement': len({r['last_readout_sha256'] for r in rows}) == 1,
        'components': stack.components, 'startup_seconds': startup,
        'warmup_seconds': warmup_seconds, 'warmup': args.warmup, 'steps': args.steps,
        'repetitions': args.repetitions, 'trials': rows,
        'graph_capture_seconds': getattr(backend, 'capture_seconds', 0),
        'modes': {'recording': False, 'dashboard': False, 'logging': False,
                  'checkpoints': False, 'component_profiling': False},
        'input_policy': 'fixed encoded initial observations; unit weights; independent per-decision seeds'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, default=str)
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
