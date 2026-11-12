"""Repeated, state-matched training measurements with production safety checks.

Run with ``python -m flylatro.benchmarks.migration``. This measures workload B,
not CLI logging/checkpoint/dashboard overhead. CUDA events span the whole trial
and therefore include GPU idle gaps; they are not a sum of kernel durations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np

from flylatro.benchmarks.cli import _metrics
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack


def source_identity() -> dict:
    files = sorted(Path('src/flylatro').rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path).encode() + b'\0' + path.read_bytes())
    return {
        'python_source_sha256': digest.hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
    }


def measure(stack, *, warmup: int, steps: int, repetitions: int, device: str) -> dict:
    if warmup < 0 or min(steps, repetitions) < 1:
        raise ValueError('warmup must be non-negative; steps and repetitions must be positive')
    trainer = stack.trainer
    batch = stack.env.num_envs
    initial_decisions = trainer.state.environment_decisions
    if initial_decisions + (warmup + steps) * batch > trainer.config.max_environment_decisions:
        raise ValueError('warmup plus measured steps exceeds configured exposure budget')
    cuda = device.startswith('cuda')
    if cuda:
        import torch
    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)
    if warmup:
        trainer.train(stop_after_environment_decisions=initial_decisions + warmup * batch)
    synchronize()
    # Serialize once to ensure every restore owns its mutable arrays and tensors.
    snapshot_start = time.perf_counter()
    snapshot = pickle.dumps(trainer.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    snapshot_seconds = time.perf_counter() - snapshot_start
    target = trainer.state.environment_decisions + steps * batch
    trials = []
    for repetition in range(repetitions):
        trainer.load_state_dict(pickle.loads(snapshot))
        synchronize()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        latencies = []
        last_metrics = {}
        previous = time.perf_counter()
        def observed(_trainer, metrics):
            nonlocal previous, last_metrics
            now = time.perf_counter()
            latencies.append(now - previous)
            previous = now
            last_metrics = metrics
        started = previous
        if cuda:
            start_event.record()
        trainer.train(on_step=observed, stop_after_environment_decisions=target)
        if cuda:
            end_event.record()
        synchronize()
        seconds = time.perf_counter() - started
        trial = {
            'repetition': repetition,
            **_metrics(steps * batch, seconds),
            'total_decisions': steps * batch,
            'per_learner_decisions_per_second': steps / seconds,
            'vector_step_latency_seconds': {'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)), 'max': max(latencies), 'samples': latencies},
            'final_weight_sha256': trainer.agent.plasticity.state.weight_sha256,
            'last_metrics': last_metrics,
        }
        if cuda:
            trial.update(cuda_trial_elapsed_ms=start_event.elapsed_time(end_event),
                gpu_reserved_bytes=torch.cuda.memory_reserved(device),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        trials.append(trial)
    return {'batch_size': batch, 'warmup_vector_steps': warmup, 'measured_vector_steps': steps,
        'repetitions': repetitions, 'snapshot_capture_and_serialization_seconds': snapshot_seconds,
        'snapshot_bytes': len(snapshot), 'trials': trials,
        'repeat_weight_agreement': len({t['final_weight_sha256'] for t in trials}) == 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--repetitions', type=int, default=3)
    from flylatro.fly.execution import EXECUTION_BACKENDS
    parser.add_argument('--execution', choices=EXECUTION_BACKENDS)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite benchmark evidence: {args.output}')
    measured_source = source_identity()
    started = time.perf_counter()
    config = PlasticExperimentConfig.load(args.config)
    if args.execution is not None:
        config = replace(config, fly=replace(config.fly, execution=args.execution))
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('batch size must be positive')
        per_learner_budget = config.training.max_environment_decisions // config.environment.num_envs
        config = replace(config, environment=replace(config.environment, num_envs=args.batch_size),
            training=replace(config.training, max_environment_decisions=per_learner_budget * args.batch_size))
    config.validate()
    config.require_heavy_opt_in(args.heavy)
    stack = build_plastic_stack(config)
    startup_seconds = time.perf_counter() - started
    report = measure(stack, warmup=args.warmup, steps=args.steps,
                     repetitions=args.repetitions, device=config.fly.device)
    import torch
    backend = getattr(stack.agent.processor, "backend", None)
    report["graph_capture"] = {"count": getattr(backend, "capture_count", 0),
        "seconds": getattr(backend, "capture_seconds", 0.0)}
    report.update(schema='performance-migration-training-v1', workload='B-training-decision',
        startup_seconds=startup_seconds, components=stack.components,
        config_text=args.config.read_text(), effective_config=asdict(config), source=measured_source,
        source_changed_during_measurement=source_identity()["python_source_sha256"] != measured_source["python_source_sha256"],
        modes={'dashboard': False, 'durable_logging': False, 'periodic_checkpoints': False,
               'component_profiling': False, 'safety_checks': True,
               'detailed_plasticity': stack.trainer.record_detailed_plasticity},
        hardware_software={'platform': platform.platform(), 'python': sys.version,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': config.fly.device,
            'gpu': torch.cuda.get_device_name(config.fly.device) if config.fly.device.startswith('cuda') else None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2, sort_keys=True, default=str)
        output.write('\n')
    print(json.dumps({'output': str(args.output), 'repeat_weight_agreement': report['repeat_weight_agreement']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
