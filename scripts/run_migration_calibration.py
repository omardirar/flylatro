"""Fresh performance-backend evidence with the original motor artifact retained.

Serial jobs record exact commands/output and preserve failing gates. This creates
an explicit qualification configuration; it does not change the default backend.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from flylatro.analysis.evidence import experiment_identity
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--qualified-config', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--training-report', type=Path, required=True)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
original = PlasticExperimentConfig.load(a.config)
config = replace(original, calibration=replace(original.calibration,
    representation_pre_report=str((a.output_dir / 'representation-pre.json').resolve()),
    representation_post_report=str((a.output_dir / 'representation-post.json').resolve()),
    motor_calibration_report=str((a.output_dir / 'motor-report.json').resolve()),
    reachability_report=str((a.output_dir / 'reachability.json').resolve()),
    plasticity_report=str((a.output_dir / 'plasticity.json').resolve()),
    specificity_report=str((a.output_dir / 'specificity.json').resolve()),
    benchmark_report=str((a.output_dir / 'benchmark.json').resolve())))
document = asdict(config)
document.pop('source_path')
with a.qualified_config.open('x') as stream:
    stream.write(render_toml(document))
config = PlasticExperimentConfig.load(a.qualified_config)

# Bind recorded repeated timing to the identical model/mapping/build identities.
timing = json.loads(a.training_report.read_text())
if timing['source_changed_during_measurement'] or not timing['repeat_weight_agreement']:
    raise ValueError('measured training source or restored repetitions are inconsistent')
if timing['effective_config']['fly'] != config.to_dict()['fly']:
    raise ValueError('timing report fly configuration differs from qualification')
for section in ('motor', 'plasticity', 'reinforcement', 'safety'):
    if timing['effective_config'][section] != config.to_dict()[section]:
        raise ValueError('timing report ' + section + ' differs from qualification')
benchmark = {'benchmark': 'plastic_end_to_end', 'version': 'plastic-end-to-end-benchmark-v3',
    'decisions_per_second': statistics.median(t['decisions_per_second'] for t in timing['trials']),
    'wall_seconds': statistics.median(t['wall_seconds'] for t in timing['trials']),
    'decisions': timing['measured_vector_steps'] * timing['batch_size'],
    'batch_size': timing['batch_size'],
    'execution_mode': 'shared-fixed-sparse-plus-batched-plastic',
    'peak_memory_bytes': max(t['peak_memory_bytes'] for t in timing['trials']),
    'sampling': 'median of three sustained, restored workload-B measurements after warmup',
    'source_report': str(a.training_report.resolve()), 'source_report_sha256': sha256_file(a.training_report),
    'modes': timing['modes'], 'evidence_identity': experiment_identity(config, timing['components'],
        report_kind='benchmark', report_version='plastic-end-to-end-benchmark-v3').to_dict()}
(a.output_dir / 'benchmark.json').write_text(json.dumps(benchmark, indent=2) + '\n')

cache = str(a.output_dir / 'unit-activity.npz')
common = ['--config', str(a.qualified_config), '--repeats', '3', '--batch-size', '8',
          '--activity-cache', cache, '--heavy']
def module(name, args):
    return [sys.executable, '-m', name, *args]
jobs = [
    ('representation-pre', module('flylatro.analysis.representation_cli', [*common,
        '--stage', 'pre', '--output', str(a.output_dir / 'representation-pre.json'),
        '--reachability-output', str(a.output_dir / 'reachability-pre.json')])),
    ('motor-revalidation', module('flylatro.analysis.motor_calibration_cli', [*common,
        '--validate-existing', '--calibration-seed', '12000000',
        '--output', str(a.output_dir / 'motor-unchanged.json'),
        '--report', str(a.output_dir / 'motor-report.json')])),
    ('representation-post', module('flylatro.analysis.representation_cli', [*common,
        '--stage', 'post', '--output', str(a.output_dir / 'representation-post.json'),
        '--reachability-output', str(a.output_dir / 'reachability.json')])),
    ('plasticity', module('flylatro.analysis.calibration_cli', ['--config', str(a.qualified_config),
        '--mode', 'integrated', '--decisions', '20', '--output', str(a.output_dir / 'plasticity.json'), '--heavy'])),
    ('specificity', module('flylatro.analysis.specificity_cli', ['--config', str(a.qualified_config),
        '--decisions', '20', '--output', str(a.output_dir / 'specificity.json'), '--heavy'])),
    ('initial-preflight', module('flylatro.analysis.preflight_cli', ['--config', str(a.qualified_config),
        '--profile', 'initial', '--output', str(a.output_dir / 'preflight.json')])),
]
state = {'created_at': datetime.now(timezone.utc).isoformat(), 'jobs': [], 'status': 'running',
    'qualified_config': str(a.qualified_config.resolve()),
    'motor_policy': 'preserve original pools, normalization, exploration and artifact identity'}
def save():
    temporary = a.output_dir / 'execution.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(a.output_dir / 'execution.json')
try:
    for name, command in jobs:
        row = {'name': name, 'command': command, 'status': 'running'}
        state['jobs'].append(row)
        save()
        started = time.perf_counter()
        with (a.output_dir / (name + '.stdout')).open('x') as stdout, (a.output_dir / (name + '.stderr')).open('x') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            row['pid'] = process.pid
            save()
            row['returncode'] = process.wait()
        row['wall_seconds'] = time.perf_counter() - started
        row['status'] = 'passed' if row['returncode'] == 0 else 'reported_gate_failure'
        save()
        print(name, row['returncode'], flush=True)
        if row['returncode'] not in (0, 1, 2):
            raise RuntimeError(name + ' failed unexpectedly; inspect stderr')
        if row['returncode'] and not (a.output_dir / ('preflight.json' if name == 'initial-preflight'
                else 'motor-report.json' if name == 'motor-revalidation' else name + '.json')).exists():
            raise RuntimeError(name + ' failed without a gate report; inspect stderr')
    state['status'] = ('passed' if all(row['returncode'] == 0 for row in state['jobs'])
        else 'evidence_complete_with_failed_gates')
except BaseException as error:
    state['status'] = 'failed'
    state['error'] = str(error)
    raise
finally:
    save()
