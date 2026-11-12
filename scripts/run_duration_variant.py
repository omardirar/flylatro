"""Separately named shorter-duration experiment with fresh gates and evaluation.

Never include these measurements in equivalent-model acceleration. Existing
motor pools, normalization, rewards and learning equations remain unchanged.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--variant-config', type=Path, required=True)
p.add_argument('--qualified-config', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--duration-ms', type=float, default=50)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
original = PlasticExperimentConfig.load(a.config)
config = replace(original, name=f'experimental-event-duration-{a.duration_ms:g}ms-v1',
    fly=replace(original.fly, duration_ms=a.duration_ms))
document = asdict(config)
document.pop('source_path')
with a.variant_config.open('x') as stream:
    stream.write(render_toml(document))
state = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
    'scope': 'explicit model variant; not equivalent-model acceleration',
    'decision_duration_ms': a.duration_ms, 'jobs': []}
def save():
    temporary = a.output_dir / 'execution.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(a.output_dir / 'execution.json')
def run(name, command):
    row = {'name': name, 'command': list(map(str,command)), 'status': 'running'}
    state['jobs'].append(row)
    save()
    started = time.perf_counter()
    with (a.output_dir / (name + '.stdout')).open('x') as stdout, (a.output_dir / (name + '.stderr')).open('x') as stderr:
        process = subprocess.Popen(row['command'], stdout=stdout, stderr=stderr)
        row['pid'] = process.pid
        save()
        row['returncode'] = process.wait()
    row['wall_seconds'] = time.perf_counter() - started
    row['status'] = 'passed' if row['returncode'] == 0 else 'failed'
    save()
    print(name, row['returncode'], flush=True)
    if row['returncode']:
        raise RuntimeError(name + ' failed; retain its safety/gate evidence')
try:
    run('pilot-timing', [sys.executable,'-m','flylatro.benchmarks.migration',
        '--config',a.variant_config,'--warmup','10','--steps','100','--repetitions','3',
        '--heavy','--output',a.output_dir/'pilot-timing.json'])
    run('fresh-calibration', [sys.executable,'scripts/run_migration_calibration.py',
        '--config',a.variant_config,'--qualified-config',a.qualified_config,
        '--output-dir',a.output_dir/'calibration','--training-report',a.output_dir/'pilot-timing.json'])
    preflight = json.loads((a.output_dir/'calibration/preflight-v2.json').read_text()) if (a.output_dir/'calibration/preflight-v2.json').exists() else json.loads((a.output_dir/'calibration/preflight.json').read_text())
    if preflight['failed']:
        state['status'] = 'rejected_by_fresh_gates'
        state['failed'] = preflight['failed']
        save()
        raise SystemExit(0)
    run('learning-run', [sys.executable,'-m','flylatro.learning.cli','--config',a.qualified_config,
        '--run-dir',a.output_dir/'training','--stop-after-decisions','1000',
        '--heavy','--dashboard','--dashboard-refresh','1','--no-tensorboard'])
    summary = json.loads((a.output_dir/'training/run-summary.json').read_text())
    run('frozen-trained-evaluation', [sys.executable,'-m','flylatro.evaluation.plastic_cli',
        '--config',a.qualified_config,'--checkpoint',summary['checkpoint'],'--output-dir',a.output_dir/'evaluation',
        '--episodes','16','--batch-size','4','--win-ante','1','--max-vector-steps','1000','--heavy'])
    state['status'] = 'completed'
except SystemExit:
    raise
except BaseException as error:
    state['status'] = 'failed'
    state['error'] = str(error)
    raise
finally:
    save()
