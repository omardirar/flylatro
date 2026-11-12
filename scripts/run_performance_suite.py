"""Serial GPU measurements with fresh outputs and durable per-job status."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--execution', default='cuda_native_graph_blocks')
p.add_argument('--batches', default='1,2,4,8,16')
p.add_argument('--steps', type=int, default=100)
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--workload', choices=['training','neural','cli'], default='training')
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
state = {'created_at': datetime.now(timezone.utc).isoformat(), 'jobs': [], 'status': 'running'}
def save():
    path = a.output_dir / 'suite.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(path)
try:
    jobs = []
    if a.workload == 'cli':
        for repetition in range(a.repetitions):
            directory = a.output_dir / f'cli-{repetition}'
            jobs.append((f'cli-{repetition}', [sys.executable, '-m', 'flylatro.learning.cli',
                '--config', str(a.config), '--execution', a.execution, '--heavy',
                '--run-dir', str(directory), '--stop-after-decisions', str(a.steps),
                '--no-tensorboard', '--dashboard', '--dashboard-refresh', '2']))
    else:
        module = 'flylatro.benchmarks.migration' if a.workload == 'training' else 'flylatro.benchmarks.neural'
        for batch in (int(value) for value in a.batches.split(',')):
            jobs.append((f'batch-{batch}', [sys.executable, '-m', module,
                '--config', str(a.config), '--execution', a.execution, '--batch-size', str(batch),
                '--warmup', '10', '--steps', str(a.steps), '--repetitions', str(a.repetitions), '--heavy',
                '--output', str(a.output_dir / f'batch-{batch}.json')]))
    for name, command in jobs:
        row = {'name': name, 'command': command, 'status': 'running'}
        state['jobs'].append(row); save()
        started = time.perf_counter()
        with (a.output_dir / f'{name}.stdout').open('x') as stdout, (a.output_dir / f'{name}.stderr').open('x') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            row['pid'] = process.pid; save()
            row['returncode'] = process.wait()
        row['wall_seconds'] = time.perf_counter() - started
        row['status'] = 'passed' if row['returncode'] == 0 else 'failed'
        save()
        if row['returncode']:
            raise RuntimeError(f'{name} failed; see captured stderr')
        print(name, row['wall_seconds'], flush=True)
    state['status'] = 'completed'
except BaseException as error:
    state['status'] = 'failed'; state['error'] = str(error)
    raise
finally:
    save()
