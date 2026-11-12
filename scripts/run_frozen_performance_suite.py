"""Serial reference/native frozen evaluation and calibration measurements."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--reference-repetitions', type=int)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
state = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'jobs': []}
def save():
    temporary = a.output_dir / 'suite.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(a.output_dir / 'suite.json')
try:
    for execution, batches in [('torch_reference', [1]), ('cuda_native_event_graph_blocks', [1, 2, 4, 8, 16])]:
        for workload in ['evaluation', 'calibration']:
            for batch in batches:
                name = f'{execution}-{workload}-{batch}'
                command = [sys.executable, 'scripts/benchmark_frozen_workloads.py',
                    '--config', str(a.config), '--execution', execution, '--workload', workload,
                    '--batch-size', str(batch), '--repetitions', str(
                        a.reference_repetitions if execution == 'torch_reference' and a.reference_repetitions is not None else a.repetitions),
                    '--episodes', '16', '--heavy', '--output', str(a.output_dir / (name + '.json'))]
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
                row['status'] = 'passed' if row['returncode'] == 0 else 'failed'
                save()
                print(name, row['returncode'], flush=True)
                if row['returncode']:
                    raise RuntimeError(name + ' failed; inspect captured stderr')
    state['status'] = 'completed'
except BaseException as error:
    state['status'] = 'failed'
    state['error'] = str(error)
    raise
finally:
    save()
