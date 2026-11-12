"""Serial final GPU acceptance jobs; preserve exact commands and failure evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--batch-config', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
def module(name, *args):
    return [sys.executable, '-m', name, *map(str, args)]
def script(name, *args):
    return [sys.executable, 'scripts/' + name, *map(str, args)]
common = ['--config', a.config, '--heavy']
jobs = [
    ('native-tests', module('pytest', 'tests/test_native_cuda.py', '-q'), (0,)),
    ('reference-neural', module('flylatro.benchmarks.neural', *common, '--execution', 'torch_reference',
        '--batch-size', '1', '--warmup', '10', '--steps', '100', '--repetitions', '3',
        '--output', a.output_dir / 'reference-neural.json'), (0,)),
    ('eager-event-training', module('flylatro.benchmarks.migration', *common, '--execution', 'cuda_native_event',
        '--warmup', '10', '--steps', '100', '--repetitions', '3',
        '--output', a.output_dir / 'eager-event-training.json'), (0,)),
    ('sparse-layouts', script('benchmark_sparse_layouts.py', *common,
        '--output', a.output_dir / 'sparse-layouts.json'), (0,)),
    ('cusparse-algorithms', script('benchmark_cusparse.py', *common,
        '--output', a.output_dir / 'cusparse-algorithms.json'), (0,)),
    ('event-timeline', script('profile_execution.py', *common, '--execution', 'cuda_native_event_graph_blocks',
        '--mode', 'timeline', '--output-dir', a.output_dir / 'timeline'), (0,)),
    # This deliberately retains the strict full-buffer comparison. Historical
    # atomic plastic accumulation can fail bitwise reference replay; exit 2 is
    # evidence to inspect, never a reason to loosen comparison or hide results.
    ('historical-comparison', script('qualify_execution.py', *common,
        '--executions', 'torch_reference,cuda_native_event_graph_blocks', '--steps', '100',
        '--output-dir', a.output_dir / 'historical-comparison'), (0, 2)),
    ('reference-cli', script('run_performance_suite.py', '--config', a.config,
        '--execution', 'torch_reference', '--workload', 'cli', '--steps', '100', '--repetitions', '3',
        '--output-dir', a.output_dir / 'reference-cli'), (0,)),
    ('current-cli', script('run_performance_suite.py', '--config', a.config,
        '--execution', 'cuda_native_event_graph_blocks', '--workload', 'cli', '--steps', '1000', '--repetitions', '3',
        '--output-dir', a.output_dir / 'current-cli'), (0,)),
    ('single-complete', module('flylatro.learning.cli', '--config', a.config,
        '--run-dir', a.output_dir / 'single-complete', '--heavy', '--dashboard', '--dashboard-refresh', '1', '--no-tensorboard'), (0,)),
    ('batch-complete', module('flylatro.learning.cli', '--config', a.batch_config,
        '--run-dir', a.output_dir / 'batch-complete', '--heavy', '--dashboard', '--dashboard-refresh', '1', '--no-tensorboard'), (0,)),
    ('cli-artifact-audit', script('verify_cli_artifacts.py',
        '--run-dir', a.output_dir / 'single-complete', '--run-dir', a.output_dir / 'batch-complete',
        '--run-dir', a.output_dir / 'current-cli/cli-0', '--run-dir', a.output_dir / 'current-cli/cli-1',
        '--run-dir', a.output_dir / 'current-cli/cli-2', '--output', a.output_dir / 'cli-audit.json'), (0,)),
]
state = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'jobs': []}
def save():
    temporary = a.output_dir / 'suite.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(a.output_dir / 'suite.json')
try:
    for name, command, accepted in jobs:
        row = {'name': name, 'command': command, 'accepted_returncodes': accepted, 'status': 'running'}
        state['jobs'].append(row)
        save()
        start = time.perf_counter()
        with (a.output_dir / (name + '.stdout')).open('x') as stdout, (a.output_dir / (name + '.stderr')).open('x') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            row['pid'] = process.pid
            save()
            row['returncode'] = process.wait()
        row['wall_seconds'] = time.perf_counter() - start
        row['status'] = 'passed' if row['returncode'] == 0 else 'comparison_mismatch' if row['returncode'] in accepted else 'failed'
        save()
        print(name, row['status'], round(row['wall_seconds'], 3), flush=True)
        if row['returncode'] not in accepted:
            raise RuntimeError(name + ' failed; inspect captured stderr')
    state['status'] = 'completed'
except BaseException as error:
    state['status'] = 'failed'
    state['error'] = str(error)
    raise
finally:
    save()
