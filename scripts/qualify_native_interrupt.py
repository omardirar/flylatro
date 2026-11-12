"""Real GPU CLI interruption, audit flush and exact resumed future qualification."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
commands = []
def command(directory, resume=None):
    args = [sys.executable, '-m', 'flylatro.learning.cli', '--config', str(a.config),
        '--run-dir', str(directory), '--stop-after-decisions', '100',
        '--heavy', '--no-tensorboard', '--dashboard', '--dashboard-refresh', '1']
    if resume is not None:
        args += ['--resume', str(resume)]
    commands.append(args)
    return args
baseline = a.output_dir / 'uninterrupted'
resumed = a.output_dir / 'interrupted'
with (a.output_dir / 'baseline.stdout').open('x') as stdout, (a.output_dir / 'baseline.stderr').open('x') as stderr:
    subprocess.run(command(baseline), stdout=stdout, stderr=stderr, check=True)
with (a.output_dir / 'interrupt.stdout').open('x') as stdout, (a.output_dir / 'interrupt.stderr').open('x') as stderr:
    process = subprocess.Popen(command(resumed), stdout=stdout, stderr=stderr)
    barrier = resumed / 'plastic-checkpoint-000000000050.pkl.manifest.json'
    deadline = time.monotonic() + 120
    while not barrier.exists():
        if process.poll() is not None:
            raise AssertionError('training ended before the interruption barrier')
        if time.monotonic() >= deadline:
            raise TimeoutError('interruption barrier was not reached; own process remains identifiable by command/PID')
        time.sleep(.02)
    process.send_signal(signal.SIGINT)
    code = process.wait(timeout=120)
    if code != 130:
        raise AssertionError(f'interrupted CLI returned {code}, expected 130')
summary = json.loads((resumed / 'run-summary.json').read_text())
stopped = summary['environment_decisions']
if not 50 <= stopped < 100 or summary['status'] != 'interrupted':
    raise AssertionError('interruption did not commit a decision-boundary checkpoint')
checkpoint = Path(summary['checkpoint'])
for name in ('training-actions.jsonl', 'synthetic-reinforcement-events.jsonl', 'metrics.jsonl'):
    if len((resumed / name).read_text().splitlines()) != stopped:
        raise AssertionError('interrupted checkpoint lacks matching durable audit records')
with (a.output_dir / 'resume.stdout').open('x') as stdout, (a.output_dir / 'resume.stderr').open('x') as stderr:
    subprocess.run(command(resumed, checkpoint), stdout=stdout, stderr=stderr, check=True)
expected = json.loads((baseline / 'run-summary.json').read_text())
actual = json.loads((resumed / 'run-summary.json').read_text())
exact = {name: (baseline / name).read_bytes() == (resumed / name).read_bytes()
    for name in ('training-actions.jsonl', 'synthetic-reinforcement-events.jsonl', 'training-episodes.jsonl')}
exact['final_weights'] = expected['plastic_weight_sha256'] == actual['plastic_weight_sha256']
exact['completed_exposure'] = expected['environment_decisions'] == actual['environment_decisions'] == 100
(a.output_dir / 'qualification.json').write_text(json.dumps({'status': 'PASS' if all(exact.values()) else 'FAIL',
    'interrupted_at': stopped, 'interrupt_exit_code': code, 'exact': exact, 'commands': commands}, indent=2) + '\n')
print(a.output_dir, exact)
raise SystemExit(0 if all(exact.values()) else 2)
