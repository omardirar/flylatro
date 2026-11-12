"""Read-only consistency audit of completed training logs and every checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from flylatro.evaluation.state_hash import hash_observation_row
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_manifest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir', type=Path, action='append', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
reports = []
for directory in a.run_dir:
    summary = json.loads((directory / 'run-summary.json').read_text())
    manifest = json.loads((directory / 'run-manifest.json').read_text())
    count = summary['environment_decisions']
    batch = manifest['config']['environment']['num_envs']
    def rows(name):
        return [json.loads(line) for line in (directory / name).read_text().splitlines()]
    actions = rows('training-actions.jsonl')
    reinforcement = rows('synthetic-reinforcement-events.jsonl')
    metrics = rows('metrics.jsonl')
    episodes = rows('training-episodes.jsonl')
    expected = list(range(batch, count + 1, batch))
    if [r['environment_decisions'] for r in actions] != expected:
        raise AssertionError('actions missing, reordered or duplicated')
    if [r['environment_decisions'] for r in reinforcement] != expected:
        raise AssertionError('reinforcement audit differs from action exposure')
    if [r['step'] for r in metrics] != expected:
        raise AssertionError('metrics audit differs from action exposure')
    if len(episodes) != summary['episodes']:
        raise AssertionError('durable episodes differ from checkpoint counters')
    for field, name in [('executed_action_schedule_sha256', 'training-actions.jsonl'),
                       ('synthetic_reinforcement_event_log_sha256', 'synthetic-reinforcement-events.jsonl')]:
        actual = sha256_file(directory / name)
        if summary['components'][field] != actual or manifest['components'][field] != actual:
            raise AssertionError('final audit hash differs between products')
    checkpoints = []
    for file in sorted(directory.glob('plastic-checkpoint-*.pkl')):
        checked = load_plastic_manifest(file)
        with file.open('rb') as stream:
            payload = pickle.load(stream)
        state = payload['trainer']['training_state']
        exposure = state['environment_decisions']
        if not 0 < exposure <= count or exposure % batch:
            raise AssertionError('checkpoint counter is outside durable audit')
        efficacy = np.asarray(payload['trainer']['plastic_state']['efficacy'])
        weight_hash = hashlib.sha256(efficacy.astype('<f4', copy=False).tobytes()).hexdigest()
        if weight_hash != checked['plastic_weight_sha256']:
            raise AssertionError('checkpoint weights differ from hash-verified manifest')
        counters = np.asarray(payload['trainer']['plastic_state']['decision_count'])
        if not np.all(counters == state['vector_steps']):
            raise AssertionError('per-learner learning exposure differs from environment counter')
        observation_check = 'no following decision in this stage'
        if exposure < count:
            hashes = [hash_observation_row(payload['trainer']['observations'], row) for row in range(batch)]
            if hashes != actions[exposure // batch]['state_hashes_before']:
                raise AssertionError('checkpoint observations differ from subsequent audited decision')
            observation_check = 'exact next audited input'
        checkpoints.append({'path': str(file), 'environment_decisions': exposure,
            'checkpoint_sha256': checked['checkpoint_sha256'], 'weight_sha256': weight_hash,
            'observations': observation_check})
    final = next(row for row in checkpoints if Path(row['path']).resolve() == Path(summary['checkpoint']).resolve())
    if final['weight_sha256'] != summary['plastic_weight_sha256']:
        raise AssertionError('final checkpoint and summary disagree')
    reports.append({'run_dir': str(directory), 'environment_decisions': count, 'batch_size': batch,
        'action_records': len(actions), 'reinforcement_records': len(reinforcement),
        'metric_records': len(metrics), 'episode_records': len(episodes), 'checkpoints': checkpoints})
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('x') as stream:
    json.dump({'schema': 'cli-artifact-consistency-v1', 'status': 'PASS', 'runs': reports}, stream, indent=2)
print(a.output)
