"""Compare real batched training with separately trained, identity-matched flies."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.benchmarks.migration import source_identity

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--steps', type=int, default=16)
p.add_argument('--heavy', action='store_true')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
if min(a.batch_size, a.steps) < 1:
    raise ValueError('positive batch and step counts required')
config = PlasticExperimentConfig.load(a.config)
config.require_heavy_opt_in(a.heavy)
def digest(value):
    array = np.asarray(state_numpy(value))
    h = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()
def row_record(stack, row):
    t = stack.trainer
    b = stack.agent.processor.backend
    return {
        'actions': {key: value[row:row+1].tolist() for key, value in t.last_executed_actions.items()},
        'before': t.last_state_hashes_before[row], 'after': t.last_state_hashes_after[row],
        'learning': {key: digest(value[row:row+1])
            for key, value in stack.agent.plasticity.state.state_dict().items()},
        'neural': [digest(value[:, row:row+1] if index == 1 else value[row:row+1])
            for index, value in enumerate(b._state)],
        'rng': digest(b._generators[row].get_state()),
    }
batched_config = replace(config,
    environment=replace(config.environment, num_envs=a.batch_size),
    training=replace(config.training, max_environment_decisions=config.training.max_environment_decisions * a.batch_size))
batched = build_plastic_stack(batched_config)
expected = []
batched.trainer.train(stop_after_environment_decisions=a.steps * a.batch_size,
    on_step=lambda trainer, metrics: expected.append([row_record(batched, row) for row in range(a.batch_size)]))
components = batched.components
del batched
import torch
torch.cuda.synchronize()
torch.cuda.empty_cache()
results = []
for row in range(a.batch_size):
    selected = replace(config, environment=replace(config.environment, num_envs=1),
        training=replace(config.training, training_seed_offset=config.training.training_seed_offset + row,
            base_fly_seed=config.training.base_fly_seed + row))
    stack = build_plastic_stack(selected)
    observed = []
    stack.trainer.train(stop_after_environment_decisions=a.steps,
        on_step=lambda trainer, metrics: observed.append(row_record(stack, 0)))
    mismatches = [{key: {'batched': expected[step][row][key], 'sequential': actual[key]}
        for key in actual if expected[step][row][key] != actual[key]}
        for step, actual in enumerate(observed)]
    results.append({'learner': row, 'exact': not any(mismatches), 'mismatches': mismatches})
    del stack
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
report = {'source': source_identity(), 'components': components, 'batch_size': a.batch_size,
    'steps': a.steps, 'results': results, 'exact_all_learners': all(row['exact'] for row in results),
    'scope': 'closed-loop independent training; exact actions, environment hashes, learning tensors, six neural buffers and row RNG; aggregate metrics inherently differ'}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('x') as stream:
    json.dump(report, stream, indent=2)
print(a.output, report['exact_all_learners'])
raise SystemExit(0 if report['exact_all_learners'] else 2)
