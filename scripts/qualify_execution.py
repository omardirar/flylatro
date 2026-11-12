"""Source-complete controlled trajectory and decision-boundary resume qualification.

This intentionally copies diagnostic state and is not a throughput benchmark.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.checkpoints import save_plastic_checkpoint, load_plastic_checkpoint
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.benchmarks.migration import source_identity


def array_digest(value):
    values = np.asarray(state_numpy(value))
    digest = hashlib.sha256(str((values.dtype.str, values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def observe(stack, metrics):
    trainer = stack.trainer
    return {
        'actions': {key: value.tolist() for key, value in trainer.last_executed_actions.items()},
        'states_before': list(trainer.last_state_hashes_before),
        'states_after': list(trainer.last_state_hashes_after),
        'metrics': metrics,
        'learning': {key: array_digest(value) for key, value in stack.agent.plasticity.state.state_dict().items()},
        'neural': {str(index): array_digest(value) for index, value in enumerate(stack.agent.processor.backend._state)},
        'rng': [array_digest(generator.get_state()) for generator in stack.agent.processor.backend._generators],
    }


def run_steps(stack, steps):
    rows = []
    target = stack.trainer.state.environment_decisions + steps * stack.env.num_envs
    stack.trainer.train(stop_after_environment_decisions=target,
        on_step=lambda trainer, metrics: rows.append(observe(stack, metrics)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--executions', default='torch_reference,torch_csr,cuda_native,cuda_native_graph,cuda_native_graph_blocks')
    p.add_argument('--heavy', action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = PlasticExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    import torch
    result = {'source': source_identity(), 'steps': args.steps, 'results': {},
        'comparison_reference': args.executions.split(',')[0],
        'scope': 'diagnostic full-state per-step hashes; no throughput claims'}
    reference = None
    for execution in args.executions.split(','):
        selected = replace(config, fly=replace(config.fly, execution=execution))
        stack = build_plastic_stack(selected)
        initial = pickle.dumps(stack.trainer.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        trajectory = run_steps(stack, args.steps)
        if reference is None:
            reference = trajectory
        mismatches = [{key: {'reference': expected[key], 'actual': actual[key]}
                       for key in expected if expected[key] != actual[key]}
                      for expected, actual in zip(reference, trajectory, strict=True)]
        path = args.output_dir / f'{execution}-checkpoint.pkl'
        save_plastic_checkpoint(path, stack.trainer, experiment_config=selected.to_dict(),
            component_metadata=stack.components)
        expected_future = run_steps(stack, 3)
        load_plastic_checkpoint(path, stack.trainer, expected_components=stack.components)
        resumed_future = run_steps(stack, 3)
        # Exercise restore after graph construction and previous replay history.
        stack.trainer.load_state_dict(pickle.loads(initial))
        repeated = run_steps(stack, args.steps)
        result['results'][execution] = {'components': stack.components,
            'exact_reference_trajectory': not any(mismatches), 'mismatches': mismatches,
            'exact_checkpoint_resume': expected_future == resumed_future,
            'exact_initial_restore': repeated == trajectory, 'trajectory': trajectory,
            'resume_mismatches': [{key: {'expected': a[key], 'actual': b[key]}
                for key in a if a[key] != b[key]}
                for a, b in zip(expected_future, resumed_future, strict=True)],
            'restore_mismatches': [{key: {'expected': a[key], 'actual': b[key]}
                for key in a if a[key] != b[key]}
                for a, b in zip(trajectory, repeated, strict=True)]}
        (args.output_dir / 'qualification.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
        print(execution, result['results'][execution]['exact_reference_trajectory'],
            result['results'][execution]['exact_checkpoint_resume'], flush=True)
        del stack
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return 0 if all(row['exact_reference_trajectory'] and row['exact_checkpoint_resume']
                    and row['exact_initial_restore'] for row in result['results'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
