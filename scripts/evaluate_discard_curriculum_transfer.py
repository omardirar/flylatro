"""Freeze curriculum weights and restore free actions in native Ante-1 games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import efficacy_like
from evaluate_motor_sampling import evaluate_policy, POLICIES
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.seeds import SeedPlan


def validate_state(efficacy, initial, *, replicates, edges, minimum, maximum, discard_edges):
    if efficacy.shape != initial.shape or efficacy.shape != (2*replicates, edges):
        raise ValueError('curriculum state shape differs')
    if (not np.isfinite(efficacy).all() or not np.isfinite(initial).all()
            or np.any(initial != 1) or np.any(efficacy < minimum) or np.any(efficacy > maximum)):
        raise ValueError('invalid curriculum efficacy or initial state')
    if not np.array_equal(efficacy[1::2], initial[1::2]):
        raise ValueError('frozen curriculum weights changed')
    if np.any((efficacy-initial)[:, ~np.asarray(discard_edges, dtype=bool)]):
        raise ValueError('non-DISCARD curriculum weights changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum-dir', type=Path, required=True)
    parser.add_argument('--condition', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--seed-offset', type=int, default=200000)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report_path = args.curriculum_dir/'report.json'
    state_path = args.curriculum_dir/'diagnostic-final-state.npz'
    curriculum = json.loads(report_path.read_text())
    if curriculum.get('status') != 'completed' or not curriculum['non_discard_efficacy_unchanged']:
        raise ValueError('completed isolated-DISCARD curriculum required')
    for path, digest in curriculum['source_hashes'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f'curriculum source changed: {path}')
    condition, corpus = load_condition(args.condition)
    config = PlasticExperimentConfig.load(Path(condition['config'])); config.require_heavy_opt_in(args.heavy)
    stack = build_condition_stack(config, condition, corpus)
    load_plastic_checkpoint(args.checkpoint, stack.trainer, expected_components=stack.components)
    with np.load(state_path, allow_pickle=False) as data:
        efficacy, initial = data['efficacy'].copy(), data['initial_efficacy'].copy()
    mapping = stack.agent.motor.mapping
    pools = [pool for head in ('discard_card', 'discard_card_count') for pool in mapping.routing.head_routes[head]]
    roots = [mapping.output_root_ids[i] for pool in pools for i in mapping.pool_indices[pool]]
    discard_edges = np.isin(stack.agent.plasticity.topology.post_root_ids, roots)
    validate_state(efficacy, initial, replicates=curriculum['replicates'],
        edges=stack.agent.plasticity.topology.edge_count, minimum=config.plasticity.min_efficacy,
        maximum=config.plasticity.max_efficacy, discard_edges=discard_edges)
    np.testing.assert_allclose(np.abs(efficacy-initial).mean(axis=1), curriculum['mean_absolute_drift'], atol=1e-9)
    paths = (report_path, state_path, args.condition, args.checkpoint, Path(__file__), Path(__file__).with_name('evaluate_motor_sampling.py'))
    sources = {str(path.resolve()): sha256_file(path) for path in paths}
    seeds = list(SeedPlan().seeds('validation', args.episodes, offset=args.seed_offset))
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'design.json', {'source_hashes': sources, 'seeds': seeds, 'learning_disabled': True,
        'models': 'initial and all plastic replicates', 'policies': POLICIES,
        'scope': 'free native actions; only DISCARD posts were trained; action timing was forced during curriculum'})
    started = time.perf_counter(); results = {}
    models = [('initial', initial[:1])]+[(f'learner-{i}', efficacy[2*i:2*i+1]) for i in range(curriculum['replicates'])]
    try:
        for name, values in models:
            results[name] = {}
            weights = efficacy_like(stack.agent.plasticity.state, values)
            for policy in POLICIES:
                def progress(row):
                    write_json(args.output_dir/'status.json', {'status': 'running', 'model': name, **row})
                report = evaluate_policy(stack.agent, weights, seeds, policy=policy, progress=progress)
                path = args.output_dir/f'{name}-{policy}.json';write_json(path, report)
                results[name][policy] = {'summary': report['summary'], 'report': str(path), 'sha256': sha256_file(path)}
                print(json.dumps({'model': name, 'policy': policy, **report['summary']}), flush=True)
        if any(sha256_file(Path(path)) != digest for path,digest in sources.items()):
            raise ValueError('transfer source identity changed')
        write_json(args.output_dir/'report.json', {'status': 'completed', 'results': results, 'source_hashes': sources,
            'seeds': seeds, 'source_state_unchanged': True, 'wall_seconds': time.perf_counter()-started})
        write_json(args.output_dir/'status.json', {'status': 'completed'})
    except BaseException as error:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
