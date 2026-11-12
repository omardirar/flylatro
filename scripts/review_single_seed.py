"""Review sealed single-seed runs with independent CPU replay of evaluations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import time

import numpy as np

from card_learning.audit import analyze
from card_learning.common import write_json
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from run_single_seed import read, verify


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(root, destination):
    root, destination = root.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    plan = verify(root)
    status = read(root / 'single-seed-status.json')
    require(status['status'] == 'completed', 'declared run is incomplete')
    destination.mkdir(parents=True)
    started = time.monotonic()
    report = {'status': 'running', 'plan_sha256': sha256_file(root / 'single-seed-plan.json'),
        'reviewer_sha256': sha256_file(Path(__file__)), 'source': str(root),
        'scope': 'all saved evaluation actions replayed on CPU; all episode records checked; checkpoint hashes and final anatomy budgets verified',
        'limitations': 'No replay of six million neural decisions. Shared evaluation seeds and checkpoint measurements are not independent training replicates. Training wins use stochastic choices; evaluations are greedy.',
        'learning_curve': [], 'training': {}, 'weight_health': {}}
    native_checks = [0]
    native_actions = total_episodes = 0
    try:
        for stage in plan['stages']:
            for arm in plan['arms']:
                receipt = read(root / 'receipts' / f'{arm}-{stage:09d}.json')
                path = Path(receipt['evaluation'])
                require(sha256_file(path) == receipt['sha256'], 'evaluation receipt mismatch')
                document = read(path)
                require(document['steps_per_learner'] == stage and document['arm'] == arm, 'wrong evaluation stage')
                if stage:
                    cp = Path(document['checkpoint'])
                    require(sha256_file(cp) == document['checkpoint_sha256'], 'evaluation checkpoint changed')
                    manifest = load_plastic_manifest(cp)
                    require(manifest['environment_decisions'] == stage * plan['replicates'], 'checkpoint exposure differs')
                    require(manifest['plastic_weight_sha256'] == document['repeated_seed']['weight_sha256']
                        == document['unseen_seeds']['weight_sha256'], 'evaluation weight identity differs')
                row = {'arm': arm, 'steps_per_learner': stage, 'source_sha256': receipt['sha256']}
                for split, seeds in [('repeated_seed', [plan['game_seed']]), ('unseen_seeds', plan['evaluation_seeds'])]:
                    doc = document[split]
                    require(doc['learning_disabled'] and doc['seeds'] == seeds, 'evaluation coverage or frozen status differs')
                    require(len(doc['episodes']) == len(doc['transitions']) == len(seeds) * plan['replicates'], 'evaluation count differs')
                    expected = {(i, seed) for i in range(plan['replicates']) for seed in seeds}
                    require({(r['learner'], r['seed']) for r in doc['episodes']} == expected, 'episode identities differ')
                    require({(r['learner'], r['seed']) for r in doc['transitions']} == expected, 'transition identities differ')
                    metrics = {'games': len(doc['episodes']), 'first_blind_clears': sum(e['blinds_beaten'] > 0 for e in doc['episodes']),
                        'blinds_cleared': sum(e['blinds_beaten'] for e in doc['episodes']), 'wins': sum(e['won'] for e in doc['episodes']),
                        'mean_native_return': float(np.mean([e['episode_return'] for e in doc['episodes']])), 'learners': []}
                    for learner in range(plan['replicates']):
                        groups = [g for g in doc['transitions'] if g['learner'] == learner]
                        eps = [e for e in doc['episodes'] if e['learner'] == learner]
                        data = destination / 'replay-inputs' / f'{arm}-{stage:09d}-{split}-{learner}.jsonl'
                        data.parent.mkdir(exist_ok=True)
                        with data.open('w') as stream:
                            for group in groups:
                                for action in group['actions']:
                                    require(action['seed'] == group['seed'], 'transition seed differs')
                                    stream.write(json.dumps(action, sort_keys=True) + '\n')
                        audit, decisions = analyze(data, data.stem, native_checks, native_check_states=2)
                        native_actions += audit['counts']['verified_before_hashes']
                        total_episodes += audit['counts']['episodes']
                        require(audit['counts']['episodes'] == len(eps), 'native completed episode count differs')
                        require(audit['counts'].get('wins', 0) == sum(e['won'] for e in eps), 'native win total differs')
                        require(audit['counts']['blinds_cleared'] == sum(e['blinds_beaten'] for e in eps), 'native clear total differs')
                        # Preserve individual trajectories and plain scores for a
                        # direct explanation of the repeated-seed behaviour.
                        write_json(destination / 'audits' / f'{data.stem}.json', audit)
                        if split == 'repeated_seed':
                            write_json(destination / 'repeated-decisions' / f'{arm}-{stage:09d}-{learner}.json', decisions)
                        metrics['learners'].append({'learner': learner,
                            'first_blind_clears': sum(e['blinds_beaten'] > 0 for e in eps),
                            'wins': sum(e['won'] for e in eps), 'audit': audit})
                    row[split] = metrics
                report['learning_curve'].append(row)
                print(f'replayed {arm} {stage:,}: repeated clears {row["repeated_seed"]["first_blind_clears"]}/3, unseen {row["unseen_seeds"]["first_blind_clears"]}/192', flush=True)
        config = PlasticExperimentConfig.load(Path(read(Path(next(iter(plan['arms'].values()))['condition']))['config']))
        with np.load(resolve_path(config, config.fly.artifact_path), allow_pickle=False) as z:
            edge_ids = z['kc_mbon_edge_indices']
            anatomy = z['signed_synapse_counts'][edge_ids]
            posts = z['root_ids'][z['post_indices'][edge_ids]]
        keep = anatomy >= config.fly.minimum_synapse_count
        anatomy, posts = anatomy[keep].astype(np.float64), posts[keep]
        for arm in plan['arms']:
            cp = Path(status['checkpoints'][arm]); load_plastic_manifest(cp)
            with cp.open('rb') as stream: payload = pickle.load(stream)
            state = payload['trainer']['plastic_state']
            efficacy, initial = state['efficacy'], state['initial_efficacy']
            require(efficacy.shape == (plan['replicates'], len(posts)), 'anatomical alignment differs')
            require(np.isfinite(efficacy).all() and np.all((efficacy >= config.plasticity.min_efficacy) & (efficacy <= config.plasticity.max_efficacy)), 'invalid efficacy')
            errors = np.zeros(plan['replicates']); growth = errors.copy()
            for post in np.unique(posts):
                mask = posts == post
                budget = (initial[:, mask] * anatomy[mask]).sum(axis=1)
                actual = (efficacy[:, mask] * anatomy[mask]).sum(axis=1)
                errors = np.maximum(errors, abs(actual - budget) / budget)
                growth = np.maximum(growth, (actual - budget) / budget)
            if arm == 'normalised': require(np.max(errors) < 1e-6, 'normalised budget differs')
            report['weight_health'][arm] = {'checkpoint_sha256': sha256_file(cp),
                'max_relative_post_budget_error_per_learner': errors.tolist(), 'max_post_strength_growth_per_learner': growth.tolist(),
                'lower_bound_fraction': np.mean(efficacy == config.plasticity.min_efficacy, axis=1).tolist(),
                'upper_bound_fraction': np.mean(efficacy == config.plasticity.max_efficacy, axis=1).tolist(),
                'mean_absolute_drift': np.mean(abs(efficacy - initial), axis=1).tolist(),
                'mean_efficacy': efficacy.mean(axis=1).tolist()}
            counters, windows = Counter(), {}
            expected_runs = Counter()
            seconds = 0.0
            for stage in plan['stages'][1:]:
                folder = root / 'training' / arm / f'stage-{stage:09d}'
                # This completed run must have one successful training attempt
                # per stage; an interrupted/replayed stream needs explicit audit.
                attempts = sorted(folder.glob('attempt-*'))
                require(len(attempts) == 1, 'multiple training attempts require explicit tail reconciliation')
                summary = read(attempts[0] / 'run-summary.json')
                require(summary['safety_stop'] is None and summary['environment_decisions'] == stage * plan['replicates'], 'training stage incomplete')
                seconds += summary['runtime']['runtime/wall_seconds']
                with (attempts[0] / 'training-episodes.jsonl').open() as stream:
                    for line in stream:
                        e = json.loads(line); learner = e['fly'] - 1
                        expected_runs[learner] += 1
                        require(e['run'] == expected_runs[learner], 'training episode cursor differs')
                        counters['episodes'] += 1; counters['wins'] += bool(e['won'])
                        counters['first_blind_cleared'] += bool(e['round'] > 1 or e['won'])
                        window = min((e['fly_decisions'] - 1) // 100000, 9)
                        w = windows.setdefault(str(window), Counter())
                        w['episodes'] += 1; w['wins'] += bool(e['won'])
                        w['first_blind_cleared'] += bool(e['round'] > 1 or e['won'])
                require(counters['episodes'] == summary['episodes'], 'cumulative training episode count differs')
            require(counters['episodes'] == payload['trainer']['training_state']['completed_episodes'], 'checkpoint episode total differs')
            require(counters['wins'] == payload['trainer']['training_state']['wins'], 'checkpoint training win total differs')
            report['training'][arm] = {'decisions': payload['trainer']['training_state']['environment_decisions'],
                **dict(counters), 'per_learner_episodes': dict(expected_runs), 'windows_100k_per_learner': windows,
                'training_wall_seconds': seconds}
        report.update(status='completed', evaluation_episodes_verified=total_episodes,
            evaluation_native_actions_verified=native_actions, extra_counterfactual_states_checked=native_checks[0],
            review_wall_seconds=time.monotonic() - started)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = review(args.run_dir, args.output_dir)
    print(json.dumps({k: report[k] for k in ('status', 'evaluation_episodes_verified', 'evaluation_native_actions_verified', 'training', 'weight_health')}))
