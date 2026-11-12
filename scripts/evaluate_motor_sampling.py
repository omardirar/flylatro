"""Frozen greedy versus training-sampled actions, with unchanged source state."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter, action_batch_row_to_composite
from flylatro.env.upstream_contract import GLOBAL_PHASE_OFF
from flylatro.evaluation.state_hash import hash_observation_row
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.seeds import SeedPlan, derive_seed


VERSION = 'frozen-motor-sampling-study-v1'
POLICIES = ('greedy', 'training-sampled')


def repeat_efficacy(value, count):
    if hasattr(value, 'detach'):
        return value.repeat(count, 1)
    return np.repeat(value, count, axis=0)


def evaluate_policy(agent, efficacy, seeds, *, policy, batch_size=4, max_steps=128,
                    env_factory=None, progress=None):
    if policy not in POLICIES or batch_size < 1 or max_steps < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('known policy, positive budgets and distinct seeds required')
    if np.shape(efficacy)[0] != 1:
        raise ValueError('evaluate one weight vector at a time')
    before = plastic_hashes(agent.plasticity.state)
    episodes, margins, actions_count = [], [], Counter()
    transitions = []
    factory = env_factory or (lambda n: BalatroSimAdapter(n, win_ante=1, allow_skip_blinds=False))
    for start in range(0, len(seeds), batch_size):
        batch_seeds = list(seeds[start:start+batch_size])
        env = factory(len(batch_seeds)); obs, masks = env.reset(batch_seeds)
        active = np.ones(len(batch_seeds), dtype=bool)
        counts = np.zeros(len(batch_seeds), dtype=int)
        returns, blinds = np.zeros(len(batch_seeds)), np.zeros(len(batch_seeds), dtype=int)
        per_episode = [Counter() for _ in batch_seeds]
        for step in range(max_steps):
            neural = agent.processor.process(obs,
                fly_seeds=[derive_seed('plastic-evaluation-fly', seed, step) for seed in batch_seeds],
                efficacy=repeat_efficacy(efficacy, len(batch_seeds)))
            if not np.isfinite(neural.output_activity).all():
                raise ValueError('nonfinite frozen output')
            scores = agent.motor.head_scores(neural.output_activity)['action_type']
            action = agent.motor.decode(neural.output_activity, masks, deterministic=policy == 'greedy',
                learner_ids=batch_seeds, decision_ids=counts.tolist())
            before_hashes = [hash_observation_row(obs, row) for row in range(len(batch_seeds))]
            playing = obs['global'][:, GLOBAL_PHASE_OFF+1].copy().astype(bool)
            both = masks['action_type_mask'][:, 0] & masks['action_type_mask'][:, 1]
            result = env.step(action)
            for row, seed in enumerate(batch_seeds):
                if not active[row]:
                    continue
                kind = int(action['action_type'][row]); per_episode[row][kind] += 1; actions_count[kind] += 1
                if both[row]:
                    margins.append(float(scores[row, 0]-scores[row, 1]))
                returns[row] += float(result.rewards[row])
                won = bool(result.infos[row].get('episode', {}).get('won', False))
                cleared = playing[row] and ((not result.dones[row] and bool(result.observations['global'][row, GLOBAL_PHASE_OFF+2])) or won)
                blinds[row] += int(cleared)
                transitions.append({'seed': seed, 'decision_id': int(counts[row]),
                    'state_hash_before': before_hashes[row], 'action': action_batch_row_to_composite(action, row).to_payload(),
                    'reward': float(result.rewards[row]), 'done': bool(result.dones[row]),
                    'both_play_discard_legal': bool(both[row]),
                    'play_minus_discard_score': float(scores[row, 0]-scores[row, 1]) if both[row] else None})
                counts[row] += 1
                if result.dones[row]:
                    episodes.append({'seed': seed, 'won': won, 'truncated': False,
                        'blinds': int(blinds[row]), 'return': float(result.infos[row].get('episode', {}).get('r', returns[row])),
                        'decisions': int(counts[row]), 'plays': per_episode[row][0], 'discards': per_episode[row][1]})
                    active[row] = False
            obs, masks = result.observations, result.masks
            if not active.any():
                break
        for row in np.flatnonzero(active):
            episodes.append({'seed': batch_seeds[row], 'won': False, 'truncated': True,
                'blinds': int(blinds[row]), 'return': float(returns[row]), 'decisions': int(counts[row]),
                'plays': per_episode[row][0], 'discards': per_episode[row][1]})
        if progress:
            progress({'completed_episodes': start+len(batch_seeds), 'policy': policy})
    if plastic_hashes(agent.plasticity.state) != before:
        raise ValueError('frozen policy study changed source state')
    episodes.sort(key=lambda row: row['seed'])
    if [row['seed'] for row in episodes] != list(seeds):
        raise ValueError('episode seed alignment differs')
    return {'version': VERSION, 'policy': policy, 'episodes': episodes, 'transitions': transitions,
        'summary': {'episodes': len(episodes), 'wins': sum(e['won'] for e in episodes),
            'blinds': sum(e['blinds'] for e in episodes), 'truncations': sum(e['truncated'] for e in episodes),
            'mean_return': float(np.mean([e['return'] for e in episodes])),
            'mean_decisions': float(np.mean([e['decisions'] for e in episodes])),
            'plays': actions_count[0], 'discards': actions_count[1],
            'discard_fraction': actions_count[1]/max(actions_count[0]+actions_count[1], 1),
            'play_minus_discard_mean': float(np.mean(margins)) if margins else None,
            'discard_outscores_play_fraction': float(np.mean(np.asarray(margins) < 0)) if margins else None},
        'source_state_unchanged': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed-offset', type=int, default=180000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=128)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.episodes < 1:
        raise ValueError('positive evaluation count required')
    seeds = list(SeedPlan().seeds('validation', args.episodes, offset=args.seed_offset))
    sources = {str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    for arm in ('subset-plastic', 'topk-plastic'):
        for path in (args.pilot_dir/'conditions'/arm/'condition.json', args.pilot_dir/arm/'plastic-checkpoint-final.pkl'):
            sources[str(path.resolve())] = sha256_file(path)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'design.json', {'version': VERSION, 'seeds': seeds, 'source_hashes': sources,
        'policies': POLICIES, 'models': 'initial once per arm and all three trained learners; same seeds/policies',
        'learning_disabled': True, 'max_decisions_per_episode': args.max_steps,
        'interpretation': 'diagnostic policy comparison; wins underpowered; no training or final-test use'})
    started = time.perf_counter(); results = {}
    try:
        for arm in ('subset-plastic', 'topk-plastic'):
            condition, corpus = load_condition(args.pilot_dir/'conditions'/arm/'condition.json')
            config = PlasticExperimentConfig.load(Path(condition['config'])); config.require_heavy_opt_in(args.heavy)
            stack = build_condition_stack(config, condition, corpus)
            load_plastic_checkpoint(args.pilot_dir/arm/'plastic-checkpoint-final.pkl', stack.trainer, expected_components=stack.components)
            state = stack.agent.plasticity.state
            models = [('initial', state.initial_efficacy[:1])]+[(f'learner-{i}', state.efficacy[i:i+1]) for i in range(state.learners)]
            results[arm] = {}
            for name, efficacy in models:
                results[arm][name] = {}
                for policy in POLICIES:
                    def progress(row):
                        value = {'status': 'running', 'arm': arm, 'model': name, **row}
                        write_json(args.output_dir/'status.json', value)
                        if row['completed_episodes'] % 32 == 0:
                            print(json.dumps(value), flush=True)
                    report = evaluate_policy(stack.agent, efficacy, seeds, policy=policy,
                        batch_size=args.batch_size, max_steps=args.max_steps, progress=progress)
                    path = args.output_dir/f'{arm}-{name}-{policy}.json'; write_json(path, report)
                    results[arm][name][policy] = {'summary': report['summary'], 'report': str(path), 'sha256': sha256_file(path)}
                    print(json.dumps({'arm': arm, 'model': name, 'policy': policy, **report['summary']}), flush=True)
            del stack
        if any(sha256_file(Path(p)) != h for p, h in sources.items()):
            raise ValueError('source identity changed during frozen comparison')
        write_json(args.output_dir/'report.json', {'version': VERSION, 'status': 'completed', 'results': results,
            'seeds': seeds, 'source_hashes': sources, 'source_state_unchanged': True, 'wall_seconds': time.perf_counter()-started})
        write_json(args.output_dir/'status.json', {'status': 'completed'})
    except BaseException as error:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
