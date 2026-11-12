"""Exact native opening exploration diagnostic; never updates a fly or policy."""
from itertools import combinations
from pathlib import Path
import json
import sys

import numpy as np

from card_learning.common import write_json
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.telemetry.actions import _card_label
from strategy_puzzles import action


def explore(root, output):
    if output.exists(): raise FileExistsError(output)
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    env.reset([4000]); env.step(action(2)); obs, _ = env.observe()
    snapshot = env.snapshot(0)
    sources, selected = {}, []
    for arm in ('normalised', 'original'):
        path = root / 'evaluations' / f'{arm}-001000000.json'
        sources[str(path)] = sha256_file(path)
        doc = json.loads(path.read_text())
        for group in doc['repeated_seed']['transitions']:
            row = group['actions'][1]
            assert row['state_hash_before'] == env.state_hash(obs, 0)
            assert row['action']['type'] == 'play_hand'
            selected.append(tuple(sorted(row['action']['cards'])))
    assert len(selected) == 6 and len(set(selected)) == 1
    chosen = selected[0]
    subsets = list(combinations(range(8), 5))
    incidence = np.array([[i in s for i in range(8)] for s in subsets])
    scores = []
    for subset in subsets:
        env.restore(0, snapshot)
        result = env.step(action(0, list(subset)))
        assert not result.dones[0]
        assert result.infos[0]['reward_components']['blind_clear'] == 0
        scores.append(round(float(result.rewards[0]) * 300))
    scores = np.array(scores, dtype=float)
    swaps = [s for s in subsets if len(set(s) & set(chosen)) == 4]
    distributions = {'uniform_all_subsets': np.ones(len(subsets)) / len(subsets)}
    for name, candidates in [('whole_subset_5_percent', subsets), ('single_swap_5_percent', swaps)]:
        p = np.zeros(len(subsets)); p[subsets.index(chosen)] = .95
        for s in candidates: p[subsets.index(s)] += .05 / len(candidates)
        distributions[name] = p
    report = {'status': 'completed', 'source_evaluations_sha256': sources,
        'script_sha256': sha256_file(Path(__file__)), 'game_seed': 4000,
        'native_subsets_checked': len(subsets), 'chosen_slots': chosen,
        'hand': [_card_label(obs, 0, i) for i in range(8)],
        'chosen_chips': scores[subsets.index(chosen)], 'best_chips': float(max(scores)),
        'scope': 'one opening; fixed five cards and fixed greedy neural ranking; analytical sampler mixtures, not the full stochastic neural training distribution',
        'credit_proxy': 'E[score * selected] and Cov(score, selected); omits synaptic coactivity, eligibility history, normalisation and neural response to updates',
        'not_training': True, 'distributions': {}}
    for name, p in distributions.items():
        probability = p @ incidence
        mean = float(p @ scores)
        positive_credit = (p * scores) @ incidence
        covariance = (p * (scores - mean)) @ incidence
        report['distributions'][name] = {'mean_chips': mean,
            'probability_of_best_hand': float(p[scores == scores.max()].sum()),
            'per_card': [{'slot': i, 'card': report['hand'][i],
                'selected_probability': probability[i],
                'reward_times_selection': positive_credit[i],
                'reward_selection_covariance': covariance[i],
                'conditional_advantage_chips': positive_credit[i] / probability[i] - mean}
                for i in range(8)]}
    write_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    report = explore(Path(sys.argv[1]), Path(sys.argv[2]))
    for name, result in report['distributions'].items():
        print(name, json.dumps({'mean': result['mean_chips'], 'best_probability': result['probability_of_best_hand'],
            'missing_three': result['per_card'][7]}))
