"""CPU-only observational audit of independently generated frozen traces.

Replayed choices are used solely to verify evidence and inspect visible card
patterns. They are never submitted as a learning policy or reward feature.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from flylatro.env.balatro_sim import BalatroSimAdapter, composite_actions_to_batch
from flylatro.env.types import ActionType, CompositeAction
from flylatro.evaluation.state_hash import hash_observation_row, terminal_hash


def audit(directory):
    started = time.monotonic()
    summary = json.loads((directory / 'summary.json').read_text())
    grouped = defaultdict(list)
    trace_path = directory / 'transitions.jsonl'
    for line in trace_path.read_text().splitlines():
        row = json.loads(line)
        grouped[row['seed']].append(row)
    if set(grouped) != set(summary['seeds']):
        raise ValueError('Trace seeds do not match the frozen summary')
    selections, patterns = Counter(), Counter()
    checked = unknown = available_pairs = selected_pairs = 0
    env = BalatroSimAdapter(1, strict=True, win_ante=summary['win_ante'])
    for seed in summary['seeds']:
        observations, masks = env.reset([seed])
        del masks
        rows = sorted(grouped[seed], key=lambda item: item['decision_id'])
        for index, row in enumerate(rows):
            if row['decision_id'] != index or hash_observation_row(observations, 0) != row['state_hash_before']:
                raise ValueError(f'Pre-action replay mismatch at seed {seed}, decision {index}')
            payload = row['action']
            action = CompositeAction(action_type=ActionType(payload['type']), cards=tuple(payload['cards']),
                **{key: payload.get(key) for key in ('target','joker_target','consumable_target','shop_target','pack_target')})
            if action.action_type == ActionType.PLAY_HAND:
                selections[len(action.cards)] += 1
                visible_hand_ranks = [int(np.argmax(token[:13]))
                    for token in observations['hand'][0, :int(observations['hand_len'][0])]
                    if token[37] <= 0 and np.any(token[:13])]
                available_pairs += any(count >= 2 for count in Counter(visible_hand_ranks).values())
                visible_ranks = []
                for slot in action.cards:
                    token = observations['hand'][0, slot]
                    if token[37] > 0 or not np.any(token[:13]):
                        unknown += 1
                        visible_ranks = []
                        break
                    visible_ranks.append(int(np.argmax(token[:13])))
                if visible_ranks:
                    # Rank multiplicities describe visible selected cards;
                    # they do not infer the simulator's scored poker category.
                    patterns[str(tuple(sorted(Counter(visible_ranks).values(), reverse=True)))] += 1
                    selected_pairs += any(count >= 2 for count in Counter(visible_ranks).values())
            step = env.step(composite_actions_to_batch([action]))
            after = terminal_hash(step.infos[0]) if step.dones[0] else hash_observation_row(step.observations, 0)
            if after != row['state_hash_after'] or bool(step.dones[0]) != row['done']:
                raise ValueError(f'Post-action replay mismatch at seed {seed}, decision {index}')
            if bool(step.dones[0]) != (index == len(rows) - 1):
                raise ValueError('Trace terminal boundary differs')
            checked += 1
            observations = step.observations
    return {'directory': str(directory), 'status': 'hash-verified-observational-audit',
        'transitions_checked': checked, 'episodes_checked': len(grouped),
        'visible_play_card_counts': dict(selections), 'visible_rank_multiplicities': dict(patterns),
        'plays_with_hidden_or_rankless_cards': unknown,
        'plays_with_available_visible_rank_pair': available_pairs,
        'plays_with_selected_visible_rank_pair': selected_pairs,
        'source_transitions_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'wall_seconds': time.monotonic() - started,
        'interpretation': 'Offline replay audit only; no policy, reward, training state or hidden card identities were changed. Visible rank pairs are descriptive, not optimal-play labels or a full scored poker category.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [audit(path) for path in args.evaluation_dir]
    with args.output.open('x') as stream:
        json.dump({'audits': reports}, stream, indent=2)
        stream.write('\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
