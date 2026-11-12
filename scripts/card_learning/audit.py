"""Verified frozen-log replay; plain first-Small-Blind mistakes, never policy input.

Only complete, native Ante-1 evaluation logs are accepted. Immediate-score
regret is descriptive, not a claim of optimal long-term strategy.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from flylatro.analysis.card_selection import score_plays
from flylatro.env.balatro_sim import BalatroSimAdapter, composite_actions_to_batch
from flylatro.env.types import ActionType, CompositeAction
from flylatro.evaluation.state_hash import terminal_hash
from flylatro.telemetry.actions import _card_label

BASE = {'high_card': (5, 1), 'pair': (10, 2), 'two_pair': (20, 2),
        'three_kind': (30, 3), 'straight': (30, 4), 'flush': (35, 4),
        'full_house': (40, 4), 'four_kind': (60, 7), 'straight_flush': (100, 8)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(hand, slots):
    cards = [hand[i] for i in slots]
    counts = Counter(rank for rank, suit in cards)
    groups = sorted(counts.values(), reverse=True)
    ranks = sorted(counts)
    straight = len(cards) == 5 and len(ranks) == 5 and (ranks[-1] - ranks[0] == 4 or ranks == [2, 3, 4, 5, 14])
    flush = len(cards) == 5 and len({s for r, s in cards}) == 1
    if straight and flush:
        name, scoring = 'straight_flush', list(counts)
    elif groups[0] == 4:
        name, scoring = 'four_kind', [r for r, n in counts.items() if n == 4]
    elif groups == [3, 2]:
        name, scoring = 'full_house', list(counts)
    elif flush:
        name, scoring = 'flush', list(counts)
    elif straight:
        name, scoring = 'straight', list(counts)
    elif groups[0] == 3:
        name, scoring = 'three_kind', [r for r, n in counts.items() if n == 3]
    elif groups.count(2) == 2:
        name, scoring = 'two_pair', [r for r, n in counts.items() if n == 2]
    elif groups[0] == 2:
        name, scoring = 'pair', [r for r, n in counts.items() if n == 2]
    else:
        name, scoring = 'high_card', [max(counts)]
    chips, mult = BASE[name]
    return (chips + sum((11 if r == 14 else min(r, 10)) * counts[r] for r in scoring)) * mult, name


def analyze(path, label, simulator_checks, native_check_states=20):
    data = path.read_bytes()
    by_seed = defaultdict(list)
    for line in data.splitlines():
        row = json.loads(line)
        by_seed[row['seed']].append(row)
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    stats = Counter()
    values = defaultdict(list)
    decisions = []
    examples = defaultdict(list)
    slots_play, slots_discard = Counter(), Counter()
    kinds, counts = Counter(), Counter()
    for seed, rows in sorted(by_seed.items()):
        obs, masks = env.reset([seed])
        episode_positive = episode_negative = 0.0
        first_blind_positive = first_blind_negative = 0.0
        first_blind_progress = 0.0
        episode_clears = 0
        for expected_id, row in enumerate(rows):
            require(row['decision_id'] == expected_id, 'decision IDs are not contiguous')
            require(env.state_hash(obs, 0) == row['state_hash_before'], f'{label} seed {seed} decision {expected_id}: before hash differs')
            stats['verified_before_hashes'] += 1
            payload = dict(row['action'])
            payload['action_type'] = ActionType(payload.pop('type'))
            payload['cards'] = tuple(payload['cards'])
            action = CompositeAction(**payload)
            batch = composite_actions_to_batch([action])
            first = env.run_info(0)['round'] == 1 and action.action_type in (ActionType.PLAY_HAND, ActionType.DISCARD)
            detail = None
            if first:
                n = int(obs['hand_len'][0])
                hand_features = obs['hand'][0, :n]
                require(not np.any(obs['joker_ids']), 'plain first-blind audit found a Joker')
                require(np.all(hand_features[:, [17, 26, 31]] == 1), 'plain first-blind audit found modified cards')
                require(np.all(hand_features[:, [i for i in range(17, 38) if i not in (17, 26, 31)]] == 0), 'plain first-blind audit found modified cards')
                hand = [(int(c[:13].argmax()) + 2, int(c[13:17].argmax())) for c in hand_features]
                all_scores = {subset: score(hand, subset) for k in range(1, min(5, n) + 1) for subset in combinations(range(n), k)}
                best = max(v[0] for v in all_scores.values())
                chosen_score, kind = score(hand, action.cards)
                target = math.expm1(float(obs['blind'][0, 35]))
                scored = math.expm1(float(obs['blind'][0, 36]))
                remaining = target - scored
                require(abs(target - 300) < 0.01, 'first-blind target differs from pinned 300 chips')
                hands_left = round(float(obs['global'][0, 1]) * 4)
                discards_left = round(float(obs['global'][0, 2]) * 4)
                stats['first_blind_decisions'] += 1
                detail = dict(seed=seed, decision_id=expected_id, action=action.to_payload(),
                    hand=[_card_label(obs, 0, i) for i in range(n)],
                    selected=[_card_label(obs, 0, i) for i in action.cards],
                    selected_as_play_score=chosen_score, selected_as_play_kind=kind,
                    best_play_score=best, best_plays=[list(k) for k, v in all_scores.items() if v[0] == best],
                    hands_left=hands_left, discards_left=discards_left, score_so_far=round(scored), target=round(target),
                    best_play_clears=best >= remaining - 0.01,
                    selected_play_clears=chosen_score >= remaining - 0.01)
                if simulator_checks[0] < native_check_states and expected_id == 1:
                    native = score_plays(env, env.snapshot(0), obs, masks)
                    for subset, native_reward in native.items():
                        predicted_score = all_scores[subset][0]
                        predicted_reward = min(predicted_score, remaining) / target
                        if predicted_score >= remaining - 0.01:
                            predicted_reward += 0.575
                        require(abs(native_reward - predicted_reward) < 2e-5, f'native subset score differs: {subset}')
                        stats['counterfactual_native_play_checks'] += 1
                    simulator_checks[0] += 1
                key = 'plays' if action.action_type == ActionType.PLAY_HAND else 'discards'
                stats[key] += 1
                counts[f'{key}_{len(action.cards)}'] += 1
                (slots_play if key == 'plays' else slots_discard).update(action.cards)
                values[f'{key}_selected_score'].append(chosen_score)
                values[f'{key}_best_score'].append(best)
                if key == 'plays':
                    kinds[kind] += 1
                    stats['plays_best_score_selected'] += chosen_score == best
                    stats['plays_inferior_to_available'] += chosen_score < best
                    stats['plays_missed_available_clear'] += best >= remaining - 0.01 and chosen_score < remaining - 0.01
                    stats['plays_with_discards_left'] += discards_left > 0
                    stats['last_hand_plays_with_discards_left'] += hands_left == 1 and discards_left > 0
                    stats['last_hand_losing_play_with_discards_left'] += hands_left == 1 and discards_left > 0 and chosen_score < remaining - 0.01
                    if chosen_score < best and len(examples['inferior_plays']) < 4:
                        examples['inferior_plays'].append(detail)
                    if best >= remaining - 0.01 and chosen_score < remaining - 0.01 and len(examples['missed_clear']) < 4:
                        examples['missed_clear'].append(detail)
                    random_score = np.mean([np.mean([v[0] for s, v in all_scores.items() if len(s) == k]) for k in range(1, min(5, n) + 1)])
                    values['uniform_count_then_uniform_subset_score'].append(float(random_score))
                    same_count = [v[0] for s, v in all_scores.items() if len(s) == len(action.cards)]
                    values['uniform_subset_with_recorded_count_score'].append(float(np.mean(same_count)))
                    stats['plays_inferior_to_available_same_count'] += chosen_score < max(same_count)
                    stats['high_card_plays_with_pair_available'] += kind == 'high_card' and len({r for r, s in hand}) < n
                    values['play_score_over_best'].append(chosen_score / best)
                else:
                    stats['discards_with_available_clear'] += best >= remaining - 0.01
                    kept = [i for i in range(n) if i not in action.cards]
                    best_kept = max(score(hand, s)[0] for k in range(1, min(5, len(kept)) + 1) for s in combinations(kept, k))
                    stats['discards_break_all_current_best_plays'] += best_kept < best
                    if best >= remaining - 0.01 and len(examples['discard_with_clear']) < 4:
                        examples['discard_with_clear'].append(detail)
            result = env.step(batch)
            require(bool(result.dones[0]) == row['done'], 'terminal flag differs from native replay')
            require(abs(float(result.rewards[0]) - row['reward']) < 1e-6, 'reward differs from native replay')
            after_hash = terminal_hash(result.infos[0]) if row['done'] else env.state_hash(result.observations, 0)
            require(after_hash == row['state_hash_after'], f'{label} seed {seed} decision {expected_id}: after hash differs')
            stats['verified_after_hashes'] += 1
            positive = row['synthetic_appetitive_reinforcement']
            negative = row['synthetic_aversive_reinforcement']
            episode_positive += positive
            episode_negative += negative
            cleared = row['reward_components'].get('blind_clear', 0) > 0
            episode_clears += cleared
            if first:
                first_blind_positive += positive
                first_blind_negative += negative
                first_blind_progress += row['reward_components'].get('blind_progress', 0)
                detail.update(reward=row['reward'], reward_components=row['reward_components'],
                    appetitive=positive, aversive=negative, done=row['done'])
                decisions.append(detail)
                if row['done']:
                    stats['first_blind_failures'] += 1
                    stats['first_blind_failures_with_unused_discards'] += detail['discards_left'] > 0
                    stats['terminal_scoring_plays_with_zero_progress'] += detail['selected_as_play_score'] > 0 and row['reward_components']['blind_progress'] == 0
                    values['failed_first_blind_net_pulse'].append(first_blind_positive - first_blind_negative)
                    values['failed_first_blind_progress_fraction'].append(first_blind_progress)
            if row['done']:
                stats['episodes'] += 1
                stats['wins'] += result.infos[0]['episode']['won']
                if not result.infos[0]['episode']['won']:
                    values['failed_episode_net_pulse'].append(episode_positive - episode_negative)
                    stats['failed_episodes_net_positive_pulse'] += episode_positive > episode_negative
                stats['blinds_cleared'] += episode_clears
            obs, masks = result.observations, result.masks
        require(rows[-1]['done'], 'audit found an incomplete episode')
    return dict(label=label, source=str(path), source_sha256=hashlib.sha256(data).hexdigest(),
        counts=dict(stats), means={k: float(np.mean(v)) for k, v in values.items()},
        failed_first_blind_net_positive=sum(v > 0 for v in values['failed_first_blind_net_pulse']),
        play_types=dict(kinds), card_counts=dict(counts), selected_play_slots=dict(slots_play),
        selected_discard_slots=dict(slots_discard), examples=dict(examples)), decisions



def audit_files(paths, output, *, native_check_states=20):
    from .common import write_json, identity
    if native_check_states < 0:
        raise ValueError('native check count cannot be negative')
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('audit needs distinct transition files')
    if output.exists():
        raise FileExistsError(output)
    for path in paths:
        summary_path = path.parent / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get('win_ante') != 1 or summary.get('seed_stream') != 'validation':
                raise ValueError('this audit supports validation Ante-1 evaluation only')
        raw = path.read_bytes()
        if not raw or not raw.endswith(b'\n'):
            raise ValueError('audit requires complete newline-terminated logs')
    output.mkdir(parents=True)
    report = {'version': 'first-blind-decision-audit-v1', 'status': 'running',
        'implementation': identity(), 'scope': 'plain first Small Blind; immediate-score comparisons exclude Jokers and upgrades',
        'limitations': 'same-state random subsets are diagnostics, not a random full-game policy; training replicates remain distinct from evaluation roots',
        'arms': []}
    try:
        with (output / 'decisions.jsonl').open('w') as stream:
            for path in paths:
                summary, decisions = analyze(path, path.parent.name, [0], native_check_states)
                report['arms'].append(summary)
                for row in decisions:
                    stream.write(json.dumps(dict(source=str(path), **row), sort_keys=True) + '\n')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'report.json', report)
    return report
