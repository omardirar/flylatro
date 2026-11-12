"""Natural last-hand discard puzzles and visible-information reference outcomes.

Only ordinary native actions construct puzzles. Hidden futures are sampled
after a choice; no deck order or RNG state enters observations or the scorer.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from card_learning.common import write_json
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch, SIMULATOR_VERSION
from flylatro.seeds import SeedPlan, derive_seed
from run_training_bottlenecks import best_visible_play


VERSION = 'native-last-hand-strategy-puzzles-v1'


def action(kind, cards=(), *, batch=1):
    result = empty_action_batch(batch)
    result['action_type'][:] = kind; result['n_cards'][:] = len(cards)
    if cards: result['cards'][:, :len(cards)] = cards
    return result


def resample_future(env, row, seed):
    before, masks = env.observe()
    digest = env.state_hash(before, row)
    env._env.redeterminize(int(row), int(seed))
    after, new_masks = env.observe()
    if env.state_hash(after, row) != digest or any(not np.array_equal(masks[k][row], new_masks[k][row]) for k in masks):
        raise ValueError('hidden resampling altered observable state or legal actions')


def resources(obs, row=0):
    return (round(float(obs['global'][row, 1]) * 4), round(float(obs['global'][row, 2]) * 4))


def remaining(obs, row=0):
    return float(np.expm1(obs['blind'][row, 35]) - np.expm1(obs['blind'][row, 36]))


def construct_case(env, seed):
    env.reset([seed]); env.step(action(2))
    rng = np.random.default_rng(derive_seed(VERSION, seed, 'prefix'))
    prefix = [{'kind': 2, 'cards': []}]
    # Red Deck has four discards/hands. Six legal one-card actions leave one
    # of each, and three single-card scores cannot clear the first 300 blind.
    for _ in range(3):
        for kind in (1, 0):
            obs, _ = env.observe()
            slot = int(rng.integers(int(obs['hand_len'][0])))
            result = env.step(action(kind, [slot]))
            if result.dones[0] or result.infos[0]['reward_components'].get('blind_clear', 0):
                raise ValueError('one-card prefix unexpectedly completed a blind')
            prefix.append({'kind': kind, 'cards': [slot]})
    obs, _ = env.observe()
    if resources(obs) != (1, 1): raise ValueError('native prefix did not leave one hand/discard')
    n = int(obs['hand_len'][0])
    suits = obs['hand'][0, :n, 13:17].argmax(axis=1)
    suit = int(np.bincount(suits, minlength=4).argmax())
    suited = np.flatnonzero(suits == suit)
    if len(suited) not in (4, 5): return None
    chips, _ = best_visible_play(obs['hand'][0, :n])
    family = 'play-now' if chips >= remaining(obs) - .01 else 'draw-needed'
    if (family == 'play-now' and len(suited) != 5) or (family == 'draw-needed' and len(suited) != 4):
        return None
    ranks = obs['hand'][0, :n, :13].argmax(axis=1) + 2
    values = np.where(ranks == 14, 11, np.minimum(ranks, 10))
    keep = sorted(suited.tolist(), key=lambda i: (-int(values[i]), i))[:4]
    preserve = [i for i in range(n) if i not in keep]
    damage = suited[:4].tolist()
    if len(preserve) != 4 or set(preserve) == set(damage): raise ValueError('invalid equal-count alternatives')
    # Visible-pattern curriculum: even a low new card of this suit must make
    # a sufficiently strong plain flush. No actual future draw is inspected.
    if family == 'draw-needed' and (35 + int(values[keep].sum()) + 2) * 4 < remaining(obs) - .01:
        return None
    candidates = [preserve, damage]
    rng.shuffle(candidates)  # no universally correct candidate index
    snapshot = env.snapshot(0)
    return {'seed': int(seed), 'family': family, 'suit': suit, 'prefix': prefix,
        'snapshot': base64.b64encode(snapshot).decode(), 'snapshot_sha256': hashlib.sha256(snapshot).hexdigest(),
        'state_hash': env.state_hash(obs, 0), 'candidates': candidates,
        'visible_best_chips': chips, 'remaining_target': remaining(obs), 'drawpile_size': float(obs['drawpile_counts'].sum())}


def build(output, *, per_suit=3, max_attempts=10000):
    if per_suit < 1: raise ValueError('positive quota required')
    output = output.resolve()
    if output.exists(): raise FileExistsError(output)
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    cases = []
    for split, stream, offset in [('training', 'training', 150000), ('validation', 'validation', 650000),
                                  ('fresh', 'validation', 670000)]:
        counts = Counter()
        for seed in SeedPlan().seeds(stream, max_attempts, offset=offset):
            case = construct_case(env, seed)
            if case is None: continue
            group = (case['family'], case['suit'])
            if counts[group] >= per_suit: continue
            case['split'] = split; cases.append(case); counts[group] += 1
            if len(counts) == 8 and min(counts.values()) == per_suit: break
        else: raise ValueError(f'could not meet visible-state quota for {split}: {counts}')
    write_json(output, {'version': VERSION, 'simulator': SIMULATOR_VERSION, 'per_suit': per_suit,
        'construction': 'legal one-card native prefixes; one hand/discard left; visible suit/score filters only',
        'assistance': 'two curated equal-count discard alternatives; reference PLAY uses current visible cards only',
        'hidden_draws_used_for_selection': False, 'final_test_used': False, 'cases': cases})
    return PuzzleCorpus(output)


class PuzzleCorpus:
    def __init__(self, path):
        self.path = Path(path).resolve(); raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest(); self.document = json.loads(raw)
        if self.document['version'] != VERSION or self.document['simulator'] != SIMULATOR_VERSION:
            raise ValueError('puzzle version differs')
        self.cases = self.document['cases']; seen = set()
        for c in self.cases:
            if c['seed'] in seen: raise ValueError('puzzle seed overlap')
            seen.add(c['seed']); c['_snapshot'] = base64.b64decode(c['snapshot'], validate=True)
            if hashlib.sha256(c['_snapshot']).hexdigest() != c['snapshot_sha256']:
                raise ValueError('puzzle snapshot changed')
            if any(len(set(s)) != 4 or min(s) < 0 or max(s) >= 8 for s in c['candidates']):
                raise ValueError('invalid discard alternatives')
        for split in ('training', 'validation', 'fresh'):
            groups = Counter((c['family'], c['suit']) for c in self.cases if c['split'] == split)
            if len(groups) != 8 or any(n != self.document['per_suit'] for n in groups.values()):
                raise ValueError('puzzle quotas differ')

    def indices(self, split):
        return [i for i, c in enumerate(self.cases) if c['split'] == split]

    def restore(self, env, row, index):
        env.restore(row, self.cases[index]['_snapshot'])
        obs, _ = env.observe()
        if env.state_hash(obs, row) != self.cases[index]['state_hash'] or resources(obs, row) != (1, 1):
            raise ValueError('restored puzzle state differs')


class Reference:
    """Native one-discard/one-PLAY evaluator; no future-order access."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
        self.env.reset([0]); self.cache = {}; self.native_actions = 0

    def outcome(self, index, discard, draw_key):
        discard = None if discard is None else tuple(sorted(map(int, discard)))
        key = (index, discard, tuple(draw_key) if isinstance(draw_key, list) else draw_key)
        if key in self.cache: return self.cache[key]
        self.corpus.restore(self.env, 0, index)
        hidden = derive_seed(VERSION, self.corpus.cases[index]['seed'], draw_key, 'hidden-future')
        resample_future(self.env, 0, hidden)
        if discard is not None:
            result = self.env.step(action(1, discard)); self.native_actions += 1
            if result.dones[0] or result.rewards[0] != 0: raise ValueError('discard was not nonterminal/zero reward')
        obs, _ = self.env.observe()
        chips, cards = best_visible_play(obs['hand'][0, :int(obs['hand_len'][0])])
        result = self.env.step(action(0, cards)); self.native_actions += 1
        clear = result.infos[0]['reward_components'].get('blind_clear', 0) > 0
        if clear != (chips >= remaining(obs) - .01): raise ValueError('visible scorer/native clear mismatch')
        value = {'cleared': bool(clear), 'native_reward': float(result.rewards[0]),
            'native_components': result.infos[0]['reward_components'], 'play_cards': list(cards), 'chips': chips}
        self.cache[key] = value
        return value


def benchmark(corpus, output, *, draws=128):
    ref = Reference(corpus); rows = []
    for index, case in enumerate(corpus.cases):
        outcomes = []
        for choice in range(3):
            discard = None if choice == 0 else case['candidates'][choice-1]
            values = [ref.outcome(index, discard, ('benchmark', draw))['cleared'] for draw in range(draws)]
            outcomes.append(values)
        q = np.mean(outcomes, axis=1)
        rows.append({'case_index': index, 'split': case['split'], 'family': case['family'],
            'clear_probability': q.tolist(), 'outcomes': outcomes, 'best_choice': int(q.argmax()),
            'play_now_clear': bool(q[0]), 'best_value': float(q.max())})
    result = {'version': VERSION, 'corpus_sha256': corpus.sha256, 'draws': draws,
        'selection': 'all cases; no case dropped based on these outcomes', 'rows': rows,
        'native_actions': ref.native_actions, 'scope': 'one-step assisted reference, not an optimal full-run policy'}
    write_json(output, result)
    return result
