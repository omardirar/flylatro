"""Native opening snapshots and an explicitly artificial one-decision curriculum."""
from __future__ import annotations

import base64
from collections import Counter
import json
from pathlib import Path
import pickle

import numpy as np

from flylatro.env.balatro_sim import ArrayStep, BalatroSimAdapter
from flylatro.env.snapshot_identity import canonical_snapshot
from flylatro.env.upstream_contract import SIMULATOR_VERSION, empty_action_batch, validate_strict_action_batch
from flylatro.seeds import SeedPlan, derive_seed

from .common import digest, write_json


VERSION = 'opening-one-pair-corpus-v1'
TASK = 'one-decision-two-card-pair-task-v1'


def pair_slots(observations, row=0):
    n = int(observations['hand_len'][row])
    cards = observations['hand'][row, :n]
    if n != 8 or np.any(observations['joker_ids'][row]):
        raise ValueError('pair curriculum requires eight plain opening cards without Jokers')
    normal = np.zeros((8, 21), dtype=np.float32)
    normal[:, [0, 9, 14]] = 1  # enhancement/edition/seal = none (17/26/31)
    if not np.array_equal(cards[:, 17:], normal):
        raise ValueError('pair curriculum does not accept modified cards')
    if not np.all(cards[:, :13].sum(axis=1) == 1) or not np.all(cards[:, 13:17].sum(axis=1) == 1):
        raise ValueError('pair curriculum requires visible rank and suit')
    ranks = cards[:, :13].argmax(axis=1)
    counts = Counter(ranks.tolist())
    if sorted(counts.values()) != [1, 1, 1, 1, 1, 1, 2]:
        raise ValueError('pair curriculum requires exactly one pair')
    rank = next(k for k, count in counts.items() if count == 2)
    return tuple(int(i) for i in np.flatnonzero(ranks == rank))


def build_corpus(output: Path, *, train_per_rank=8, validation_per_rank=4,
                 training_offset=6000, validation_offset=60000, max_attempts=100000):
    if min(train_per_rank, validation_per_rank, max_attempts) < 1:
        raise ValueError('positive corpus quotas and search budget required')
    if output.exists():
        raise FileExistsError(output)
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    rows = []
    for split, stream, offset, quota in [('training', 'training', training_offset, train_per_rank),
                                         ('validation', 'validation', validation_offset, validation_per_rank)]:
        counts = Counter()
        seeds = SeedPlan().seeds(stream, max_attempts, offset=offset)
        for seed in seeds:
            env.reset([seed])
            action = empty_action_batch(1); action['action_type'][:] = 2; action['n_cards'][:] = 0
            result = env.step(action)
            obs = result.observations
            try:
                slots = pair_slots(obs)
            except ValueError:
                continue
            rank = int(obs['hand'][0, slots[0], :13].argmax()) + 2
            if counts[rank] == quota:
                continue
            snapshot = canonical_snapshot(env.snapshot(0), simulator_version=SIMULATOR_VERSION)
            rows.append({'split': split, 'seed': seed, 'pair_rank': rank,
                'state_hash': env.state_hash(obs, 0), 'snapshot_sha256': digest(snapshot),
                'snapshot': base64.b64encode(snapshot).decode('ascii')})
            counts[rank] += 1
            if len(counts) == 13 and all(v == quota for v in counts.values()):
                break
        else:
            raise ValueError(f'corpus search exhausted before {split} rank quotas were met')
    document = {'version': VERSION, 'simulator_version': SIMULATOR_VERSION,
        'construction': 'opening eight-card hands with exactly one pair; equal quota per pair rank; no action targets or reward labels',
        'training_per_rank': train_per_rank, 'validation_per_rank': validation_per_rank,
        'training_offset': training_offset, 'validation_offset': validation_offset,
        'display_order': 'independent seeded permutation per trial; actions translated back to snapshot slots',
        'chance_pair_selection': 1 / 28, 'cases': rows, 'final_test_seeds_used': False}
    write_json(output, document)
    return document


class PairCorpus:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = digest(raw)
        self.document = json.loads(raw)
        if self.document.get('version') != VERSION or self.document.get('simulator_version') != SIMULATOR_VERSION:
            raise ValueError('unknown pair corpus or simulator version')
        self.cases = self.document['cases']
        seen = set()
        counts = {'training': Counter(), 'validation': Counter()}
        for case in self.cases:
            split, seed = case['split'], case['seed']
            if split not in counts or seed in seen:
                raise ValueError('pair corpus split or duplicate seed invalid')
            bounds = getattr(SeedPlan(), split)
            if not bounds.start <= seed < bounds.stop:
                raise ValueError('pair corpus seed outside declared split')
            seen.add(seed)
            snapshot = base64.b64decode(case['snapshot'], validate=True)
            if digest(snapshot) != case['snapshot_sha256']:
                raise ValueError('pair snapshot digest differs')
            case['_snapshot'] = snapshot
            counts[split][case['pair_rank']] += 1
        for split, key in [('training', 'training_per_rank'), ('validation', 'validation_per_rank')]:
            if counts[split] != Counter({rank: self.document[key] for rank in range(2, 15)}):
                raise ValueError('pair corpus rank quotas differ')

    def indices(self, split):
        if split not in ('training', 'validation'):
            raise ValueError('pair task supports training/validation only')
        return [i for i, case in enumerate(self.cases) if case['split'] == split]

    def restore(self, env, row, index):
        case = self.cases[index]
        env.restore(row, case['_snapshot'])
        obs, masks = env.observe()
        if env.state_hash(obs, row) != case['state_hash']:
            raise ValueError('pair snapshot observation hash differs')
        slots = pair_slots(obs, row)
        rank = int(obs['hand'][row, slots[0], :13].argmax()) + 2
        if rank != case['pair_rank']:
            raise ValueError('pair snapshot rank differs')


class PairTaskEnvironment:
    """Each action scores in a real opening state, then resets to another case.

    Curriculum boundaries are not run failures or Balatro wins. The only reward
    component is native blind progress; pair labels are diagnostic info only.
    """
    allow_skip_blinds = False
    simulator_version = SIMULATOR_VERSION

    def __init__(self, base, corpus: PairCorpus):
        self.base, self.corpus = base, corpus
        self.num_envs = base.num_envs
        self._beta = base._beta
        self.indices = corpus.indices('training')
        self.seeds = [0] * self.num_envs
        self.trials = [0] * self.num_envs
        self.case_indices = [0] * self.num_envs
        self.orders = [np.arange(10) for _ in range(self.num_envs)]

    def _start(self, row):
        seed, trial = self.seeds[row], self.trials[row]
        rng = np.random.default_rng(derive_seed(TASK, seed, trial))
        index = self.indices[int(rng.integers(len(self.indices)))]
        self.corpus.restore(self.base, row, index)
        self.case_indices[row] = index
        self.orders[row] = np.r_[rng.permutation(8), 8, 9]

    def reset(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError('pair task needs one seed per learner')
        self.base.reset(seeds)
        self.seeds = [int(s) for s in seeds]
        self.trials = [0] * self.num_envs
        for row in range(self.num_envs):
            self._start(row)
        return self.observe()

    def observe(self):
        obs, masks = self.base.observe()
        obs = {k: v.copy() for k, v in obs.items()}
        masks = {k: np.zeros_like(v) for k, v in masks.items()}
        masks['action_type_mask'][:, 0] = True
        masks['card_select_mask'][:, :8] = True
        for row, order in enumerate(self.orders):
            obs['hand'][row] = obs['hand'][row, order].copy()
        return obs, masks

    def step(self, actions):
        before, masks = self.observe()
        validate_strict_action_batch(actions, masks)
        if np.any(actions['n_cards'] != 2):
            raise ValueError('pair task requires exactly two cards')
        translated = {k: v.copy() for k, v in actions.items()}
        for row, order in enumerate(self.orders):
            translated['cards'][row, :2] = order[actions['cards'][row, :2]]
        result = self.base.step(translated)
        infos = []
        for row, raw_info in enumerate(result.infos):
            if result.dones[row] or raw_info['reward_components'].get('blind_clear', 0):
                raise ValueError('pair task unexpectedly ended or cleared a native blind')
            chosen = actions['cards'][row, :2]
            same_rank = before['hand'][row, chosen, :13].argmax(axis=1)
            infos.append({'reward_components': dict(raw_info['reward_components']),
                'episode': {'r': float(result.rewards[row]), 'l': 1, 'ante': 1, 'task_boundary': True},
                'task': {'version': TASK, 'pair_selected': bool(same_rank[0] == same_rank[1]),
                         'case_index': self.case_indices[row], 'trial': self.trials[row]}})
            self.trials[row] += 1
            self._start(row)
        obs, masks = self.observe()
        return ArrayStep(obs, masks, result.rewards, np.ones(self.num_envs, dtype=bool), tuple(infos))

    def snapshot_all(self):
        return tuple(pickle.dumps({'version': TASK, 'corpus_sha256': self.corpus.sha256,
            'base': self.base.snapshot(row), 'seed': self.seeds[row], 'trial': self.trials[row],
            'case_index': self.case_indices[row], 'order': self.orders[row].tolist()})
            for row in range(self.num_envs))

    def restore_all(self, snapshots):
        if len(snapshots) != self.num_envs:
            raise ValueError('pair snapshot count differs')
        for row, snapshot in enumerate(snapshots):
            # Trusted local checkpoint, same contract as production checkpoints.
            value = pickle.loads(snapshot)
            if value['version'] != TASK or value['corpus_sha256'] != self.corpus.sha256:
                raise ValueError('pair task checkpoint corpus differs')
            if value['case_index'] not in self.indices or sorted(value['order'][:8]) != list(range(8)) or value['order'][8:] != [8, 9]:
                raise ValueError('invalid pair checkpoint selection or permutation')
            self.base.restore(row, value['base'])
            self.seeds[row], self.trials[row] = int(value['seed']), int(value['trial'])
            self.case_indices[row], self.orders[row] = value['case_index'], np.asarray(value['order'])

    def run_seed(self, row):
        return self.base.run_seed(row)

    def run_info(self, row):
        return self.base.run_info(row)

    def set_win_ante(self, ante):
        if ante != 1:
            raise ValueError('pair task requires fixed Ante 1')
        self.base.set_win_ante(ante)

    def set_shaping_beta(self, beta):
        self.base.set_shaping_beta(beta)
        self._beta = beta
