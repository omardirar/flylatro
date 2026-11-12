"""Assisted strategy choices with native delayed outcomes and local plasticity.

The two discard alternatives and the reference PLAY are explicit task aids.
Only the fly's action-type and DISCARD-card KC->MBON efficacies learn.
"""
from __future__ import annotations

import numpy as np

from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed
from run_local_normalisation import Study as BaseStudy
from run_training_bottlenecks import best_visible_play
from strategy_puzzles import action, resample_future
from train_discard_play_curriculum import permute_rows, translate


VERSION = 'native-discard-strategy-learning-v1'


class StrategyMotor:
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def decode_strategy(self, activity, masks, candidates, *, ids, key, sampled=False, free_cards=False):
        scores = self.base.head_scores(activity)
        result = empty_action_batch(len(ids)); result['n_cards'][:] = 0
        choices, forced_discards = [], []
        for row, rep in enumerate(ids):
            legal = masks['action_type_mask'][row].copy(); legal[2:] = False
            rng = np.random.default_rng(derive_seed(VERSION, key, int(rep), 'type'))
            kind = self.base._choose(scores['action_type'][row], legal, not sampled, rng)
            values = np.asarray([scores['discard_card'][row, subset].mean() for subset in candidates[row]])
            if not np.isfinite(values).all(): raise ValueError('nonfinite strategy membership scores')
            candidate = int(values.argmax()); forced_discards.append(candidate)
            if sampled:
                r = np.random.default_rng(derive_seed(VERSION, key, int(rep), 'discard-exploration'))
                if r.random() < self.mapping.exploration_epsilon: candidate = int(r.integers(2))
            result['action_type'][row] = kind
            choices.append(0 if kind == 0 else candidate + 1)
            if kind == 1:
                raw_scores = scores['discard_card'][row].copy()
                raw_scores[~masks['card_select_mask'][row]] = -np.inf
                cards = (np.argsort(-raw_scores, kind='stable')[:4] if free_cards else candidates[row][candidate])
                result['n_cards'][row] = 4; result['cards'][row, :4] = cards
        return result, choices, forced_discards

    def selected_pool_mask(self, actions, masks):
        pools = np.zeros((len(actions['action_type']), self.mapping.routing.pool_count), dtype=bool)
        for row, kind in enumerate(actions['action_type']):
            if kind not in (0, 1) or not masks['action_type_mask'][row, kind]:
                raise ValueError('invalid strategy action')
            pools[row, self.mapping.routing.head_routes['action_type'][kind]] = True
            if kind == 1:
                if actions['n_cards'][row] != 4: raise ValueError('discard count must stay fixed')
                cards = actions['cards'][row, :4]
                if len(set(cards.tolist())) != 4 or np.any(cards < 0) or np.any(cards >= 8):
                    raise ValueError('invalid discard membership')
                pools[row, np.asarray(self.mapping.routing.head_routes['discard_card'])[cards]] = True
        return pools


def decay_helper_trace(local, discarded):
    """One zero-credit helper decision advances only the DISCARD learners."""
    if local.is_torch:
        import torch
        rows = torch.as_tensor(discarded, dtype=torch.bool, device=local.state.device)
    else:
        rows = np.asarray(discarded, dtype=bool)
    local.state.eligibility[rows] *= local.config.eligibility_decay
    local.state.decision_count[rows] += 1


class StrategyStudy(BaseStudy):
    def __init__(self, stack, corpus, *, replicates=3):
        super().__init__(stack, corpus, replicates=replicates)
        self.motor = StrategyMotor(stack.agent.motor.base)
        self.credit = SelectedMotorCredit(self.motor, self.local)
        mapping = self.motor.mapping
        selected_pools = list(mapping.routing.head_routes['discard_card'])
        selected_pools += [mapping.routing.head_routes['action_type'][i] for i in (0, 1)]
        roots = [mapping.output_root_ids[i] for pool in selected_pools if pool >= 0 for i in mapping.pool_indices[pool]]
        self.allowed = np.isin(self.local.topology.post_root_ids, roots)

    def choose(self, indices, key, *, initial=False, sampled=False, free_cards=False):
        for row, index in enumerate(indices): self.corpus.restore(self.env, row, index)
        raw, masks = self.env.observe()
        orders = [np.r_[np.random.default_rng(derive_seed(VERSION, key, int(rep), index, 'display')).permutation(8), 8, 9]
            for rep, index in zip(self.ids, indices, strict=True)]
        obs, legal = permute_rows(raw, masks, orders)
        candidates = []
        for index, order in zip(indices, orders, strict=True):
            inverse = np.argsort(order)
            candidates.append([inverse[subset].tolist() for subset in self.corpus.cases[index]['candidates']])
        efficacy = self.local.state.initial_efficacy if initial else self.local.state.efficacy
        neural = self.agent.processor.process(obs,
            fly_seeds=[derive_seed(VERSION, key, int(rep), index, 'neural') for rep, index in zip(self.ids, indices, strict=True)],
            efficacy=efficacy)
        if not np.isfinite(neural.output_activity).all(): raise ValueError('nonfinite strategy activity')
        selected, choices, forced = self.motor.decode_strategy(neural.output_activity, legal, candidates,
            ids=self.ids, key=key, sampled=sampled, free_cards=free_cards)
        return selected, choices, forced, neural, orders, legal, raw

    def train_step(self):
        cases = self.corpus.indices('training')
        indices = [cases[derive_seed(VERSION, self.steps, int(rep), 'case') % len(cases)] for rep in self.ids]
        key = ('training', self.steps)
        selected, choices, forced, neural, orders, legal, raw = self.choose(indices, key, sampled=True)
        self.local.record_activity(neural.edge_pre_activity, neural.edge_post_activity,
            edge_credit=self.credit.mask(selected, legal))
        executed = translate(selected, orders)
        immediate = selected['action_type'] == 0
        for row in range(self.n):
            # Resample only AFTER choosing. Even a future RNG implementation
            # change cannot expose the new hidden draw to this neural decision.
            hidden = derive_seed(VERSION, key, int(self.ids[row]), indices[row], 'hidden-future')
            resample_future(self.env, row, hidden)
            if immediate[row]:
                _, cards = best_visible_play(raw['hand'][row, :int(raw['hand_len'][row])])
                executed['n_cards'][row] = len(cards); executed['cards'][row] = -1
                executed['cards'][row, :len(cards)] = cards
        first = self.env.step(executed)
        outcomes = [dict(info) if immediate[row] else None for row, info in enumerate(first.infos)]
        helper_cards = [executed['cards'][row, :int(executed['n_cards'][row])].tolist() if immediate[row] else None
                        for row in range(self.n)]
        if not immediate.all():
            after, masks = self.env.observe(); follow = empty_action_batch(self.n); follow['n_cards'][:] = 0
            for row in range(self.n):
                if immediate[row]:
                    # Inactive rows must still receive a legal vector-env
                    # action. These auxiliary actions never supply learning.
                    follow['action_type'][row] = int(np.flatnonzero(masks['action_type_mask'][row])[0])
                    if follow['action_type'][row] in (0, 1): raise ValueError('inactive row unexpectedly still selecting a hand')
                else:
                    if first.dones[row] or first.rewards[row] != 0: raise ValueError('invalid discard transition')
                    _, cards = best_visible_play(after['hand'][row, :int(after['hand_len'][row])])
                    follow['action_type'][row] = 0; follow['n_cards'][row] = len(cards)
                    follow['cards'][row, :len(cards)] = cards; helper_cards[row] = list(cards)
            second = self.env.step(follow)
            for row in np.flatnonzero(~immediate): outcomes[row] = dict(second.infos[row])
            decay_helper_trace(self.local, ~immediate)
        pulses = [self.agent.reinforcement.map(info['reward_components'], info) for info in outcomes]
        positive = np.asarray([p.appetitive for p in pulses], dtype=np.float32)
        negative = np.asarray([p.aversive for p in pulses], dtype=np.float32)
        positive[self.roles == 2] = 0; negative[self.roles == 2] = 0
        events = self.local.apply_dopamine(positive, negative)
        update = np.asarray([e.absolute_change for e in events]); self.update_l1 += update
        self.local.state.reset_fast_traces(eligibility=True)
        row = {'step': self.steps, 'case_indices': indices, 'choices': choices,
            'discard_sets_native': [None if immediate[i] else executed['cards'][i, :4].tolist() for i in range(self.n)],
            'helper_play_cards_native': helper_cards,
            'cleared': [bool(info['reward_components'].get('blind_clear', 0) > 0) for info in outcomes],
            'appetitive': [p.appetitive for p in pulses], 'aversive': [p.aversive for p in pulses],
            'update_l1': update.tolist(), 'forced_discard_choices': forced}
        self.steps += 1
        return row

    def decisions(self, split, repeats=2, *, initial=False, free_cards=False):
        before = self.local.state.weight_sha256
        rows = []
        for index in self.corpus.indices(split):
            for repeat in range(repeats):
                selected, choices, forced, _, orders, _, _ = self.choose([index] * self.n,
                    ('evaluation', split, repeat), initial=initial, free_cards=free_cards)
                native = translate(selected, orders)
                rows.append({'case_index': index, 'repeat': repeat, 'choices': choices,
                    'forced_discard_choices': forced,
                    'discard_sets_native': [None if native['action_type'][i] == 0 else native['cards'][i, :4].tolist()
                                            for i in range(self.n)]})
        if self.local.state.weight_sha256 != before: raise ValueError('evaluation changed weights')
        return rows


def evaluate_choices(study, rows, benchmark):
    q = {r['case_index']: np.asarray(r['clear_probability']) for r in benchmark['rows']}
    result = {}
    for role, name in enumerate(('original', 'normalised', 'frozen')):
        lanes = np.flatnonzero(study.roles == role)
        values = np.asarray([[q[r['case_index']][r['choices'][lane]] for lane in lanes] for r in rows])
        best = np.asarray([q[r['case_index']].max() for r in rows])
        result[name] = {'estimated_clear_probability': values.mean(axis=0).tolist(),
            'mean_clear_probability': float(values.mean()),
            'mean_reference_regret': float(np.mean(best[:, None] - values))}
        for family in ('play-now', 'draw-needed'):
            selected = [r for r in rows if study.corpus.cases[r['case_index']]['family'] == family]
            types = np.asarray([[r['choices'][lane] == 0 for lane in lanes] for r in selected])
            forced_values = np.asarray([[q[r['case_index']][1 + r['forced_discard_choices'][lane]] for lane in lanes] for r in selected])
            result[name][family] = {'play_fraction': types.mean(axis=0).tolist(),
                'forced_discard_quality': forced_values.mean(axis=0).tolist()}
    return result
