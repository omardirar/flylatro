"""Explicit, stateless card exploration; no poker rules or learned parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from flylatro.env.upstream_contract import empty_action_batch, validate_strict_action_batch
from flylatro.seeds import derive_seed


@dataclass(frozen=True)
class CardExploration:
    probability: float = 0.05
    seed: int = 241024
    version: str = 'uniform-card-subset-exploration-v1'

    def __post_init__(self):
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError('card exploration probability must be in [0, 1]')
        if self.version != 'uniform-card-subset-exploration-v1':
            raise ValueError('unknown card exploration version')


class CardSubsetMotor:
    """Decorate the existing fixed motor without changing type/count RNG draws.

    Only PLAY and DISCARD membership is randomized. Greedy evaluation is exactly
    the underlying decoder. Actual selections reach selected_pool_mask/credit.
    """
    trainable_parameter_count = 0

    def __init__(self, base, exploration: CardExploration, *, pair_task=False):
        self.base = base
        self.exploration = exploration
        self.pair_task = bool(pair_task)
        self.last_choices = ()

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def experiment(self):
        return {'card_exploration': asdict(self.exploration),
                'pair_task': self.pair_task, 'external_trainable_parameter_count': 0,
                'greedy_evaluation': 'no subset sampling',
                'scope': 'PLAY and DISCARD only; type/count and targets unchanged outside pair task'}

    def decode(self, activity, masks, *, deterministic=True, learner_ids=None,
               decision_ids=None, record_choices=False):
        if self.pair_task:
            scores = self.base.head_scores(activity)['card'].copy()
            scores[~masks['card_select_mask']] = -np.inf
            actions = empty_action_batch(len(scores))
            actions['action_type'][:] = 0
            actions['n_cards'][:] = 2
            for row in range(len(scores)):
                if masks['card_select_mask'][row].sum() < 3:
                    raise ValueError('pair task requires competing two-card subsets')
                actions['cards'][row, :2] = np.argsort(-scores[row], kind='stable')[:2]
        else:
            actions = self.base.decode(activity, masks, deterministic=deterministic,
                learner_ids=learner_ids, decision_ids=decision_ids, record_choices=record_choices)
        if not deterministic and self.exploration.probability > 0:
            batch = len(actions['action_type'])
            if learner_ids is None or decision_ids is None:
                raise ValueError('subset exploration needs explicit learner and decision IDs')
            if np.shape(learner_ids) != (batch,) or np.shape(decision_ids) != (batch,):
                raise ValueError('subset exploration IDs must match batch')
            for row in range(batch):
                if int(actions['action_type'][row]) not in (0, 1):
                    continue
                rng = np.random.default_rng(derive_seed(self.exploration.version,
                    self.exploration.seed, int(learner_ids[row]), int(decision_ids[row])))
                if rng.random() < self.exploration.probability:
                    available = np.flatnonzero(masks['card_select_mask'][row])
                    count = int(actions['n_cards'][row])
                    # Canonical slot order: only membership, not card order, is
                    # this experimental factor. The simulator sorts selections.
                    actions['cards'][row, :count] = np.sort(rng.choice(available, count, replace=False))
        validate_strict_action_batch(actions, masks)
        self.last_choices = tuple({'action_type': int(actions['action_type'][row]),
            'card_count': int(actions['n_cards'][row]),
            'cards': tuple(int(c) for c in actions['cards'][row, :int(actions['n_cards'][row])])}
            for row in range(len(actions['action_type'])))
        return actions

    def selected_pool_mask(self, actions, masks):
        if not self.pair_task:
            return self.base.selected_pool_mask(actions, masks)
        validate_strict_action_batch(actions, masks)
        selected = np.zeros((len(actions['action_type']), self.mapping.routing.pool_count), dtype=bool)
        for row in range(len(selected)):
            if actions['action_type'][row] != 0 or actions['n_cards'][row] != 2:
                raise ValueError('pair task forces PLAY and count two')
            for card in actions['cards'][row, :2]:
                selected[row, self.mapping.routing.head_routes['card'][int(card)]] = True
        return selected

    def rng_state(self):
        return {'version': 'card-learning-motor-state-v1', 'experiment': self.experiment,
                'base': self.base.rng_state()}

    def load_rng_state(self, state):
        if state.get('version') != 'card-learning-motor-state-v1' or state.get('experiment') != self.experiment:
            raise ValueError('checkpoint card exploration or task differs')
        self.base.load_rng_state(state['base'])


def install_motor(stack, exploration, *, pair_task=False):
    motor = CardSubsetMotor(stack.agent.motor, exploration, pair_task=pair_task)
    stack.agent.motor = motor
    if stack.agent.credit is None:
        raise ValueError('card learning experiments require choice-gated credit')
    stack.agent.credit.motor = motor
    stack.components['card_learning_motor'] = motor.experiment
    return stack
