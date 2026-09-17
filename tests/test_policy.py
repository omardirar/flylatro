from __future__ import annotations

import numpy as np
import pytest

from flylatro.env.types import ACTION_INDEX, ACTION_TYPES, ActionMask, ActionType
from flylatro.policy.structured import StructuredLinearPolicy


def only_discard_mask() -> ActionMask:
    action_types = np.zeros(len(ACTION_TYPES), dtype=np.bool_)
    cards = np.zeros((len(ACTION_TYPES), 5), dtype=np.bool_)
    min_cards = np.zeros(len(ACTION_TYPES), dtype=np.int64)
    max_cards = np.zeros(len(ACTION_TYPES), dtype=np.int64)
    targets = np.zeros((len(ACTION_TYPES), 3), dtype=np.bool_)
    requires_target = np.zeros(len(ACTION_TYPES), dtype=np.bool_)
    index = ACTION_INDEX[ActionType.DISCARD]
    action_types[index] = True
    cards[index, [1, 3, 4]] = True
    min_cards[index] = 2
    max_cards[index] = 2
    return ActionMask(
        action_types, cards, min_cards, max_cards, targets, requires_target
    )


def test_structured_policy_never_emits_an_illegal_composite_action() -> None:
    policy = StructuredLinearPolicy(feature_size=6, max_cards=5, max_targets=3)
    mask = only_discard_mask()

    (decision,) = policy.act(
        np.ones((1, 6), dtype=np.float64), [mask], deterministic=True
    )

    assert decision.action.action_type is ActionType.DISCARD
    assert len(decision.action.cards) == 2
    assert mask.allows(decision.action)
    assert decision.action_probability > 0
    assert sum(decision.action_type_probabilities.values()) == pytest.approx(1.0)
    assert policy.trainable_parameter_count < 2_000


def test_stochastic_policy_remains_masked() -> None:
    policy = StructuredLinearPolicy(feature_size=4, max_cards=5, max_targets=3)
    mask = only_discard_mask()

    decisions = [
        policy.act(np.zeros((1, 4)), [mask])[0].action for _ in range(30)
    ]

    assert all(mask.allows(action) for action in decisions)


def test_policy_seed_reproduces_stochastic_action_sequence() -> None:
    first = StructuredLinearPolicy(
        feature_size=4, max_cards=5, max_targets=3, seed=99
    )
    second = StructuredLinearPolicy(
        feature_size=4, max_cards=5, max_targets=3, seed=99
    )
    mask = only_discard_mask()
    features = np.ones((1, 4))

    first_actions = [first.act(features, [mask])[0].action for _ in range(10)]
    second_actions = [second.act(features, [mask])[0].action for _ in range(10)]

    assert first_actions == second_actions
