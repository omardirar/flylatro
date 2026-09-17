from __future__ import annotations

import numpy as np
import pytest

from flylatro.env.upstream_contract import MASK_SPEC, UpstreamActionType
from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.evaluation.baselines import RandomLegalPolicy, evaluate_action_baseline


def masks():
    result = {
        key: np.zeros((3, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    result["action_type_mask"][0, int(UpstreamActionType.PLAY_HAND)] = True
    result["card_select_mask"][0, :4] = True
    result["action_type_mask"][1, int(UpstreamActionType.BUY_SHOP)] = True
    result["shop_target_mask"][1, [2, 5]] = True
    result["action_type_mask"][2, int(UpstreamActionType.USE_CONSUMABLE)] = True
    result["card_select_mask"][2, :3] = True
    result["consumable_target_mask"][2, [0, 2]] = True
    return result


def test_random_legal_policy_fills_only_required_action_fields() -> None:
    policy = RandomLegalPolicy(seed=10)

    output = policy.act(masks())

    assert output.actions["n_cards"][0] >= 1
    assert output.actions["shop_target"][1] in (2, 5)
    assert output.actions["consumable_target"][2] in (0, 2)
    assert output.actions["joker_target"][0] == -1
    assert (output.action_probabilities > 0).all()


def test_random_legal_policy_seed_is_reproducible() -> None:
    first = RandomLegalPolicy(seed=4).act(masks())
    second = RandomLegalPolicy(seed=4).act(masks())

    for key in first.actions:
        np.testing.assert_array_equal(first.actions[key], second.actions[key])
    np.testing.assert_array_equal(first.action_probabilities, second.action_probabilities)


def test_conventional_hidden_policy_remains_modest() -> None:
    torch = pytest.importorskip("torch")
    from flylatro.fly.processors import DirectObservationProcessor
    from flylatro.policy.torch_structured import TorchStructuredPolicy

    processor = DirectObservationProcessor()
    policy = TorchStructuredPolicy(processor.output_size, hidden_size=64)

    assert policy.hidden_size == 64
    assert policy.trainable_parameter_count < 500_000


def test_random_baseline_uses_standard_evaluation_contract() -> None:
    env = MockArrayBalatroEnv(2, blind_target=18, initial_hands=3)
    policy = RandomLegalPolicy(seed=22)

    result = evaluate_action_baseline(
        env, (100, 101), policy.act, max_vector_steps=10
    )

    assert len(result.episodes) == 2
    assert result.decision_count == len(result.transitions)
    assert all(transition.action["type"] for transition in result.transitions)
