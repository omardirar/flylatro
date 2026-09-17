from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flylatro.env.upstream_contract import (
    CONSUMABLE_SLOTS,
    HAND_MAX,
    JOKER_SLOTS,
    N_ACTION_TYPES,
    PACK_SLOTS,
    SHOP_SLOTS,
    UpstreamActionType,
)
from flylatro.policy.torch_structured import TorchStructuredPolicy


def masks_for_heads():
    batch = 5
    masks = {
        "action_type_mask": torch.zeros(batch, N_ACTION_TYPES, dtype=torch.bool),
        "card_select_mask": torch.zeros(batch, HAND_MAX, dtype=torch.bool),
        "joker_target_mask": torch.zeros(batch, JOKER_SLOTS, dtype=torch.bool),
        "consumable_target_mask": torch.zeros(
            batch, CONSUMABLE_SLOTS, dtype=torch.bool
        ),
        "shop_target_mask": torch.zeros(batch, SHOP_SLOTS, dtype=torch.bool),
        "pack_target_mask": torch.zeros(batch, PACK_SLOTS, dtype=torch.bool),
    }
    types = (
        UpstreamActionType.PLAY_HAND,
        UpstreamActionType.SELL_JOKER,
        UpstreamActionType.USE_CONSUMABLE,
        UpstreamActionType.BUY_SHOP,
        UpstreamActionType.PICK_PACK,
    )
    for row, action_type in enumerate(types):
        masks["action_type_mask"][row, int(action_type)] = True
    masks["card_select_mask"][0, :3] = True
    masks["card_select_mask"][2, :3] = True
    masks["joker_target_mask"][1, [1, 3]] = True
    masks["consumable_target_mask"][2, [0, 2]] = True
    masks["shop_target_mask"][3, [2, 4]] = True
    masks["pack_target_mask"][4, [1, 3]] = True
    return masks


def test_torch_policy_masks_all_structured_heads_and_recomputes_logprob() -> None:
    torch.manual_seed(7)
    policy = TorchStructuredPolicy(feature_size=16)
    features = torch.randn(5, 16)
    masks = masks_for_heads()

    sampled = policy(features, masks)
    recomputed = policy(features, masks, actions=sampled.actions)

    torch.testing.assert_close(sampled.log_probability, recomputed.log_probability)
    assert sampled.actions["n_cards"][0] >= 1
    assert sampled.actions["joker_target"][1].item() in (1, 3)
    assert sampled.actions["consumable_target"][2].item() in (0, 2)
    assert sampled.actions["shop_target"][3].item() in (2, 4)
    assert sampled.actions["pack_target"][4].item() in (1, 3)
    assert sampled.actions["shop_target"][0].item() == -1
    assert torch.isfinite(sampled.log_probability).all()
    assert torch.isfinite(sampled.value).all()


def test_illegal_action_types_have_exactly_zero_probability() -> None:
    policy = TorchStructuredPolicy(feature_size=8)
    masks = masks_for_heads()

    output = policy(torch.ones(5, 8), masks, deterministic=True)

    probabilities = output.action_type_probabilities.detach().numpy()
    legal = masks["action_type_mask"].numpy()
    assert np.count_nonzero(probabilities[~legal]) == 0
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_policy_and_critic_are_autodiff_backed_and_small() -> None:
    policy = TorchStructuredPolicy(feature_size=32)
    features = torch.randn(5, 32)
    output = policy(features, masks_for_heads())

    loss = -output.log_probability.mean() + output.value.square().mean()
    loss.backward()

    assert policy.action_type_head.weight.grad is not None
    assert policy.value_head.weight.grad is not None
    assert policy.trainable_parameter_count < 10_000

