"""Autodiff-backed, deliberately linear structured actor and critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from flylatro.env.upstream_contract import (
    CONSUMABLE_SLOTS,
    HAND_MAX,
    JOKER_SLOTS,
    MAX_CARD_PICKS,
    N_ACTION_TYPES,
    PACK_SLOTS,
    SHOP_SLOTS,
    ActionDict,
    MaskDict,
    UpstreamActionType,
)


TensorDict = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TorchPolicyOutput:
    actions: TensorDict
    log_probability: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    action_type_logits: torch.Tensor
    action_type_probabilities: torch.Tensor


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        if logits.shape != mask.shape:
            raise ValueError("logits and mask shapes differ")
        if mask.dtype is not torch.bool:
            raise ValueError("categorical mask must be boolean")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every categorical row needs a legal choice")
        self.mask = mask
        self.logits = logits.masked_fill(~mask, -torch.inf)
        self.distribution = Categorical(logits=self.logits)

    @property
    def probabilities(self) -> torch.Tensor:
        return self.distribution.probs

    def sample(self, deterministic: bool) -> torch.Tensor:
        if deterministic:
            return self.logits.argmax(dim=-1)
        return self.distribution.sample()

    def log_prob(self, choice: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(choice)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()


class TorchStructuredPolicy(nn.Module):
    """One-layer structured readout with no conventional hidden network."""

    policy_version = "torch-structured-linear-v1"

    def __init__(self, feature_size: int, *, hidden_size: int = 0) -> None:
        super().__init__()
        if feature_size < 1:
            raise ValueError("feature_size must be positive")
        self.feature_size = feature_size
        if hidden_size < 0:
            raise ValueError("hidden_size cannot be negative")
        self.hidden_size = hidden_size
        if hidden_size:
            self.trunk = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.Tanh())
            head_size = hidden_size
        else:
            self.trunk = nn.Identity()
            head_size = feature_size
        self.action_type_head = nn.Linear(head_size, N_ACTION_TYPES)
        self.card_head = nn.Linear(
            head_size, MAX_CARD_PICKS * (HAND_MAX + 1)
        )
        self.joker_head = nn.Linear(head_size, JOKER_SLOTS)
        self.consumable_head = nn.Linear(head_size, CONSUMABLE_SLOTS)
        self.shop_head = nn.Linear(head_size, SHOP_SLOTS)
        self.pack_head = nn.Linear(head_size, PACK_SLOTS)
        self.value_head = nn.Linear(head_size, 1)
        self._reset_parameters()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        features: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        *,
        actions: Mapping[str, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> TorchPolicyOutput:
        if features.ndim != 2 or features.shape[1] != self.feature_size:
            raise ValueError(
                f"features must have shape [batch, {self.feature_size}]"
            )
        batch_size = features.shape[0]
        _validate_tensor_masks(masks, batch_size)
        summary = self.trunk(features)
        type_logits = self.action_type_head(summary)
        type_distribution = MaskedCategorical(
            type_logits, masks["action_type_mask"]
        )
        if actions is None:
            action_type = type_distribution.sample(deterministic)
        else:
            action_type = actions["action_type"].long()
        log_probability = type_distribution.log_prob(action_type)
        entropy = type_distribution.entropy()
        output_actions: TensorDict = {
            "action_type": action_type,
            "cards": torch.full(
                (batch_size, MAX_CARD_PICKS),
                -1,
                dtype=torch.long,
                device=features.device,
            ),
            "n_cards": torch.zeros(
                batch_size, dtype=torch.long, device=features.device
            ),
            "joker_target": torch.full(
                (batch_size,), -1, dtype=torch.long, device=features.device
            ),
            "consumable_target": torch.full(
                (batch_size,), -1, dtype=torch.long, device=features.device
            ),
            "shop_target": torch.full(
                (batch_size,), -1, dtype=torch.long, device=features.device
            ),
            "pack_target": torch.full(
                (batch_size,), -1, dtype=torch.long, device=features.device
            ),
        }
        card_logits = self.card_head(summary).reshape(
            batch_size, MAX_CARD_PICKS, HAND_MAX + 1
        )
        card_logp, card_entropy, cards, n_cards = self._card_actions(
            card_logits,
            masks["card_select_mask"],
            action_type,
            given=actions,
            deterministic=deterministic,
        )
        output_actions["cards"] = cards
        output_actions["n_cards"] = n_cards
        log_probability = log_probability + card_logp
        entropy = entropy + card_entropy

        target_specs = (
            (
                "joker_target",
                self.joker_head(summary),
                masks["joker_target_mask"],
                action_type == int(UpstreamActionType.SELL_JOKER),
            ),
            (
                "consumable_target",
                self.consumable_head(summary),
                masks["consumable_target_mask"],
                (action_type == int(UpstreamActionType.USE_CONSUMABLE))
                | (action_type == int(UpstreamActionType.SELL_CONSUMABLE)),
            ),
            (
                "shop_target",
                self.shop_head(summary),
                masks["shop_target_mask"],
                action_type == int(UpstreamActionType.BUY_SHOP),
            ),
            (
                "pack_target",
                self.pack_head(summary),
                masks["pack_target_mask"],
                action_type == int(UpstreamActionType.PICK_PACK),
            ),
        )
        for key, logits, mask, required in target_specs:
            choice, head_logp, head_entropy = _conditional_target(
                logits,
                mask,
                required,
                given=None if actions is None else actions[key],
                deterministic=deterministic,
            )
            output_actions[key] = choice
            log_probability = log_probability + head_logp
            entropy = entropy + head_entropy

        if actions is not None:
            for key in output_actions:
                if not torch.equal(output_actions[key], actions[key].long()):
                    raise ValueError(f"given action field {key!r} violates policy contract")
        return TorchPolicyOutput(
            actions=output_actions,
            log_probability=log_probability,
            entropy=entropy,
            value=self.value_head(summary).squeeze(-1),
            action_type_logits=type_logits,
            action_type_probabilities=type_distribution.probabilities,
        )

    def _card_actions(
        self,
        logits: torch.Tensor,
        base_mask: torch.Tensor,
        action_type: torch.Tensor,
        *,
        given: Mapping[str, torch.Tensor] | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = logits.shape[0]
        device = logits.device
        needs_cards = (
            (action_type == int(UpstreamActionType.PLAY_HAND))
            | (action_type == int(UpstreamActionType.DISCARD))
            | (action_type == int(UpstreamActionType.USE_CONSUMABLE))
        )
        min_picks = torch.where(
            (action_type == int(UpstreamActionType.PLAY_HAND))
            | (action_type == int(UpstreamActionType.DISCARD)),
            torch.ones_like(action_type),
            torch.zeros_like(action_type),
        )
        remaining = base_mask.clone()
        active = needs_cards.clone()
        cards = torch.full(
            (batch_size, MAX_CARD_PICKS), -1, dtype=torch.long, device=device
        )
        n_cards = torch.zeros(batch_size, dtype=torch.long, device=device)
        logp = torch.zeros(batch_size, dtype=logits.dtype, device=device)
        entropy = torch.zeros_like(logp)
        stop_index = HAND_MAX
        rows = torch.arange(batch_size, device=device)

        for step in range(MAX_CARD_PICKS):
            step_mask = torch.zeros(
                (batch_size, HAND_MAX + 1), dtype=torch.bool, device=device
            )
            step_mask[:, :HAND_MAX] = remaining
            stop_legal = n_cards >= min_picks
            step_mask[:, stop_index] = stop_legal
            step_mask[~active] = False
            step_mask[~active, stop_index] = True
            no_choice = ~step_mask.any(dim=1)
            step_mask[no_choice, stop_index] = True
            distribution = MaskedCategorical(logits[:, step], step_mask)
            if given is None:
                choice = distribution.sample(deterministic)
            else:
                expected_count = given["n_cards"].long()
                choice = torch.where(
                    step < expected_count,
                    given["cards"][:, step].long(),
                    torch.full_like(expected_count, stop_index),
                )
            chosen_logp = distribution.log_prob(choice)
            logp = logp + torch.where(active, chosen_logp, 0.0)
            entropy = entropy + torch.where(active, distribution.entropy(), 0.0)
            picked = active & (choice != stop_index)
            if bool(picked.any()):
                cards[picked, step] = choice[picked]
                n_cards[picked] += 1
                remaining[rows[picked], choice[picked]] = False
            active = active & (choice != stop_index)

        return logp, entropy, cards, n_cards

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)


def masks_to_torch(masks: MaskDict, device: torch.device | str) -> TensorDict:
    return {
        key: torch.as_tensor(value, dtype=torch.bool, device=device)
        for key, value in masks.items()
    }


def actions_to_numpy(actions: Mapping[str, torch.Tensor]) -> ActionDict:
    return {
        key: value.detach().cpu().numpy().astype(np.int64, copy=False)
        for key, value in actions.items()
    }


def actions_to_torch(
    actions: ActionDict, device: torch.device | str
) -> TensorDict:
    return {
        key: torch.as_tensor(value, dtype=torch.long, device=device)
        for key, value in actions.items()
    }


def _conditional_target(
    logits: torch.Tensor,
    mask: torch.Tensor,
    required: torch.Tensor,
    *,
    given: torch.Tensor | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    effective_mask = mask.clone()
    effective_mask[~required] = False
    effective_mask[~required, 0] = True
    if bool((required & ~mask.any(dim=1)).any()):
        raise ValueError("required target head has no legal choice")
    distribution = MaskedCategorical(logits, effective_mask)
    if given is None:
        sampled = distribution.sample(deterministic)
        choice = torch.where(required, sampled, -1)
    else:
        choice = given.long()
    distribution_choice = torch.where(required, choice, 0)
    logp = torch.where(required, distribution.log_prob(distribution_choice), 0.0)
    entropy = torch.where(required, distribution.entropy(), 0.0)
    return choice, logp, entropy


def _validate_tensor_masks(
    masks: Mapping[str, torch.Tensor], batch_size: int
) -> None:
    expected = {
        "action_type_mask": (batch_size, N_ACTION_TYPES),
        "card_select_mask": (batch_size, HAND_MAX),
        "joker_target_mask": (batch_size, JOKER_SLOTS),
        "consumable_target_mask": (batch_size, CONSUMABLE_SLOTS),
        "shop_target_mask": (batch_size, SHOP_SLOTS),
        "pack_target_mask": (batch_size, PACK_SLOTS),
    }
    if set(masks) != set(expected):
        raise ValueError("policy mask keys do not match upstream contract")
    for key, shape in expected.items():
        if masks[key].shape != shape or masks[key].dtype is not torch.bool:
            raise ValueError(f"invalid tensor mask {key}")
