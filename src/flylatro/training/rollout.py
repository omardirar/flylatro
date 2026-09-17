"""Tensor rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FlatRollout:
    features: torch.Tensor
    masks: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    old_log_probability: torch.Tensor
    old_value: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def select(self, indices: torch.Tensor) -> "FlatRollout":
        return FlatRollout(
            features=self.features[indices],
            masks={key: value[indices] for key, value in self.masks.items()},
            actions={key: value[indices] for key, value in self.actions.items()},
            old_log_probability=self.old_log_probability[indices],
            old_value=self.old_value[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
        )


class RolloutBuffer:
    def __init__(self) -> None:
        self.features: list[torch.Tensor] = []
        self.masks: dict[str, list[torch.Tensor]] = {}
        self.actions: dict[str, list[torch.Tensor]] = {}
        self.log_probabilities: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []

    def add(
        self,
        *,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
        log_probability: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.features.append(features.detach())
        for key, tensor in masks.items():
            self.masks.setdefault(key, []).append(tensor.detach())
        for key, tensor in actions.items():
            self.actions.setdefault(key, []).append(tensor.detach())
        self.log_probabilities.append(log_probability.detach())
        self.values.append(value.detach())
        self.rewards.append(reward.detach())
        self.dones.append(done.detach())

    def finish(
        self,
        last_value: torch.Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> FlatRollout:
        if not self.features:
            raise RuntimeError("cannot finish an empty rollout")
        values = torch.stack(self.values)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(last_value)
        for step in reversed(range(len(self.features))):
            next_value = last_value if step == len(self.features) - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step].to(rewards.dtype)
            delta = rewards[step] + gamma * next_value * nonterminal - values[step]
            last_advantage = (
                delta + gamma * gae_lambda * nonterminal * last_advantage
            )
            advantages[step] = last_advantage
        returns = advantages + values
        return FlatRollout(
            features=_flatten(torch.stack(self.features)),
            masks={
                key: _flatten(torch.stack(tensors))
                for key, tensors in self.masks.items()
            },
            actions={
                key: _flatten(torch.stack(tensors))
                for key, tensors in self.actions.items()
            },
            old_log_probability=_flatten(torch.stack(self.log_probabilities)),
            old_value=_flatten(values),
            advantages=_flatten(advantages),
            returns=_flatten(returns),
        )


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, *tensor.shape[2:])

