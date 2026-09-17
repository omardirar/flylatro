"""A deliberately small, structured trainable readout.

The NumPy implementation supports the M5 integration slice without requiring
a heavyweight ML runtime.  Its parameter arrays form the future actor/critic
boundary; PPO will provide an autodiff-backed implementation of this same
interface rather than adding capacity before the fixed fly processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.env.types import ACTION_TYPES, ActionMask, CompositeAction


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: CompositeAction
    action_probability: float
    value: float
    action_type_probabilities: dict[str, float]


class StructuredLinearPolicy:
    """Direct linear actor heads and critic over fly features."""

    policy_version = "structured-linear-v1"

    def __init__(
        self,
        feature_size: int,
        *,
        max_cards: int = 8,
        max_targets: int = 8,
        seed: int = 2203,
    ) -> None:
        if feature_size < 1 or max_cards < 1 or max_targets < 1:
            raise ValueError("policy dimensions must be positive")
        self.feature_size = feature_size
        self.max_cards = max_cards
        self.max_targets = max_targets
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(feature_size)
        action_count = len(ACTION_TYPES)
        self.action_weights = self._rng.normal(
            0.0, scale, (feature_size, action_count)
        )
        self.action_bias = np.zeros(action_count, dtype=np.float64)
        self.card_weights = self._rng.normal(
            0.0, scale, (feature_size, action_count, max_cards)
        )
        self.card_bias = np.zeros((action_count, max_cards), dtype=np.float64)
        self.count_weights = self._rng.normal(
            0.0, scale, (feature_size, action_count, max_cards)
        )
        self.count_bias = np.zeros((action_count, max_cards), dtype=np.float64)
        self.target_weights = self._rng.normal(
            0.0, scale, (feature_size, action_count, max_targets)
        )
        self.target_bias = np.zeros((action_count, max_targets), dtype=np.float64)
        self.value_weights = self._rng.normal(0.0, scale, feature_size)
        self.value_bias = np.zeros(1, dtype=np.float64)

    @property
    def trainable_parameter_count(self) -> int:
        arrays = (
            self.action_weights,
            self.action_bias,
            self.card_weights,
            self.card_bias,
            self.count_weights,
            self.count_bias,
            self.target_weights,
            self.target_bias,
            self.value_weights,
            self.value_bias,
        )
        return sum(array.size for array in arrays)

    def act(
        self,
        features: FloatArray,
        masks: Sequence[ActionMask],
        *,
        deterministic: bool = False,
    ) -> tuple[PolicyDecision, ...]:
        if features.ndim != 2 or features.shape[1] != self.feature_size:
            raise ValueError(
                f"features must have shape [batch, {self.feature_size}]"
            )
        if len(masks) != features.shape[0]:
            raise ValueError("one action mask is required per feature row")
        action_logits = features @ self.action_weights + self.action_bias
        card_logits = (
            np.einsum("bf,fac->bac", features, self.card_weights) + self.card_bias
        )
        count_logits = (
            np.einsum("bf,fac->bac", features, self.count_weights) + self.count_bias
        )
        target_logits = (
            np.einsum("bf,fat->bat", features, self.target_weights)
            + self.target_bias
        )
        values = features @ self.value_weights + self.value_bias[0]

        decisions = []
        for batch_index, mask in enumerate(masks):
            self._check_mask_capacity(mask)
            action_probs = _masked_softmax(
                action_logits[batch_index], mask.action_types
            )
            action_index = self._select(action_probs, deterministic)
            probability = float(action_probs[action_index])

            minimum = int(mask.min_cards[action_index])
            maximum = int(mask.max_cards[action_index])
            chosen_cards: tuple[int, ...] = ()
            if maximum > 0:
                legal_counts = np.zeros(self.max_cards, dtype=np.bool_)
                legal_counts[minimum - 1 : maximum] = True
                count_probs = _masked_softmax(
                    count_logits[batch_index, action_index], legal_counts
                )
                count_index = self._select(count_probs, deterministic)
                count = count_index + 1
                probability *= float(count_probs[count_index])
                chosen_cards, card_probability = self._select_without_replacement(
                    card_logits[batch_index, action_index, : mask.max_card_slots],
                    mask.cards[action_index],
                    count,
                    deterministic,
                )
                probability *= card_probability

            target: int | None = None
            if mask.requires_target[action_index]:
                target_probs = _masked_softmax(
                    target_logits[
                        batch_index, action_index, : mask.max_target_slots
                    ],
                    mask.targets[action_index],
                )
                target = self._select(target_probs, deterministic)
                probability *= float(target_probs[target])

            decisions.append(
                PolicyDecision(
                    action=CompositeAction(
                        action_type=ACTION_TYPES[action_index],
                        cards=chosen_cards,
                        target=target,
                    ),
                    action_probability=probability,
                    value=float(values[batch_index]),
                    action_type_probabilities={
                        action_type.value: float(action_probs[index])
                        for index, action_type in enumerate(ACTION_TYPES)
                    },
                )
            )
        return tuple(decisions)

    def _select(self, probabilities: FloatArray, deterministic: bool) -> int:
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self._rng.choice(len(probabilities), p=probabilities))

    def _select_without_replacement(
        self,
        logits: FloatArray,
        legal: NDArray[np.bool_],
        count: int,
        deterministic: bool,
    ) -> tuple[tuple[int, ...], float]:
        remaining = legal.copy()
        chosen: list[int] = []
        probability = 1.0
        for _ in range(count):
            probabilities = _masked_softmax(logits, remaining)
            selected = self._select(probabilities, deterministic)
            chosen.append(selected)
            probability *= float(probabilities[selected])
            remaining[selected] = False
        return tuple(chosen), probability

    def _check_mask_capacity(self, mask: ActionMask) -> None:
        if mask.max_card_slots > self.max_cards:
            raise ValueError("action mask exceeds policy card capacity")
        if mask.max_target_slots > self.max_targets:
            raise ValueError("action mask exceeds policy target capacity")


def _masked_softmax(logits: FloatArray, legal: NDArray[np.bool_]) -> FloatArray:
    if logits.shape != legal.shape:
        raise ValueError("logits and legal mask must have matching shapes")
    if not np.any(legal):
        raise ValueError("cannot sample from an empty legal mask")
    result = np.zeros_like(logits, dtype=np.float64)
    legal_logits = logits[legal]
    shifted = legal_logits - np.max(legal_logits)
    exponentials = np.exp(shifted)
    result[legal] = exponentials / exponentials.sum()
    return result

