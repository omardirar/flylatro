"""Stable data contracts at the Balatro boundary.

These types intentionally contain no upstream-simulator objects.  A future
adapter can translate an upstream observation and action representation here
without leaking that dependency through the rest of Flylatro.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Suit(str, Enum):
    SPADES = "spades"
    HEARTS = "hearts"
    CLUBS = "clubs"
    DIAMONDS = "diamonds"


class GamePhase(str, Enum):
    SELECTING_HAND = "selecting_hand"
    SHOP = "shop"
    TERMINAL = "terminal"


class ActionType(str, Enum):
    PLAY_HAND = "play_hand"
    DISCARD = "discard"
    SELECT_BLIND = "select_blind"
    SKIP_BLIND = "skip_blind"
    CASH_OUT = "cash_out"
    BUY = "buy"
    REROLL = "reroll"
    END_SHOP = "end_shop"
    USE_CONSUMABLE = "use_consumable"
    SELL_JOKER = "sell_joker"
    SELL_CONSUMABLE = "sell_consumable"
    PICK_PACK = "pick_pack"
    SKIP_PACK = "skip_pack"


ACTION_TYPES: tuple[ActionType, ...] = tuple(ActionType)
ACTION_INDEX = {action_type: index for index, action_type in enumerate(ACTION_TYPES)}


@dataclass(frozen=True, slots=True)
class CardObservation:
    """Simulator-neutral card information needed by the first encoder."""

    rank: int
    suit: Suit

    def __post_init__(self) -> None:
        if not 2 <= self.rank <= 14:
            raise ValueError(f"card rank must be in [2, 14], got {self.rank}")

    def to_payload(self) -> dict[str, Any]:
        return {"rank": self.rank, "suit": self.suit.value}


@dataclass(frozen=True, slots=True)
class BalatroObservation:
    """A compact observation contract, extendable as the real adapter matures."""

    episode_id: str
    seed: int
    decision_id: int
    ante: int
    round: int
    phase: GamePhase
    hand: tuple[CardObservation, ...]
    money: int
    score: float
    blind_target: float
    hands_remaining: int
    discards_remaining: int
    shop_slots: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "decision_id": self.decision_id,
            "ante": self.ante,
            "round": self.round,
            "phase": self.phase.value,
            "hand": [card.to_payload() for card in self.hand],
            "money": self.money,
            "score": self.score,
            "blind_target": self.blind_target,
            "hands_remaining": self.hands_remaining,
            "discards_remaining": self.discards_remaining,
            "shop_slots": self.shop_slots,
        }

    def state_hash(self) -> str:
        payload = self.to_payload()
        # Episode IDs are recorder metadata, not simulator state.  Omitting
        # them lets a trace replay in a different vector slot without a false
        # divergence.
        payload.pop("episode_id")
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CompositeAction:
    """A complete action, including all card or target selections."""

    action_type: ActionType
    cards: tuple[int, ...] = ()
    target: int | None = None
    joker_target: int | None = None
    consumable_target: int | None = None
    shop_target: int | None = None
    pack_target: int | None = None

    def __post_init__(self) -> None:
        if len(set(self.cards)) != len(self.cards):
            raise ValueError("card selections must be unique")
        if any(card < 0 for card in self.cards):
            raise ValueError("card indices must be non-negative")
        targets = (
            self.target,
            self.joker_target,
            self.consumable_target,
            self.shop_target,
            self.pack_target,
        )
        if any(target is not None and target < 0 for target in targets):
            raise ValueError("targets must be non-negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.action_type.value,
            "cards": list(self.cards),
            "target": self.target,
            "joker_target": self.joker_target,
            "consumable_target": self.consumable_target,
            "shop_target": self.shop_target,
            "pack_target": self.pack_target,
        }


BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ActionMask:
    """Conditional masks for structured Balatro actions.

    Arrays are indexed first by ``ACTION_TYPES``.  Card/count/target choices
    are decoded only after the action type has been selected, so the fly is
    evaluated once per state rather than once per candidate action.
    """

    action_types: BoolArray
    cards: BoolArray
    min_cards: IntArray
    max_cards: IntArray
    targets: BoolArray
    requires_target: BoolArray

    def __post_init__(self) -> None:
        action_count = len(ACTION_TYPES)
        if self.action_types.shape != (action_count,):
            raise ValueError("action_types has the wrong shape")
        if self.cards.ndim != 2 or self.cards.shape[0] != action_count:
            raise ValueError("cards must have shape [action_types, card_slots]")
        if self.targets.ndim != 2 or self.targets.shape[0] != action_count:
            raise ValueError("targets must have shape [action_types, target_slots]")
        if self.min_cards.shape != (action_count,) or self.max_cards.shape != (
            action_count,
        ):
            raise ValueError("card bounds have the wrong shape")
        if self.requires_target.shape != (action_count,):
            raise ValueError("requires_target has the wrong shape")
        if not np.any(self.action_types):
            raise ValueError("at least one action type must be legal")
        for index, enabled in enumerate(self.action_types):
            if not enabled:
                continue
            available = int(np.count_nonzero(self.cards[index]))
            minimum = int(self.min_cards[index])
            maximum = int(self.max_cards[index])
            if minimum < 0 or maximum < minimum or maximum > available:
                raise ValueError(f"invalid card bounds for {ACTION_TYPES[index].value}")
            if maximum > 0 and minimum == 0:
                raise ValueError(
                    f"{ACTION_TYPES[index].value} cannot have optional card selection"
                )
            if self.requires_target[index] and not np.any(self.targets[index]):
                raise ValueError(
                    f"{ACTION_TYPES[index].value} requires an available target"
                )

    @property
    def max_card_slots(self) -> int:
        return self.cards.shape[1]

    @property
    def max_target_slots(self) -> int:
        return self.targets.shape[1]

    def allows(self, action: CompositeAction) -> bool:
        index = ACTION_INDEX[action.action_type]
        if not bool(self.action_types[index]):
            return False
        count = len(action.cards)
        if not int(self.min_cards[index]) <= count <= int(self.max_cards[index]):
            return False
        if any(card >= self.max_card_slots or not self.cards[index, card] for card in action.cards):
            return False
        if self.requires_target[index]:
            return (
                action.target is not None
                and action.target < self.max_target_slots
                and bool(self.targets[index, action.target])
            )
        return action.target is None
