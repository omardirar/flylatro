"""Pinned array contract for ``balatroagent`` commit 38ae214.

The constants intentionally live in Flylatro so the rest of the project does
not import the upstream training package.  Contract tests compare these shapes
and dtypes at the adapter boundary.  Reserved action slots remain represented
because checkpoints and rollouts bake their indices in.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Mapping

import numpy as np


UPSTREAM_REPOSITORY = "https://github.com/jahankazimi078/balatroagent"
UPSTREAM_COMMIT = "38ae214317009952db4d22a98dc0765cef79370a"
SIMULATOR_VERSION = f"balatroagent-{UPSTREAM_COMMIT[:12]}"

HAND_MAX = 10
JOKER_SLOTS = 6
CONSUMABLE_SLOTS = 3
SHOP_SLOTS = 6
PACK_SLOTS = 5
MAX_CARD_PICKS = 5
N_ACTION_TYPES = 25

F_CARD = 38
F_JOKER_FEAT = 11
F_SHOP_FEAT = 13
F_BLIND = 38
F_GLOBAL = 64
N_DECK_SLOTS = 52
F_DECK_AGG = 18
JOKER_VOCAB = 160
CONSUMABLE_VOCAB = 60
SHOP_VOCAB = 400

GLOBAL_MONEY = 0
GLOBAL_HANDS_LEFT = 1
GLOBAL_DISCARDS_LEFT = 2
GLOBAL_ANTE_OFF = 3
GLOBAL_ROUND = 12
GLOBAL_PHASE_OFF = 13
N_PHASES = 6
BLIND_REQ_OFF = 35
BLIND_SCORED_OFF = 36


class UpstreamActionType(IntEnum):
    PLAY_HAND = 0
    DISCARD = 1
    SELECT_BLIND = 2
    SKIP_BLIND = 3
    CASH_OUT = 4
    BUY_SHOP = 5
    REROLL = 6
    LEAVE_SHOP = 7
    USE_CONSUMABLE = 8
    SELL_JOKER = 9
    SELL_CONSUMABLE = 10
    PICK_PACK = 11
    SKIP_PACK = 12
    MOVE_JOKER = 13


OBS_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "hand": ((HAND_MAX, F_CARD), np.dtype(np.float32)),
    "hand_len": ((), np.dtype(np.int64)),
    "joker_ids": ((JOKER_SLOTS,), np.dtype(np.int64)),
    "joker_feats": ((JOKER_SLOTS, F_JOKER_FEAT), np.dtype(np.float32)),
    "consumable_ids": ((CONSUMABLE_SLOTS,), np.dtype(np.int64)),
    "consumables_len": ((), np.dtype(np.int64)),
    "shop_ids": ((SHOP_SLOTS,), np.dtype(np.int64)),
    "shop_feats": ((SHOP_SLOTS, F_SHOP_FEAT), np.dtype(np.float32)),
    "blind": ((F_BLIND,), np.dtype(np.float32)),
    "global": ((F_GLOBAL,), np.dtype(np.float32)),
    "deck_counts": ((N_DECK_SLOTS,), np.dtype(np.float32)),
    "deck_aggregates": ((F_DECK_AGG,), np.dtype(np.float32)),
    "drawpile_counts": ((N_DECK_SLOTS,), np.dtype(np.float32)),
}

MASK_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "action_type_mask": ((N_ACTION_TYPES,), np.dtype(np.bool_)),
    "card_select_mask": ((HAND_MAX,), np.dtype(np.bool_)),
    "joker_target_mask": ((JOKER_SLOTS,), np.dtype(np.bool_)),
    "consumable_target_mask": ((CONSUMABLE_SLOTS,), np.dtype(np.bool_)),
    "shop_target_mask": ((SHOP_SLOTS,), np.dtype(np.bool_)),
    "pack_target_mask": ((PACK_SLOTS,), np.dtype(np.bool_)),
}

ACTION_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "action_type": ((), np.dtype(np.int64)),
    "cards": ((MAX_CARD_PICKS,), np.dtype(np.int64)),
    "n_cards": ((), np.dtype(np.int64)),
    "joker_target": ((), np.dtype(np.int64)),
    "consumable_target": ((), np.dtype(np.int64)),
    "shop_target": ((), np.dtype(np.int64)),
    "pack_target": ((), np.dtype(np.int64)),
}

ObsDict = dict[str, np.ndarray]
MaskDict = dict[str, np.ndarray]
ActionDict = dict[str, np.ndarray]


def validate_batch(
    spec: Mapping[str, tuple[tuple[int, ...], np.dtype]],
    batch: Mapping[str, np.ndarray],
    num_envs: int,
    label: str,
) -> None:
    missing = spec.keys() - batch.keys()
    extra = batch.keys() - spec.keys()
    if missing or extra:
        raise ValueError(
            f"{label}: missing keys {sorted(missing)}, extra keys {sorted(extra)}"
        )
    for key, (row_shape, dtype) in spec.items():
        array = batch[key]
        expected_shape = (num_envs, *row_shape)
        if array.shape != expected_shape:
            raise ValueError(
                f"{label}[{key}] shape {array.shape} != {expected_shape}"
            )
        if array.dtype != dtype:
            raise ValueError(f"{label}[{key}] dtype {array.dtype} != {dtype}")


def empty_action_batch(num_envs: int) -> ActionDict:
    batch: ActionDict = {}
    for key, (row_shape, dtype) in ACTION_SPEC.items():
        batch[key] = np.full((num_envs, *row_shape), -1, dtype=dtype)
    return batch


#: Action types the pinned referee lets carry card selections
#: (``sim/py/src/action.rs::needs_cards``).
CARD_CONSUMING_ACTION_TYPES: tuple[int, ...] = (
    int(UpstreamActionType.PLAY_HAND),
    int(UpstreamActionType.DISCARD),
    int(UpstreamActionType.USE_CONSUMABLE),
)

#: Minimum ``n_cards`` per card-consuming type (``card_pick_min``).  A play or
#: discard needs at least one card; a consumable may target none.
CARD_PICK_MINIMUM: dict[int, int] = {
    int(UpstreamActionType.PLAY_HAND): 1,
    int(UpstreamActionType.DISCARD): 1,
    int(UpstreamActionType.USE_CONSUMABLE): 0,
}

#: Which pointer field each action type must fill, and which mask legalizes it.
#: Every other pointer must be exactly ``-1``.
ACTION_POINTER_REQUIREMENTS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("joker_target", "joker_target_mask", (int(UpstreamActionType.SELL_JOKER),)),
    (
        "consumable_target",
        "consumable_target_mask",
        (
            int(UpstreamActionType.USE_CONSUMABLE),
            int(UpstreamActionType.SELL_CONSUMABLE),
        ),
    ),
    ("shop_target", "shop_target_mask", (int(UpstreamActionType.BUY_SHOP),)),
    ("pack_target", "pack_target_mask", (int(UpstreamActionType.PICK_PACK),)),
)


def validate_strict_action_row(
    actions: ActionDict, masks: MaskDict, row: int
) -> None:
    """Mirror the pinned simulator's strict referee for one batch row.

    This is a faithful local re-implementation of
    ``balatroagent@38ae214 sim/py/src/action.rs::validate``, which
    ``BalatroVecEnv`` runs before every step when ``strict=True``.  Reproducing
    it here lets the fixed motor interface be checked against the real contract
    without the compiled extension:

    * the action type must be legal in the emitted mask;
    * every pointer field must be ``-1`` unless the type needs it, and legal in
      its mask when it does;
    * a type that consumes no cards must carry ``n_cards == 0`` — **not** the
      ``-1`` padding value — and an all-``-1`` ``cards`` row;
    * a card-consuming type needs ``n_cards`` in ``[minimum, MAX_CARD_PICKS]``,
      ``-1`` padding past ``n_cards``, and distinct legal card indices.

    Leniency the simulator applies *after* validation (Psychic short plays,
    Cerulean Bell forced cards, consumable target legalization and the
    documented unusable-consumable no-op) is deliberately not reproduced: it
    never rejects an action, so it cannot turn a valid action into an error.
    """

    action_type = int(actions["action_type"][row])
    if not 0 <= action_type < N_ACTION_TYPES:
        raise ValueError(f"row {row}: illegal action_type {action_type}")
    if not bool(masks["action_type_mask"][row, action_type]):
        raise ValueError(f"row {row}: illegal action_type {action_type}")
    for field, mask_name, needed_by in ACTION_POINTER_REQUIREMENTS:
        value = int(actions[field][row])
        if action_type not in needed_by:
            if value != -1:
                raise ValueError(
                    f"row {row}: {field}={value} must be -1 for type {action_type}"
                )
            continue
        mask = masks[mask_name][row]
        if value < 0 or value >= len(mask) or not bool(mask[value]):
            raise ValueError(
                f"row {row}: illegal {field}={value} for type {action_type}"
            )
    count = int(actions["n_cards"][row])
    cards = [int(value) for value in actions["cards"][row]]
    if action_type not in CARD_CONSUMING_ACTION_TYPES:
        if count != 0 or any(value != -1 for value in cards):
            raise ValueError(
                f"row {row}: card params must be empty for type {action_type} "
                f"(n_cards={count}, cards={cards})"
            )
        return
    minimum = CARD_PICK_MINIMUM[action_type]
    if count < minimum or count > MAX_CARD_PICKS:
        raise ValueError(
            f"row {row}: n_cards={count} out of [{minimum},{MAX_CARD_PICKS}] "
            f"for type {action_type}"
        )
    if any(value != -1 for value in cards[count:]):
        raise ValueError(f"row {row}: cards not -1-padded past n_cards")
    select = masks["card_select_mask"][row]
    for position, value in enumerate(cards[:count]):
        if value < 0 or value >= len(select) or not bool(select[value]):
            raise ValueError(f"row {row}: illegal card pick {value}")
        if value in cards[:position]:
            raise ValueError(f"row {row}: duplicate card pick {value}")


def validate_strict_action_batch(actions: ActionDict, masks: MaskDict) -> None:
    """Apply the pinned strict referee to every row of a batch."""

    batch = int(actions["action_type"].shape[0])
    for row in range(batch):
        validate_strict_action_row(actions, masks, row)
