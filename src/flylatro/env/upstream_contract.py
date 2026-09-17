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
