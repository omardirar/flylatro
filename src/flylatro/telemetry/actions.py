"""Describe submitted actions from observable pre-step tokens, for display only.

Offsets and item vocabulary follow balatroagent@38ae214's consts.rs/encode.rs.
No simulator snapshots, hidden card identities or strategy features are read.
"""
from __future__ import annotations

import numpy as np

from flylatro.env.upstream_contract import UpstreamActionType
from flylatro.telemetry.item_names import CONSUMABLE_NAMES, JOKER_NAMES, PACK_NAMES, VOUCHER_NAMES


_RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace")
_SUITS = ("S", "H", "C", "D")


def _card_label(observations: dict, row: int, slot: int) -> str:
    hand = observations.get("hand")
    if hand is None or slot < 0 or slot >= hand.shape[1]:
        return f"card slot {slot + 1} (unavailable)"
    card = hand[row, slot]
    if card[37] > 0:
        return f"face-down card [{slot + 1}]"
    if card[23] > 0:  # Stone: enhancement 6; the rank/suit block is zeroed.
        return "Stone card"
    rank = _RANKS[int(np.argmax(card[:13]))] if np.any(card[:13]) else "unknown rank"
    suit = _SUITS[int(np.argmax(card[13:17]))] if np.any(card[13:17]) else "?"
    return f"{rank} {suit}"


def _consumable_name(index: int) -> str:
    if not 0 <= index < len(CONSUMABLE_NAMES):
        return f"consumable ID {index + 1}"
    kind = "Tarot" if index < 22 else "Planet" if index < 34 else "Spectral"
    return f"{kind}: {CONSUMABLE_NAMES[index]}"


def _item_label(observations: dict, row: int, slot: int, field: str) -> str:
    tokens = observations.get(field)
    if tokens is None or slot < 0 or slot >= tokens.shape[1]:
        return f"{field.removesuffix('_ids')} slot {slot + 1} (unavailable)"
    token = int(tokens[row, slot])
    if field == "joker_ids":
        return f"Joker: {JOKER_NAMES[token - 1]}" if 1 <= token <= len(JOKER_NAMES) else f"joker ID {token}"
    if field == "consumable_ids":
        return _consumable_name(token - 1)
    if 1 <= token <= len(JOKER_NAMES):
        return f"Joker: {JOKER_NAMES[token - 1]}"
    if 160 <= token < 160 + len(CONSUMABLE_NAMES):
        return _consumable_name(token - 160)
    if 220 <= token <= 271:
        rank, suit = divmod(token - 220, 4)
        return f"card: {_RANKS[rank]} {_SUITS[suit]}"
    if 280 <= token < 280 + len(VOUCHER_NAMES):
        return f"Voucher: {VOUCHER_NAMES[token - 280]}"
    if 320 <= token < 320 + len(PACK_NAMES):
        return PACK_NAMES[token - 320]
    return f"item ID {token} (slot {slot + 1})"


def describe_action(actions: dict, observations: dict, row: int) -> str:
    action = int(actions["action_type"][row])
    count = int(actions["n_cards"][row])
    cards = ", ".join(_card_label(observations, row, int(slot))
        for slot in actions["cards"][row, :max(0, count)])
    if action == UpstreamActionType.PLAY_HAND:
        return "Played " + (cards or "selected cards")
    if action == UpstreamActionType.DISCARD:
        return "Discarded " + (cards or "selected cards")
    targets = {
        UpstreamActionType.BUY_SHOP: ("Bought", "shop_ids", "shop_target"),
        UpstreamActionType.PICK_PACK: ("Picked", "shop_ids", "pack_target"),
        UpstreamActionType.SELL_JOKER: ("Sold", "joker_ids", "joker_target"),
        UpstreamActionType.SELL_CONSUMABLE: ("Sold", "consumable_ids", "consumable_target"),
        UpstreamActionType.USE_CONSUMABLE: ("Used", "consumable_ids", "consumable_target"),
    }
    if action in targets:
        verb, field, pointer = targets[action]
        description = verb + " " + _item_label(observations, row, int(actions[pointer][row]), field)
        return description + (" on " + cards if action == UpstreamActionType.USE_CONSUMABLE and cards else "")
    return {
        UpstreamActionType.SELECT_BLIND: "Selected blind",
        UpstreamActionType.SKIP_BLIND: "Skipped blind",
        UpstreamActionType.CASH_OUT: "Cashed out",
        UpstreamActionType.REROLL: "Rerolled shop",
        UpstreamActionType.LEAVE_SHOP: "Left shop",
        UpstreamActionType.SKIP_PACK: "Skipped pack",
    }.get(action, f"Action type {action}")
