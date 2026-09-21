"""Field-aware, zero-parameter Balatro features for plastic-brain experiments.

The pinned upstream environment already normalizes several fields.  This
module deliberately describes every source position before transforming it so
categorical indicators are never accidentally divided by a blanket scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import (
    CONSUMABLE_SLOTS,
    CONSUMABLE_VOCAB,
    HAND_MAX,
    JOKER_SLOTS,
    JOKER_VOCAB,
    SHOP_SLOTS,
    SHOP_VOCAB,
    ObsDict,
)


ENCODER_VERSION = "plastic-balatro-field-aware-v2"


@dataclass(frozen=True, slots=True)
class FeatureChannel:
    name: str
    source_field: str
    source_index: tuple[int, ...]
    semantic_class: str
    transform: str
    rail: str = "single"


def feature_channels() -> tuple[FeatureChannel, ...]:
    channels: list[FeatureChannel] = []

    def add(field: str, indices: list[tuple[int, ...]], cls: str, transform: str) -> None:
        for index in indices:
            suffix = ":".join(map(str, index))
            channels.append(FeatureChannel(f"{field}:{suffix}", field, index, cls, transform))

    def rails(field: str, indices: list[tuple[int, ...]], cls: str) -> None:
        for index in indices:
            suffix = ":".join(map(str, index))
            for rail in ("positive", "negative"):
                channels.append(
                    FeatureChannel(
                        f"{field}:{suffix}:{rail}", field, index, cls,
                        "signed_log_saturating", rail,
                    )
                )

    add("hand", [(s, i) for s in range(HAND_MAX) for i in range(38)], "categorical_or_boolean", "identity_binary")
    add("hand_len", [()], "bounded_count", "linear_divide_10")
    for field, slots, vocabulary in (
        ("joker_ids", JOKER_SLOTS, JOKER_VOCAB),
        ("consumable_ids", CONSUMABLE_SLOTS, CONSUMABLE_VOCAB),
        ("shop_ids", SHOP_SLOTS, SHOP_VOCAB),
    ):
        for slot in range(slots):
            for category in range(vocabulary):
                channels.append(
                    FeatureChannel(
                        f"{field}:{slot}:category:{category}", field,
                        (slot, category), "categorical_id", "one_hot",
                    )
                )
    add("joker_feats", [(s, i) for s in range(JOKER_SLOTS) for i in range(5)], "categorical", "identity_binary")
    add("joker_feats", [(s, 5) for s in range(JOKER_SLOTS)], "upstream_log1p_nonnegative", "log_saturating")
    rails("joker_feats", [(s, i) for s in range(JOKER_SLOTS) for i in range(6, 10)], "upstream_signed_log1p")
    add("joker_feats", [(s, 10) for s in range(JOKER_SLOTS)], "boolean", "identity_binary")
    add("consumables_len", [()], "bounded_count", "linear_divide_3")
    add("shop_feats", [(s, i) for s in range(SHOP_SLOTS) for i in range(7)], "categorical", "identity_binary")
    add("shop_feats", [(s, 7) for s in range(SHOP_SLOTS)], "upstream_log1p_nonnegative", "log_saturating")
    add("shop_feats", [(s, i) for s in range(SHOP_SLOTS) for i in range(8, 13)], "categorical", "identity_binary")
    add("blind", [(i,) for i in range(35)], "categorical", "identity_binary")
    add("blind", [(i,) for i in range(35, 38)], "upstream_log1p_nonnegative", "log_saturating")
    rails("global", [(0,)], "upstream_signed_log1p")
    add("global", [(1,), (2,)], "upstream_normalized_ratio", "identity_bounded")
    add("global", [(i,) for i in range(3, 12)], "categorical", "identity_binary")
    add("global", [(12,)], "upstream_normalized_ratio", "identity_bounded")
    add("global", [(i,) for i in range(13, 19)], "categorical", "identity_binary")
    add("global", [(i,) for i in range(19, 43)], "upstream_log1p_nonnegative", "log_saturating")
    add("global", [(43,)], "upstream_log1p_nonnegative", "log_saturating")
    add("global", [(44,), (45,)], "upstream_normalized_ratio", "identity_bounded")
    add("global", [(46,)], "upstream_log1p_nonnegative", "log_saturating")
    add("global", [(47,)], "upstream_normalized_ratio", "identity_bounded")
    add("global", [(i,) for i in range(48, 64)], "reserved_zero", "identity_zero")
    add("deck_counts", [(i,) for i in range(52)], "raw_count", "log1p_divide_log1p_8")
    add("deck_aggregates", [(i,) for i in range(18)], "upstream_log1p_nonnegative", "log_saturating")
    add("drawpile_counts", [(i,) for i in range(52)], "raw_count", "log1p_divide_log1p_8")
    return tuple(channels)


PLASTIC_FEATURE_CHANNELS = feature_channels()


def feature_names() -> tuple[str, ...]:
    return tuple(channel.name for channel in PLASTIC_FEATURE_CHANNELS)


def channel_manifest() -> dict[str, object]:
    payload = [asdict(channel) for channel in PLASTIC_FEATURE_CHANNELS]
    return {
        "version": ENCODER_VERSION,
        "channel_count": len(payload),
        "channels": payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def observation_features(observations: ObsDict) -> NDArray[np.float32]:
    batch = next(iter(observations.values())).shape[0]
    columns: list[NDArray[np.float32]] = []
    for channel in PLASTIC_FEATURE_CHANNELS:
        values = observations[channel.source_field]
        if channel.transform == "one_hot":
            slot, category = channel.source_index
            column = (values[:, slot] == category).astype(np.float32)
        else:
            column = np.asarray(values[(slice(None), *channel.source_index)], dtype=np.float32)
            if channel.transform == "identity_binary":
                column = np.clip(column, 0.0, 1.0)
            elif channel.transform == "identity_bounded":
                column = np.clip(column, 0.0, 1.0)
            elif channel.transform == "identity_zero":
                if np.any(column != 0):
                    raise ValueError(f"reserved channel {channel.name} must be zero")
                column = np.zeros_like(column)
            elif channel.transform == "linear_divide_10":
                column = np.clip(column / 10.0, 0.0, 1.0)
            elif channel.transform == "linear_divide_3":
                column = np.clip(column / 3.0, 0.0, 1.0)
            elif channel.transform == "log1p_divide_log1p_8":
                column = np.clip(np.log1p(np.maximum(column, 0.0)) / np.log1p(8.0), 0.0, 1.0)
            elif channel.transform == "log_saturating":
                column = np.maximum(column, 0.0)
                column = column / (1.0 + column)
            elif channel.transform == "signed_log_saturating":
                magnitude = np.abs(column) / (1.0 + np.abs(column))
                column = magnitude * ((column >= 0) if channel.rail == "positive" else (column < 0))
            else:  # pragma: no cover - channel table and executor are kept exhaustive
                raise RuntimeError(f"unknown feature transform {channel.transform}")
        columns.append(column.astype(np.float32, copy=False))
    result = np.stack(columns, axis=1)
    if result.shape != (batch, len(PLASTIC_FEATURE_CHANNELS)) or not np.isfinite(result).all():
        raise ValueError("plastic feature encoder produced invalid output")
    return result
