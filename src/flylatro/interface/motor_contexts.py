"""Where each fixed motor pool is actually interpreted.

A motor pool only means anything in the states where the structured head it
drives is read.  A neuron with a large dynamic range across *all* calibration
states may be constant in the shop, and a pool group chosen on that evidence
would look calibrated while carrying no usable shop signal.

This module derives motor *interpretation contexts* from the environment's own
legality masks alone.  It contains no reward, no outcome, no heuristic, no
strategy and no trainable parameter; it is a fixed, versioned reading of the
pinned action contract:

* ``action_type`` — every state where a supported action type is legal;
* ``card_count`` / ``card_slot`` — states where a card-consuming action type
  (``PLAY_HAND``, ``DISCARD``, ``USE_CONSUMABLE``) is legal;
* ``shop_target`` / ``pack_target`` / ``joker_target`` / ``consumable_target``
  — the four mutually exclusive contexts in which the *shared* contextual slot
  pools are interpreted.

The contextual pools are deliberately reused across those four heads (ADR
0011); that is an engineering interface convention, not a claim about fly
behaviour.  Reuse is only defensible if the shared pools carry usable activity
in *each* context where they are read, which is exactly what these windows let
calibration and diagnostics measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import (
    MASK_SPEC,
    MAX_CARD_PICKS,
    MaskDict,
    UpstreamActionType,
)
from flylatro.interface.motor import HEAD_SIZES, SUPPORTED_ACTION_TYPES


MOTOR_CONTEXT_VERSION = "motor-interpretation-contexts-v1"

#: Action types that consume hand-card selections in the pinned contract.
CARD_CONSUMING_ACTION_TYPES: tuple[int, ...] = (
    int(UpstreamActionType.PLAY_HAND),
    int(UpstreamActionType.DISCARD),
    int(UpstreamActionType.USE_CONSUMABLE),
)


@dataclass(frozen=True, slots=True)
class MotorContextSpec:
    """One context in which a routing group's pools are interpreted."""

    name: str
    group: str
    head: str
    description: str


MOTOR_CONTEXTS: tuple[MotorContextSpec, ...] = (
    MotorContextSpec(
        name="action_type",
        group="action_type",
        head="action_type",
        description="states where at least one represented action type is legal",
    ),
    MotorContextSpec(
        name="card_count",
        group="card_count",
        head="card_count",
        description="states where a card-consuming action type is legal",
    ),
    MotorContextSpec(
        name="card_slot",
        group="card_slot",
        head="card",
        description="states where hand cards can be selected",
    ),
    MotorContextSpec(
        name="shop_target",
        group="context_slot",
        head="shop",
        description="states where BUY_SHOP is legal (shop slot interpretation)",
    ),
    MotorContextSpec(
        name="pack_target",
        group="context_slot",
        head="pack",
        description="states where PICK_PACK is legal (pack slot interpretation)",
    ),
    MotorContextSpec(
        name="joker_target",
        group="context_slot",
        head="joker",
        description="states where SELL_JOKER is legal (joker slot interpretation)",
    ),
    MotorContextSpec(
        name="consumable_target",
        group="context_slot",
        head="consumable",
        description=(
            "states where USE_CONSUMABLE or SELL_CONSUMABLE is legal "
            "(consumable slot interpretation)"
        ),
    ),
)

MOTOR_CONTEXT_NAMES: tuple[str, ...] = tuple(spec.name for spec in MOTOR_CONTEXTS)

#: Contexts whose pools are shared by the fixed contextual routing scheme.
CONTEXTUAL_SLOT_CONTEXTS: tuple[str, ...] = tuple(
    spec.name for spec in MOTOR_CONTEXTS if spec.group == "context_slot"
)


@dataclass(frozen=True, slots=True)
class MotorContextWindow:
    """Which samples exercise a context, and which options compete there."""

    spec: MotorContextSpec
    relevant: NDArray[np.bool_]
    option_legal: NDArray[np.bool_]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def relevant_indices(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.relevant).astype(np.int64)

    @property
    def relevant_states(self) -> int:
        return int(np.count_nonzero(self.relevant))

    @property
    def competing_states(self) -> int:
        """Relevant states offering more than one legal option."""

        counts = self.option_legal.sum(axis=1)
        return int(np.count_nonzero(self.relevant & (counts > 1)))

    @property
    def active_options(self) -> NDArray[np.int64]:
        """Option indices that are legal in at least one relevant state."""

        if not self.relevant.any():
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self.option_legal[self.relevant].any(axis=0)).astype(
            np.int64
        )

    def counts(self) -> dict[str, Any]:
        return {
            "group": self.spec.group,
            "head": self.spec.head,
            "relevant_states": self.relevant_states,
            "states_with_competing_options": self.competing_states,
            "active_option_count": int(self.active_options.size),
            "option_width": int(self.option_legal.shape[1]),
        }


def motor_context_windows(
    masks: Mapping[str, NDArray[Any]],
    *,
    order: Sequence[int] | None = None,
    supported_action_types: Sequence[int] = SUPPORTED_ACTION_TYPES,
) -> dict[str, MotorContextWindow]:
    """Derive every motor interpretation context from legality masks only.

    ``order`` repeats/reorders corpus rows so that a sample-major activity
    matrix (repeats x states) can be aligned with its states' masks.
    """

    missing = set(MASK_SPEC) - set(masks)
    if missing:
        raise ValueError(f"motor contexts need the full mask contract; missing {sorted(missing)}")
    selected = (
        np.arange(int(np.asarray(masks["action_type_mask"]).shape[0]), dtype=np.int64)
        if order is None
        else np.asarray(order, dtype=np.int64)
    )
    action_mask = np.asarray(masks["action_type_mask"], dtype=bool)[selected]
    card_mask = np.asarray(masks["card_select_mask"], dtype=bool)[selected]
    joker_mask = np.asarray(masks["joker_target_mask"], dtype=bool)[selected]
    consumable_mask = np.asarray(masks["consumable_target_mask"], dtype=bool)[selected]
    shop_mask = np.asarray(masks["shop_target_mask"], dtype=bool)[selected]
    pack_mask = np.asarray(masks["pack_target_mask"], dtype=bool)[selected]

    supported = np.zeros(action_mask.shape[1], dtype=bool)
    supported[list(supported_action_types)] = True
    legal_types = action_mask & supported

    def legal(action_type: UpstreamActionType) -> NDArray[np.bool_]:
        return legal_types[:, int(action_type)]

    use_consumable = legal(UpstreamActionType.USE_CONSUMABLE)
    cards_relevant = (
        legal(UpstreamActionType.PLAY_HAND)
        | legal(UpstreamActionType.DISCARD)
        | use_consumable
    )
    # The exact-count head mirrors ``FixedMotorInterface.decode``: count zero is
    # only legal for a consumable that targets no hand card, and the maximum is
    # the number of selectable cards.
    selectable = np.minimum(card_mask.sum(axis=1), MAX_CARD_PICKS)
    counts = np.arange(HEAD_SIZES["card_count"], dtype=np.int64)
    count_legal = counts[None, :] <= selectable[:, None]
    count_legal[:, 0] &= use_consumable

    windows = {
        "action_type": (legal_types.any(axis=1), legal_types),
        "card_count": (cards_relevant & count_legal.any(axis=1), count_legal),
        "card_slot": (cards_relevant & card_mask.any(axis=1), card_mask),
        "shop_target": (legal(UpstreamActionType.BUY_SHOP), shop_mask),
        "pack_target": (legal(UpstreamActionType.PICK_PACK), pack_mask),
        "joker_target": (legal(UpstreamActionType.SELL_JOKER), joker_mask),
        "consumable_target": (
            use_consumable | legal(UpstreamActionType.SELL_CONSUMABLE),
            consumable_mask,
        ),
    }
    return {
        spec.name: MotorContextWindow(
            spec=spec,
            relevant=np.asarray(windows[spec.name][0], dtype=bool),
            option_legal=np.asarray(windows[spec.name][1], dtype=bool),
        )
        for spec in MOTOR_CONTEXTS
    }


def motor_context_counts(masks: MaskDict) -> dict[str, dict[str, Any]]:
    """Compact per-context evidence counts for corpus coverage reporting."""

    windows = motor_context_windows(masks)
    return {name: window.counts() for name, window in windows.items()}


def group_relevant_rows(
    windows: Mapping[str, MotorContextWindow], group: str
) -> NDArray[np.int64]:
    """Rows in which *any* context of a routing group is interpreted."""

    relevant: NDArray[np.bool_] | None = None
    for window in windows.values():
        if window.spec.group != group:
            continue
        relevant = window.relevant if relevant is None else (relevant | window.relevant)
    if relevant is None:
        raise KeyError(f"no motor context belongs to routing group {group!r}")
    return np.flatnonzero(relevant).astype(np.int64)
