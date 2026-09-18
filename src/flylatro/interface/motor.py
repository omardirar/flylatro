"""Deterministic zero-parameter decoder from neural populations to actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import (
    ACTION_SPEC,
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
    empty_action_batch,
    validate_batch,
)


HEAD_SIZES: dict[str, int] = {
    "action_type": N_ACTION_TYPES,
    "card": HAND_MAX,
    # Index is the exact selected-card count. Zero is required for consumables
    # that do not target a hand card; PLAY/DISCARD mask it out.
    "card_count": MAX_CARD_PICKS + 1,
    "joker": JOKER_SLOTS,
    "consumable": CONSUMABLE_SLOTS,
    "shop": SHOP_SLOTS,
    "pack": PACK_SLOTS,
}


@dataclass(frozen=True, slots=True)
class MotorMapping:
    version: str
    mode: str
    output_root_ids: NDArray[np.int64]
    pools: Mapping[str, tuple[tuple[int, ...], ...]]
    exploration_epsilon: float = 0.0
    exploration_temperature: float = 1.0
    exploration_seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"mbon_direct", "whole_brain"}:
            raise ValueError("mode must be mbon_direct or whole_brain")
        if not 0 <= self.exploration_epsilon <= 1:
            raise ValueError("exploration_epsilon must be in [0, 1]")
        if self.exploration_temperature <= 0:
            raise ValueError("exploration_temperature must be positive")
        if set(self.pools) != set(HEAD_SIZES):
            raise ValueError("motor mapping heads are incomplete")
        for name, size in HEAD_SIZES.items():
            pools = self.pools[name]
            if len(pools) != size or any(not pool for pool in pools):
                raise ValueError(f"motor head {name} has invalid pools")
            if any(
                index < 0 or index >= len(self.output_root_ids)
                for pool in pools
                for index in pool
            ):
                raise ValueError(f"motor head {name} references an invalid output")

    @property
    def sha256(self) -> str:
        payload = {
            "version": self.version,
            "mode": self.mode,
            "pools": {name: [list(pool) for pool in value] for name, value in self.pools.items()},
            "exploration_epsilon": self.exploration_epsilon,
            "exploration_temperature": self.exploration_temperature,
            "exploration_seed": self.exploration_seed,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(self.output_root_ids.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    def to_manifest(self) -> dict[str, object]:
        """Return the exact fixed output populations and exploration rule."""

        return {
            "version": self.version,
            "mode": self.mode,
            "output_root_ids": self.output_root_ids.tolist(),
            "pools": {
                name: [list(pool) for pool in values]
                for name, values in self.pools.items()
            },
            "exploration_epsilon": self.exploration_epsilon,
            "exploration_temperature": self.exploration_temperature,
            "exploration_seed": self.exploration_seed,
            "sha256": self.sha256,
        }

    @classmethod
    def round_robin(
        cls,
        output_root_ids: NDArray[np.int64],
        *,
        mode: str,
        pool_width: int = 1,
        exploration_epsilon: float = 0.0,
        exploration_temperature: float = 1.0,
        exploration_seed: int = 0,
    ) -> "MotorMapping":
        roots = np.asarray(output_root_ids, dtype=np.int64)
        required = sum(HEAD_SIZES.values()) * pool_width
        if pool_width < 1 or len(roots) < required:
            raise ValueError(
                f"motor mapping requires at least {required} output neurons"
            )
        cursor = 0
        heads: dict[str, tuple[tuple[int, ...], ...]] = {}
        for name, size in HEAD_SIZES.items():
            head_pools = []
            for _ in range(size):
                head_pools.append(tuple(range(cursor, cursor + pool_width)))
                cursor += pool_width
            heads[name] = tuple(head_pools)
        return cls(
            version="fixed-structured-population-motor-v1",
            mode=mode,
            output_root_ids=roots.copy(),
            pools=heads,
            exploration_epsilon=exploration_epsilon,
            exploration_temperature=exploration_temperature,
            exploration_seed=exploration_seed,
        )


class FixedMotorInterface:
    trainable_parameter_count = 0

    def __init__(self, mapping: MotorMapping) -> None:
        self.mapping = mapping
        self._rng = np.random.default_rng(mapping.exploration_seed)

    def decode(
        self,
        activity: NDArray[np.floating],
        masks: MaskDict,
        *,
        deterministic: bool = True,
    ) -> ActionDict:
        values = np.asarray(activity, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mapping.output_root_ids):
            raise ValueError("motor activity has the wrong shape")
        batch = values.shape[0]
        validate_batch(
            {
                key: value
                for key, value in {
                    "action_type_mask": ((N_ACTION_TYPES,), np.dtype(np.bool_)),
                    "card_select_mask": ((HAND_MAX,), np.dtype(np.bool_)),
                    "joker_target_mask": ((JOKER_SLOTS,), np.dtype(np.bool_)),
                    "consumable_target_mask": ((CONSUMABLE_SLOTS,), np.dtype(np.bool_)),
                    "shop_target_mask": ((SHOP_SLOTS,), np.dtype(np.bool_)),
                    "pack_target_mask": ((PACK_SLOTS,), np.dtype(np.bool_)),
                }.items()
            },
            masks,
            batch,
            "motor masks",
        )
        scores = {
            name: np.stack(
                [values[:, pool].mean(axis=1) for pool in pools], axis=1
            )
            for name, pools in self.mapping.pools.items()
        }
        actions = empty_action_batch(batch)
        for row in range(batch):
            action_type = self._choose(
                scores["action_type"][row],
                masks["action_type_mask"][row],
                deterministic,
            )
            actions["action_type"][row] = action_type
            if action_type in (
                int(UpstreamActionType.PLAY_HAND),
                int(UpstreamActionType.DISCARD),
            ):
                available = masks["card_select_mask"][row]
                maximum = min(MAX_CARD_PICKS, int(available.sum()))
                count_mask = np.arange(MAX_CARD_PICKS + 1) <= maximum
                count_mask[0] = False
                count = self._choose(
                    scores["card_count"][row], count_mask, deterministic
                )
                card_scores = scores["card"][row].copy()
                card_scores[~available] = -np.inf
                chosen = np.argsort(-card_scores, kind="stable")[:count]
                actions["n_cards"][row] = count
                actions["cards"][row, :count] = chosen
            elif action_type == int(UpstreamActionType.USE_CONSUMABLE):
                available = masks["card_select_mask"][row]
                maximum = min(MAX_CARD_PICKS, int(available.sum()))
                count_mask = np.arange(MAX_CARD_PICKS + 1) <= maximum
                count = self._choose(
                    scores["card_count"][row], count_mask, deterministic
                )
                card_scores = scores["card"][row].copy()
                card_scores[~available] = -np.inf
                chosen = np.argsort(-card_scores, kind="stable")[:count]
                actions["n_cards"][row] = count
                actions["cards"][row, :count] = chosen
            target_specs = (
                (
                    int(UpstreamActionType.SELL_JOKER),
                    "joker_target",
                    "joker",
                    "joker_target_mask",
                ),
                (
                    int(UpstreamActionType.USE_CONSUMABLE),
                    "consumable_target",
                    "consumable",
                    "consumable_target_mask",
                ),
                (
                    int(UpstreamActionType.SELL_CONSUMABLE),
                    "consumable_target",
                    "consumable",
                    "consumable_target_mask",
                ),
                (
                    int(UpstreamActionType.BUY_SHOP),
                    "shop_target",
                    "shop",
                    "shop_target_mask",
                ),
                (
                    int(UpstreamActionType.PICK_PACK),
                    "pack_target",
                    "pack",
                    "pack_target_mask",
                ),
            )
            for expected_type, field, head, mask_name in target_specs:
                if action_type == expected_type:
                    actions[field][row] = self._choose(
                        scores[head][row], masks[mask_name][row], deterministic
                    )
                    break
        validate_batch(ACTION_SPEC, actions, batch, "motor actions")
        return actions

    def _choose(
        self,
        scores: NDArray[np.float64],
        legal: NDArray[np.bool_],
        deterministic: bool,
    ) -> int:
        if not np.any(legal):
            raise ValueError("motor head has no legal output")
        legal_indices = np.flatnonzero(legal)
        legal_scores = scores[legal_indices]
        if deterministic:
            return int(legal_indices[int(np.argmax(legal_scores))])
        if self._rng.random() < self.mapping.exploration_epsilon:
            return int(self._rng.choice(legal_indices))
        shifted = legal_scores / self.mapping.exploration_temperature
        shifted -= shifted.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        return int(self._rng.choice(legal_indices, p=probabilities))

    def rng_state(self) -> dict[str, object]:
        return self._rng.bit_generator.state

    def load_rng_state(self, state: dict[str, object]) -> None:
        self._rng.bit_generator.state = state
