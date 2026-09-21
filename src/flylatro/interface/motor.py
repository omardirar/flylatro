"""Deterministic zero-parameter decoder from neural populations to actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
from flylatro.seeds import derive_seed


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
    selection_method: str = "legacy-round-robin"
    calibration_sha256: str | None = None
    calibration_stats: Mapping[str, object] | None = None

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
            "selection_method": self.selection_method,
            "calibration_sha256": self.calibration_sha256,
            "calibration_stats": self.calibration_stats,
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
            "selection_method": self.selection_method,
            "calibration_sha256": self.calibration_sha256,
            "calibration_stats": self.calibration_stats,
            "sha256": self.sha256,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "MotorMapping":
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = cls(
            version=str(payload["version"]),
            mode=str(payload["mode"]),
            output_root_ids=np.asarray(payload["output_root_ids"], dtype=np.int64),
            pools={
                name: tuple(tuple(int(index) for index in pool) for pool in pools)
                for name, pools in payload["pools"].items()
            },
            exploration_epsilon=float(payload["exploration_epsilon"]),
            exploration_temperature=float(payload["exploration_temperature"]),
            exploration_seed=int(payload["exploration_seed"]),
            selection_method=str(payload.get("selection_method", "unknown")),
            calibration_sha256=payload.get("calibration_sha256"),
            calibration_stats=payload.get("calibration_stats"),
        )
        if payload.get("sha256") != mapping.sha256:
            raise ValueError("motor mapping artifact hash mismatch")
        return mapping

    @classmethod
    def from_reward_free_calibration(
        cls,
        output_root_ids: NDArray[np.int64],
        activity_hz: NDArray[np.floating],
        *,
        mode: str,
        pool_width: int = 3,
        high_rate_hz: float = 200.0,
        exploration_epsilon: float = 0.0,
        exploration_temperature: float = 1.0,
        exploration_seed: int = 0,
        calibration_metadata: Mapping[str, object] | None = None,
    ) -> "MotorMapping":
        roots = np.asarray(output_root_ids, dtype=np.int64)
        values = np.asarray(activity_hz, dtype=np.float64)
        required = sum(HEAD_SIZES.values()) * pool_width
        if values.ndim != 2 or values.shape[1] != len(roots):
            raise ValueError("calibration activity must be sample-by-output")
        if pool_width < 2:
            raise ValueError("calibrated motor pools must contain at least two neurons")
        if values.shape[0] < 2 or len(roots) < required or not np.isfinite(values).all():
            raise ValueError("insufficient finite reward-free motor calibration data")
        dynamic = np.ptp(values, axis=0)
        variance = values.var(axis=0)
        active_fraction = np.mean(values > 0, axis=0)
        high_fraction = np.mean(values >= high_rate_hz, axis=0)
        eligible = (active_fraction > 0) & (high_fraction < 1.0)
        score = dynamic + np.sqrt(variance)
        # Stable root-ID tie break makes the artifact reproducible.
        order = np.lexsort((roots, -score))
        order = order[eligible[order]]
        if len(order) < required:
            raise ValueError(
                f"motor calibration found {len(order)} usable outputs; needs {required}"
            )
        selected = order[:required]
        cursor = 0
        heads: dict[str, tuple[tuple[int, ...], ...]] = {}
        for name, size in HEAD_SIZES.items():
            pools = []
            for _ in range(size):
                pools.append(tuple(int(v) for v in selected[cursor:cursor + pool_width]))
                cursor += pool_width
            heads[name] = tuple(pools)
        raw = {
            "sample_count": int(values.shape[0]),
            "high_rate_hz": high_rate_hz,
            "pool_width": pool_width,
            "usable_output_count": int(len(order)),
            "selected_root_ids": roots[selected].tolist(),
            "selected_dynamic_range_hz": dynamic[selected].tolist(),
            "selected_variance_hz2": variance[selected].tolist(),
            "reward_used": False,
            "state_sample_set": dict(calibration_metadata or {}),
        }
        calibration_hash = hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            version="reward-free-neural-motor-calibration-v1",
            mode=mode,
            output_root_ids=roots.copy(),
            pools=heads,
            exploration_epsilon=exploration_epsilon,
            exploration_temperature=exploration_temperature,
            exploration_seed=exploration_seed,
            selection_method="reward-free-activity-dynamic-range",
            calibration_sha256=calibration_hash,
            calibration_stats=raw,
        )

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

    def decode(
        self,
        activity: NDArray[np.floating],
        masks: MaskDict,
        *,
        deterministic: bool = True,
        learner_ids: NDArray[np.integer] | tuple[int, ...] | None = None,
        decision_ids: NDArray[np.integer] | tuple[int, ...] | None = None,
    ) -> ActionDict:
        values = np.asarray(activity, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mapping.output_root_ids):
            raise ValueError("motor activity has the wrong shape")
        batch = values.shape[0]
        learner_keys = np.arange(batch) if learner_ids is None else np.asarray(learner_ids)
        decision_keys = np.zeros(batch, dtype=np.int64) if decision_ids is None else np.asarray(decision_ids)
        if learner_keys.shape != (batch,) or decision_keys.shape != (batch,):
            raise ValueError("motor learner and decision IDs must match batch")
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
            rng = np.random.default_rng(
                derive_seed(
                    "motor-exploration",
                    self.mapping.exploration_seed,
                    int(learner_keys[row]),
                    int(decision_keys[row]),
                )
            )
            action_type = self._choose(
                scores["action_type"][row],
                masks["action_type_mask"][row],
                deterministic,
                rng,
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
                    scores["card_count"][row], count_mask, deterministic, rng
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
                    scores["card_count"][row], count_mask, deterministic, rng
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
                        scores[head][row], masks[mask_name][row], deterministic, rng
                    )
                    break
        validate_batch(ACTION_SPEC, actions, batch, "motor actions")
        return actions

    def _choose(
        self,
        scores: NDArray[np.float64],
        legal: NDArray[np.bool_],
        deterministic: bool,
        rng: np.random.Generator,
    ) -> int:
        if not np.any(legal):
            raise ValueError("motor head has no legal output")
        legal_indices = np.flatnonzero(legal)
        legal_scores = scores[legal_indices]
        if deterministic:
            return int(legal_indices[int(np.argmax(legal_scores))])
        if rng.random() < self.mapping.exploration_epsilon:
            return int(rng.choice(legal_indices))
        shifted = legal_scores / self.mapping.exploration_temperature
        shifted -= shifted.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        return int(rng.choice(legal_indices, p=probabilities))

    def rng_state(self) -> dict[str, object]:
        return {"stateless": True, "version": "per-learner-decision-v1"}

    def load_rng_state(self, state: dict[str, object]) -> None:
        if not state:
            raise ValueError("motor RNG checkpoint metadata is missing")
