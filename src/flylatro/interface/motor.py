"""Deterministic zero-parameter decoder from neural populations to actions.

Three properties matter scientifically and are enforced here:

* the interface contains no trainable parameter and no fitted state that was
  chosen using reward, win rate or action correctness;
* the neural populations that drive mutually exclusive contexts may be reused,
  because reusing an output across contexts is a fixed interpretation rule, not
  a learned one, while options that compete inside one comparison must be
  driven by distinct populations;
* permanently reserved action slots of the pinned simulator contract consume no
  biological output at all and can never be selected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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

#: Action types this V1 motor interface can actually express with the pinned
#: `ACTION_SPEC`.  `MOVE_JOKER` (13) needs a source *and* destination slot that
#: the pinned action contract cannot carry, and 14-24 are reserved index-
#: stability placeholders with no upstream semantics at all.  They stay in the
#: contract width so mask/action indices never shift, but they receive no
#: neural population and can never be chosen.
SUPPORTED_ACTION_TYPES: tuple[int, ...] = tuple(
    int(value)
    for value in (
        UpstreamActionType.PLAY_HAND,
        UpstreamActionType.DISCARD,
        UpstreamActionType.SELECT_BLIND,
        UpstreamActionType.SKIP_BLIND,
        UpstreamActionType.CASH_OUT,
        UpstreamActionType.BUY_SHOP,
        UpstreamActionType.REROLL,
        UpstreamActionType.LEAVE_SHOP,
        UpstreamActionType.USE_CONSUMABLE,
        UpstreamActionType.SELL_JOKER,
        UpstreamActionType.SELL_CONSUMABLE,
        UpstreamActionType.PICK_PACK,
        UpstreamActionType.SKIP_PACK,
    )
)
RESERVED_ACTION_TYPES: tuple[int, ...] = tuple(
    value for value in range(N_ACTION_TYPES) if value not in SUPPORTED_ACTION_TYPES
)

#: Heads whose options are read at most once per decision and never at the same
#: time as each other.  They therefore share one contextual slot population:
#: "slot 2" means shop slot 2, pack slot 2, joker slot 2 or consumable slot 2
#: depending on the already-selected action type.
CONTEXTUAL_SLOT_HEADS: tuple[str, ...] = ("joker", "consumable", "shop", "pack")

MOTOR_ROUTING_VERSION = "contextual-motor-routing-v1"


@dataclass(frozen=True, slots=True)
class MotorRouting:
    """Fixed, zero-parameter map from structured action options to pool IDs.

    ``head_routes[head][option]`` is the index of the neural pool that scores
    that option, or ``-1`` when the option is a reserved contract slot with no
    biological representation.
    """

    version: str
    group_names: tuple[str, ...]
    group_pool_counts: tuple[int, ...]
    head_routes: Mapping[str, tuple[int, ...]]
    supported_action_types: tuple[int, ...] = SUPPORTED_ACTION_TYPES
    reserved_action_types: tuple[int, ...] = RESERVED_ACTION_TYPES

    def __post_init__(self) -> None:
        if len(self.group_names) != len(self.group_pool_counts):
            raise ValueError("routing groups and pool counts must align")
        if set(self.head_routes) != set(HEAD_SIZES):
            raise ValueError("routing must cover every structured action head")
        for head, size in HEAD_SIZES.items():
            route = self.head_routes[head]
            if len(route) != size:
                raise ValueError(f"routing head {head} must cover {size} options")
            if any(value < -1 or value >= self.pool_count for value in route):
                raise ValueError(f"routing head {head} references an unknown pool")
        for head, route in self.head_routes.items():
            if head == "action_type":
                continue
            if any(value < 0 for value in route):
                raise ValueError(f"head {head} must represent every contract option")
        action_route = self.head_routes["action_type"]
        for value in self.reserved_action_types:
            if action_route[value] != -1:
                raise ValueError("reserved action slots must not own a neural pool")
        for value in self.supported_action_types:
            if action_route[value] < 0:
                raise ValueError("supported action types require a neural pool")
        for head in ("action_type", "card_count", "card"):
            represented = [value for value in self.head_routes[head] if value >= 0]
            if len(set(represented)) != len(represented):
                raise ValueError(
                    f"head {head} competes simultaneously and needs distinct pools"
                )

    @property
    def pool_count(self) -> int:
        return int(sum(self.group_pool_counts))

    @property
    def pool_groups(self) -> tuple[str, ...]:
        groups: list[str] = []
        for name, count in zip(self.group_names, self.group_pool_counts, strict=True):
            groups.extend([name] * count)
        return tuple(groups)

    def group_pool_ids(self, group: str) -> tuple[int, ...]:
        return tuple(
            index for index, name in enumerate(self.pool_groups) if name == group
        )

    def represented_options(self, head: str) -> tuple[int, ...]:
        return tuple(
            index for index, pool in enumerate(self.head_routes[head]) if pool >= 0
        )

    def reuse_summary(self) -> dict[str, Any]:
        """Describe exactly which heads share a pool and why that is legal."""

        owners: dict[int, list[str]] = {}
        for head, route in self.head_routes.items():
            for option, pool in enumerate(route):
                if pool >= 0:
                    owners.setdefault(pool, []).append(f"{head}:{option}")
        shared = {
            str(pool): sorted(names)
            for pool, names in owners.items()
            if len({name.split(":", 1)[0] for name in names}) > 1
        }
        return {
            "pool_count": self.pool_count,
            "groups": dict(zip(self.group_names, self.group_pool_counts, strict=True)),
            "shared_pools": shared,
            "contextual_slot_heads": list(CONTEXTUAL_SLOT_HEADS),
            "rule": (
                "pools are distinct inside one simultaneous comparison; they are "
                "reused across heads that the already-selected action type makes "
                "mutually exclusive"
            ),
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "group_names": list(self.group_names),
            "group_pool_counts": list(self.group_pool_counts),
            "head_routes": {
                head: list(route) for head, route in sorted(self.head_routes.items())
            },
            "supported_action_types": list(self.supported_action_types),
            "reserved_action_types": list(self.reserved_action_types),
            "full_action_space_width": N_ACTION_TYPES,
            "scientifically_represented_action_count": len(self.supported_action_types),
            "reuse": self.reuse_summary(),
            "sha256": self.sha256,
        }

    @property
    def sha256(self) -> str:
        payload = {
            "version": self.version,
            "group_names": list(self.group_names),
            "group_pool_counts": list(self.group_pool_counts),
            "head_routes": {
                head: list(route) for head, route in sorted(self.head_routes.items())
            },
            "supported_action_types": list(self.supported_action_types),
            "reserved_action_types": list(self.reserved_action_types),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "MotorRouting":
        routing = cls(
            version=str(payload["version"]),
            group_names=tuple(str(value) for value in payload["group_names"]),
            group_pool_counts=tuple(int(value) for value in payload["group_pool_counts"]),
            head_routes={
                head: tuple(int(value) for value in route)
                for head, route in payload["head_routes"].items()
            },
            supported_action_types=tuple(
                int(value) for value in payload["supported_action_types"]
            ),
            reserved_action_types=tuple(
                int(value) for value in payload["reserved_action_types"]
            ),
        )
        if payload.get("sha256") not in (None, routing.sha256):
            raise ValueError("motor routing hash mismatch")
        return routing

    @classmethod
    def contextual_v1(cls) -> "MotorRouting":
        """The fixed V1 routing scheme.

        Groups, in pool order: supported action types, exact card counts, hand
        card slots, and one shared contextual target slot group.
        """

        groups = (
            ("action_type", len(SUPPORTED_ACTION_TYPES)),
            ("card_count", HEAD_SIZES["card_count"]),
            ("card_slot", HEAD_SIZES["card"]),
            (
                "context_slot",
                max(HEAD_SIZES[head] for head in CONTEXTUAL_SLOT_HEADS),
            ),
        )
        offsets: dict[str, int] = {}
        cursor = 0
        for name, count in groups:
            offsets[name] = cursor
            cursor += count
        action_route = [-1] * N_ACTION_TYPES
        for position, action_type in enumerate(SUPPORTED_ACTION_TYPES):
            action_route[action_type] = offsets["action_type"] + position
        head_routes: dict[str, tuple[int, ...]] = {
            "action_type": tuple(action_route),
            "card_count": tuple(
                offsets["card_count"] + index for index in range(HEAD_SIZES["card_count"])
            ),
            "card": tuple(
                offsets["card_slot"] + index for index in range(HEAD_SIZES["card"])
            ),
        }
        for head in CONTEXTUAL_SLOT_HEADS:
            head_routes[head] = tuple(
                offsets["context_slot"] + index for index in range(HEAD_SIZES[head])
            )
        return cls(
            version=MOTOR_ROUTING_VERSION,
            group_names=tuple(name for name, _ in groups),
            group_pool_counts=tuple(count for _, count in groups),
            head_routes=head_routes,
        )


CONTEXTUAL_ROUTING = MotorRouting.contextual_v1()
MOTOR_POOL_COUNT = CONTEXTUAL_ROUTING.pool_count

MOTOR_NORMALIZATION_VERSION = "reward-free-median-iqr-v1"


@dataclass(frozen=True, slots=True)
class MotorNormalization:
    """Fixed reward-free baseline/scale so competing pools compare fairly.

    ``normalized = (pool_activity - baseline) / scale`` with a robust
    median/IQR estimator.  Both vectors are frozen at calibration time, derive
    from reward-free observable states only, and are hashed into the artifact.
    """

    version: str
    baseline: NDArray[np.float64]
    scale: NDArray[np.float64]
    minimum_scale_hz: float = 0.5
    statistics: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.baseline.ndim != 1 or self.baseline.shape != self.scale.shape:
            raise ValueError("motor normalization vectors must align")
        if not np.isfinite(self.baseline).all() or not np.isfinite(self.scale).all():
            raise ValueError("motor normalization must be finite")
        if self.minimum_scale_hz <= 0:
            raise ValueError("minimum_scale_hz must be positive")
        if np.any(self.scale < self.minimum_scale_hz - 1e-12):
            raise ValueError("motor normalization scale is below the configured floor")

    @classmethod
    def identity(cls, pool_count: int, *, minimum_scale_hz: float = 0.5) -> "MotorNormalization":
        return cls(
            version="identity-uncalibrated-v1",
            baseline=np.zeros(pool_count, dtype=np.float64),
            scale=np.full(pool_count, max(1.0, minimum_scale_hz), dtype=np.float64),
            minimum_scale_hz=minimum_scale_hz,
        )

    @classmethod
    def from_pool_activity(
        cls,
        pool_activity: NDArray[np.floating],
        *,
        minimum_scale_hz: float = 0.5,
    ) -> "MotorNormalization":
        """Robust median/IQR statistics over reward-free calibration states."""

        values = np.asarray(pool_activity, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("normalization needs a sample-by-pool activity matrix")
        median = np.median(values, axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        iqr = q75 - q25
        # 1.349 converts a Gaussian IQR to a standard deviation. The floor keeps
        # an inert pool from being amplified into a dominant score.
        robust = iqr / 1.349
        scale = np.maximum(robust, minimum_scale_hz)
        statistics = tuple(
            {
                "pool": int(index),
                "mean_hz": float(values[:, index].mean()),
                "std_hz": float(values[:, index].std()),
                "median_hz": float(median[index]),
                "iqr_hz": float(iqr[index]),
                "robust_scale_hz": float(robust[index]),
                "applied_scale_hz": float(scale[index]),
                "dynamic_range_hz": float(np.ptp(values[:, index])),
                "min_hz": float(values[:, index].min()),
                "max_hz": float(values[:, index].max()),
                "scale_floored": bool(robust[index] < minimum_scale_hz),
            }
            for index in range(values.shape[1])
        )
        return cls(
            version=MOTOR_NORMALIZATION_VERSION,
            baseline=median,
            scale=scale,
            minimum_scale_hz=float(minimum_scale_hz),
            statistics=statistics,
        )

    def apply(self, pool_activity: NDArray[np.floating]) -> NDArray[np.float64]:
        values = np.asarray(pool_activity, dtype=np.float64)
        if values.shape[-1] != self.baseline.shape[0]:
            raise ValueError("pool activity does not match the normalization width")
        return (values - self.baseline) / self.scale

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "baseline": [float(value) for value in self.baseline],
            "scale": [float(value) for value in self.scale],
            "minimum_scale_hz": self.minimum_scale_hz,
            "statistics": [dict(entry) for entry in self.statistics],
            "rule": "normalized_score = (pool_activity - baseline) / scale",
            "reward_used": False,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "MotorNormalization":
        return cls(
            version=str(payload["version"]),
            baseline=np.asarray(payload["baseline"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            minimum_scale_hz=float(payload.get("minimum_scale_hz", 0.5)),
            statistics=tuple(dict(entry) for entry in payload.get("statistics", ())),
        )

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {"version": self.version, "minimum_scale_hz": self.minimum_scale_hz},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(self.baseline.astype("<f8", copy=False).tobytes())
        digest.update(self.scale.astype("<f8", copy=False).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MotorMapping:
    version: str
    mode: str
    output_root_ids: NDArray[np.int64]
    pool_indices: tuple[tuple[int, ...], ...]
    routing: MotorRouting = CONTEXTUAL_ROUTING
    normalization: MotorNormalization | None = None
    exploration_epsilon: float = 0.0
    exploration_temperature: float = 1.0
    exploration_seed: int = 0
    selection_method: str = "unspecified"
    candidate_set_sha256: str | None = None
    calibration_sha256: str | None = None
    calibration_stats: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"mbon_direct", "whole_brain"}:
            raise ValueError("mode must be mbon_direct or whole_brain")
        if not 0 <= self.exploration_epsilon <= 1:
            raise ValueError("exploration_epsilon must be in [0, 1]")
        if self.exploration_temperature <= 0:
            raise ValueError("exploration_temperature must be positive")
        if len(self.pool_indices) != self.routing.pool_count:
            raise ValueError("one neural pool is required per routed pool ID")
        if any(not pool for pool in self.pool_indices):
            raise ValueError("every motor pool needs at least one neural output")
        limit = len(self.output_root_ids)
        if any(index < 0 or index >= limit for pool in self.pool_indices for index in pool):
            raise ValueError("motor pool references an invalid neural output")
        for group in self.routing.group_names:
            used: list[int] = []
            for pool_id in self.routing.group_pool_ids(group):
                used.extend(self.pool_indices[pool_id])
            if len(set(used)) != len(used):
                raise ValueError(
                    f"competing pools in group {group} must use distinct neurons"
                )
        if self.normalization is not None and (
            self.normalization.baseline.shape[0] != self.routing.pool_count
        ):
            raise ValueError("normalization width does not match the routing")

    @property
    def pool_width(self) -> int:
        return min(len(pool) for pool in self.pool_indices)

    @property
    def pools(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        """Head-aligned neural pools; reserved contract slots yield ``()``."""

        return {
            head: tuple(
                self.pool_indices[pool] if pool >= 0 else ()
                for pool in self.routing.head_routes[head]
            )
            for head in HEAD_SIZES
        }

    def represented_pools(self) -> dict[str, dict[int, tuple[int, ...]]]:
        """Only the options that own a biological population."""

        return {
            head: {
                option: self.pool_indices[pool]
                for option, pool in enumerate(self.routing.head_routes[head])
                if pool >= 0
            }
            for head in HEAD_SIZES
        }

    @property
    def structure_sha256(self) -> str:
        """Identity of the fixed interface itself.

        Covers the candidate universe, pool structure, routing and calibrated
        normalization, and deliberately excludes exploration, which is recorded
        per-learner interface state rather than part of the mapping. Matched
        conditions must share this hash even when their exploration seeds
        differ by replicate.
        """

        payload = {
            "version": self.version,
            "mode": self.mode,
            "pool_indices": [list(pool) for pool in self.pool_indices],
            "routing_sha256": self.routing.sha256,
            "normalization_sha256": (
                None if self.normalization is None else self.normalization.sha256
            ),
            "selection_method": self.selection_method,
            "candidate_set_sha256": self.candidate_set_sha256,
            "calibration_sha256": self.calibration_sha256,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(self.output_root_ids.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    @property
    def sha256(self) -> str:
        """Identity of the serialized artifact, exploration settings included."""

        payload = {
            "structure_sha256": self.structure_sha256,
            "exploration_epsilon": self.exploration_epsilon,
            "exploration_temperature": self.exploration_temperature,
            "exploration_seed": self.exploration_seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def selected_root_ids(self) -> list[int]:
        return [
            int(self.output_root_ids[index])
            for pool in self.pool_indices
            for index in pool
        ]

    def to_manifest(self) -> dict[str, Any]:
        """Return the exact fixed output populations and interpretation rule."""

        return {
            "version": self.version,
            "mode": self.mode,
            "output_root_ids": self.output_root_ids.tolist(),
            "pool_indices": [list(pool) for pool in self.pool_indices],
            "pool_root_ids": [
                [int(self.output_root_ids[index]) for index in pool]
                for pool in self.pool_indices
            ],
            "routing": self.routing.to_manifest(),
            "normalization": (
                None if self.normalization is None else self.normalization.to_manifest()
            ),
            "exploration_epsilon": self.exploration_epsilon,
            "exploration_temperature": self.exploration_temperature,
            "exploration_seed": self.exploration_seed,
            "selection_method": self.selection_method,
            "candidate_set_sha256": self.candidate_set_sha256,
            "calibration_sha256": self.calibration_sha256,
            "calibration_stats": self.calibration_stats,
            "action_space": {
                "full_width": N_ACTION_TYPES,
                "scientifically_represented": list(SUPPORTED_ACTION_TYPES),
                "reserved_unrepresented": list(RESERVED_ACTION_TYPES),
                "note": (
                    "reserved slots keep contract index stability and receive no "
                    "neural population; they can never be selected"
                ),
            },
            "trainable_parameter_count": 0,
            "structure_sha256": self.structure_sha256,
            "sha256": self.sha256,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "MotorMapping":
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = cls.from_manifest(payload)
        if payload.get("sha256") != mapping.sha256:
            raise ValueError("motor mapping artifact hash mismatch")
        return mapping

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "MotorMapping":
        normalization = payload.get("normalization")
        return cls(
            version=str(payload["version"]),
            mode=str(payload["mode"]),
            output_root_ids=np.asarray(payload["output_root_ids"], dtype=np.int64),
            pool_indices=tuple(
                tuple(int(index) for index in pool) for pool in payload["pool_indices"]
            ),
            routing=MotorRouting.from_manifest(payload["routing"]),
            normalization=(
                None if normalization is None else MotorNormalization.from_manifest(normalization)
            ),
            exploration_epsilon=float(payload["exploration_epsilon"]),
            exploration_temperature=float(payload["exploration_temperature"]),
            exploration_seed=int(payload["exploration_seed"]),
            selection_method=str(payload.get("selection_method", "unknown")),
            candidate_set_sha256=payload.get("candidate_set_sha256"),
            calibration_sha256=payload.get("calibration_sha256"),
            calibration_stats=payload.get("calibration_stats"),
        )

    def with_exploration(
        self,
        *,
        epsilon: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> "MotorMapping":
        """Exploration is recorded interface state, never learned state."""

        return replace(
            self,
            exploration_epsilon=(
                self.exploration_epsilon if epsilon is None else float(epsilon)
            ),
            exploration_temperature=(
                self.exploration_temperature if temperature is None else float(temperature)
            ),
            exploration_seed=(
                self.exploration_seed if seed is None else int(seed)
            ),
        )

    @classmethod
    def contiguous_pools(
        cls,
        output_root_ids: NDArray[np.int64],
        *,
        mode: str,
        pool_width: int = 2,
        exploration_epsilon: float = 0.0,
        exploration_temperature: float = 1.0,
        exploration_seed: int = 0,
        routing: MotorRouting = CONTEXTUAL_ROUTING,
    ) -> "MotorMapping":
        """Uncalibrated structural fallback for synthetic development only."""

        roots = np.asarray(output_root_ids, dtype=np.int64)
        required = routing.pool_count * pool_width
        if pool_width < 1 or len(roots) < required:
            raise ValueError(
                f"motor mapping requires at least {required} output neurons"
            )
        pools = tuple(
            tuple(range(index * pool_width, (index + 1) * pool_width))
            for index in range(routing.pool_count)
        )
        return cls(
            version="fixed-structured-population-motor-v2",
            mode=mode,
            output_root_ids=roots.copy(),
            pool_indices=pools,
            routing=routing,
            normalization=MotorNormalization.identity(routing.pool_count),
            exploration_epsilon=exploration_epsilon,
            exploration_temperature=exploration_temperature,
            exploration_seed=exploration_seed,
            selection_method="uncalibrated-contiguous-structural-bootstrap",
        )


class FixedMotorInterface:
    """Zero-parameter decoder; exploration is seeded, recorded interface state."""

    trainable_parameter_count = 0

    def __init__(self, mapping: MotorMapping) -> None:
        self.mapping = mapping
        self.reserved_action_legal_count = 0
        #: Diagnostic capture of the structured choice made per batch row.
        self.record_choices = False
        self.last_choices: tuple[dict[str, Any], ...] = ()
        self._supported_mask = np.zeros(N_ACTION_TYPES, dtype=np.bool_)
        self._supported_mask[list(mapping.routing.supported_action_types)] = True
        self._routes = {
            head: np.asarray(route, dtype=np.int64)
            for head, route in mapping.routing.head_routes.items()
        }
        self._normalization = mapping.normalization or MotorNormalization.identity(
            mapping.routing.pool_count
        )

    def pool_activity(self, activity: NDArray[np.floating]) -> NDArray[np.float64]:
        """Mean raw Hz per motor pool, before calibrated normalization."""

        values = np.asarray(activity, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mapping.output_root_ids):
            raise ValueError("motor activity has the wrong shape")
        return np.stack(
            [values[:, list(pool)].mean(axis=1) for pool in self.mapping.pool_indices],
            axis=1,
        )

    def head_scores(self, activity: NDArray[np.floating]) -> dict[str, NDArray[np.float64]]:
        """Calibrated, normalized score per structured action option."""

        normalized = self._normalization.apply(self.pool_activity(activity))
        scores: dict[str, NDArray[np.float64]] = {}
        for head, route in self._routes.items():
            safe = np.where(route >= 0, route, 0)
            values = normalized[:, safe]
            if np.any(route < 0):
                values = values.copy()
                values[:, route < 0] = -np.inf
            scores[head] = values
        return scores

    def decode(
        self,
        activity: NDArray[np.floating],
        masks: MaskDict,
        *,
        deterministic: bool = True,
        learner_ids: NDArray[np.integer] | Sequence[int] | None = None,
        decision_ids: NDArray[np.integer] | Sequence[int] | None = None,
        record_choices: bool = False,
    ) -> ActionDict:
        values = np.asarray(activity, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mapping.output_root_ids):
            raise ValueError("motor activity has the wrong shape")
        batch = values.shape[0]
        learner_keys = np.arange(batch) if learner_ids is None else np.asarray(learner_ids)
        decision_keys = (
            np.zeros(batch, dtype=np.int64)
            if decision_ids is None
            else np.asarray(decision_ids)
        )
        if learner_keys.shape != (batch,) or decision_keys.shape != (batch,):
            raise ValueError("motor learner and decision IDs must match batch")
        validate_batch(
            {
                "action_type_mask": ((N_ACTION_TYPES,), np.dtype(np.bool_)),
                "card_select_mask": ((HAND_MAX,), np.dtype(np.bool_)),
                "joker_target_mask": ((JOKER_SLOTS,), np.dtype(np.bool_)),
                "consumable_target_mask": ((CONSUMABLE_SLOTS,), np.dtype(np.bool_)),
                "shop_target_mask": ((SHOP_SLOTS,), np.dtype(np.bool_)),
                "pack_target_mask": ((PACK_SLOTS,), np.dtype(np.bool_)),
            },
            masks,
            batch,
            "motor masks",
        )
        scores = self.head_scores(values)
        actions = empty_action_batch(batch)
        # `empty_action_batch` pads every field with -1, but the pinned
        # simulator's strict referee requires `n_cards == 0` (not -1) for every
        # action type that does not consume cards, and rejects the step
        # otherwise.  Card-consuming branches overwrite this below.
        actions["n_cards"][:] = 0
        choices: list[dict[str, int]] = []
        for row in range(batch):
            rng = np.random.default_rng(
                derive_seed(
                    "motor-exploration",
                    self.mapping.exploration_seed,
                    int(learner_keys[row]),
                    int(decision_keys[row]),
                )
            )
            legal_types = np.asarray(masks["action_type_mask"][row], dtype=np.bool_)
            reserved_legal = int(np.count_nonzero(legal_types & ~self._supported_mask))
            self.reserved_action_legal_count += reserved_legal
            representable = legal_types & self._supported_mask
            if not representable.any():
                raise ValueError(
                    "no scientifically represented action type is legal; legal "
                    f"reserved slots={np.flatnonzero(legal_types).tolist()}"
                )
            action_type = self._choose(
                scores["action_type"][row], representable, deterministic, rng
            )
            actions["action_type"][row] = action_type
            chosen: dict[str, int] = {"action_type": action_type}
            if action_type in (
                int(UpstreamActionType.PLAY_HAND),
                int(UpstreamActionType.DISCARD),
                int(UpstreamActionType.USE_CONSUMABLE),
            ):
                available = masks["card_select_mask"][row]
                maximum = min(MAX_CARD_PICKS, int(available.sum()))
                count_mask = np.arange(MAX_CARD_PICKS + 1) <= maximum
                if action_type != int(UpstreamActionType.USE_CONSUMABLE):
                    count_mask[0] = False
                count = self._choose(
                    scores["card_count"][row], count_mask, deterministic, rng
                )
                card_scores = scores["card"][row].copy()
                card_scores[~available] = -np.inf
                selected = np.argsort(-card_scores, kind="stable")[:count]
                actions["n_cards"][row] = count
                actions["cards"][row, :count] = selected
                chosen["card_count"] = count
                chosen["cards"] = tuple(int(value) for value in selected)
            for expected_type, field, head, mask_name in _TARGET_SPECS:
                if action_type == expected_type:
                    target = self._choose(
                        scores[head][row], masks[mask_name][row], deterministic, rng
                    )
                    actions[field][row] = target
                    chosen[head] = target
                    break
            choices.append(chosen)
        validate_batch(ACTION_SPEC, actions, batch, "motor actions")
        if record_choices or self.record_choices:
            self.last_choices = tuple(choices)
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
        if not np.isfinite(legal_scores).any():
            raise ValueError("every legal motor option is unrepresented")
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


_TARGET_SPECS: tuple[tuple[int, str, str, str], ...] = (
    (int(UpstreamActionType.SELL_JOKER), "joker_target", "joker", "joker_target_mask"),
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
    (int(UpstreamActionType.BUY_SHOP), "shop_target", "shop", "shop_target_mask"),
    (int(UpstreamActionType.PICK_PACK), "pack_target", "pack", "pack_target_mask"),
)
