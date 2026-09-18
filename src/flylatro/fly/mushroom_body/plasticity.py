"""Bounded dopamine-gated three-factor KC->MBON plasticity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


@dataclass(frozen=True, slots=True)
class PlasticityConfig:
    version: str = "three-factor-kc-mbon-v1"
    learning_rate: float = 0.02
    eligibility_decay: float = 0.90
    dopamine_decay: float = 0.0
    min_efficacy: float = 0.20
    max_efficacy: float = 2.00
    activity_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.learning_rate < 0:
            raise ValueError("learning_rate cannot be negative")
        if not 0 <= self.eligibility_decay <= 1:
            raise ValueError("eligibility_decay must be in [0, 1]")
        if not 0 <= self.dopamine_decay <= 1:
            raise ValueError("dopamine_decay must be in [0, 1]")
        if not 0 <= self.min_efficacy < self.max_efficacy:
            raise ValueError("invalid efficacy bounds")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PlasticityEvent:
    learner: int
    appetitive: float
    aversive: float
    changed_synapses: int
    absolute_change: float
    lower_bound_hits: int
    upper_bound_hits: int
    before_hash: str
    after_hash: str
    changed_edge_indices: tuple[int, ...]
    efficacy_changes: tuple[float, ...]


class ThreeFactorPlasticity:
    """One plastic state per learner over a shared sparse topology.

    Activity arrays are edge-aligned. This avoids constructing dense KC x MBON
    matrices and lets both MBON-direct and whole-brain simulators gather only
    the anatomical connections that can change.
    """

    def __init__(
        self,
        topology: PlasticEdgeTopology,
        state: PlasticEdgeState,
        config: PlasticityConfig | None = None,
    ) -> None:
        if topology.edge_count != state.edge_count:
            raise ValueError("plastic state does not match topology")
        self.topology = topology
        self.state = state
        self.config = config or PlasticityConfig()
        if np.any(state.efficacy < self.config.min_efficacy) or np.any(
            state.efficacy > self.config.max_efficacy
        ):
            raise ValueError("initial efficacy is outside configured bounds")

    def record_activity(
        self,
        edge_pre_activity: NDArray[np.floating],
        edge_post_activity: NDArray[np.floating],
    ) -> None:
        pre = np.asarray(edge_pre_activity, dtype=np.float32)
        post = np.asarray(edge_post_activity, dtype=np.float32)
        expected = self.state.efficacy.shape
        if pre.shape != expected or post.shape != expected:
            raise ValueError(f"edge activity must have shape {expected}")
        if not np.isfinite(pre).all() or not np.isfinite(post).all():
            raise ValueError("edge activity must be finite")
        pre = np.maximum(pre, 0.0)
        post = np.maximum(post, 0.0)
        coincidence = pre * post
        normalizer = np.maximum(
            np.max(coincidence, axis=1, keepdims=True),
            self.config.activity_epsilon,
        )
        coincidence = coincidence / normalizer
        self.state.eligibility *= self.config.eligibility_decay
        self.state.eligibility += coincidence
        self.state.decision_count += 1

    def apply_dopamine(
        self,
        appetitive: NDArray[np.floating] | float,
        aversive: NDArray[np.floating] | float,
        *,
        plasticity_enabled: bool = True,
        include_sparse_changes: bool = False,
    ) -> tuple[PlasticityEvent, ...]:
        learners = self.state.learners
        positive = np.broadcast_to(
            np.asarray(appetitive, dtype=np.float32), (learners,)
        ).copy()
        negative = np.broadcast_to(
            np.asarray(aversive, dtype=np.float32), (learners,)
        ).copy()
        if np.any(positive < 0) or np.any(negative < 0):
            raise ValueError("dopamine channel magnitudes cannot be negative")
        before_all = self.state.efficacy.copy()
        before_hashes = [_row_hash(row) for row in before_all]
        self.state.dopamine *= self.config.dopamine_decay
        self.state.dopamine[:, 0] += positive
        self.state.dopamine[:, 1] += negative
        if plasticity_enabled and self.config.learning_rate > 0:
            signed_dopamine = self.state.dopamine[:, 0] - self.state.dopamine[:, 1]
            delta = (
                self.config.learning_rate
                * signed_dopamine[:, None]
                * self.state.eligibility
            )
            proposed = self.state.efficacy + delta
            lower = proposed < self.config.min_efficacy
            upper = proposed > self.config.max_efficacy
            self.state.lower_bound_hits += lower.sum(axis=1, dtype=np.int64)
            self.state.upper_bound_hits += upper.sum(axis=1, dtype=np.int64)
            np.clip(
                proposed,
                self.config.min_efficacy,
                self.config.max_efficacy,
                out=self.state.efficacy,
            )
        difference = self.state.efficacy - before_all
        changed = np.count_nonzero(difference, axis=1)
        self.state.update_count += changed > 0
        return tuple(
            PlasticityEvent(
                learner=index,
                appetitive=float(positive[index]),
                aversive=float(negative[index]),
                changed_synapses=int(changed[index]),
                absolute_change=float(np.abs(difference[index]).sum()),
                lower_bound_hits=int(self.state.lower_bound_hits[index]),
                upper_bound_hits=int(self.state.upper_bound_hits[index]),
                before_hash=before_hashes[index],
                after_hash=_row_hash(self.state.efficacy[index]),
                changed_edge_indices=(
                    tuple(int(edge) for edge in np.flatnonzero(difference[index]))
                    if include_sparse_changes
                    else ()
                ),
                efficacy_changes=(
                    tuple(
                        float(value)
                        for value in difference[index][difference[index] != 0]
                    )
                    if include_sparse_changes
                    else ()
                ),
            )
            for index in range(learners)
        )

    def effective_anatomical_weights(self) -> NDArray[np.float32]:
        return (
            self.state.efficacy
            * self.topology.anatomical_weights[None, :]
        ).astype(np.float32, copy=False)


def _row_hash(values: NDArray[np.float32]) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()
