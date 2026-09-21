"""Bounded dopamine-gated three-factor KC->MBON plasticity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.mushroom_body.state import PlasticEdgeState, state_numpy
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


@dataclass(frozen=True, slots=True)
class PlasticityConfig:
    version: str = "three-factor-global-v1"
    learning_rate: float = 0.02
    eligibility_decay: float = 0.90
    dopamine_decay: float = 0.0
    min_efficacy: float = 0.20
    max_efficacy: float = 2.00
    kc_reference_hz: float = 1.0
    mbon_reference_hz: float = 1.0
    max_eligibility: float = 4.0

    def __post_init__(self) -> None:
        if self.learning_rate < 0:
            raise ValueError("learning_rate cannot be negative")
        if not 0 <= self.eligibility_decay <= 1:
            raise ValueError("eligibility_decay must be in [0, 1]")
        if not 0 <= self.dopamine_decay <= 1:
            raise ValueError("dopamine_decay must be in [0, 1]")
        if not 0 <= self.min_efficacy < self.max_efficacy:
            raise ValueError("invalid efficacy bounds")
        if self.kc_reference_hz <= 0 or self.mbon_reference_hz <= 0:
            raise ValueError("activity reference rates must be positive")
        if self.max_eligibility <= 0:
            raise ValueError("max_eligibility must be positive")

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
    eligible_synapses: int
    max_absolute_eligibility: float
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
        efficacy = state_numpy(state.efficacy)
        if np.any(efficacy < self.config.min_efficacy) or np.any(
            efficacy > self.config.max_efficacy
        ):
            raise ValueError("initial efficacy is outside configured bounds")

    def record_activity(
        self,
        edge_pre_activity: NDArray[np.floating],
        edge_post_activity: NDArray[np.floating],
    ) -> None:
        if getattr(self.state, "is_torch", False):
            self._record_activity_torch(edge_pre_activity, edge_post_activity)
            return
        pre = np.asarray(edge_pre_activity, dtype=np.float32)
        post = np.asarray(edge_post_activity, dtype=np.float32)
        expected = self.state.efficacy.shape
        if pre.shape != expected or post.shape != expected:
            raise ValueError(f"edge activity must have shape {expected}")
        if not np.isfinite(pre).all() or not np.isfinite(post).all():
            raise ValueError("edge activity must be finite")
        pre = np.maximum(pre, 0.0)
        post = np.maximum(post, 0.0)
        # Absolute, fixed-reference scaling preserves magnitude across
        # decisions: a uniformly weak response remains weak rather than being
        # promoted to 1.0 by within-decision maximum normalization.
        pre_scaled = np.clip(pre / self.config.kc_reference_hz, 0.0, 1.0)
        post_scaled = np.clip(post / self.config.mbon_reference_hz, 0.0, 1.0)
        coincidence = pre_scaled * post_scaled
        self.state.eligibility *= self.config.eligibility_decay
        self.state.eligibility += coincidence
        np.clip(
            self.state.eligibility,
            -self.config.max_eligibility,
            self.config.max_eligibility,
            out=self.state.eligibility,
        )
        self.state.decision_count += 1

    def _record_activity_torch(self, edge_pre_activity: object, edge_post_activity: object) -> None:
        import torch

        pre = torch.as_tensor(edge_pre_activity, dtype=torch.float32, device=self.state.device)
        post = torch.as_tensor(edge_post_activity, dtype=torch.float32, device=self.state.device)
        if tuple(pre.shape) != tuple(self.state.efficacy.shape) or tuple(post.shape) != tuple(pre.shape):
            raise ValueError(f"edge activity must have shape {tuple(self.state.efficacy.shape)}")
        if not bool(torch.isfinite(pre).all()) or not bool(torch.isfinite(post).all()):
            raise ValueError("edge activity must be finite")
        with torch.no_grad():
            coincidence = torch.clamp(pre, min=0) / self.config.kc_reference_hz
            coincidence.clamp_(0, 1)
            post_scaled = torch.clamp(post, min=0) / self.config.mbon_reference_hz
            post_scaled.clamp_(0, 1)
            coincidence.mul_(post_scaled)
            self.state.eligibility.mul_(self.config.eligibility_decay).add_(coincidence)
            self.state.eligibility.clamp_(-self.config.max_eligibility, self.config.max_eligibility)
            self.state.decision_count.add_(1)

    def apply_dopamine(
        self,
        appetitive: NDArray[np.floating] | float,
        aversive: NDArray[np.floating] | float,
        *,
        plasticity_enabled: bool = True,
        include_sparse_changes: bool = False,
    ) -> tuple[PlasticityEvent, ...]:
        if getattr(self.state, "is_torch", False):
            return self._apply_dopamine_torch(
                appetitive, aversive,
                plasticity_enabled=plasticity_enabled,
                include_sparse_changes=include_sparse_changes,
            )
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
                eligible_synapses=int(np.count_nonzero(self.state.eligibility[index])),
                max_absolute_eligibility=float(np.max(np.abs(self.state.eligibility[index]))),
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

    def _apply_dopamine_torch(
        self,
        appetitive: object,
        aversive: object,
        *,
        plasticity_enabled: bool,
        include_sparse_changes: bool,
    ) -> tuple[PlasticityEvent, ...]:
        import torch

        learners = self.state.learners
        positive = torch.as_tensor(appetitive, dtype=torch.float32, device=self.state.device).broadcast_to((learners,)).clone()
        negative = torch.as_tensor(aversive, dtype=torch.float32, device=self.state.device).broadcast_to((learners,)).clone()
        if bool((positive < 0).any()) or bool((negative < 0).any()):
            raise ValueError("reinforcement channel magnitudes cannot be negative")
        before = self.state.efficacy.detach().clone()
        before_np = before.cpu().numpy()
        with torch.no_grad():
            self.state.dopamine.mul_(self.config.dopamine_decay)
            self.state.dopamine[:, 0].add_(positive)
            self.state.dopamine[:, 1].add_(negative)
            if plasticity_enabled and self.config.learning_rate > 0:
                signed = self.state.dopamine[:, 0] - self.state.dopamine[:, 1]
                proposed = self.state.efficacy + self.config.learning_rate * signed[:, None] * self.state.eligibility
                lower = proposed < self.config.min_efficacy
                upper = proposed > self.config.max_efficacy
                self.state.lower_bound_hits.add_(lower.sum(dim=1))
                self.state.upper_bound_hits.add_(upper.sum(dim=1))
                self.state.efficacy.copy_(proposed.clamp(self.config.min_efficacy, self.config.max_efficacy))
            difference = self.state.efficacy - before
            changed = torch.count_nonzero(difference, dim=1)
            self.state.update_count.add_((changed > 0).to(torch.int64))
        diff_np = difference.detach().cpu().numpy()
        positive_np = positive.cpu().numpy()
        negative_np = negative.cpu().numpy()
        efficacy_np = self.state.efficacy.detach().cpu().numpy()
        eligibility_np = self.state.eligibility.detach().cpu().numpy()
        return tuple(
            PlasticityEvent(
                learner=index,
                appetitive=float(positive_np[index]),
                aversive=float(negative_np[index]),
                changed_synapses=int(np.count_nonzero(diff_np[index])),
                absolute_change=float(np.abs(diff_np[index]).sum()),
                lower_bound_hits=int(self.state.lower_bound_hits[index].item()),
                upper_bound_hits=int(self.state.upper_bound_hits[index].item()),
                eligible_synapses=int(np.count_nonzero(eligibility_np[index])),
                max_absolute_eligibility=float(np.max(np.abs(eligibility_np[index]))),
                before_hash=_row_hash(before_np[index]),
                after_hash=_row_hash(efficacy_np[index]),
                changed_edge_indices=(tuple(int(edge) for edge in np.flatnonzero(diff_np[index])) if include_sparse_changes else ()),
                efficacy_changes=(tuple(float(value) for value in diff_np[index][diff_np[index] != 0]) if include_sparse_changes else ()),
            )
            for index in range(learners)
        )


def _row_hash(values: NDArray[np.float32]) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()
