"""Bounded dopamine-gated three-factor KC->MBON plasticity.

Routine training keeps efficacy, eligibility and reinforcement traces resident
on the simulation device.  Per-decision metrics are Torch reductions and reach
the host as one small scalar block, so a normal CUDA step never copies a full
plastic vector.  Per-edge detail and weight hashes are opt-in and belong to
checkpoints, audits, short gate runs and explicit recordings.
"""

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
class PlasticityEventDetail:
    """Opt-in per-edge detail; never produced on the routine training path."""

    before_hash: str
    after_hash: str
    changed_edge_indices: tuple[int, ...]
    efficacy_changes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PlasticityEvent:
    """Lightweight scalar summary of one reinforcement application."""

    learner: int
    appetitive: float
    aversive: float
    changed_synapses: int
    absolute_change: float
    maximum_absolute_change: float
    lower_bound_hits: int
    upper_bound_hits: int
    eligible_synapses: int
    max_absolute_eligibility: float
    mean_absolute_eligibility: float
    detail: PlasticityEventDetail | None = None

    @property
    def detailed(self) -> bool:
        return self.detail is not None

    @property
    def before_hash(self) -> str | None:
        return None if self.detail is None else self.detail.before_hash

    @property
    def after_hash(self) -> str | None:
        return None if self.detail is None else self.detail.after_hash

    @property
    def changed_edge_indices(self) -> tuple[int, ...]:
        return () if self.detail is None else self.detail.changed_edge_indices

    @property
    def efficacy_changes(self) -> tuple[float, ...]:
        return () if self.detail is None else self.detail.efficacy_changes


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

    @property
    def is_torch(self) -> bool:
        return bool(getattr(self.state, "is_torch", False))

    def record_activity(
        self,
        edge_pre_activity: NDArray[np.floating],
        edge_post_activity: NDArray[np.floating],
    ) -> None:
        if self.is_torch:
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
        with torch.no_grad():
            # One fused device-side finiteness reduction; no host copy of the
            # full activity vectors on the routine path.
            if not bool(torch.isfinite(pre).all() & torch.isfinite(post).all()):
                raise ValueError("edge activity must be finite")
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
        detail: bool = False,
    ) -> tuple[PlasticityEvent, ...]:
        """Apply one synthetic reinforcement pulse to every learner.

        ``detail`` additionally produces before/after weight hashes, changed
        edge IDs and per-edge deltas.  It requires host copies of the full
        plastic vectors and is intended for short gate runs, audits, selected
        recordings and diagnostics, not for normal training.
        """

        if self.is_torch:
            return self._apply_dopamine_torch(
                appetitive,
                aversive,
                plasticity_enabled=plasticity_enabled,
                detail=detail,
            )
        return self._apply_dopamine_numpy(
            appetitive,
            aversive,
            plasticity_enabled=plasticity_enabled,
            detail=detail,
        )

    def _apply_dopamine_numpy(
        self,
        appetitive: NDArray[np.floating] | float,
        aversive: NDArray[np.floating] | float,
        *,
        plasticity_enabled: bool,
        detail: bool,
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
        before = self.state.efficacy.copy() if detail else None
        self.state.dopamine *= self.config.dopamine_decay
        self.state.dopamine[:, 0] += positive
        self.state.dopamine[:, 1] += negative
        if plasticity_enabled and self.config.learning_rate > 0:
            signed_dopamine = self.state.dopamine[:, 0] - self.state.dopamine[:, 1]
            proposed = self.state.efficacy + (
                self.config.learning_rate * signed_dopamine[:, None] * self.state.eligibility
            )
            lower = proposed < self.config.min_efficacy
            upper = proposed > self.config.max_efficacy
            self.state.lower_bound_hits += lower.sum(axis=1, dtype=np.int64)
            self.state.upper_bound_hits += upper.sum(axis=1, dtype=np.int64)
            np.clip(
                proposed,
                self.config.min_efficacy,
                self.config.max_efficacy,
                out=proposed,
            )
            difference = proposed - self.state.efficacy
            self.state.efficacy[...] = proposed
        else:
            difference = np.zeros_like(self.state.efficacy)
        changed = np.count_nonzero(difference, axis=1)
        self.state.update_count += changed > 0
        absolute = np.abs(difference)
        eligibility = np.abs(self.state.eligibility)
        return tuple(
            PlasticityEvent(
                learner=index,
                appetitive=float(positive[index]),
                aversive=float(negative[index]),
                changed_synapses=int(changed[index]),
                absolute_change=float(absolute[index].sum()),
                maximum_absolute_change=float(absolute[index].max(initial=0.0)),
                lower_bound_hits=int(self.state.lower_bound_hits[index]),
                upper_bound_hits=int(self.state.upper_bound_hits[index]),
                eligible_synapses=int(np.count_nonzero(self.state.eligibility[index])),
                max_absolute_eligibility=float(eligibility[index].max(initial=0.0)),
                mean_absolute_eligibility=float(eligibility[index].mean()),
                detail=(
                    _numpy_detail(before[index], self.state.efficacy[index], difference[index])
                    if before is not None
                    else None
                ),
            )
            for index in range(learners)
        )

    def _apply_dopamine_torch(
        self,
        appetitive: object,
        aversive: object,
        *,
        plasticity_enabled: bool,
        detail: bool,
    ) -> tuple[PlasticityEvent, ...]:
        import torch

        learners = self.state.learners
        device = self.state.device
        with torch.no_grad():
            positive = (
                torch.as_tensor(appetitive, dtype=torch.float32, device=device)
                .broadcast_to((learners,))
                .clone()
            )
            negative = (
                torch.as_tensor(aversive, dtype=torch.float32, device=device)
                .broadcast_to((learners,))
                .clone()
            )
            if bool((positive < 0).any() | (negative < 0).any()):
                raise ValueError("reinforcement channel magnitudes cannot be negative")
            before = self.state.efficacy.detach().clone() if detail else None
            self.state.dopamine.mul_(self.config.dopamine_decay)
            self.state.dopamine[:, 0].add_(positive)
            self.state.dopamine[:, 1].add_(negative)
            eligibility = self.state.eligibility.abs()
            if plasticity_enabled and self.config.learning_rate > 0:
                signed = self.state.dopamine[:, 0] - self.state.dopamine[:, 1]
                proposed = self.state.efficacy + (
                    self.config.learning_rate * signed[:, None] * self.state.eligibility
                )
                lower = (proposed < self.config.min_efficacy).sum(dim=1)
                upper = (proposed > self.config.max_efficacy).sum(dim=1)
                self.state.lower_bound_hits.add_(lower)
                self.state.upper_bound_hits.add_(upper)
                proposed.clamp_(self.config.min_efficacy, self.config.max_efficacy)
                difference = proposed - self.state.efficacy
                self.state.efficacy.copy_(proposed)
                absolute = difference.abs()
                changed = (difference != 0).sum(dim=1)
                change_sum = absolute.sum(dim=1)
                change_max = absolute.amax(dim=1)
            else:
                difference = None
                zeros = torch.zeros(learners, dtype=torch.float32, device=device)
                changed = torch.zeros(learners, dtype=torch.int64, device=device)
                change_sum = zeros
                change_max = zeros
            self.state.update_count.add_((changed > 0).to(torch.int64))
            # Exactly one small device-to-host transfer per learning step.
            summary = torch.stack(
                (
                    changed.to(torch.float64),
                    change_sum.to(torch.float64),
                    change_max.to(torch.float64),
                    (self.state.eligibility != 0).sum(dim=1).to(torch.float64),
                    eligibility.amax(dim=1).to(torch.float64),
                    eligibility.mean(dim=1).to(torch.float64),
                    self.state.lower_bound_hits.to(torch.float64),
                    self.state.upper_bound_hits.to(torch.float64),
                    positive.to(torch.float64),
                    negative.to(torch.float64),
                ),
                dim=1,
            )
            rows = summary.cpu().numpy()
            details: list[PlasticityEventDetail | None] = [None] * learners
            if before is not None:
                after_np = self.state.efficacy.detach().cpu().numpy()
                before_np = before.cpu().numpy()
                diff_np = (
                    difference.detach().cpu().numpy()
                    if difference is not None
                    else np.zeros_like(after_np)
                )
                details = [
                    _numpy_detail(before_np[index], after_np[index], diff_np[index])
                    for index in range(learners)
                ]
        return tuple(
            PlasticityEvent(
                learner=index,
                appetitive=float(rows[index, 8]),
                aversive=float(rows[index, 9]),
                changed_synapses=int(rows[index, 0]),
                absolute_change=float(rows[index, 1]),
                maximum_absolute_change=float(rows[index, 2]),
                lower_bound_hits=int(rows[index, 6]),
                upper_bound_hits=int(rows[index, 7]),
                eligible_synapses=int(rows[index, 3]),
                max_absolute_eligibility=float(rows[index, 4]),
                mean_absolute_eligibility=float(rows[index, 5]),
                detail=details[index],
            )
            for index in range(learners)
        )

    def effective_anatomical_weights(self) -> NDArray[np.float32]:
        """Anatomical magnitude times current efficacy, for NumPy or Torch state."""

        efficacy = state_numpy(self.state.efficacy).astype(np.float32, copy=False)
        return (efficacy * self.topology.anatomical_weights[None, :]).astype(
            np.float32, copy=False
        )

    def routine_metrics(self) -> dict[str, float]:
        """Per-step telemetry computed without copying any full plastic vector.

        On the Torch path every quantity is a device reduction and the whole
        block reaches the host as one small tensor.
        """

        if not self.is_torch:
            efficacy = self.state.efficacy
            initial = self.state.initial_efficacy
            eligibility = np.abs(self.state.eligibility)
            drift = np.abs(efficacy - initial)
            return {
                "mean_efficacy": float(efficacy.mean()),
                "absolute_change": float(drift.sum()),
                "mean_absolute_drift": float(drift.mean()),
                "max_absolute_drift": float(drift.max(initial=0.0)),
                "eligibility_mean_absolute": float(eligibility.mean()),
                "eligibility_max_absolute": float(eligibility.max(initial=0.0)),
                "reinforcement_trace_mean_absolute": float(
                    np.abs(self.state.dopamine).mean()
                ),
                "lower_bound_fraction": float(
                    np.mean(efficacy <= self.config.min_efficacy)
                ),
                "upper_bound_fraction": float(
                    np.mean(efficacy >= self.config.max_efficacy)
                ),
            }
        import torch

        with torch.no_grad():
            efficacy = self.state.efficacy
            drift = (efficacy - self.state.initial_efficacy).abs()
            eligibility = self.state.eligibility.abs()
            block = torch.stack(
                (
                    efficacy.mean(),
                    drift.sum(),
                    drift.mean(),
                    drift.amax(),
                    eligibility.mean(),
                    eligibility.amax(),
                    self.state.dopamine.abs().mean(),
                    (efficacy <= self.config.min_efficacy).to(torch.float64).mean(),
                    (efficacy >= self.config.max_efficacy).to(torch.float64).mean(),
                )
            ).to(torch.float64)
            values = block.cpu().numpy()
        names = (
            "mean_efficacy",
            "absolute_change",
            "mean_absolute_drift",
            "max_absolute_drift",
            "eligibility_mean_absolute",
            "eligibility_max_absolute",
            "reinforcement_trace_mean_absolute",
            "lower_bound_fraction",
            "upper_bound_fraction",
        )
        return {name: float(values[index]) for index, name in enumerate(names)}

    def weight_audit(self) -> dict[str, object]:
        """Explicit host-side weight hash for checkpoints and periodic audits."""

        efficacy = state_numpy(self.state.efficacy)
        return {
            "plastic_weight_sha256": self.state.weight_sha256,
            "per_learner_weight_sha256": [
                _row_hash(efficacy[index]) for index in range(efficacy.shape[0])
            ],
            "plastic_topology_sha256": self.topology.sha256,
            "plasticity_rule_sha256": self.config.sha256,
        }


def _numpy_detail(
    before: NDArray[np.float32],
    after: NDArray[np.float32],
    difference: NDArray[np.float32],
) -> PlasticityEventDetail:
    indices = np.flatnonzero(difference)
    return PlasticityEventDetail(
        before_hash=_row_hash(before),
        after_hash=_row_hash(after),
        changed_edge_indices=tuple(int(edge) for edge in indices),
        efficacy_changes=tuple(float(value) for value in difference[indices]),
    )


def _row_hash(values: NDArray[np.float32]) -> str:
    return hashlib.sha256(
        np.asarray(values).astype("<f4", copy=False).tobytes()
    ).hexdigest()
