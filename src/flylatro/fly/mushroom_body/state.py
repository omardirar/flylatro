"""Mutable per-learner plastic state, separate from shared fixed anatomy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class PlasticEdgeState:
    initial_efficacy: NDArray[np.float32]
    efficacy: NDArray[np.float32]
    eligibility: NDArray[np.float32]
    dopamine: NDArray[np.float32]
    update_count: NDArray[np.int64]
    decision_count: NDArray[np.int64]
    episode_count: NDArray[np.int64]
    lower_bound_hits: NDArray[np.int64]
    upper_bound_hits: NDArray[np.int64]

    @classmethod
    def initialize(
        cls,
        edge_count: int,
        *,
        learners: int = 1,
        initial_efficacy: float = 1.0,
    ) -> "PlasticEdgeState":
        if edge_count < 1 or learners < 1:
            raise ValueError("edge_count and learners must be positive")
        shape = (learners, edge_count)
        initial = np.full(shape, initial_efficacy, dtype=np.float32)
        return cls(
            initial_efficacy=initial.copy(),
            efficacy=initial.copy(),
            eligibility=np.zeros(shape, dtype=np.float32),
            dopamine=np.zeros((learners, 2), dtype=np.float32),
            update_count=np.zeros(learners, dtype=np.int64),
            decision_count=np.zeros(learners, dtype=np.int64),
            episode_count=np.zeros(learners, dtype=np.int64),
            lower_bound_hits=np.zeros(learners, dtype=np.int64),
            upper_bound_hits=np.zeros(learners, dtype=np.int64),
        )

    @property
    def learners(self) -> int:
        return self.efficacy.shape[0]

    @property
    def edge_count(self) -> int:
        return self.efficacy.shape[1]

    @property
    def weight_sha256(self) -> str:
        return hashlib.sha256(
            self.efficacy.astype("<f4", copy=False).tobytes()
        ).hexdigest()

    def reset_fast_traces(self, *, eligibility: bool = False) -> None:
        """Dopamine is fast; eligibility is retained unless explicitly reset."""

        self.dopamine.fill(0.0)
        if eligibility:
            self.eligibility.fill(0.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "initial_efficacy": self.initial_efficacy.copy(),
            "efficacy": self.efficacy.copy(),
            "eligibility": self.eligibility.copy(),
            "dopamine": self.dopamine.copy(),
            "update_count": self.update_count.copy(),
            "decision_count": self.decision_count.copy(),
            "episode_count": self.episode_count.copy(),
            "lower_bound_hits": self.lower_bound_hits.copy(),
            "upper_bound_hits": self.upper_bound_hits.copy(),
        }

    @classmethod
    def from_state_dict(cls, values: dict[str, Any]) -> "PlasticEdgeState":
        state = cls(
            initial_efficacy=np.asarray(values["initial_efficacy"], dtype=np.float32),
            efficacy=np.asarray(values["efficacy"], dtype=np.float32),
            eligibility=np.asarray(values["eligibility"], dtype=np.float32),
            dopamine=np.asarray(values["dopamine"], dtype=np.float32),
            update_count=np.asarray(values["update_count"], dtype=np.int64),
            decision_count=np.asarray(values["decision_count"], dtype=np.int64),
            episode_count=np.asarray(values["episode_count"], dtype=np.int64),
            lower_bound_hits=np.asarray(values["lower_bound_hits"], dtype=np.int64),
            upper_bound_hits=np.asarray(values["upper_bound_hits"], dtype=np.int64),
        )
        shape = state.efficacy.shape
        if state.efficacy.ndim != 2 or any(
            array.shape != shape
            for array in (state.initial_efficacy, state.eligibility)
        ):
            raise ValueError("checkpoint plastic edge arrays are inconsistent")
        if state.dopamine.shape != (shape[0], 2):
            raise ValueError("checkpoint dopamine state is inconsistent")
        return state
