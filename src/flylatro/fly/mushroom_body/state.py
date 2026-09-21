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

    def reset_fast_traces(
        self,
        *,
        learners: NDArray[np.integer] | list[int] | tuple[int, ...] | None = None,
        eligibility: bool = False,
    ) -> None:
        """Reset selected learners' fast traces while preserving efficacy."""

        selected = slice(None) if learners is None else np.asarray(learners, dtype=np.int64)
        self.dopamine[selected] = 0.0
        if eligibility:
            self.eligibility[selected] = 0.0

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


class TorchPlasticEdgeState:
    """No-autograd plastic state resident on the real simulation device."""

    is_torch = True

    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, value)

    @classmethod
    def initialize(
        cls,
        edge_count: int,
        *,
        learners: int = 1,
        initial_efficacy: float = 1.0,
        device: str = "cpu",
    ) -> "TorchPlasticEdgeState":
        import torch

        if edge_count < 1 or learners < 1:
            raise ValueError("edge_count and learners must be positive")
        shape = (learners, edge_count)
        initial = torch.full(shape, initial_efficacy, dtype=torch.float32, device=device)
        return cls(
            initial_efficacy=initial.clone(),
            efficacy=initial.clone(),
            eligibility=torch.zeros(shape, dtype=torch.float32, device=device),
            dopamine=torch.zeros((learners, 2), dtype=torch.float32, device=device),
            update_count=torch.zeros(learners, dtype=torch.int64, device=device),
            decision_count=torch.zeros(learners, dtype=torch.int64, device=device),
            episode_count=torch.zeros(learners, dtype=torch.int64, device=device),
            lower_bound_hits=torch.zeros(learners, dtype=torch.int64, device=device),
            upper_bound_hits=torch.zeros(learners, dtype=torch.int64, device=device),
        )

    @property
    def device(self) -> Any:
        return self.efficacy.device

    @property
    def learners(self) -> int:
        return int(self.efficacy.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.efficacy.shape[1])

    @property
    def weight_sha256(self) -> str:
        return hashlib.sha256(
            self.efficacy.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest()

    def reset_fast_traces(
        self,
        *,
        learners: NDArray[np.integer] | list[int] | tuple[int, ...] | None = None,
        eligibility: bool = False,
    ) -> None:
        import torch

        selected: Any = slice(None)
        if learners is not None:
            selected = torch.as_tensor(learners, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            self.dopamine[selected] = 0.0
            if eligibility:
                self.eligibility[selected] = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name).detach().cpu().numpy()
            for name in (
                "initial_efficacy", "efficacy", "eligibility", "dopamine",
                "update_count", "decision_count", "episode_count",
                "lower_bound_hits", "upper_bound_hits",
            )
        }

    @classmethod
    def from_state_dict(
        cls, values: dict[str, Any], *, device: str = "cpu"
    ) -> "TorchPlasticEdgeState":
        import torch

        float_names = {"initial_efficacy", "efficacy", "eligibility", "dopamine"}
        tensors = {
            name: torch.as_tensor(
                value,
                dtype=torch.float32 if name in float_names else torch.int64,
                device=device,
            ).clone()
            for name, value in values.items()
        }
        state = cls(**tensors)
        if state.efficacy.ndim != 2 or state.dopamine.shape != (state.learners, 2):
            raise ValueError("checkpoint torch plastic state is inconsistent")
        return state


def state_numpy(value: Any) -> np.ndarray:
    """Convert a state tensor/array only at diagnostics or serialization edges."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
