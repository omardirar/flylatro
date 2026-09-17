"""Interfaces implemented by real and lightweight Balatro backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.env.types import ActionMask, BalatroObservation, CompositeAction


@dataclass(frozen=True, slots=True)
class VectorStep:
    observations: tuple[BalatroObservation, ...]
    rewards: NDArray[np.float64]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    reward_components: tuple[dict[str, float], ...]
    infos: tuple[dict[str, Any], ...]


class BalatroEnv(Protocol):
    """Vector environment contract owned by Flylatro.

    ``None`` is passed for an environment whose one-shot episode has already
    terminated.  Real training adapters may instead auto-reset, but must keep
    terminal observations available to the rollout recorder.
    """

    @property
    def num_envs(self) -> int: ...

    def reset(self, seeds: Sequence[int]) -> tuple[BalatroObservation, ...]: ...

    def legal_action_masks(self) -> tuple[ActionMask | None, ...]: ...

    def step(self, actions: Sequence[CompositeAction | None]) -> VectorStep: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, snapshot: dict[str, Any]) -> None: ...

