"""Held-out Ante curriculum with strict seed-stream separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from flylatro.seeds import SeedPlan


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    ladder: tuple[int, ...] = (1, 2, 3, 5, 8)
    promotion_win_rate: float = 0.7
    evaluation_frequency_updates: int = 100
    evaluation_episodes: int = 256
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if not self.ladder or tuple(sorted(set(self.ladder))) != self.ladder:
            raise ValueError("curriculum ladder must be strictly increasing")
        if self.ladder[-1] != 8 or not all(1 <= ante <= 8 for ante in self.ladder):
            raise ValueError("curriculum must terminate at Ante 8")
        if not 0 <= self.promotion_win_rate <= 1:
            raise ValueError("promotion win rate must be in [0, 1]")
        if min(self.evaluation_frequency_updates, self.evaluation_episodes) < 1:
            raise ValueError("curriculum evaluation settings must be positive")


@dataclass(frozen=True, slots=True)
class CurriculumDecision:
    evaluated: bool
    promoted: bool
    previous_ante: int
    current_ante: int
    win_rate: float | None


class AnteCurriculum:
    def __init__(
        self, config: CurriculumConfig, seed_plan: SeedPlan | None = None
    ) -> None:
        self.config = config
        self.seed_plan = seed_plan or SeedPlan()
        self.index = 0
        self.last_evaluation_update = -1
        self.evaluation_count = 0

    @property
    def current_ante(self) -> int:
        return self.config.ladder[self.index]

    @property
    def complete(self) -> bool:
        return self.index == len(self.config.ladder) - 1

    def maybe_evaluate(
        self,
        update_index: int,
        evaluator: Callable[[int, tuple[int, ...]], float],
    ) -> CurriculumDecision:
        previous = self.current_ante
        if self.complete or (
            update_index > 0
            and update_index % self.config.evaluation_frequency_updates != 0
        ):
            return CurriculumDecision(False, False, previous, previous, None)
        offset = (
            self.config.seed_offset
            + self.evaluation_count * self.config.evaluation_episodes
        )
        seeds = self.seed_plan.seeds(
            "curriculum", self.config.evaluation_episodes, offset=offset
        )
        win_rate = float(evaluator(previous, seeds))
        if not 0 <= win_rate <= 1:
            raise ValueError("curriculum evaluator returned an invalid win rate")
        self.last_evaluation_update = update_index
        self.evaluation_count += 1
        promoted = win_rate >= self.config.promotion_win_rate
        if promoted:
            self.index += 1
        return CurriculumDecision(
            True, promoted, previous, self.current_ante, win_rate
        )

    def state_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "last_evaluation_update": self.last_evaluation_update,
            "evaluation_count": self.evaluation_count,
            "current_ante": self.current_ante,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        index = int(state["index"])
        if not 0 <= index < len(self.config.ladder):
            raise ValueError("invalid curriculum index")
        if int(state["current_ante"]) != self.config.ladder[index]:
            raise ValueError("curriculum state does not match configured ladder")
        self.index = index
        self.last_evaluation_update = int(state["last_evaluation_update"])
        self.evaluation_count = int(state["evaluation_count"])

