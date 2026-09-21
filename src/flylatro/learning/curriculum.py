"""Held-out Ante curriculum driven by exposure counts, not optimiser updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from flylatro.seeds import SeedPlan


@dataclass(frozen=True, slots=True)
class PlasticCurriculumConfig:
    ladder: tuple[int, ...] = (1, 2, 3, 5, 8)
    promotion_win_rate: float = 0.70
    evaluation_every_decisions: int = 10_000
    evaluation_episodes: int = 256

    def __post_init__(self) -> None:
        if not self.ladder or not 1 <= self.ladder[0] <= 8 or tuple(sorted(set(self.ladder))) != self.ladder or (
            len(self.ladder) > 1 and self.ladder[-1] != 8
        ):
            raise ValueError(
                "curriculum ladder must be increasing and either contain one fixed Ante or end at 8"
            )
        if not 0 <= self.promotion_win_rate <= 1:
            raise ValueError("promotion_win_rate must be in [0, 1]")
        if min(self.evaluation_every_decisions, self.evaluation_episodes) < 1:
            raise ValueError("curriculum cadence and episodes must be positive")


@dataclass(frozen=True, slots=True)
class PlasticCurriculumDecision:
    evaluated: bool
    promoted: bool
    previous_ante: int
    current_ante: int
    win_rate: float | None


class PlasticAnteCurriculum:
    def __init__(self, config: PlasticCurriculumConfig) -> None:
        self.config = config
        self.index = 0
        self.evaluation_count = 0
        self.last_evaluation_decisions = -1

    @property
    def current_ante(self) -> int:
        return self.config.ladder[self.index]

    @property
    def complete(self) -> bool:
        return self.index == len(self.config.ladder) - 1

    def maybe_evaluate(
        self,
        environment_decisions: int,
        evaluator: Callable[[int, tuple[int, ...]], float],
    ) -> PlasticCurriculumDecision:
        previous = self.current_ante
        due = (
            environment_decisions > 0
            and environment_decisions % self.config.evaluation_every_decisions == 0
            and environment_decisions != self.last_evaluation_decisions
        )
        if self.complete or not due:
            return PlasticCurriculumDecision(False, False, previous, previous, None)
        offset = self.evaluation_count * self.config.evaluation_episodes
        seeds = SeedPlan().seeds(
            "curriculum", self.config.evaluation_episodes, offset=offset
        )
        win_rate = float(evaluator(previous, seeds))
        if not 0 <= win_rate <= 1:
            raise ValueError("curriculum evaluator returned invalid win rate")
        self.evaluation_count += 1
        self.last_evaluation_decisions = environment_decisions
        promoted = win_rate >= self.config.promotion_win_rate
        if promoted:
            self.index += 1
        return PlasticCurriculumDecision(
            True, promoted, previous, self.current_ante, win_rate
        )

    def state_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "current_ante": self.current_ante,
            "evaluation_count": self.evaluation_count,
            "last_evaluation_decisions": self.last_evaluation_decisions,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        index = int(state["index"])
        if not 0 <= index < len(self.config.ladder):
            raise ValueError("invalid curriculum checkpoint index")
        if int(state["current_ante"]) != self.config.ladder[index]:
            raise ValueError("curriculum checkpoint does not match ladder")
        self.index = index
        self.evaluation_count = int(state["evaluation_count"])
        self.last_evaluation_decisions = int(state["last_evaluation_decisions"])
