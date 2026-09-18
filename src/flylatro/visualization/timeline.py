"""Shared decision clock for game replay, neural activity, and presentation cues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TimelineConfig:
    neural_duration_ms: float = 50.0
    playback_seconds_per_decision: float = 1.0
    inter_decision_pause_seconds: float = 0.15

    def __post_init__(self) -> None:
        if self.neural_duration_ms <= 0 or self.playback_seconds_per_decision <= 0:
            raise ValueError("timeline durations must be positive")
        if self.inter_decision_pause_seconds < 0:
            raise ValueError("timeline pause cannot be negative")

    @property
    def slowdown(self) -> float:
        return self.playback_seconds_per_decision / (self.neural_duration_ms / 1000.0)


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    decision_id: int
    start_seconds: float
    neural_end_seconds: float
    end_seconds: float
    action_type: str
    cursor_cue: str
    selected_action_probability: float | None
    action_type_probabilities: tuple[float, ...]
    reward: float
    dopamine_appetitive: float
    dopamine_aversive: float


@dataclass(frozen=True, slots=True)
class Timeline:
    config: TimelineConfig
    entries: tuple[TimelineEntry, ...]

    @property
    def duration_seconds(self) -> float:
        return self.entries[-1].end_seconds if self.entries else 0.0

    def neural_time_to_video(self, decision_id: int, simulated_ms: float) -> float:
        entry = self.entries[decision_id]
        bounded = min(max(simulated_ms, 0.0), self.config.neural_duration_ms)
        return entry.start_seconds + bounded / 1000.0 * self.config.slowdown

    def to_payload(self) -> dict[str, Any]:
        return {
            "neural_time_slowdown": self.config.slowdown,
            "duration_seconds": self.duration_seconds,
            "entries": [
                {
                    "decision_id": item.decision_id,
                    "start_seconds": item.start_seconds,
                    "neural_end_seconds": item.neural_end_seconds,
                    "end_seconds": item.end_seconds,
                    "action_type": item.action_type,
                    "cursor_cue": item.cursor_cue,
                    "selected_action_probability": item.selected_action_probability,
                    "action_type_probabilities": item.action_type_probabilities,
                    "reward": item.reward,
                    "dopamine_appetitive": item.dopamine_appetitive,
                    "dopamine_aversive": item.dopamine_aversive,
                }
                for item in self.entries
            ],
        }


def build_timeline(
    decisions: Sequence[Mapping[str, Any]],
    config: TimelineConfig | None = None,
) -> Timeline:
    config = config or TimelineConfig()
    entries = []
    clock = 0.0
    for expected_id, decision in enumerate(decisions):
        decision_id = int(decision["decision_id"])
        if decision_id != expected_id:
            raise ValueError("timeline requires contiguous decisions from zero")
        action_type = str(decision["action"]["type"])
        neural_end = clock + config.playback_seconds_per_decision
        end = neural_end + config.inter_decision_pause_seconds
        entries.append(
            TimelineEntry(
                decision_id=decision_id,
                start_seconds=clock,
                neural_end_seconds=neural_end,
                end_seconds=end,
                action_type=action_type,
                cursor_cue=_cursor_cue(action_type),
                selected_action_probability=(
                    float(decision["action_probability"])
                    if decision.get("action_probability") is not None else None
                ),
                action_type_probabilities=tuple(
                    float(value)
                    for value in decision.get("action_type_probabilities", ())
                ),
                reward=float(decision.get("reward", 0.0)),
                dopamine_appetitive=float(
                    decision.get("dopamine_appetitive", 0.0)
                ),
                dopamine_aversive=float(
                    decision.get("dopamine_aversive", 0.0)
                ),
            )
        )
        clock = end
    return Timeline(config, tuple(entries))


def _cursor_cue(action_type: str) -> str:
    if action_type in {"play_hand", "discard"}:
        return "hand"
    if action_type in {"buy", "reroll", "end_shop"}:
        return "shop"
    if action_type in {"pick_pack", "skip_pack"}:
        return "pack"
    if action_type in {"sell_joker", "sell_consumable", "use_consumable"}:
        return "inventory"
    return "primary-control"
