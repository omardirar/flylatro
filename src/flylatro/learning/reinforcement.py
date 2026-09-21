"""Fixed strategy-neutral synthetic reinforcement mapping and controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ReinforcementPulse:
    appetitive: float
    aversive: float
    events: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.appetitive < 0 or self.aversive < 0:
            raise ValueError("synthetic reinforcement magnitudes cannot be negative")
        if not np.isfinite((self.appetitive, self.aversive)).all():
            raise ValueError("synthetic reinforcement magnitudes must be finite")


#: Predeclared reinforcement-shaping sensitivity conditions.
#:
#: These exist so the experiment can answer whether apparent learning depends
#: on dense blind-progress shaping.  They are declared **before** any result is
#: seen and must never be selected by whichever one eventually scores best.
SENSITIVITY_CONDITIONS: dict[str, dict[str, float]] = {
    "primary-progress": {"progress_scale": 0.05},
    "reduced-progress": {"progress_scale": 0.01},
    "terminal-or-clear-only": {"progress_scale": 0.0},
}


@dataclass(frozen=True, slots=True)
class ReinforcementConfig:
    version: str = "balatro-outcome-synthetic-reinforcement-v2"
    condition: str = "primary-progress"
    progress_scale: float = 0.05
    blind_clear_pulse: float = 0.40
    ante_clear_pulse: float = 0.80
    run_failure_pulse: float = 0.60
    ante8_success_pulse: float = 1.50
    max_progress_component: float = 1.0
    max_total_pulse: float = 2.0

    def __post_init__(self) -> None:
        numeric = (
            self.progress_scale,
            self.blind_clear_pulse,
            self.ante_clear_pulse,
            self.run_failure_pulse,
            self.ante8_success_pulse,
            self.max_progress_component,
            self.max_total_pulse,
        )
        if any(value < 0 or not np.isfinite(value) for value in numeric):
            raise ValueError("reinforcement parameters must be finite and non-negative")
        expected = SENSITIVITY_CONDITIONS.get(self.condition)
        if expected is None:
            raise ValueError(
                "reinforcement.condition must be one of "
                f"{sorted(SENSITIVITY_CONDITIONS)}; ad-hoc shaping magnitudes are "
                "not predeclared experiments"
            )
        mismatched = {
            name: (getattr(self, name), value)
            for name, value in expected.items()
            if getattr(self, name) != value
        }
        if mismatched:
            raise ValueError(
                f"reinforcement condition {self.condition!r} fixes {mismatched}; "
                "declare a new named sensitivity condition instead of retuning it"
            )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class ReinforcementMapper:
    trainable_parameter_count = 0

    def __init__(self, config: ReinforcementConfig | None = None) -> None:
        self.config = config or ReinforcementConfig()

    def map(
        self,
        reward_components: dict[str, float],
        info: dict[str, Any] | None = None,
    ) -> ReinforcementPulse:
        """Map only generic outcome components, never action/hand strategy."""

        info = info or {}
        events: list[str] = []
        progress = max(
            0.0,
            float(
                reward_components.get(
                    "blind_progress",
                    reward_components.get("delta_capped_blind_progress", 0.0),
                )
            ),
        )
        positive = (
            min(progress, self.config.max_progress_component)
            * self.config.progress_scale
        )
        if progress > 0:
            events.append("progress")
        if float(reward_components.get("blind_clear", 0.0)) > 0 or bool(
            info.get("blind_cleared", False)
        ):
            positive += self.config.blind_clear_pulse
            events.append("blind_clear")
        if float(reward_components.get("ante_clear", 0.0)) > 0 or bool(
            info.get("ante_cleared", False)
        ):
            positive += self.config.ante_clear_pulse
            events.append("ante_clear")
        won = bool(
            info.get("won", False)
            or (isinstance(info.get("episode"), dict) and info["episode"].get("won"))
            or float(reward_components.get("win", 0.0)) > 0
        )
        reached_ante = int(
            info.get(
                "ante",
                info.get("episode", {}).get("ante", 0)
                if isinstance(info.get("episode"), dict)
                else 0,
            )
        )
        if won and reached_ante >= 8:
            positive += self.config.ante8_success_pulse
            events.append("ante8_success")
        failed = bool(
            info.get("run_failed", False)
            or (
                isinstance(info.get("episode"), dict)
                and info["episode"].get("won") is False
            )
        )
        negative = self.config.run_failure_pulse if failed else 0.0
        if failed:
            events.append("run_failure")
        return ReinforcementPulse(
            appetitive=min(positive, self.config.max_total_pulse),
            aversive=min(negative, self.config.max_total_pulse),
            events=tuple(events),
        )


def sensitivity_condition(name: str, **overrides: float) -> ReinforcementConfig:
    """Build one predeclared reinforcement-shaping sensitivity condition."""

    if name not in SENSITIVITY_CONDITIONS:
        raise ValueError(f"unknown predeclared sensitivity condition: {name}")
    return ReinforcementConfig(
        condition=name, **SENSITIVITY_CONDITIONS[name], **overrides
    )


def shuffled_pulse_schedule(
    pulses: Sequence[ReinforcementPulse], *, seed: int
) -> tuple[ReinforcementPulse, ...]:
    """Offline deterministic temporal shuffle preserving the exact pulse multiset."""

    if len(pulses) < 2:
        raise ValueError("shuffled-reward control requires at least two pulses")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(pulses))
    if np.array_equal(permutation, np.arange(len(pulses))):
        permutation = np.roll(permutation, 1)
    return tuple(pulses[int(index)] for index in permutation)


# Compatibility import for older local checkpoints/tests. Public logs and
# manifests use the honest synthetic-reinforcement terminology.
DopaminePulse = ReinforcementPulse
