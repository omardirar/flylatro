"""Default-on training stops using scalar/device-reduced health telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


TRAINING_SAFETY_VERSION = "plastic-training-safety-v1"


@dataclass(frozen=True, slots=True)
class TrainingSafetyConfig:
    enabled: bool = True
    maximum_bound_fraction: float = 0.10
    maximum_mean_absolute_drift: float = 0.50
    minimum_kc_active_fraction: float = 0.01
    minimum_motor_active_fraction: float = 0.01
    silence_patience_steps: int = 8

    def __post_init__(self) -> None:
        for name in ("maximum_bound_fraction", "minimum_kc_active_fraction", "minimum_motor_active_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"safety {name} must be finite and in [0, 1]")
        if not math.isfinite(self.maximum_mean_absolute_drift) or self.maximum_mean_absolute_drift <= 0:
            raise ValueError("safety maximum_mean_absolute_drift must be finite and positive")
        if type(self.silence_patience_steps) is not int or self.silence_patience_steps < 1:
            raise ValueError("safety silence_patience_steps must be positive")
        if not isinstance(self.enabled, bool):
            raise ValueError("safety enabled must be a boolean")


class TrainingSafetyStop(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__("training safety stop: " + ", ".join(report["failed"]))


class TrainingSafetyMonitor:
    """Observe health only; never alter efficacy, rewards, masks or choices."""

    def __init__(self, config: TrainingSafetyConfig, *, maximum_eligibility: float = 4.0) -> None:
        self.config = config
        self.maximum_eligibility = maximum_eligibility
        self.silent_steps = {"kc": 0, "motor": 0}
        self.failure: dict[str, Any] | None = None

    def observe(self, metrics: Mapping[str, float], *, state_only: bool = False) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None
        failed: list[str] = []
        if any(not math.isfinite(float(value)) for value in metrics.values()) or metrics.get("plasticity/finite_state", 0.0) != 1.0:
            failed.append("non_finite_state")
        if metrics.get("plasticity/max_learner_bound_fraction", math.inf) > self.config.maximum_bound_fraction:
            failed.append("bound_saturation")
        if metrics.get("plasticity/max_learner_mean_absolute_drift", math.inf) > self.config.maximum_mean_absolute_drift:
            failed.append("efficacy_drift")
        if metrics.get("plasticity/efficacy_outside_bounds", 1.0) != 0.0:
            failed.append("efficacy_outside_bounds")
        if metrics.get("plasticity/eligibility_max", metrics.get("plasticity/eligibility_max_absolute", math.inf)) > self.maximum_eligibility + 1e-7:
            failed.append("eligibility_outside_bounds")
        if not state_only:
            if metrics.get("neural/finite_activity", 0.0) != 1.0:
                failed.append("non_finite_neural_activity")
            for population, minimum in (
                ("kc", self.config.minimum_kc_active_fraction),
                ("motor", self.config.minimum_motor_active_fraction),
            ):
                activity = metrics.get(f"neural/min_learner_{population}_active_fraction", math.nan)
                self.silent_steps[population] = self.silent_steps[population] + 1 if activity < minimum or not math.isfinite(activity) else 0
                if self.silent_steps[population] >= self.config.silence_patience_steps:
                    failed.append(f"{population}_activity_collapse")
        if failed:
            self.failure = {
                "version": TRAINING_SAFETY_VERSION,
                "status": "SAFETY_STOPPED", "failed": failed,
                "thresholds": asdict(self.config), "silent_steps": dict(self.silent_steps),
                "maximum_eligibility": self.maximum_eligibility,
                "metrics": {key: float(value) if math.isfinite(float(value)) else None for key, value in metrics.items()},
            }
        return self.failure

    def state_dict(self) -> dict[str, Any]:
        return {"silent_steps": dict(self.silent_steps), "failure": self.failure}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.silent_steps = {name: int(state.get("silent_steps", {}).get(name, 0)) for name in ("kc", "motor")}
        self.failure = state.get("failure")
