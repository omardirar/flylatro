"""Dependency-free TOML configuration with conservative development limits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    num_envs: int
    hand_size: int
    blind_target: float
    initial_hands: int
    initial_discards: int


@dataclass(frozen=True, slots=True)
class FlyConfig:
    neuron_count: int
    graph_seed: int
    edge_probability: float
    duration_ms: float
    microbatch_size: int
    readout_count: int


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    seed: int
    deterministic: bool


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed_start: int
    max_decisions: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    environment: EnvironmentConfig
    fly: FlyConfig
    policy: PolicyConfig
    run: RunConfig

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        try:
            config = cls(
                environment=EnvironmentConfig(**raw["environment"]),
                fly=FlyConfig(**raw["fly"]),
                policy=PolicyConfig(**raw["policy"]),
                run=RunConfig(**raw["run"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid Flylatro config {path}: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        env = self.environment
        fly = self.fly
        if env.num_envs < 1 or env.hand_size < 1:
            raise ValueError("environment sizes must be positive")
        if env.blind_target <= 0 or env.initial_hands < 1:
            raise ValueError("mock episode limits must be positive")
        if env.initial_discards < 0:
            raise ValueError("initial_discards cannot be negative")
        if fly.neuron_count < 2 or fly.readout_count < 1:
            raise ValueError("fly and readout sizes must be positive")
        if fly.readout_count > fly.neuron_count:
            raise ValueError("readout_count exceeds neuron_count")
        if fly.duration_ms <= 0 or fly.microbatch_size < 1:
            raise ValueError("fly duration and microbatch must be positive")
        if self.run.max_decisions < env.num_envs:
            raise ValueError("max_decisions must allow at least one vector step")

    def seeds(self) -> tuple[int, ...]:
        return tuple(
            range(self.run.seed_start, self.run.seed_start + self.environment.num_envs)
        )

    def check_development_limits(self, *, allow_large: bool = False) -> None:
        if allow_large:
            return
        reasons = []
        if self.environment.num_envs > 64:
            reasons.append("num_envs > 64")
        if self.fly.neuron_count > 10_000:
            reasons.append("neuron_count > 10,000")
        if self.run.max_decisions > 10_000:
            reasons.append("max_decisions > 10,000")
        if reasons:
            joined = ", ".join(reasons)
            raise ValueError(
                f"development safety limit exceeded ({joined}); pass the explicit "
                "large-run opt-in only on suitable hardware"
            )


def config_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "environment": {
            field: getattr(config.environment, field)
            for field in config.environment.__dataclass_fields__
        },
        "fly": {
            field: getattr(config.fly, field) for field in config.fly.__dataclass_fields__
        },
        "policy": {
            field: getattr(config.policy, field)
            for field in config.policy.__dataclass_fields__
        },
        "run": {
            field: getattr(config.run, field) for field in config.run.__dataclass_fields__
        },
    }

