"""Exposure-budgeted sequential trainer for internally plastic flies."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Callable, Sequence

import numpy as np

from flylatro.env.balatro_sim import ArrayBalatroEnv
from flylatro.env.upstream_contract import ActionDict
from flylatro.env.upstream_contract import N_ACTION_TYPES
from flylatro.evaluation.state_hash import hash_observation_row, terminal_hash
from flylatro.learning.agent import LearningResult, PlasticDecision, PlasticFlyAgent
from flylatro.learning.reinforcement import DopaminePulse
from flylatro.learning.reward_schedule import MatchedActionStep
from flylatro.seeds import derive_seed


@dataclass(frozen=True, slots=True)
class PlasticTrainingConfig:
    max_environment_decisions: int = 100
    deterministic_motor: bool = False
    plasticity_enabled: bool = True
    checkpoint_every_decisions: int = 100
    base_fly_seed: int = 2001

    def __post_init__(self) -> None:
        if self.max_environment_decisions < 1:
            raise ValueError("max_environment_decisions must be positive")
        if self.checkpoint_every_decisions < 1:
            raise ValueError("checkpoint cadence must be positive")


@dataclass(slots=True)
class PlasticTrainingState:
    environment_decisions: int = 0
    vector_steps: int = 0
    completed_episodes: int = 0
    episode_returns: list[float] = field(default_factory=list)
    episode_antes: list[int] = field(default_factory=list)
    wins: int = 0
    plasticity_events: int = 0
    action_type_counts: list[int] = field(
        default_factory=lambda: [0] * N_ACTION_TYPES
    )


class PlasticTrainer:
    def __init__(
        self,
        env: ArrayBalatroEnv,
        agent: PlasticFlyAgent,
        config: PlasticTrainingConfig,
        *,
        training_seeds: Sequence[int],
        dopamine_schedule: Sequence[DopaminePulse] | None = None,
        action_schedule: Sequence[MatchedActionStep] | None = None,
    ) -> None:
        if len(training_seeds) != env.num_envs:
            raise ValueError("one training seed is required per environment")
        if env.num_envs != agent.plasticity.state.learners:
            raise ValueError("each environment must have an independent fly state")
        if config.max_environment_decisions % env.num_envs:
            raise ValueError(
                "exposure budget must be divisible by the number of independent flies"
            )
        self.env = env
        self.agent = agent
        self.config = config
        self.training_seeds = tuple(int(seed) for seed in training_seeds)
        self.dopamine_schedule = (
            tuple(dopamine_schedule) if dopamine_schedule is not None else None
        )
        self.action_schedule = (
            action_schedule if action_schedule is not None else None
        )
        self.observations, self.masks = env.reset(self.training_seeds)
        self.state = PlasticTrainingState()
        self.last_decision: PlasticDecision | None = None
        self.last_learning: LearningResult | None = None
        self.last_executed_actions: ActionDict | None = None
        self.last_state_hashes_before: tuple[str, ...] = ()
        self.last_state_hashes_after: tuple[str, ...] = ()
        self.last_scheduled_ante: int | None = None
        self.record_sparse_changes = False

    def step(self) -> dict[str, float]:
        scheduled = (
            self.action_schedule[self.state.vector_steps]
            if self.action_schedule is not None
            else None
        )
        if scheduled is not None and scheduled.curriculum_ante is not None:
            self.env.set_win_ante(scheduled.curriculum_ante)
        self.last_scheduled_ante = (
            scheduled.curriculum_ante if scheduled is not None else None
        )
        hashes_before = tuple(
            hash_observation_row(self.observations, row)
            for row in range(self.env.num_envs)
        )
        if scheduled is not None and scheduled.state_hashes_before:
            if hashes_before != scheduled.state_hashes_before:
                raise RuntimeError("matched action schedule diverged before environment step")
        fly_seeds = tuple(
            derive_seed(
                "plastic-fly-decision",
                self.config.base_fly_seed + learner,
                self.state.vector_steps,
            )
            for learner in range(self.env.num_envs)
        )
        decision = self.agent.act(
            self.observations,
            self.masks,
            fly_seeds=fly_seeds,
            deterministic_motor=self.config.deterministic_motor,
        )
        executed_actions = (
            scheduled.actions
            if scheduled is not None
            else decision.actions
        )
        step = self.env.step(executed_actions)
        hashes_after = tuple(
            terminal_hash(step.infos[row])
            if bool(step.dones[row])
            else hash_observation_row(step.observations, row)
            for row in range(self.env.num_envs)
        )
        if scheduled is not None and scheduled.state_hashes_after:
            if hashes_after != scheduled.state_hashes_after:
                raise RuntimeError("matched action schedule diverged after environment step")
        override = None
        if self.dopamine_schedule is not None:
            start = self.state.environment_decisions
            stop = start + self.env.num_envs
            override = self.dopamine_schedule[start:stop]
        learning = self.agent.learn(
            step.infos,
            plasticity_enabled=self.config.plasticity_enabled,
            override_pulses=override,
            include_sparse_changes=self.record_sparse_changes,
        )
        self.last_decision = decision
        self.last_learning = learning
        self.last_executed_actions = executed_actions
        self.last_state_hashes_before = hashes_before
        self.last_state_hashes_after = hashes_after
        self.observations, self.masks = step.observations, step.masks
        self.state.vector_steps += 1
        self.state.environment_decisions += self.env.num_envs
        self.state.plasticity_events += sum(
            event.changed_synapses > 0 for event in learning.events
        )
        action_counts = np.bincount(
            np.asarray(executed_actions["action_type"], dtype=np.int64),
            minlength=N_ACTION_TYPES,
        )
        self.state.action_type_counts = (
            np.asarray(self.state.action_type_counts, dtype=np.int64) + action_counts
        ).tolist()
        for info in step.infos:
            episode = info.get("episode")
            if not isinstance(episode, dict):
                continue
            self.state.completed_episodes += 1
            self.state.episode_returns.append(float(episode.get("r", 0.0)))
            self.state.episode_antes.append(int(episode.get("ante", 0)))
            self.state.wins += bool(episode.get("won", False))
        efficacy = self.agent.plasticity.state.efficacy
        eligibility = self.agent.plasticity.state.eligibility
        dopamine = self.agent.plasticity.state.dopamine
        action_probabilities = np.asarray(
            self.state.action_type_counts, dtype=np.float64
        )
        action_probabilities /= action_probabilities.sum()
        observed = action_probabilities[action_probabilities > 0]
        action_entropy = -float(np.sum(observed * np.log(observed)))
        return {
            "training/environment_decisions": float(self.state.environment_decisions),
            "training/completed_episodes": float(self.state.completed_episodes),
            "training/win_rate": (
                self.state.wins / self.state.completed_episodes
                if self.state.completed_episodes
                else 0.0
            ),
            "plasticity/changed_event_count": float(self.state.plasticity_events),
            "plasticity/event_absolute_change": float(
                sum(event.absolute_change for event in learning.events)
            ),
            "plasticity/mean_efficacy": float(efficacy.mean()),
            "plasticity/absolute_change": float(
                np.abs(efficacy - self.agent.plasticity.state.initial_efficacy).sum()
            ),
            "plasticity/eligibility_mean": float(np.abs(eligibility).mean()),
            "plasticity/dopamine_trace_mean_absolute": float(
                np.abs(dopamine).mean()
            ),
            "reinforcement/appetitive_mean": float(
                np.mean([pulse.appetitive for pulse in learning.pulses])
            ),
            "reinforcement/aversive_mean": float(
                np.mean([pulse.aversive for pulse in learning.pulses])
            ),
            "plasticity/lower_bound_fraction": float(
                np.mean(efficacy <= self.agent.plasticity.config.min_efficacy)
            ),
            "plasticity/upper_bound_fraction": float(
                np.mean(efficacy >= self.agent.plasticity.config.max_efficacy)
            ),
            "behaviour/action_type_entropy": action_entropy,
            "behaviour/action_types_observed": float(len(observed)),
            "behaviour/dominant_action_fraction": float(
                action_probabilities.max()
            ),
            "neural/kc_active_fraction": float(
                np.mean(decision.neural.kc_activity > 0)
            ),
            "neural/mbon_mean_activity": float(decision.neural.mbon_activity.mean()),
            "neural/descending_mean_activity": float(
                decision.neural.descending_activity.mean()
            ),
        }

    def train(
        self,
        *,
        on_step: Callable[["PlasticTrainer", dict[str, float]], None] | None = None,
    ) -> PlasticTrainingState:
        if (
            self.dopamine_schedule is not None
            and len(self.dopamine_schedule) < self.config.max_environment_decisions
        ):
            raise ValueError(
                "shuffled dopamine schedule is shorter than the exposure budget"
            )
        required_steps = self.config.max_environment_decisions // self.env.num_envs
        if self.action_schedule is not None and len(self.action_schedule) < required_steps:
            raise ValueError("matched action schedule is shorter than exposure budget")
        while (
            self.state.environment_decisions
            < self.config.max_environment_decisions
        ):
            metrics = self.step()
            if on_step is not None:
                on_step(self, metrics)
        return self.state

    def state_dict(self) -> dict[str, Any]:
        if not hasattr(self.env, "snapshot_all"):
            raise TypeError("environment does not support exact snapshots")
        rng: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        try:
            import torch
        except ImportError:
            pass
        else:
            rng["torch"] = torch.get_rng_state()
            rng["torch_cuda"] = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            )
        return {
            "training_state": {
                "environment_decisions": self.state.environment_decisions,
                "vector_steps": self.state.vector_steps,
                "completed_episodes": self.state.completed_episodes,
                "episode_returns": list(self.state.episode_returns),
                "episode_antes": list(self.state.episode_antes),
                "wins": self.state.wins,
                "plasticity_events": self.state.plasticity_events,
                "action_type_counts": list(self.state.action_type_counts),
            },
            "plastic_state": self.agent.plasticity.state.state_dict(),
            "motor_rng_state": self.agent.motor.rng_state(),
            "observations": self.observations,
            "masks": self.masks,
            "environment_snapshots": self.env.snapshot_all(),
            "training_seeds": self.training_seeds,
            "rng": rng,
        }

    def load_state_dict(self, values: dict[str, Any]) -> None:
        if tuple(values["training_seeds"]) != self.training_seeds:
            raise ValueError("checkpoint training seeds differ")
        restored = type(self.agent.plasticity.state).from_state_dict(
            values["plastic_state"]
        )
        if restored.efficacy.shape != self.agent.plasticity.state.efficacy.shape:
            raise ValueError("checkpoint plastic state shape differs")
        self.agent.plasticity.state = restored
        self.agent.motor.load_rng_state(values["motor_rng_state"])
        random.setstate(values["rng"]["python"])
        np.random.set_state(values["rng"]["numpy"])
        if "torch" in values["rng"]:
            import torch

            torch.set_rng_state(values["rng"]["torch"].cpu())
            if torch.cuda.is_available() and values["rng"].get("torch_cuda"):
                torch.cuda.set_rng_state_all(values["rng"]["torch_cuda"])
        state = values["training_state"]
        self.state = PlasticTrainingState(
            environment_decisions=int(state["environment_decisions"]),
            vector_steps=int(state["vector_steps"]),
            completed_episodes=int(state["completed_episodes"]),
            episode_returns=list(state["episode_returns"]),
            episode_antes=list(state["episode_antes"]),
            wins=int(state["wins"]),
            plasticity_events=int(state["plasticity_events"]),
            action_type_counts=[
                int(value)
                for value in state.get("action_type_counts", [0] * N_ACTION_TYPES)
            ],
        )
        self.env.reset(self.training_seeds)
        self.env.restore_all(values["environment_snapshots"])
        self.observations = values["observations"]
        self.masks = values["masks"]
