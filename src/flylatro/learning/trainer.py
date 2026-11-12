"""Exposure-budgeted sequential trainer for internally plastic flies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from typing import Any, Callable, Sequence

import numpy as np

from flylatro.env.balatro_sim import ArrayBalatroEnv, ArrayStep
from flylatro.env.upstream_contract import ActionDict
from flylatro.env.upstream_contract import N_ACTION_TYPES
from flylatro.evaluation.state_hash import hash_observation_row, terminal_hash
from flylatro.learning.agent import LearningResult, PlasticDecision, PlasticFlyAgent
from flylatro.learning.reinforcement import DopaminePulse
from flylatro.learning.reward_schedule import MatchedActionStep
from flylatro.seeds import derive_seed
from flylatro.learning.safety import TRAINING_SAFETY_VERSION, TrainingSafetyConfig, TrainingSafetyMonitor, TrainingSafetyStop


@dataclass(frozen=True, slots=True)
class PlasticTrainingConfig:
    max_environment_decisions: int = 100
    deterministic_motor: bool = False
    plasticity_enabled: bool = True
    checkpoint_every_decisions: int = 100
    base_fly_seed: int = 2001
    episode_history_limit: int = 512

    def __post_init__(self) -> None:
        if self.max_environment_decisions < 1:
            raise ValueError("max_environment_decisions must be positive")
        if self.checkpoint_every_decisions < 1:
            raise ValueError("checkpoint cadence must be positive")
        if self.episode_history_limit < 1:
            raise ValueError("episode history limit must be positive")


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
    episode_history_start: int = 0
    episode_return_sum: float = 0.0
    best_episode_return: float | None = None
    best_episode_ante: int = 0


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
        safety: TrainingSafetyConfig = TrainingSafetyConfig(),
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
        self.last_step: ArrayStep | None = None
        self.last_executed_actions: ActionDict | None = None
        self.last_state_hashes_before: tuple[str, ...] = ()
        self.last_state_hashes_after: tuple[str, ...] = ()
        self.last_scheduled_ante: int | None = None
        self.record_detailed_plasticity = False
        self._motor_columns = np.unique(np.concatenate([
            np.asarray(pool, dtype=np.int64) for pool in self.agent.motor.mapping.pool_indices
        ]))
        self.safety = TrainingSafetyMonitor(safety, maximum_eligibility=agent.plasticity.config.max_eligibility)

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
        # The motor exploration RNG needs a cheap, deterministic, CPU-side
        # decision identifier. Reading the plastic state's `decision_count`
        # tensor would convert a CUDA scalar to a Python int on every decision
        # and synchronize the device. `vector_steps` is the same quantity on
        # this path (one decision per learner per vector step), lives on the
        # host, is checkpointed, and is restored exactly on resume.
        decision_ids = (self.state.vector_steps,) * self.env.num_envs
        decision = self.agent.act(
            self.observations,
            self.masks,
            fly_seeds=fly_seeds,
            deterministic_motor=self.config.deterministic_motor,
            motor_learner_ids=self.training_seeds,
            motor_decision_ids=decision_ids,
            record_eligibility=self.agent.credit is None,
        )
        executed_actions = (
            scheduled.actions
            if scheduled is not None
            else decision.actions
        )
        step = self.env.step(executed_actions)
        if self.agent.credit is not None:
            self.agent.record_choice_activity(decision.neural, executed_actions, self.masks)
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
            detail=self.record_detailed_plasticity,
        )
        self.last_decision = decision
        self.last_learning = learning
        self.last_step = step
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
            episode_return = float(episode.get("r", 0.0))
            episode_ante = int(episode.get("ante", 0))
            self.state.episode_returns.append(episode_return)
            self.state.episode_antes.append(episode_ante)
            self.state.episode_return_sum += episode_return
            self.state.best_episode_return = (episode_return if self.state.best_episode_return is None
                else max(self.state.best_episode_return, episode_return))
            self.state.best_episode_ante = max(self.state.best_episode_ante, episode_ante)
            excess = len(self.state.episode_returns) - self.config.episode_history_limit
            if excess > 0:
                del self.state.episode_returns[:excess]
                del self.state.episode_antes[:excess]
                self.state.episode_history_start += excess
            self.state.wins += bool(episode.get("won", False))
        # Device-resident reductions: routine training never copies a full
        # efficacy, eligibility or reinforcement-trace vector to the host.
        plastic = self.agent.plasticity.routine_metrics()
        action_probabilities = np.asarray(
            self.state.action_type_counts, dtype=np.float64
        )
        action_probabilities /= action_probabilities.sum()
        observed = action_probabilities[action_probabilities > 0]
        action_entropy = -float(np.sum(observed * np.log(observed)))
        neural = decision.neural.health_summary
        if neural is None:
            neural = {
                'kc_active_fraction': float(np.mean(decision.neural.kc_activity > 0)),
                'mbon_mean_activity': float(decision.neural.mbon_activity.mean()),
                'descending_mean_activity': float(decision.neural.descending_activity.mean()),
                'min_learner_kc_active_fraction': float(np.mean(decision.neural.kc_activity > 0, axis=1).min()),
                'finite_activity': float(all(np.isfinite(values).all() for values in (
                    decision.neural.kc_activity, decision.neural.mbon_activity,
                    decision.neural.descending_activity, decision.neural.output_activity))),
            }
        neural = {**neural, 'min_learner_motor_active_fraction': float(
            np.mean(decision.neural.output_activity[:, self._motor_columns] > 0, axis=1).min())}
        return {
            **(self.agent.credit.metrics if self.agent.credit is not None else {}),
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
            "plasticity/mean_efficacy": plastic["mean_efficacy"],
            "plasticity/absolute_change": plastic["absolute_change"],
            "plasticity/mean_absolute_drift": plastic["mean_absolute_drift"],
            "plasticity/max_absolute_drift": plastic["max_absolute_drift"],
            "plasticity/eligibility_mean": plastic["eligibility_mean_absolute"],
            "plasticity/eligibility_max": plastic["eligibility_max_absolute"],
            "plasticity/dopamine_trace_mean_absolute": plastic[
                "reinforcement_trace_mean_absolute"
            ],
            "plasticity/event_changed_synapses": float(
                sum(event.changed_synapses for event in learning.events)
            ),
            "plasticity/event_max_absolute_change": float(
                max(
                    (event.maximum_absolute_change for event in learning.events),
                    default=0.0,
                )
            ),
            "plasticity/eligible_synapses": float(
                sum(event.eligible_synapses for event in learning.events)
            ),
            "reinforcement/appetitive_mean": float(
                np.mean([pulse.appetitive for pulse in learning.pulses])
            ),
            "reinforcement/aversive_mean": float(
                np.mean([pulse.aversive for pulse in learning.pulses])
            ),
            "plasticity/lower_bound_fraction": plastic["lower_bound_fraction"],
            "plasticity/upper_bound_fraction": plastic["upper_bound_fraction"],
            **{f"plasticity/{name}": plastic[name] for name in (
                "finite_state", "efficacy_outside_bounds", "max_learner_bound_fraction", "max_learner_mean_absolute_drift",
                "net_strengthened_fraction", "net_weakened_fraction", "max_learner_near_bound_fraction",
            )},
            "behaviour/action_type_entropy": action_entropy,
            "behaviour/action_types_observed": float(len(observed)),
            "behaviour/dominant_action_fraction": float(
                action_probabilities.max()
            ),
            **{f'neural/{name}': value for name, value in neural.items()},
        }

    def train(
        self,
        *,
        on_step: Callable[["PlasticTrainer", dict[str, float]], None] | None = None,
        stop_after_environment_decisions: int | None = None,
    ) -> PlasticTrainingState:
        target = self.config.max_environment_decisions if stop_after_environment_decisions is None else stop_after_environment_decisions
        if not self.state.environment_decisions <= target <= self.config.max_environment_decisions or target < 1 or target % self.env.num_envs:
            raise ValueError("stage stop must be aligned, not precede restored exposure and not exceed the declared budget")
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
        state_metrics = {f"plasticity/{key}": value for key, value in self.agent.plasticity.routine_metrics().items()}
        state_metrics["training/environment_decisions"] = float(self.state.environment_decisions)
        if self.safety.observe(state_metrics, state_only=True) is not None:
            raise TrainingSafetyStop(self.safety.failure)
        while (
            self.state.environment_decisions
            < target
        ):
            try:
                metrics = self.step()
            except ValueError as error:
                if not self.safety.config.enabled or "finite" not in str(error):
                    raise
                self.safety.failure = {
                    "version": TRAINING_SAFETY_VERSION, "status": "SAFETY_STOPPED",
                    "failed": ["non_finite_neural_or_plastic_input"], "error": str(error),
                    "metrics": {"training/environment_decisions": self.state.environment_decisions},
                }
                raise TrainingSafetyStop(self.safety.failure) from error
            self.safety.observe(metrics)
            if on_step is not None:
                on_step(self, metrics)
            if self.safety.failure is not None:
                raise TrainingSafetyStop(self.safety.failure)
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
            "training_state": asdict(self.state),
            "plastic_state": self.agent.plasticity.state.state_dict(),
            "motor_rng_state": self.agent.motor.rng_state(),
            "observations": self.observations,
            "masks": self.masks,
            "environment_snapshots": self.env.snapshot_all(),
            "training_seeds": self.training_seeds,
            "safety_state": self.safety.state_dict(),
            "rng": rng,
        }

    def load_state_dict(self, values: dict[str, Any]) -> None:
        if tuple(values["training_seeds"]) != self.training_seeds:
            raise ValueError("checkpoint training seeds differ")
        state_type = type(self.agent.plasticity.state)
        restored = (
            state_type.from_state_dict(
                values["plastic_state"], device=str(self.agent.plasticity.state.device)
            )
            if getattr(self.agent.plasticity.state, "is_torch", False)
            else state_type.from_state_dict(values["plastic_state"])
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
            episode_history_start=int(state.get("episode_history_start", 0)),
            episode_return_sum=float(state.get("episode_return_sum", sum(state["episode_returns"]))),
            best_episode_return=state.get("best_episode_return", max(state["episode_returns"], default=None)),
            best_episode_ante=int(state.get("best_episode_ante", max(state["episode_antes"], default=0))),
        )
        excess = len(self.state.episode_returns) - self.config.episode_history_limit
        if excess > 0:
            del self.state.episode_returns[:excess]
            del self.state.episode_antes[:excess]
            self.state.episode_history_start += excess
        self.env.reset(self.training_seeds)
        self.env.restore_all(values["environment_snapshots"])
        self.observations = values["observations"]
        self.masks = values["masks"]
        self.safety.load_state_dict(values.get("safety_state", {}))
