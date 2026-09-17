"""Backend-independent, one-shot vector episode runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from flylatro.agent import FlyAgent
from flylatro.env.contracts import BalatroEnv
from flylatro.env.types import CompositeAction
from flylatro.replay.recorder import JsonlTransitionRecorder, TransitionRecord


@dataclass(frozen=True, slots=True)
class RunSummary:
    episodes: int
    wins: int
    decisions: int
    total_reward: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes


def run_episodes(
    env: BalatroEnv,
    agent: FlyAgent,
    seeds: Sequence[int],
    *,
    recorder: JsonlTransitionRecorder | None = None,
    deterministic_policy: bool = False,
    max_decisions: int = 1_000,
) -> RunSummary:
    """Run exactly one episode per vector slot and record every transition."""

    if max_decisions < env.num_envs:
        raise ValueError("max_decisions is too small for one action per environment")
    observations = env.reset(seeds)
    active = np.ones(env.num_envs, dtype=np.bool_)
    wins = np.zeros(env.num_envs, dtype=np.bool_)
    total_reward = 0.0
    decision_count = 0

    while np.any(active):
        masks = env.legal_action_masks()
        active_indices = [index for index, enabled in enumerate(active) if enabled]
        active_observations = [observations[index] for index in active_indices]
        active_masks = []
        for index in active_indices:
            mask = masks[index]
            if mask is None:
                raise RuntimeError("active environment returned no legal-action mask")
            active_masks.append(mask)
        decisions = agent.act(
            active_observations,
            active_masks,
            deterministic=deterministic_policy,
        )
        if decision_count + len(decisions) > max_decisions:
            raise RuntimeError(
                "run reached max_decisions before all episodes terminated"
            )
        actions: list[CompositeAction | None] = [None] * env.num_envs
        hashes_before: dict[int, str] = {}
        decisions_by_env = {}
        for index, mask, decision in zip(
            active_indices, active_masks, decisions, strict=True
        ):
            action = decision.policy.action
            if not mask.allows(action):
                raise RuntimeError("policy produced an illegal action")
            actions[index] = action
            hashes_before[index] = observations[index].state_hash()
            decisions_by_env[index] = decision

        step = env.step(actions)
        for index in active_indices:
            before = observations[index]
            decision = decisions_by_env[index]
            total_reward += float(step.rewards[index])
            if recorder is not None:
                recorder.record(
                    TransitionRecord(
                        env_index=index,
                        episode_id=before.episode_id,
                        balatro_seed=before.seed,
                        decision_id=before.decision_id,
                        ante=before.ante,
                        round=before.round,
                        game_state=before.phase.value,
                        action=decision.policy.action,
                        reward=float(step.rewards[index]),
                        reward_components=step.reward_components[index],
                        action_probability=decision.policy.action_probability,
                        action_type_probabilities=(
                            decision.policy.action_type_probabilities
                        ),
                        value=decision.policy.value,
                        state_hash_before=hashes_before[index],
                        state_hash_after=step.observations[index].state_hash(),
                        terminated=bool(step.terminated[index]),
                        truncated=bool(step.truncated[index]),
                        fly_total_spikes=decision.fly_total_spikes,
                        fly_active_neurons=decision.fly_active_neurons,
                    )
                )
            if step.terminated[index] or step.truncated[index]:
                active[index] = False
                wins[index] = bool(step.infos[index].get("won", False))
        decision_count += len(decisions)
        observations = step.observations

    return RunSummary(
        episodes=env.num_envs,
        wins=int(wins.sum()),
        decisions=decision_count,
        total_reward=total_reward,
    )
