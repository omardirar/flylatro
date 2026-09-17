"""Fixed-seed evaluation with successful-run and trace identification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from statistics import median
from typing import Any, Sequence

import numpy as np
import torch

from flylatro.env.balatro_sim import (
    ArrayBalatroEnv,
    action_batch_row_to_composite,
)
from flylatro.env.upstream_contract import ObsDict
from flylatro.env.upstream_contract import (
    BLIND_SCORED_OFF,
    GLOBAL_ANTE_OFF,
    GLOBAL_DISCARDS_LEFT,
    GLOBAL_HANDS_LEFT,
    GLOBAL_MONEY,
    GLOBAL_PHASE_OFF,
    GLOBAL_ROUND,
)
from flylatro.fly.processors import ObservationProcessor
from flylatro.policy.torch_structured import (
    TorchStructuredPolicy,
    actions_to_numpy,
    masks_to_torch,
)
from flylatro.seeds import derive_seed


@dataclass(frozen=True, slots=True)
class EvaluationTransition:
    seed: int
    env_index: int
    decision_id: int
    action: dict[str, Any]
    reward: float
    reward_components: dict[str, float]
    value: float
    action_probability: float
    action_type_probabilities: tuple[float, ...]
    state_hash_before: str
    state_hash_after: str
    state_signature_before: dict[str, Any]
    done: bool


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    seed: int
    won: bool
    ante: int
    length: int
    decisions: int
    episode_return: float
    score: float | None
    balatro_seed: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: tuple[EpisodeEvaluation, ...]
    transitions: tuple[EvaluationTransition, ...]

    @property
    def win_rate(self) -> float:
        return sum(episode.won for episode in self.episodes) / len(self.episodes)

    @property
    def mean_ante(self) -> float:
        return float(np.mean([episode.ante for episode in self.episodes]))

    @property
    def median_ante(self) -> float:
        return float(median(episode.ante for episode in self.episodes))

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean([episode.length for episode in self.episodes]))

    @property
    def decision_count(self) -> int:
        return sum(episode.decisions for episode in self.episodes)

    @property
    def successful_seeds(self) -> tuple[int, ...]:
        return tuple(episode.seed for episode in self.episodes if episode.won)

    def metrics(self) -> dict[str, float | None]:
        scores = [episode.score for episode in self.episodes if episode.score is not None]
        return {
            "evaluation/episodes": float(len(self.episodes)),
            "evaluation/win_rate": self.win_rate,
            "evaluation/mean_ante": self.mean_ante,
            "evaluation/median_ante": self.median_ante,
            "evaluation/mean_episode_length": self.mean_episode_length,
            "evaluation/decision_count": float(self.decision_count),
            "evaluation/mean_score": float(np.mean(scores)) if scores else None,
        }


def evaluate_policy(
    env: ArrayBalatroEnv,
    processor: ObservationProcessor,
    policy: TorchStructuredPolicy,
    seeds: Sequence[int],
    *,
    device: str = "cpu",
    deterministic: bool = True,
    max_vector_steps: int = 10_000,
    neural_recorder: object | None = None,
) -> EvaluationResult:
    if len(seeds) != env.num_envs:
        raise ValueError("evaluation needs exactly one seed per environment")
    if max_vector_steps < 1:
        raise ValueError("max_vector_steps must be positive")
    observations, masks = env.reset(seeds)
    run_seeds = tuple(
        str(env.run_seed(index)) if hasattr(env, "run_seed") else str(seeds[index])
        for index in range(env.num_envs)
    )
    active = np.ones(env.num_envs, dtype=np.bool_)
    returns = np.zeros(env.num_envs, dtype=np.float64)
    decisions = np.zeros(env.num_envs, dtype=np.int64)
    episodes: list[EpisodeEvaluation] = []
    transitions: list[EvaluationTransition] = []
    original_training = policy.training
    policy.eval()
    try:
        for vector_step in range(max_vector_steps):
            fly_seeds = tuple(
                derive_seed("evaluation-fly", int(seed), vector_step)
                for seed in seeds
            )
            with torch.no_grad():
                if neural_recorder is not None:
                    record_method = getattr(processor, "process_with_events", None)
                    if record_method is None:
                        raise TypeError("processor does not support neural event recording")
                    features = record_method(
                        observations,
                        fly_seeds=fly_seeds,
                        decision_ids=tuple(int(value) for value in decisions),
                        recorder=neural_recorder,
                    ).to(device)
                else:
                    features = processor.process(
                        observations, fly_seeds=fly_seeds
                    ).to(device)
                output = policy(
                    features,
                    masks_to_torch(masks, device),
                    deterministic=deterministic,
                )
            action_batch = actions_to_numpy(output.actions)
            hashes_before = [
                hash_observation_row(observations, index)
                for index in range(env.num_envs)
            ]
            step = env.step(action_batch)
            for index in range(env.num_envs):
                if not active[index]:
                    continue
                action = action_batch_row_to_composite(action_batch, index)
                probability = float(
                    torch.exp(output.log_probability[index]).detach().cpu()
                )
                after_hash = (
                    terminal_hash(step.infos[index])
                    if step.dones[index]
                    else hash_observation_row(step.observations, index)
                )
                transitions.append(
                    EvaluationTransition(
                        seed=int(seeds[index]),
                        env_index=index,
                        decision_id=int(decisions[index]),
                        action=action.to_payload(),
                        reward=float(step.rewards[index]),
                        reward_components=dict(
                            step.infos[index].get("reward_components", {})
                        ),
                        value=float(output.value[index].detach().cpu()),
                        action_probability=probability,
                        action_type_probabilities=tuple(
                            float(value)
                            for value in output.action_type_probabilities[index]
                            .detach()
                            .cpu()
                        ),
                        state_hash_before=hashes_before[index],
                        state_hash_after=after_hash,
                        state_signature_before=observation_state_signature(
                            observations, index
                        ),
                        done=bool(step.dones[index]),
                    )
                )
                returns[index] += float(step.rewards[index])
                decisions[index] += 1
                if step.dones[index]:
                    episode = step.infos[index].get("episode", {})
                    episodes.append(
                        EpisodeEvaluation(
                            seed=int(seeds[index]),
                            won=bool(episode.get("won", False)),
                            ante=int(episode.get("ante", 0)),
                            length=int(episode.get("l", decisions[index])),
                            decisions=int(decisions[index]),
                            episode_return=float(episode.get("r", returns[index])),
                            score=(
                                float(episode["score"])
                                if "score" in episode
                                else None
                            ),
                            balatro_seed=run_seeds[index],
                        )
                    )
                    active[index] = False
            observations, masks = step.observations, step.masks
            if not active.any():
                break
        else:
            unfinished = [int(seeds[i]) for i in np.flatnonzero(active)]
            raise RuntimeError(
                f"evaluation exceeded max_vector_steps; unfinished seeds: {unfinished}"
            )
    finally:
        policy.train(original_training)
    seed_order = {int(seed): index for index, seed in enumerate(seeds)}
    episodes.sort(key=lambda episode: seed_order[episode.seed])
    return EvaluationResult(tuple(episodes), tuple(transitions))


def hash_observation_row(observations: ObsDict, index: int) -> str:
    digest = hashlib.sha256()
    for key in sorted(observations):
        row = np.ascontiguousarray(observations[key][index])
        digest.update(key.encode())
        digest.update(str(row.dtype).encode())
        digest.update(np.asarray(row.shape, dtype="<i8").tobytes())
        digest.update(row.tobytes())
    return digest.hexdigest()


def terminal_hash(info: dict[str, Any]) -> str:
    import json

    terminal = {
        "episode": info.get("episode", {}),
        "seed": info.get("seed"),
        "terminal": True,
    }
    return hashlib.sha256(
        json.dumps(terminal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def observation_state_signature(
    observations: ObsDict, index: int
) -> dict[str, int | float | str]:
    """Decode the state fields shared by the simulator and live-game API."""

    global_values = observations["global"][index]
    phases = (
        "BLIND_SELECT", "SELECTING_HAND", "ROUND_EVAL", "SHOP",
        "SMODS_BOOSTER_OPENED", "GAME_OVER",
    )
    phase = int(np.argmax(global_values[GLOBAL_PHASE_OFF : GLOBAL_PHASE_OFF + 6]))
    ante = int(np.argmax(global_values[GLOBAL_ANTE_OFF : GLOBAL_ANTE_OFF + 9])) + 1
    money_encoded = float(global_values[GLOBAL_MONEY])
    money = int(round(np.sign(money_encoded) * np.expm1(abs(money_encoded))))
    return {
        "state": phases[phase],
        "ante_num": ante,
        "round_num": int(round(float(global_values[GLOBAL_ROUND]) * 24.0)),
        "money": money,
        "hands_left": int(round(float(global_values[GLOBAL_HANDS_LEFT]) * 4.0)),
        "discards_left": int(
            round(float(global_values[GLOBAL_DISCARDS_LEFT]) * 4.0)
        ),
        "chips": int(
            round(np.expm1(float(observations["blind"][index, BLIND_SCORED_OFF])))
        ),
    }
