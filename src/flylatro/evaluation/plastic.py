"""Frozen evaluation for a learned plastic fly with no synaptic updates."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

import numpy as np

from flylatro.env.balatro_sim import ArrayBalatroEnv, action_batch_row_to_composite
from flylatro.evaluation.state_hash import hash_observation_row, terminal_hash
from flylatro.learning.agent import PlasticFlyAgent
from flylatro.seeds import derive_seed


@dataclass(frozen=True, slots=True)
class PlasticEpisodeEvaluation:
    seed: int
    won: bool
    ante: int
    decisions: int
    episode_return: float
    score: float | None
    balatro_seed: str | None = None


@dataclass(frozen=True, slots=True)
class PlasticEvaluationTransition:
    seed: int
    decision_id: int
    action: dict[str, Any]
    reward: float
    reward_components: dict[str, float]
    state_hash_before: str
    state_hash_after: str
    dopamine_appetitive: float
    dopamine_aversive: float
    done: bool


@dataclass(frozen=True, slots=True)
class PlasticEvaluationResult:
    episodes: tuple[PlasticEpisodeEvaluation, ...]
    transitions: tuple[PlasticEvaluationTransition, ...]
    frozen_weight_sha256: str

    def metrics(self) -> dict[str, float]:
        scores = [episode.score for episode in self.episodes if episode.score is not None]
        antes = [episode.ante for episode in self.episodes]
        return {
            "evaluation/episodes": float(len(self.episodes)),
            "evaluation/win_rate": float(np.mean([episode.won for episode in self.episodes])),
            "evaluation/mean_ante": float(np.mean(antes)),
            "evaluation/median_ante": float(median(antes)),
            "evaluation/mean_episode_length": float(
                np.mean([episode.decisions for episode in self.episodes])
            ),
            "evaluation/decision_count": float(
                sum(episode.decisions for episode in self.episodes)
            ),
            "evaluation/mean_score": float(np.mean(scores)) if scores else 0.0,
        }


def evaluate_plastic_fly(
    env: ArrayBalatroEnv,
    agent: PlasticFlyAgent,
    seeds: Sequence[int],
    *,
    max_vector_steps: int = 10_000,
    neural_recorder: object | None = None,
) -> PlasticEvaluationResult:
    if len(seeds) != env.num_envs or env.num_envs != agent.plasticity.state.learners:
        raise ValueError("evaluation needs one seed per independent frozen fly")
    before_hash = agent.plasticity.state.weight_sha256
    before_eligibility = agent.plasticity.state.eligibility.copy()
    before_dopamine = agent.plasticity.state.dopamine.copy()
    observations, masks = env.reset(seeds)
    run_seeds = tuple(
        str(env.run_seed(index)) if hasattr(env, "run_seed") else str(seeds[index])
        for index in range(env.num_envs)
    )
    active = np.ones(env.num_envs, dtype=np.bool_)
    returns = np.zeros(env.num_envs, dtype=np.float64)
    decisions = np.zeros(env.num_envs, dtype=np.int64)
    episodes: list[PlasticEpisodeEvaluation] = []
    transitions: list[PlasticEvaluationTransition] = []
    for vector_step in range(max_vector_steps):
        fly_seeds = tuple(
            derive_seed("plastic-evaluation-fly", int(seed), vector_step)
            for seed in seeds
        )
        output = agent.act(
            observations,
            masks,
            fly_seeds=fly_seeds,
            deterministic_motor=True,
            record_eligibility=False,
        )
        hashes_before = [
            hash_observation_row(observations, row) for row in range(env.num_envs)
        ]
        step = env.step(output.actions)
        pulses = tuple(
            agent.reinforcement.map(
                dict(info.get("reward_components", {})), info
            )
            for info in step.infos
        )
        for row in range(env.num_envs):
            if not active[row]:
                continue
            if neural_recorder is not None:
                _record_neural_state(
                    neural_recorder,
                    agent,
                    output.neural,
                    row=row,
                    decision_id=int(decisions[row]),
                    pulse=pulses[row],
                )
            done = bool(step.dones[row])
            transitions.append(
                PlasticEvaluationTransition(
                    seed=int(seeds[row]),
                    decision_id=int(decisions[row]),
                    action=action_batch_row_to_composite(output.actions, row).to_payload(),
                    reward=float(step.rewards[row]),
                    reward_components=dict(
                        step.infos[row].get("reward_components", {})
                    ),
                    state_hash_before=hashes_before[row],
                    state_hash_after=(
                        terminal_hash(step.infos[row])
                        if done
                        else hash_observation_row(step.observations, row)
                    ),
                    dopamine_appetitive=pulses[row].appetitive,
                    dopamine_aversive=pulses[row].aversive,
                    done=done,
                )
            )
            returns[row] += float(step.rewards[row])
            decisions[row] += 1
            if done:
                episode = step.infos[row].get("episode", {})
                episodes.append(
                    PlasticEpisodeEvaluation(
                        seed=int(seeds[row]),
                        won=bool(episode.get("won", False)),
                        ante=int(episode.get("ante", 0)),
                        decisions=int(decisions[row]),
                        episode_return=float(episode.get("r", returns[row])),
                        score=(
                            float(episode["score"]) if "score" in episode else None
                        ),
                        balatro_seed=run_seeds[row],
                    )
                )
                active[row] = False
        observations, masks = step.observations, step.masks
        if not active.any():
            break
    else:
        raise RuntimeError("frozen evaluation exceeded max_vector_steps")
    if agent.plasticity.state.weight_sha256 != before_hash:
        raise RuntimeError("evaluation changed frozen plastic weights")
    np.testing.assert_array_equal(agent.plasticity.state.eligibility, before_eligibility)
    np.testing.assert_array_equal(agent.plasticity.state.dopamine, before_dopamine)
    return PlasticEvaluationResult(
        episodes=tuple(episodes),
        transitions=tuple(transitions),
        frozen_weight_sha256=before_hash,
    )


def _record_neural_state(
    recorder: object,
    agent: PlasticFlyAgent,
    neural: object,
    *,
    row: int,
    decision_id: int,
    pulse: object,
) -> None:
    processor = agent.processor

    def record_population(root_attr: str, activity: np.ndarray, role: str) -> None:
        roots = np.asarray(getattr(processor, root_attr), dtype=np.int64)
        values = np.asarray(activity[row], dtype=np.float32)
        if len(roots) != len(values):
            raise RuntimeError(f"recording population {role} is misaligned")
        recorder.record(
            decision_id=decision_id,
            times_ms=np.zeros(len(roots), dtype=np.float32),
            neuron_ids=roots,
            roles=np.full(len(roots), role, dtype=object),
            activities=values,
            event_kind="activity",
        )

    record_population("kc_root_ids", neural.kc_activity, "kc")
    record_population("mbon_root_ids", neural.mbon_activity, "mbon")
    record_population("dan_root_ids", neural.dan_activity, "dan")
    record_population(
        "descending_root_ids", neural.descending_activity, "descending"
    )
    for root_attr, magnitude in (
        ("pam_root_ids", pulse.appetitive),
        ("ppl1_root_ids", pulse.aversive),
    ):
        roots = np.asarray(getattr(processor, root_attr), dtype=np.int64)
        if magnitude > 0 and len(roots):
            recorder.record(
                decision_id=decision_id,
                times_ms=np.full(len(roots), neural.duration_ms, dtype=np.float32),
                neuron_ids=roots,
                roles=np.full(len(roots), "dan", dtype=object),
                activities=np.full(len(roots), magnitude, dtype=np.float32),
                event_kind="dopamine",
            )
    delta = (
        agent.plasticity.state.efficacy[row]
        - agent.plasticity.state.initial_efficacy[row]
    )
    changed = np.flatnonzero(delta)
    if len(changed):
        selected = changed[np.argsort(np.abs(delta[changed]))[-256:]]
        recorder.record(
            decision_id=decision_id,
            times_ms=np.full(len(selected), neural.duration_ms, dtype=np.float32),
            neuron_ids=agent.plasticity.topology.post_root_ids[selected],
            roles=np.full(len(selected), "plasticity", dtype=object),
            activities=delta[selected].astype(np.float32),
            event_kind="weight_snapshot",
        )
