"""Non-imitation baselines sharing Flylatro's action contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from typing import Callable, Sequence

from flylatro.env.upstream_contract import (
    HAND_MAX,
    MAX_CARD_PICKS,
    ActionDict,
    MaskDict,
    UpstreamActionType,
    empty_action_batch,
)
from flylatro.env.balatro_sim import (
    ArrayBalatroEnv,
    action_batch_row_to_composite,
)
from flylatro.evaluation.evaluator import (
    EpisodeEvaluation,
    EvaluationResult,
    EvaluationTransition,
    hash_observation_row,
    observation_state_signature,
    terminal_hash,
)


class ExperimentCondition(str, Enum):
    REAL_CONNECTOME = "real_connectome"
    SHUFFLED_CONNECTOME = "shuffled_connectome"
    CONVENTIONAL = "conventional"
    RANDOM_LEGAL = "random_legal"
    HEURISTIC_EXTERNAL = "heuristic_external"


@dataclass(frozen=True, slots=True)
class RandomPolicyOutput:
    actions: ActionDict
    action_probabilities: np.ndarray


class RandomLegalPolicy:
    """Uniform rule-respecting floor; never consumes heuristic decisions."""

    version = "random-legal-v1"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, masks: MaskDict) -> RandomPolicyOutput:
        count = masks["action_type_mask"].shape[0]
        actions = empty_action_batch(count)
        probabilities = np.ones(count, dtype=np.float64)
        for row in range(count):
            legal_types = np.flatnonzero(masks["action_type_mask"][row])
            if not legal_types.size:
                raise ValueError("environment emitted no legal action type")
            action_type = int(self.rng.choice(legal_types))
            actions["action_type"][row] = action_type
            probabilities[row] *= 1.0 / len(legal_types)
            if action_type in (
                int(UpstreamActionType.PLAY_HAND),
                int(UpstreamActionType.DISCARD),
                int(UpstreamActionType.USE_CONSUMABLE),
            ):
                legal_cards = np.flatnonzero(masks["card_select_mask"][row])
                minimum = (
                    1
                    if action_type
                    in (
                        int(UpstreamActionType.PLAY_HAND),
                        int(UpstreamActionType.DISCARD),
                    )
                    else 0
                )
                maximum = min(MAX_CARD_PICKS, len(legal_cards))
                card_count = int(self.rng.integers(minimum, maximum + 1))
                actions["n_cards"][row] = card_count
                probabilities[row] *= 1.0 / (maximum - minimum + 1)
                if card_count:
                    selected = self.rng.choice(
                        legal_cards, size=card_count, replace=False
                    )
                    actions["cards"][row, :card_count] = selected
            target_specs = (
                (
                    UpstreamActionType.SELL_JOKER,
                    "joker_target",
                    "joker_target_mask",
                ),
                (
                    UpstreamActionType.USE_CONSUMABLE,
                    "consumable_target",
                    "consumable_target_mask",
                ),
                (
                    UpstreamActionType.SELL_CONSUMABLE,
                    "consumable_target",
                    "consumable_target_mask",
                ),
                (UpstreamActionType.BUY_SHOP, "shop_target", "shop_target_mask"),
                (UpstreamActionType.PICK_PACK, "pack_target", "pack_target_mask"),
            )
            for required_type, action_key, mask_key in target_specs:
                if action_type != int(required_type):
                    continue
                legal_targets = np.flatnonzero(masks[mask_key][row])
                if not legal_targets.size:
                    raise ValueError("required target head has no legal choice")
                actions[action_key][row] = int(self.rng.choice(legal_targets))
                probabilities[row] *= 1.0 / len(legal_targets)
        return RandomPolicyOutput(actions, probabilities)


class ExternalHeuristicBaseline:
    """Explicit boundary around upstream bot actions.

    This object is intentionally incompatible with PPO training. It may only
    be constructed by evaluation tooling with an adapter exposing the optional
    ``bot_actions`` method.
    """

    version = "upstream-heuristic-external-v1"

    def __init__(self, adapter: object) -> None:
        raw = getattr(adapter, "_env", None)
        if raw is None or not hasattr(raw, "bot_actions"):
            raise TypeError("adapter does not expose the external heuristic")
        self._raw = raw

    def act(self) -> ActionDict:
        return self._raw.bot_actions()


def evaluate_action_baseline(
    env: ArrayBalatroEnv,
    seeds: Sequence[int],
    action_source: Callable[[MaskDict], RandomPolicyOutput],
    *,
    max_vector_steps: int = 10_000,
) -> EvaluationResult:
    """Evaluate a non-learning action source under the standard protocol."""

    if len(seeds) != env.num_envs:
        raise ValueError("one baseline seed is required per environment")
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
    for _ in range(max_vector_steps):
        output = action_source(masks)
        before = [hash_observation_row(observations, row) for row in range(env.num_envs)]
        step = env.step(output.actions)
        for row in range(env.num_envs):
            if not active[row]:
                continue
            action = action_batch_row_to_composite(output.actions, row)
            type_probabilities = np.zeros(25, dtype=np.float64)
            legal_types = np.flatnonzero(masks["action_type_mask"][row])
            type_probabilities[legal_types] = 1.0 / len(legal_types)
            after = (
                terminal_hash(step.infos[row]) if step.dones[row]
                else hash_observation_row(step.observations, row)
            )
            transitions.append(
                EvaluationTransition(
                    seed=int(seeds[row]), env_index=row,
                    decision_id=int(decisions[row]), action=action.to_payload(),
                    reward=float(step.rewards[row]),
                    reward_components=dict(step.infos[row].get("reward_components", {})),
                    value=0.0,
                    action_probability=float(output.action_probabilities[row]),
                    action_type_probabilities=tuple(type_probabilities),
                    state_hash_before=before[row], state_hash_after=after,
                    state_signature_before=observation_state_signature(
                        observations, row
                    ),
                    done=bool(step.dones[row]),
                )
            )
            returns[row] += float(step.rewards[row])
            decisions[row] += 1
            if step.dones[row]:
                episode = step.infos[row].get("episode", {})
                episodes.append(
                    EpisodeEvaluation(
                        seed=int(seeds[row]), won=bool(episode.get("won", False)),
                        ante=int(episode.get("ante", 0)),
                        length=int(episode.get("l", decisions[row])),
                        decisions=int(decisions[row]),
                        episode_return=float(episode.get("r", returns[row])),
                        score=float(episode["score"]) if "score" in episode else None,
                        balatro_seed=run_seeds[row],
                    )
                )
                active[row] = False
        observations, masks = step.observations, step.masks
        if not active.any():
            break
    else:
        raise RuntimeError("baseline evaluation exceeded max_vector_steps")
    order = {int(seed): index for index, seed in enumerate(seeds)}
    episodes.sort(key=lambda item: order[item.seed])
    return EvaluationResult(tuple(episodes), tuple(transitions))


def heuristic_action_source(baseline: ExternalHeuristicBaseline):
    def source(masks: MaskDict) -> RandomPolicyOutput:
        actions = baseline.act()
        return RandomPolicyOutput(
            actions=actions,
            action_probabilities=np.ones(masks["action_type_mask"].shape[0]),
        )

    return source
