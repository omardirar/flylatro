"""Cheap auto-reset array environment matching the real simulator contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from flylatro.env.balatro_sim import ArrayStep
from flylatro.env.upstream_contract import (
    BLIND_REQ_OFF,
    BLIND_SCORED_OFF,
    GLOBAL_ANTE_OFF,
    GLOBAL_HANDS_LEFT,
    GLOBAL_PHASE_OFF,
    GLOBAL_ROUND,
    GLOBAL_DISCARDS_LEFT,
    MASK_SPEC,
    OBS_SPEC,
    ActionDict,
    MaskDict,
    ObsDict,
    UpstreamActionType,
    validate_batch,
)


@dataclass(slots=True)
class _State:
    seed: int
    rng: np.random.Generator
    cards: np.ndarray
    score: float = 0.0
    hands: int = 3
    discards: int = 2
    length: int = 0
    episode_return: float = 0.0


class MockArrayBalatroEnv:
    simulator_version = "mock-array-balatro-v1"

    def __init__(
        self,
        num_envs: int,
        *,
        blind_target: float = 25.0,
        initial_hands: int = 3,
        initial_discards: int = 2,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.blind_target = blind_target
        self.initial_hands = initial_hands
        self.initial_discards = initial_discards
        self._states: list[_State] = []
        self._next_seed = 0
        self._beta = 1.0
        self._win_ante = 1

    def reset(self, seeds: Sequence[int]) -> tuple[ObsDict, MaskDict]:
        if len(seeds) != self.num_envs:
            raise ValueError("one seed is required per environment")
        self._states = [self._new_state(int(seed)) for seed in seeds]
        self._next_seed = max(int(seed) for seed in seeds) + 1
        return self._observe(), self._masks()

    def step(self, actions: ActionDict) -> ArrayStep:
        self._require_reset()
        validate_batch(
            {
                "action_type": ((), np.dtype(np.int64)),
                "cards": ((5,), np.dtype(np.int64)),
                "n_cards": ((), np.dtype(np.int64)),
                "joker_target": ((), np.dtype(np.int64)),
                "consumable_target": ((), np.dtype(np.int64)),
                "shop_target": ((), np.dtype(np.int64)),
                "pack_target": ((), np.dtype(np.int64)),
            },
            actions,
            self.num_envs,
            "actions",
        )
        masks = self._masks()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        for index, state in enumerate(self._states):
            action_type = int(actions["action_type"][index])
            if not masks["action_type_mask"][index, action_type]:
                raise ValueError("mock received an illegal action type")
            count = int(actions["n_cards"][index])
            cards = actions["cards"][index, :count]
            if count < 1 or count > 5 or len(set(cards.tolist())) != count:
                raise ValueError("mock received an illegal card selection")
            if not masks["card_select_mask"][index, cards].all():
                raise ValueError("mock received an unavailable card")
            components = {"blind_progress": 0.0, "blind_clear": 0.0, "win": 0.0}
            if action_type == int(UpstreamActionType.PLAY_HAND):
                points = float(state.cards[cards].sum())
                remaining = max(0.0, self.blind_target - state.score)
                progress = min(points, remaining) / self.blind_target
                state.score += points
                state.hands -= 1
                components["blind_progress"] = self._beta * progress
            elif action_type == int(UpstreamActionType.DISCARD):
                state.discards -= 1
            else:
                raise ValueError("mock only supports play and discard")
            self._replace_cards(state, cards)
            state.length += 1
            won = state.score >= self.blind_target
            lost = state.hands <= 0 and not won
            if won:
                components["blind_clear"] = self._beta * 0.575
                components["win"] = 15.0
            reward = float(sum(components.values()))
            state.episode_return += reward
            rewards[index] = reward
            dones[index] = won or lost
            if dones[index]:
                infos.append(
                    {
                        "ante_cleared": bool(won),
                        "episode": {
                            "r": state.episode_return,
                            "l": state.length,
                            "ante": 2 if won else 1,
                            "won": won,
                        },
                        "reward_components": components,
                        "seed": state.seed,
                    }
                )
                self._states[index] = self._new_state(self._next_seed)
                self._next_seed += 1
            else:
                infos.append({"reward_components": components, "seed": state.seed})
        return ArrayStep(self._observe(), self._masks(), rewards, dones, tuple(infos))

    def set_shaping_beta(self, beta: float) -> None:
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        self._beta = beta

    def set_win_ante(self, win_ante: int) -> None:
        if not 1 <= win_ante <= 8:
            raise ValueError("win_ante must be in [1, 8]")
        self._win_ante = win_ante

    def snapshot(self, env_index: int) -> dict[str, Any]:
        return deepcopy(
            {
                "state": self._states[env_index],
                "next_seed": self._next_seed,
                "beta": self._beta,
                "win_ante": self._win_ante,
            }
        )

    def restore(self, env_index: int, snapshot: dict[str, Any]) -> None:
        self._states[env_index] = deepcopy(snapshot["state"])
        self._next_seed = int(snapshot["next_seed"])
        self._beta = float(snapshot["beta"])
        self._win_ante = int(snapshot["win_ante"])

    def snapshot_all(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.snapshot(index) for index in range(self.num_envs))

    def restore_all(self, snapshots: Sequence[dict[str, Any]]) -> None:
        if len(snapshots) != self.num_envs:
            raise ValueError("snapshot count does not match num_envs")
        for index, snapshot in enumerate(snapshots):
            self.restore(index, snapshot)

    def run_seed(self, env_index: int) -> str:
        return str(self._states[env_index].seed)

    def _new_state(self, seed: int) -> _State:
        rng = np.random.default_rng(seed)
        return _State(
            seed=seed,
            rng=rng,
            cards=rng.integers(2, 15, size=5, dtype=np.int64),
            hands=self.initial_hands,
            discards=self.initial_discards,
        )

    def _replace_cards(self, state: _State, cards: np.ndarray) -> None:
        state.cards[cards] = state.rng.integers(2, 15, size=len(cards))

    def _observe(self) -> ObsDict:
        observations = {
            key: np.zeros((self.num_envs, *shape), dtype=dtype)
            for key, (shape, dtype) in OBS_SPEC.items()
        }
        for row, state in enumerate(self._states):
            for card_index, rank in enumerate(state.cards):
                observations["hand"][row, card_index, int(rank) - 2] = 1.0
                observations["hand"][row, card_index, 13 + (card_index % 4)] = 1.0
                observations["hand"][row, card_index, 17] = 1.0
                observations["hand"][row, card_index, 26] = 1.0
                observations["hand"][row, card_index, 31] = 1.0
            observations["hand_len"][row] = len(state.cards)
            observations["blind"][row, BLIND_REQ_OFF] = np.log1p(self.blind_target)
            observations["blind"][row, BLIND_SCORED_OFF] = np.log1p(state.score)
            observations["global"][row, GLOBAL_HANDS_LEFT] = state.hands / 4.0
            observations["global"][row, GLOBAL_DISCARDS_LEFT] = state.discards / 4.0
            observations["global"][row, GLOBAL_ANTE_OFF] = 1.0
            observations["global"][row, GLOBAL_ROUND] = 1.0 / 24.0
            observations["global"][row, GLOBAL_PHASE_OFF + 1] = 1.0
            observations["deck_counts"][row] = 1.0
            observations["drawpile_counts"][row] = 1.0
        return observations

    def _masks(self) -> MaskDict:
        masks = {
            key: np.zeros((self.num_envs, *shape), dtype=dtype)
            for key, (shape, dtype) in MASK_SPEC.items()
        }
        for row, state in enumerate(self._states):
            if state.hands > 0:
                masks["action_type_mask"][row, int(UpstreamActionType.PLAY_HAND)] = True
            if state.discards > 0:
                masks["action_type_mask"][row, int(UpstreamActionType.DISCARD)] = True
            masks["card_select_mask"][row, : len(state.cards)] = True
        return masks

    def _require_reset(self) -> None:
        if len(self._states) != self.num_envs:
            raise RuntimeError("reset must be called before use")
