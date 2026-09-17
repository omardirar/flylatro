"""A deterministic, strategy-neutral development environment.

This is not an implementation of Balatro.  It is a deliberately small state
machine for exercising Flylatro's observation, action-mask, batching, reward,
and replay contracts before the external simulator is integrated.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from flylatro.env.contracts import VectorStep
from flylatro.env.types import (
    ACTION_INDEX,
    ACTION_TYPES,
    ActionMask,
    ActionType,
    BalatroObservation,
    CardObservation,
    CompositeAction,
    GamePhase,
    Suit,
)


@dataclass(slots=True)
class _MockState:
    episode_id: str
    seed: int
    decision_id: int
    hand: list[CardObservation]
    score: float
    hands_remaining: int
    discards_remaining: int
    terminated: bool = False
    won: bool = False


class MockBalatroEnv:
    """Small vector environment with legal play/discard actions.

    Playing cards advances a generic blind-progress score.  The reward says
    only whether progress or completion occurred; it contains no poker-hand or
    card-choice teaching signal.
    """

    simulator_version = "mock-balatro-v1"

    def __init__(
        self,
        num_envs: int = 2,
        *,
        hand_size: int = 5,
        blind_target: float = 25.0,
        initial_hands: int = 3,
        initial_discards: int = 2,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not 1 <= hand_size <= 8:
            raise ValueError("hand_size must be in [1, 8]")
        if blind_target <= 0:
            raise ValueError("blind_target must be positive")
        self._num_envs = num_envs
        self.hand_size = hand_size
        self.blind_target = float(blind_target)
        self.initial_hands = initial_hands
        self.initial_discards = initial_discards
        self._rngs: list[np.random.Generator] = []
        self._states: list[_MockState] = []

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seeds: Sequence[int]) -> tuple[BalatroObservation, ...]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        self._rngs = [np.random.default_rng(int(seed)) for seed in seeds]
        self._states = []
        for index, seed in enumerate(seeds):
            self._states.append(
                _MockState(
                    episode_id=f"mock-{index}-{int(seed)}",
                    seed=int(seed),
                    decision_id=0,
                    hand=self._deal_hand(self._rngs[index]),
                    score=0.0,
                    hands_remaining=self.initial_hands,
                    discards_remaining=self.initial_discards,
                )
            )
        return self._observations()

    def legal_action_masks(self) -> tuple[ActionMask | None, ...]:
        self._require_reset()
        masks: list[ActionMask | None] = []
        for state in self._states:
            if state.terminated:
                masks.append(None)
                continue
            enabled = np.zeros(len(ACTION_TYPES), dtype=np.bool_)
            cards = np.zeros((len(ACTION_TYPES), self.hand_size), dtype=np.bool_)
            min_cards = np.zeros(len(ACTION_TYPES), dtype=np.int64)
            max_cards = np.zeros(len(ACTION_TYPES), dtype=np.int64)
            targets = np.zeros((len(ACTION_TYPES), self.hand_size), dtype=np.bool_)
            requires_target = np.zeros(len(ACTION_TYPES), dtype=np.bool_)

            play_index = ACTION_INDEX[ActionType.PLAY_HAND]
            if state.hands_remaining > 0:
                enabled[play_index] = True
                cards[play_index, : len(state.hand)] = True
                min_cards[play_index] = 1
                max_cards[play_index] = len(state.hand)

            discard_index = ACTION_INDEX[ActionType.DISCARD]
            if state.discards_remaining > 0:
                enabled[discard_index] = True
                cards[discard_index, : len(state.hand)] = True
                min_cards[discard_index] = 1
                max_cards[discard_index] = min(3, len(state.hand))

            masks.append(
                ActionMask(
                    action_types=enabled,
                    cards=cards,
                    min_cards=min_cards,
                    max_cards=max_cards,
                    targets=targets,
                    requires_target=requires_target,
                )
            )
        return tuple(masks)

    def step(self, actions: Sequence[CompositeAction | None]) -> VectorStep:
        self._require_reset()
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        masks = self.legal_action_masks()
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        components: list[dict[str, float]] = []
        infos: list[dict[str, Any]] = []

        for index, (state, action, mask) in enumerate(
            zip(self._states, actions, masks, strict=True)
        ):
            reward_parts = {"blind_progress": 0.0, "blind_clear": 0.0, "win": 0.0}
            if state.terminated:
                if action is not None:
                    raise ValueError(f"environment {index} is already terminated")
                terminated[index] = True
                components.append(reward_parts)
                infos.append({"inactive": True, "won": state.won})
                continue
            if action is None:
                raise ValueError(f"environment {index} requires an action")
            assert mask is not None
            if not mask.allows(action):
                raise ValueError(f"illegal action for environment {index}: {action}")

            if action.action_type is ActionType.PLAY_HAND:
                points = float(sum(state.hand[card].rank for card in action.cards))
                remaining = max(0.0, self.blind_target - state.score)
                progress = min(points, remaining)
                state.score += points
                state.hands_remaining -= 1
                reward_parts["blind_progress"] = progress / self.blind_target
                self._replace_cards(index, action.cards)
                if state.score >= self.blind_target:
                    state.terminated = True
                    state.won = True
                    reward_parts["blind_clear"] = 1.0
                    reward_parts["win"] = 2.0
                elif state.hands_remaining == 0:
                    state.terminated = True
            elif action.action_type is ActionType.DISCARD:
                state.discards_remaining -= 1
                self._replace_cards(index, action.cards)
            else:  # Protected by the action mask, kept explicit for adapter bugs.
                raise ValueError(f"mock environment cannot execute {action.action_type}")

            state.decision_id += 1
            rewards[index] = sum(reward_parts.values())
            terminated[index] = state.terminated
            components.append(reward_parts)
            infos.append({"inactive": False, "won": state.won})

        return VectorStep(
            observations=self._observations(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            reward_components=tuple(components),
            infos=tuple(infos),
        )

    def snapshot(self) -> dict[str, Any]:
        self._require_reset()
        return {
            "version": self.simulator_version,
            "config": {
                "num_envs": self.num_envs,
                "hand_size": self.hand_size,
                "blind_target": self.blind_target,
                "initial_hands": self.initial_hands,
                "initial_discards": self.initial_discards,
            },
            "rng_states": [deepcopy(rng.bit_generator.state) for rng in self._rngs],
            "states": [
                {
                    "episode_id": state.episode_id,
                    "seed": state.seed,
                    "decision_id": state.decision_id,
                    "hand": [card.to_payload() for card in state.hand],
                    "score": state.score,
                    "hands_remaining": state.hands_remaining,
                    "discards_remaining": state.discards_remaining,
                    "terminated": state.terminated,
                    "won": state.won,
                }
                for state in self._states
            ],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("version") != self.simulator_version:
            raise ValueError("incompatible mock snapshot version")
        if snapshot.get("config") != {
            "num_envs": self.num_envs,
            "hand_size": self.hand_size,
            "blind_target": self.blind_target,
            "initial_hands": self.initial_hands,
            "initial_discards": self.initial_discards,
        }:
            raise ValueError("snapshot configuration does not match environment")
        raw_states = snapshot["states"]
        raw_rngs = snapshot["rng_states"]
        if len(raw_states) != self.num_envs or len(raw_rngs) != self.num_envs:
            raise ValueError("snapshot environment count does not match")
        self._rngs = []
        for raw_rng in raw_rngs:
            rng = np.random.default_rng()
            rng.bit_generator.state = deepcopy(raw_rng)
            self._rngs.append(rng)
        self._states = [
            _MockState(
                episode_id=raw["episode_id"],
                seed=int(raw["seed"]),
                decision_id=int(raw["decision_id"]),
                hand=[
                    CardObservation(rank=int(card["rank"]), suit=Suit(card["suit"]))
                    for card in raw["hand"]
                ],
                score=float(raw["score"]),
                hands_remaining=int(raw["hands_remaining"]),
                discards_remaining=int(raw["discards_remaining"]),
                terminated=bool(raw["terminated"]),
                won=bool(raw["won"]),
            )
            for raw in raw_states
        ]

    def _deal_hand(self, rng: np.random.Generator) -> list[CardObservation]:
        suits = tuple(Suit)
        return [
            CardObservation(
                rank=int(rng.integers(2, 15)),
                suit=suits[int(rng.integers(0, len(suits)))],
            )
            for _ in range(self.hand_size)
        ]

    def _replace_cards(self, env_index: int, cards: tuple[int, ...]) -> None:
        rng = self._rngs[env_index]
        suits = tuple(Suit)
        state = self._states[env_index]
        for card_index in cards:
            state.hand[card_index] = CardObservation(
                rank=int(rng.integers(2, 15)),
                suit=suits[int(rng.integers(0, len(suits)))],
            )

    def _observations(self) -> tuple[BalatroObservation, ...]:
        return tuple(self._observation(state) for state in self._states)

    def _observation(self, state: _MockState) -> BalatroObservation:
        return BalatroObservation(
            episode_id=state.episode_id,
            seed=state.seed,
            decision_id=state.decision_id,
            ante=1,
            round=1,
            phase=GamePhase.TERMINAL if state.terminated else GamePhase.SELECTING_HAND,
            hand=tuple(state.hand),
            money=0,
            score=state.score,
            blind_target=self.blind_target,
            hands_remaining=state.hands_remaining,
            discards_remaining=state.discards_remaining,
        )

    def _require_reset(self) -> None:
        if len(self._states) != self.num_envs:
            raise RuntimeError("environment must be reset before use")

