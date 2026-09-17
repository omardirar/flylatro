"""Flylatro adapter for the pinned real Rust/PyO3 Balatro simulator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Protocol, Sequence

import numpy as np

from flylatro.env.types import ActionType, CompositeAction
from flylatro.env.upstream_contract import (
    ACTION_SPEC,
    MASK_SPEC,
    OBS_SPEC,
    SIMULATOR_VERSION,
    GLOBAL_ANTE_OFF,
    GLOBAL_PHASE_OFF,
    ActionDict,
    MaskDict,
    ObsDict,
    UpstreamActionType,
    empty_action_batch,
    validate_batch,
)


@dataclass(frozen=True, slots=True)
class ArrayStep:
    observations: ObsDict
    masks: MaskDict
    rewards: np.ndarray
    dones: np.ndarray
    infos: tuple[dict[str, Any], ...]


class ArrayBalatroEnv(Protocol):
    num_envs: int
    simulator_version: str

    def reset(self, seeds: Sequence[int]) -> tuple[ObsDict, MaskDict]: ...

    def step(self, actions: ActionDict) -> ArrayStep: ...

    def set_shaping_beta(self, beta: float) -> None: ...

    def set_win_ante(self, win_ante: int) -> None: ...


class BalatroSimAdapter:
    """Validated high-throughput adapter; never exposes heuristic actions."""

    simulator_version = SIMULATOR_VERSION

    def __init__(
        self,
        num_envs: int,
        *,
        strict: bool = True,
        win_ante: int = 8,
        sim_module: Any | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not 1 <= win_ante <= 8:
            raise ValueError("win_ante must be in [1, 8]")
        if sim_module is None:
            try:
                import balatro_sim as sim_module
            except ImportError as error:
                raise ImportError(
                    "balatro_sim is not installed; install Flylatro's pinned "
                    "'balatro' extra on a machine with Rust/maturin"
                ) from error
        self._env = sim_module.BalatroVecEnv(num_envs, strict, win_ante)
        self.num_envs = num_envs
        self._beta = 1.0
        self._last_observations: ObsDict | None = None

    def reset(self, seeds: Sequence[int]) -> tuple[ObsDict, MaskDict]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        observations, masks = self._env.reset([int(seed) for seed in seeds])
        validate_batch(OBS_SPEC, observations, self.num_envs, "observations")
        validate_batch(MASK_SPEC, masks, self.num_envs, "masks")
        self._last_observations = observations
        return observations, masks

    def step(self, actions: ActionDict) -> ArrayStep:
        validate_batch(ACTION_SPEC, actions, self.num_envs, "actions")
        contiguous = {
            key: np.ascontiguousarray(value, dtype=np.int64)
            for key, value in actions.items()
        }
        before = self._last_observations
        observations, masks, rewards, dones, infos = self._env.step(contiguous)
        validate_batch(OBS_SPEC, observations, self.num_envs, "observations")
        validate_batch(MASK_SPEC, masks, self.num_envs, "masks")
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool_)
        if rewards.shape != (self.num_envs,) or dones.shape != (self.num_envs,):
            raise ValueError("upstream reward/done arrays have invalid shapes")
        if len(infos) != self.num_envs:
            raise ValueError("upstream info length does not match num_envs")
        enriched_infos = []
        for index, raw_info in enumerate(infos):
            info = dict(raw_info)
            if "reward_components" not in info and before is not None:
                info["reward_components"] = _reward_components(
                    reward=float(rewards[index]),
                    done=bool(dones[index]),
                    info=info,
                    before=before,
                    after=observations,
                    row=index,
                    beta=self._beta,
                )
            enriched_infos.append(info)
        self._last_observations = observations
        return ArrayStep(
            observations, masks, rewards, dones, tuple(enriched_infos)
        )

    def set_shaping_beta(self, beta: float) -> None:
        if not 0 <= beta <= 1:
            raise ValueError("shaping beta must be in [0, 1]")
        self._env.set_shaping_beta(float(beta))
        self._beta = float(beta)

    def set_win_ante(self, win_ante: int) -> None:
        if not 1 <= win_ante <= 8:
            raise ValueError("win_ante must be in [1, 8]")
        self._env.set_win_ante(int(win_ante))

    def observe(self) -> tuple[ObsDict, MaskDict]:
        observations, masks = self._env.observe()
        validate_batch(OBS_SPEC, observations, self.num_envs, "observations")
        validate_batch(MASK_SPEC, masks, self.num_envs, "masks")
        self._last_observations = observations
        return observations, masks

    def snapshot(self, env_index: int) -> bytes:
        return bytes(self._env.snapshot(int(env_index)))

    def restore(self, env_index: int, snapshot: bytes) -> None:
        self._env.restore(int(env_index), snapshot)
        self._last_observations = None

    def snapshot_all(self) -> tuple[bytes, ...]:
        return tuple(self.snapshot(index) for index in range(self.num_envs))

    def restore_all(self, snapshots: Sequence[bytes]) -> None:
        if len(snapshots) != self.num_envs:
            raise ValueError("snapshot count does not match num_envs")
        for index, snapshot in enumerate(snapshots):
            self.restore(index, snapshot)
        self.observe()

    def run_seed(self, env_index: int) -> str:
        return str(self._env.run_seed(int(env_index)))

    def run_info(self, env_index: int) -> dict[str, Any]:
        return dict(self._env.run_info(int(env_index)))

    @staticmethod
    def state_hash(observations: ObsDict, env_index: int) -> str:
        digest = hashlib.sha256()
        for key in sorted(OBS_SPEC):
            row = np.ascontiguousarray(observations[key][env_index])
            digest.update(key.encode("utf-8"))
            digest.update(str(row.dtype).encode("ascii"))
            digest.update(np.asarray(row.shape, dtype="<i8").tobytes())
            digest.update(row.tobytes())
        return digest.hexdigest()


_TO_UPSTREAM = {
    ActionType.PLAY_HAND: UpstreamActionType.PLAY_HAND,
    ActionType.DISCARD: UpstreamActionType.DISCARD,
    ActionType.SELECT_BLIND: UpstreamActionType.SELECT_BLIND,
    ActionType.SKIP_BLIND: UpstreamActionType.SKIP_BLIND,
    ActionType.CASH_OUT: UpstreamActionType.CASH_OUT,
    ActionType.BUY: UpstreamActionType.BUY_SHOP,
    ActionType.REROLL: UpstreamActionType.REROLL,
    ActionType.END_SHOP: UpstreamActionType.LEAVE_SHOP,
    ActionType.USE_CONSUMABLE: UpstreamActionType.USE_CONSUMABLE,
    ActionType.SELL_JOKER: UpstreamActionType.SELL_JOKER,
    ActionType.SELL_CONSUMABLE: UpstreamActionType.SELL_CONSUMABLE,
    ActionType.PICK_PACK: UpstreamActionType.PICK_PACK,
    ActionType.SKIP_PACK: UpstreamActionType.SKIP_PACK,
}
_FROM_UPSTREAM = {upstream: local for local, upstream in _TO_UPSTREAM.items()}


def composite_actions_to_batch(actions: Sequence[CompositeAction]) -> ActionDict:
    batch = empty_action_batch(len(actions))
    for index, action in enumerate(actions):
        batch["action_type"][index] = int(_TO_UPSTREAM[action.action_type])
        if len(action.cards) > batch["cards"].shape[1]:
            raise ValueError("action selects too many cards")
        batch["n_cards"][index] = len(action.cards)
        if action.cards:
            batch["cards"][index, : len(action.cards)] = action.cards
        for key in (
            "joker_target",
            "consumable_target",
            "shop_target",
            "pack_target",
        ):
            value = getattr(action, key)
            batch[key][index] = -1 if value is None else value
    return batch


def action_batch_row_to_composite(actions: ActionDict, index: int) -> CompositeAction:
    upstream_type = UpstreamActionType(int(actions["action_type"][index]))
    if upstream_type not in _FROM_UPSTREAM:
        raise ValueError(f"unsupported upstream action type {upstream_type.name}")
    count = int(actions["n_cards"][index])
    if not 0 <= count <= actions["cards"].shape[1]:
        raise ValueError("invalid n_cards in action batch")

    def optional(key: str) -> int | None:
        value = int(actions[key][index])
        return None if value < 0 else value

    return CompositeAction(
        action_type=_FROM_UPSTREAM[upstream_type],
        cards=tuple(int(card) for card in actions["cards"][index, :count]),
        joker_target=optional("joker_target"),
        consumable_target=optional("consumable_target"),
        shop_target=optional("shop_target"),
        pack_target=optional("pack_target"),
    )


def _reward_components(
    *,
    reward: float,
    done: bool,
    info: dict[str, Any],
    before: ObsDict,
    after: ObsDict,
    row: int,
    beta: float,
) -> dict[str, float]:
    """Decompose the pinned simulator's documented scalar reward exactly.

    Auto-reset hides terminal observations, so terminal clear/progress values
    are recovered from the total after subtracting the known clear/win terms.
    """

    components = {"blind_progress": 0.0, "blind_clear": 0.0, "win": 0.0}
    if reward == 0:
        return components
    before_global = before["global"][row]
    playing = bool(before_global[GLOBAL_PHASE_OFF + 1])
    if not playing:
        return components
    ante_hot = before_global[GLOBAL_ANTE_OFF : GLOBAL_ANTE_OFF + 9]
    ante = int(np.argmax(ante_hot)) + 1 if bool(np.any(ante_hot)) else 0
    won = bool(info.get("episode", {}).get("won", False)) if done else False
    if won:
        components["win"] = 15.0
    after_round_eval = (
        not done and bool(after["global"][row, GLOBAL_PHASE_OFF + 2])
    )
    if won or after_round_eval:
        components["blind_clear"] = beta * 0.5 * (1.0 + 0.15 * ante)
    components["blind_progress"] = reward - sum(components.values())
    for key, value in components.items():
        if abs(value) < 1e-9:
            components[key] = 0.0
    return components
