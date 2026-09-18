from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flylatro.env.balatro_sim import (
    BalatroSimAdapter,
    _ante_advanced,
    _ante_cleared,
    _reward_components,
    action_batch_row_to_composite,
    composite_actions_to_batch,
)
from flylatro.env.types import ActionType, CompositeAction
from flylatro.env.upstream_contract import (
    ACTION_SPEC,
    MASK_SPEC,
    OBS_SPEC,
    empty_action_batch,
    GLOBAL_ANTE_OFF,
    GLOBAL_PHASE_OFF,
)


def zero_batch(spec, count: int):
    return {
        key: np.zeros((count, *row_shape), dtype=dtype)
        for key, (row_shape, dtype) in spec.items()
    }


class FakeBalatroVecEnv:
    def __init__(self, num_envs: int, strict: bool, win_ante: int) -> None:
        self.num_envs = num_envs
        self.strict = strict
        self.win_ante = win_ante
        self.observations = zero_batch(OBS_SPEC, num_envs)
        self.masks = zero_batch(MASK_SPEC, num_envs)
        self.masks["action_type_mask"][:, 0] = True
        self.masks["card_select_mask"][:, 0] = True
        self.last_actions = None
        self.beta = 1.0

    def reset(self, seeds):
        assert len(seeds) == self.num_envs
        return self.observations, self.masks

    def step(self, actions):
        self.last_actions = actions
        return (
            self.observations,
            self.masks,
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            [{} for _ in range(self.num_envs)],
        )

    def set_shaping_beta(self, beta):
        self.beta = beta

    def set_win_ante(self, win_ante):
        self.win_ante = win_ante

    def observe(self):
        return self.observations, self.masks

    def snapshot(self, env_index):
        return bytes([env_index])

    def restore(self, env_index, snapshot):
        self.restored = (env_index, snapshot)

    def run_seed(self, env_index):
        return f"SEED{env_index}"

    def run_info(self, env_index):
        return {"seed": self.run_seed(env_index), "ante": 1, "round": 1}


def test_real_adapter_validates_the_pinned_array_contract() -> None:
    adapter = BalatroSimAdapter(
        2, sim_module=SimpleNamespace(BalatroVecEnv=FakeBalatroVecEnv)
    )
    observations, masks = adapter.reset([11, 12])
    actions = empty_action_batch(2)
    actions["action_type"][:] = 0
    actions["n_cards"][:] = 1
    actions["cards"][:, 0] = 0

    step = adapter.step(actions)

    assert step.rewards.tolist() == [1.0, 1.0]
    assert step.dones.tolist() == [False, False]
    assert adapter.run_seed(1) == "SEED1"
    assert len(adapter.state_hash(observations, 0)) == 64
    assert adapter.state_hash(observations, 0) == adapter.state_hash(observations, 1)
    assert masks["action_type_mask"][:, 0].all()


def test_adapter_rejects_contract_drift() -> None:
    adapter = BalatroSimAdapter(
        1, sim_module=SimpleNamespace(BalatroVecEnv=FakeBalatroVecEnv)
    )
    adapter.reset([1])
    malformed = empty_action_batch(1)
    malformed["cards"] = malformed["cards"].astype(np.int32)

    with pytest.raises(ValueError, match="dtype"):
        adapter.step(malformed)


@pytest.mark.parametrize(
    ("action", "target_field"),
    [
        (CompositeAction(ActionType.PLAY_HAND, cards=(1, 3)), None),
        (CompositeAction(ActionType.SELL_JOKER, joker_target=2), "joker_target"),
        (
            CompositeAction(ActionType.USE_CONSUMABLE, consumable_target=1),
            "consumable_target",
        ),
        (CompositeAction(ActionType.BUY, shop_target=4), "shop_target"),
        (CompositeAction(ActionType.PICK_PACK, pack_target=3), "pack_target"),
    ],
)
def test_complete_composite_action_round_trips(action, target_field) -> None:
    batch = composite_actions_to_batch([action])

    restored = action_batch_row_to_composite(batch, 0)

    assert restored == action
    if target_field is not None:
        assert int(batch[target_field][0]) == getattr(action, target_field)


def test_action_spec_is_filled_with_minus_one_for_unused_heads() -> None:
    batch = composite_actions_to_batch([CompositeAction(ActionType.REROLL)])

    assert set(batch) == set(ACTION_SPEC)
    assert batch["n_cards"][0] == 0
    assert (batch["cards"][0] == -1).all()
    assert batch["joker_target"][0] == -1


def test_real_reward_is_decomposed_into_strategy_neutral_components() -> None:
    before = zero_batch(OBS_SPEC, 1)
    after = zero_batch(OBS_SPEC, 1)
    before["global"][0, GLOBAL_PHASE_OFF + 1] = 1
    before["global"][0, GLOBAL_ANTE_OFF] = 1

    components = _reward_components(
        reward=15.775,
        done=True,
        info={"episode": {"won": True}},
        before=before,
        after=after,
        row=0,
        beta=1.0,
    )

    assert components["win"] == 15
    assert components["blind_clear"] == pytest.approx(0.575)
    assert components["blind_progress"] == pytest.approx(0.2)
    assert sum(components.values()) == pytest.approx(15.775)


def test_ante_clear_is_inferred_from_observable_transition_only() -> None:
    before = zero_batch(OBS_SPEC, 1)
    after = zero_batch(OBS_SPEC, 1)
    before["global"][0, GLOBAL_ANTE_OFF + 2] = 1
    after["global"][0, GLOBAL_ANTE_OFF + 3] = 1

    assert _ante_advanced(before, after, row=0, done=False)
    assert not _ante_advanced(before, after, row=0, done=True)
    assert _ante_cleared(
        before,
        after,
        info={"episode": {"won": True}},
        row=0,
        done=True,
    )
