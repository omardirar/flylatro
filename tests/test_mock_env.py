from __future__ import annotations

from flylatro.env.mock import MockBalatroEnv
from flylatro.env.types import ActionType, CompositeAction


def test_snapshot_restore_repeats_the_same_transition() -> None:
    env = MockBalatroEnv(num_envs=1)
    (before,) = env.reset([123])
    snapshot = env.snapshot()
    action = CompositeAction(ActionType.DISCARD, cards=(0, 2))

    first = env.step([action]).observations[0]
    env.restore(snapshot)
    second = env.step([action]).observations[0]

    assert before.state_hash() != first.state_hash()
    assert first.to_payload() == second.to_payload()
    assert first.state_hash() == second.state_hash()


def test_mock_reward_is_progress_based_not_hand_labelled() -> None:
    env = MockBalatroEnv(num_envs=1, blind_target=100)
    env.reset([3])
    step = env.step([CompositeAction(ActionType.PLAY_HAND, cards=(0,))])

    components = step.reward_components[0]
    assert set(components) == {"blind_progress", "blind_clear", "win"}
    assert components["blind_progress"] > 0

