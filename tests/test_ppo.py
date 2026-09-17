from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.fly.processors import FixedReservoirProcessor
from flylatro.fly.upstream_encoder import full_feature_names
from flylatro.policy.torch_structured import TorchStructuredPolicy
from flylatro.seeds import seed_everything
from flylatro.training.ppo import PPOConfig, PPOTrainer
from flylatro.training.rollout import RolloutBuffer


def make_trainer() -> PPOTrainer:
    seed_everything(13)
    env = MockArrayBalatroEnv(num_envs=2, blind_target=20, initial_hands=2)
    processor = FixedReservoirProcessor(
        len(full_feature_names()), reservoir_size=24, output_size=16, seed=5
    )
    policy = TorchStructuredPolicy(feature_size=16)
    config = PPOConfig(
        rollout_steps=4,
        update_epochs=2,
        minibatch_size=4,
        learning_rate=1e-3,
        seed=13,
    )
    return PPOTrainer(
        env, processor, policy, config, training_seeds=(100, 101), device="cpu"
    )


def test_tiny_ppo_rollout_and_update_changes_policy_parameters() -> None:
    trainer = make_trainer()
    before = {
        key: value.detach().clone() for key, value in trainer.policy.state_dict().items()
    }

    metrics = trainer.train_update()

    after = trainer.policy.state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert trainer.global_step == 8
    assert trainer.update_index == 1
    assert metrics["rollout/decisions"] == 8
    for key in (
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "train/approx_kl",
        "throughput/end_to_end_decisions_per_second",
    ):
        assert math.isfinite(metrics[key])


def test_gae_respects_terminal_boundaries() -> None:
    buffer = RolloutBuffer()
    for reward, done, value in ((1.0, False, 0.5), (2.0, True, 0.25)):
        buffer.add(
            features=torch.zeros(1, 2),
            masks={"m": torch.ones(1, 1, dtype=torch.bool)},
            actions={"a": torch.zeros(1, dtype=torch.long)},
            log_probability=torch.zeros(1),
            value=torch.tensor([value]),
            reward=torch.tensor([reward]),
            done=torch.tensor([done]),
        )

    result = buffer.finish(torch.tensor([99.0]), gamma=1.0, gae_lambda=1.0)

    torch.testing.assert_close(result.returns, torch.tensor([3.0, 2.0]))


def test_tiny_ppo_is_reproducible_for_same_seed() -> None:
    first = make_trainer()
    first.train_update()
    first_state = first.policy.state_dict()
    second = make_trainer()
    second.train_update()
    second_state = second.policy.state_dict()

    for key in first_state:
        torch.testing.assert_close(first_state[key], second_state[key])
