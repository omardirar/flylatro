from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.evaluation.evaluator import evaluate_policy
from flylatro.fly.processors import FixedReservoirProcessor
from flylatro.fly.upstream_encoder import full_feature_names
from flylatro.policy.torch_structured import TorchStructuredPolicy
from flylatro.seeds import seed_everything


def test_frozen_evaluation_reports_metrics_successes_and_complete_traces() -> None:
    seed_everything(4)
    env = MockArrayBalatroEnv(num_envs=3, blind_target=20, initial_hands=2)
    processor = FixedReservoirProcessor(
        len(full_feature_names()), reservoir_size=24, output_size=16, seed=5
    )
    policy = TorchStructuredPolicy(16)
    before = {key: value.clone() for key, value in policy.state_dict().items()}

    result = evaluate_policy(
        env,
        processor,
        policy,
        (700, 701, 702),
        deterministic=True,
        max_vector_steps=10,
    )

    assert len(result.episodes) == 3
    assert 0 <= result.win_rate <= 1
    assert result.decision_count == len(result.transitions)
    assert all(transition.action["type"] for transition in result.transitions)
    assert all(len(transition.state_hash_before) == 64 for transition in result.transitions)
    assert all(len(transition.state_hash_after) == 64 for transition in result.transitions)
    first_signature = result.transitions[0].state_signature_before
    assert first_signature["ante_num"] == 1
    assert first_signature["round_num"] == 1
    assert first_signature["hands_left"] == 2
    assert first_signature["discards_left"] == 2
    assert len(result.transitions[-1].action_type_probabilities) == 25
    assert sum(result.transitions[-1].action_type_probabilities) == pytest.approx(1.0)
    metrics = result.metrics()
    assert metrics["evaluation/episodes"] == 3
    for key, value in before.items():
        torch.testing.assert_close(policy.state_dict()[key], value)
