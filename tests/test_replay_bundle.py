from __future__ import annotations

import json
import hashlib
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.evaluation.evaluator import evaluate_policy
from flylatro.fly.processors import FixedReservoirProcessor
from flylatro.fly.upstream_encoder import full_feature_names
from flylatro.policy.torch_structured import TorchStructuredPolicy
from flylatro.replay.bundle import ReplayBundle, ReplayBundleWriter, ReplayIdentity
from flylatro.replay.verify import ReplayDivergedError, verify_replay
from flylatro.replay.package import package_showcase
from flylatro.seeds import seed_everything


def _identity() -> ReplayIdentity:
    return ReplayIdentity(
        episode_id="eval-123",
        balatro_seed=123,
        checkpoint_id="checkpoint-7",
        checkpoint_sha256="a" * 64,
        simulator_version="mock-array-balatro-v1",
        encoder_hash="b" * 64,
        feature_extractor_hash="c" * 64,
        connectome_hash="d" * 64,
        policy_version="torch-structured-linear-v1",
    )


def _write_bundle(path):
    seed_everything(8)
    processor = FixedReservoirProcessor(
        len(full_feature_names()), reservoir_size=24, output_size=16, seed=2
    )
    policy = TorchStructuredPolicy(16)
    result = evaluate_policy(
        MockArrayBalatroEnv(1, blind_target=18, initial_hands=3),
        processor,
        policy,
        (123,),
        deterministic=True,
        max_vector_steps=5,
    )
    with ReplayBundleWriter(path, _identity()) as writer:
        for transition in result.transitions:
            writer.record(
                {
                    "decision_id": transition.decision_id,
                    "action": transition.action,
                    "reward": transition.reward,
                    "reward_components": transition.reward_components,
                    "value": transition.value,
                    "action_probability": transition.action_probability,
                    "action_type_probabilities": transition.action_type_probabilities,
                    "state_hash_before": transition.state_hash_before,
                    "state_hash_after": transition.state_hash_after,
                    "done": transition.done,
                }
            )
    return ReplayBundle.open(path)


def test_bundle_round_trip_and_deterministic_verification(tmp_path) -> None:
    bundle = _write_bundle(tmp_path / "replay")

    assert bundle.identity.balatro_seed == 123
    assert len(bundle.decisions) >= 1
    assert verify_replay(
        bundle, MockArrayBalatroEnv(1, blind_target=18, initial_hands=3)
    ) == ()


def test_replay_stops_at_first_hash_mismatch(tmp_path) -> None:
    bundle = _write_bundle(tmp_path / "replay")
    bundle.decisions[0]["state_hash_before"] = "0" * 64

    with pytest.raises(ReplayDivergedError, match=r"decision 0 \(before\)"):
        verify_replay(
            bundle, MockArrayBalatroEnv(1, blind_target=18, initial_hands=3)
        )


def test_bundle_rejects_file_tampering(tmp_path) -> None:
    bundle = _write_bundle(tmp_path / "replay")
    with (bundle.path / "decisions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"decision_id": 99}) + "\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        ReplayBundle.open(bundle.path)


def test_showcase_package_embeds_verified_checkpoint_config_and_readme(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "experiment.toml"
    config.write_text('[experiment]\nname = "test"\n', encoding="utf-8")
    identity = replace(
        _identity(), checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest()
    )
    source = tmp_path / "source"
    with ReplayBundleWriter(source, identity) as writer:
        writer.record(
            {
                "decision_id": 0,
                "action": {"type": "play_hand", "cards": [0]},
                "reward": 1.0,
                "state_hash_before": "a" * 64,
                "state_hash_after": "b" * 64,
                "done": True,
            }
        )

    packaged = package_showcase(
        source, tmp_path / "packaged", checkpoint=checkpoint, config=config
    )

    assert {"checkpoint.pt", "experiment.toml", "README.md"} <= set(
        packaged.manifest["files"]
    )
    assert "Balatro seed" in (packaged.path / "README.md").read_text()
