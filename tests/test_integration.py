from __future__ import annotations

import json
from pathlib import Path

from flylatro.agent import FlyAgent
from flylatro.env.mock import MockBalatroEnv
from flylatro.env.types import ActionType, CompositeAction
from flylatro.fly.backend import TinyGraphFlyBackend
from flylatro.fly.encoder import EncoderSpec, FixedBalatroEncoder
from flylatro.fly.features import FeatureSpec, RateFeatureExtractor
from flylatro.policy.structured import StructuredLinearPolicy
from flylatro.replay.recorder import JsonlTransitionRecorder, load_records
from flylatro.runner.episodes import run_episodes


def build_agent(*, microbatch_size: int = 2) -> FlyAgent:
    encoder_spec = EncoderSpec.development_default()
    feature_spec = FeatureSpec("test-readout-v1", tuple(range(88, 96)))
    return FlyAgent(
        encoder=FixedBalatroEncoder(encoder_spec),
        fly_backend=TinyGraphFlyBackend(neuron_count=96, graph_seed=11),
        feature_extractor=RateFeatureExtractor(feature_spec),
        policy=StructuredLinearPolicy(
            feature_size=feature_spec.output_size,
            max_cards=8,
            max_targets=8,
            seed=12,
        ),
        duration_ms=4,
        microbatch_size=microbatch_size,
    )


def test_lightweight_vertical_slice_records_replayable_transitions(
    tmp_path: Path,
) -> None:
    seeds = (101, 102, 103)
    env = MockBalatroEnv(
        num_envs=3, blind_target=22, initial_hands=2, initial_discards=1
    )
    output_dir = tmp_path / "run"
    with JsonlTransitionRecorder(output_dir, {"kind": "test", "seeds": seeds}) as recorder:
        summary = run_episodes(
            env,
            build_agent(),
            seeds,
            recorder=recorder,
            deterministic_policy=True,
            max_decisions=30,
        )

    records = load_records(output_dir / "decisions.jsonl")
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert summary.episodes == 3
    assert summary.decisions == len(records)
    assert manifest["kind"] == "test"
    assert all(record["action"]["cards"] for record in records)
    assert all(len(record["state_hash_before"]) == 64 for record in records)
    assert all(len(record["state_hash_after"]) == 64 for record in records)

    for env_index, seed in enumerate(seeds):
        episode_records = [
            record for record in records if record["env_index"] == env_index
        ]
        replay = MockBalatroEnv(
            num_envs=1, blind_target=22, initial_hands=2, initial_discards=1
        )
        (observation,) = replay.reset([seed])
        for record in episode_records:
            assert observation.state_hash() == record["state_hash_before"]
            raw_action = record["action"]
            action = CompositeAction(
                ActionType(raw_action["type"]),
                cards=tuple(raw_action["cards"]),
                target=raw_action["target"],
            )
            mask = replay.legal_action_masks()[0]
            assert mask is not None and mask.allows(action)
            step = replay.step([action])
            observation = step.observations[0]
            assert observation.state_hash() == record["state_hash_after"]


def test_agent_supports_env_batch_larger_than_fly_microbatch() -> None:
    env = MockBalatroEnv(num_envs=5)
    observations = env.reset([1, 2, 3, 4, 5])
    masks = env.legal_action_masks()

    decisions = build_agent(microbatch_size=2).act(
        observations,
        [mask for mask in masks if mask is not None],
        deterministic=True,
    )

    assert len(decisions) == 5
    assert all(
        mask is not None and mask.allows(decision.policy.action)
        for mask, decision in zip(masks, decisions, strict=True)
    )

