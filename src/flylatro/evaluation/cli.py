"""Frozen checkpoint evaluation with held-out seed discipline and replay bundles."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import torch

from flylatro.evaluation.evaluator import EvaluationResult, evaluate_policy
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.replay.bundle import ReplayBundleWriter, ReplayIdentity
from flylatro.seeds import SeedPlan
from flylatro.training.config import (
    ExperimentConfig,
    build_environment,
    build_training_stack,
)
from flylatro.training.checkpoints import load_checkpoint_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--simulator-seed", type=int,
        help="evaluate one explicitly selected simulator seed (for certified reruns)",
    )
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument(
        "--record-neural", action="store_true",
        help="record full-brain events for exactly one selected evaluation episode",
    )
    parser.add_argument(
        "--unlock-final-test", action="store_true",
        help="required before consuming the reserved final-test seed stream",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    episodes = args.episodes or config.evaluation.episodes
    if args.simulator_seed is not None:
        if args.episodes not in (None, 1):
            raise ValueError("an explicit simulator seed requires exactly one episode")
        episodes = 1
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if episodes > 100 and not args.heavy:
        raise ValueError("more than 100 episodes requires --heavy")
    if args.record_neural:
        if episodes != 1 or config.processor.kind not in {"real", "shuffled"}:
            raise ValueError("neural recording requires one real-fly episode")
        config = replace(
            config, processor=replace(config.processor, record_events=True)
        )
    stream = config.evaluation.stream
    if stream == "final_test" and not args.unlock_final_test:
        raise ValueError("final-test seeds require explicit --unlock-final-test")
    if stream not in {"validation", "curriculum", "final_test", "showcase"}:
        raise ValueError(f"evaluation stream {stream!r} is not permitted")
    output_dir = args.output_dir or _default_output_dir(config)
    stack = build_training_stack(config)
    checkpoint_manifest = load_checkpoint_manifest(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location=config.runtime.device, weights_only=False
    )
    _validate_checkpoint_components(
        checkpoint.get("component_metadata", {}), stack.components
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    stack.policy.load_state_dict(checkpoint["policy"])
    stack.policy.eval()
    checkpoint_hash = sha256_file(args.checkpoint)
    seeds = (
        (args.simulator_seed,)
        if args.simulator_seed is not None
        else SeedPlan().seeds(
            stream, episodes, offset=config.evaluation.seed_offset
        )
    )
    results: list[EvaluationResult] = []
    neural_recorder = None
    neural_path = None
    if args.record_neural:
        from flylatro.replay.neural import NeuralEventRecorder

        neural_path = output_dir / "neural-events.parquet"
        neural_recorder = NeuralEventRecorder(neural_path)
    width = config.environment.num_envs
    for start in range(0, episodes, width):
        batch = seeds[start : min(start + width, episodes)]
        batch_config = replace(
            config,
            environment=replace(config.environment, num_envs=len(batch)),
        )
        env = build_environment(batch_config, win_ante=8)
        env.set_shaping_beta(config.reward.shaping_beta)
        results.append(
            evaluate_policy(
                env,
                stack.processor,
                stack.policy,
                batch,
                device=config.runtime.device,
                deterministic=config.evaluation.deterministic,
                max_vector_steps=config.evaluation.max_vector_steps,
                neural_recorder=neural_recorder,
            )
        )
    if neural_recorder is not None:
        neural_recorder.close()
    result = EvaluationResult(
        tuple(episode for item in results for episode in item.episodes),
        tuple(transition for item in results for transition in item.transitions),
    )
    candidate_dir = output_dir / "replay-candidates"
    candidate_dir.mkdir()
    for seed in result.successful_seeds:
        _write_candidate(
            candidate_dir / f"seed-{seed}", result, seed, config, stack.components,
            args.checkpoint, checkpoint_hash,
            neural_path=neural_path,
        )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config.sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_manifest": checkpoint_manifest,
        "seed_stream": stream,
        "explicit_simulator_seed": args.simulator_seed,
        "seed_range": [seeds[0], seeds[-1]],
        "metrics": result.metrics(),
        "successful_seeds": result.successful_seeds,
        "components": stack.components,
        "learning_disabled": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path.resolve()), **result.metrics()}))
    return 0


def _write_candidate(
    path: Path,
    result: EvaluationResult,
    seed: int,
    config: ExperimentConfig,
    components: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_hash: str,
    neural_path: Path | None = None,
) -> None:
    episode = next(item for item in result.episodes if item.seed == seed)
    identity = ReplayIdentity(
        episode_id=f"{config.name}-seed-{seed}",
        balatro_seed=episode.balatro_seed or str(seed),
        checkpoint_id=checkpoint_path.name,
        checkpoint_sha256=checkpoint_hash,
        simulator_version=str(components["simulator_version"]),
        encoder_hash=_component_hash(components.get("encoder_hash", components.get("encoder"))),
        feature_extractor_hash=_component_hash(
            components.get("feature_extractor_hash", components.get("processor_version"))
        ),
        connectome_hash=_component_hash(components.get("connectome_hash", components.get("connectome"))),
        policy_version=str(components["policy_version"]),
        simulator_seed=seed,
        fly_backend_version=components.get("backend_version"),
        fly_dynamics_hash=components.get("fly_dynamics_hash"),
    )
    with ReplayBundleWriter(
        path,
        identity,
        metadata={"episode": {
            "won": episode.won, "ante": episode.ante, "length": episode.length,
            "return": episode.episode_return, "score": episode.score,
        }},
    ) as writer:
        for transition in result.transitions:
            if transition.seed != seed:
                continue
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
                    "state_signature_before": transition.state_signature_before,
                    "ante": transition.state_signature_before["ante_num"],
                    "round": transition.state_signature_before["round_num"],
                    "game_state": transition.state_signature_before["state"],
                    "neural_activity_reference": (
                        f"{neural_path.name}#decision_id={transition.decision_id}"
                        if neural_path is not None else None
                    ),
                    "done": transition.done,
                }
            )
        if neural_path is not None:
            destination = writer.output_dir / neural_path.name
            shutil.copy2(neural_path, destination)
            writer.attach_neural_file(destination)


def _component_hash(value: Any) -> str:
    text = "unavailable" if value is None else str(value)
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower()):
        return text
    return hashlib.sha256(text.encode()).hexdigest()


def _validate_checkpoint_components(
    trained: dict[str, Any], current: dict[str, Any]
) -> None:
    keys = (
        "condition", "connectome_hash", "encoder_hash",
        "feature_extractor_hash", "policy_version", "policy_hidden_size",
    )
    differences = {
        key: {"checkpoint": trained.get(key), "evaluation": current.get(key)}
        for key in keys
        if trained.get(key) != current.get(key)
    }
    if differences:
        raise ValueError(
            "evaluation components differ from checkpoint: "
            + json.dumps(differences, sort_keys=True)
        )


def _default_output_dir(config: ExperimentConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(config.runtime.output_root) / f"eval-{config.name}-{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
