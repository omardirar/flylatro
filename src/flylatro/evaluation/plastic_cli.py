"""Evaluate a frozen plastic fly on held-out Balatro seed streams."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import shutil
from typing import Sequence

from flylatro.evaluation.plastic import PlasticEvaluationResult, evaluate_plastic_fly
from flylatro.learning.checkpoints import load_plastic_checkpoint, load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.seeds import SeedPlan
from flylatro.replay.bundle import ReplayBundleWriter, ReplayIdentity
from flylatro.fly.flywire_artifact import sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument(
        "--seed-stream",
        choices=("validation", "curriculum", "final_test", "showcase"),
        default="validation",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--max-vector-steps", type=int, default=30_000)
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--record-neural", action="store_true")
    parser.add_argument(
        "--record-spikes",
        action="store_true",
        help="persist real time-resolved spike events for one frozen showcase",
    )
    parser.add_argument(
        "--condition",
        choices=(
            "plastic_real",
            "no_plasticity",
            "kc_mbon_shuffled",
            "whole_brain_shuffled",
            "shuffled_reward",
        ),
        default="plastic_real",
    )
    parser.add_argument("--sensory-mapping-seed", type=int)
    parser.add_argument("--output-mode", choices=("mbon_direct", "whole_brain"))
    parser.add_argument(
        "--reinforcement-schedule", "--dopamine-schedule",
        dest="reinforcement_schedule", metavar="REINFORCEMENT_SCHEDULE", type=Path,
    )
    parser.add_argument("--action-schedule", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.episodes > 100 and not args.heavy:
        raise ValueError("more than 100 evaluation episodes requires --heavy")
    if args.seed_stream == "final_test" and not args.unlock_final_test:
        raise ValueError("final-test seeds require --unlock-final-test")
    if args.record_neural and args.episodes != 1:
        raise ValueError("detailed neural recording requires exactly one episode")
    if args.record_spikes and (not args.record_neural or args.seed_stream != "showcase"):
        raise ValueError("--record-spikes requires --record-neural and seed-stream=showcase")
    config = PlasticExperimentConfig.load(args.config)
    config = _apply_evaluation_condition(config, args)
    config.require_heavy_opt_in(args.heavy)
    if config.environment.num_envs != 1:
        raise ValueError(
            "primary frozen evaluation currently requires the recommended one-fly config"
        )
    stack = build_plastic_stack(config)
    if args.record_spikes:
        if config.fly.backend != "flywire":
            raise ValueError("time-resolved showcase spikes require the real FlyWire backend")
        stack.agent.processor.backend.record_events = True
    load_plastic_checkpoint(
        args.checkpoint,
        stack.trainer,
        expected_components=stack.components,
    )
    stack.env.set_win_ante(8)
    checkpoint_manifest = load_plastic_manifest(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seeds = SeedPlan().seeds(
        args.seed_stream, args.episodes, offset=args.seed_offset
    )
    results = []
    neural_recorder = None
    neural_path = args.output_dir / "neural-plasticity-events.parquet"
    if args.record_neural:
        from flylatro.replay.neural import NeuralEventRecorder

        neural_recorder = NeuralEventRecorder(neural_path)
    for seed in seeds:
        result = evaluate_plastic_fly(
            stack.env,
            stack.agent,
            (seed,),
            max_vector_steps=args.max_vector_steps,
            neural_recorder=neural_recorder,
        )
        results.append(result)
    if neural_recorder is not None:
        neural_recorder.close()
    combined = PlasticEvaluationResult(
        episodes=tuple(episode for result in results for episode in result.episodes),
        transitions=tuple(
            transition for result in results for transition in result.transitions
        ),
        frozen_weight_sha256=results[0].frozen_weight_sha256,
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "plastic-brain-v1",
        "learning_disabled": True,
        "plastic_weight_sha256": combined.frozen_weight_sha256,
        "checkpoint_manifest": checkpoint_manifest,
        "config_sha256": config.sha256,
        "seed_stream": args.seed_stream,
        "seed_range": [seeds[0], seeds[-1]],
        "metrics": combined.metrics(),
        "components": stack.components,
        "neural_recording": str(neural_path) if args.record_neural else None,
        "neural_recording_mode": (
            "time_resolved_real_spikes" if args.record_spikes else "decision_aggregate" if args.record_neural else None
        ),
        "reinforcement_semantics": "synthetic appetitive/aversive outcome channels; not simulated PAM/PPL1 spikes",
        "neural_activity_unit": (
            "spikes_per_second_hz"
            if config.fly.backend == "flywire"
            else "synthetic_rate_hz_proxy"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "transitions.jsonl").open("w", encoding="utf-8") as stream:
        for transition in combined.transitions:
            stream.write(json.dumps(asdict(transition), sort_keys=True) + "\n")
    candidates = args.output_dir / "replay-candidates"
    candidates.mkdir()
    for episode in combined.episodes:
        if not episode.won:
            continue
        candidate = candidates / f"seed-{episode.seed}"
        identity = ReplayIdentity(
            episode_id=f"{config.name}-seed-{episode.seed}",
            balatro_seed=episode.balatro_seed or str(episode.seed),
            checkpoint_id=args.checkpoint.name,
            checkpoint_sha256=sha256_file(args.checkpoint),
            simulator_version=str(stack.env.simulator_version),
            encoder_hash=_hash_component(
                stack.components["sensory_mapping_sha256"]
            ),
            feature_extractor_hash=_hash_component(
                stack.components["plastic_topology_sha256"]
            ),
            connectome_hash=_hash_component(
                stack.components["fly_connectivity_sha256"]
            ),
            policy_version="fixed-zero-parameter-motor-interface-v1",
            simulator_seed=episode.seed,
            fly_backend_version=str(stack.components["fly_backend"]),
            fly_dynamics_hash=stack.components["fly_dynamics_sha256"],
            architecture="plastic-brain-v1",
            plastic_weight_hash=combined.frozen_weight_sha256,
            plasticity_rule_hash=stack.components["plasticity_rule_sha256"],
            population_hash=_hash_component(stack.components["population_sha256"]),
            sensory_mapping_hash=_hash_component(
                stack.components["sensory_mapping_sha256"]
            ),
            motor_mapping_hash=stack.components["motor_mapping_sha256"],
            reinforcement_mapping_hash=stack.components[
                "reinforcement_mapping_sha256"
            ],
            artifact_hash=_hash_component(stack.components["artifact_sha256"]),
        )
        with ReplayBundleWriter(
            candidate,
            identity,
            metadata={
                "episode": asdict(episode),
                "learning_disabled": True,
                "neural_duration_ms": float(stack.agent.processor.duration_ms),
                "neural_recording_mode": summary["neural_recording_mode"],
                "reinforcement_semantics": summary["reinforcement_semantics"],
            },
        ) as writer:
            for transition in combined.transitions:
                if transition.seed == episode.seed:
                    writer.record(asdict(transition))
            if args.record_neural:
                destination = writer.output_dir / neural_path.name
                shutil.copy2(neural_path, destination)
                writer.attach_neural_file(destination)
    print(json.dumps(summary["metrics"], sort_keys=True))
    return 0


def _hash_component(value: object) -> str:
    text = str(value)
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower()):
        return text
    return hashlib.sha256(text.encode()).hexdigest()


def _apply_evaluation_condition(
    config: PlasticExperimentConfig, args: argparse.Namespace
) -> PlasticExperimentConfig:
    fly = config.fly
    training = config.training
    if args.sensory_mapping_seed is not None:
        fly = replace(fly, sensory_mapping_seed=args.sensory_mapping_seed)
    if args.output_mode is not None:
        fly = replace(fly, mode=args.output_mode)
    if args.condition == "no_plasticity":
        if args.action_schedule is None and not training.action_schedule_path:
            raise ValueError(
                "no_plasticity evaluation requires the training --action-schedule"
            )
        training = replace(
            training,
            condition="no_plasticity",
            plasticity_enabled=False,
            action_schedule_path=str(
                args.action_schedule or training.action_schedule_path
            ),
        )
    elif args.condition in {"kc_mbon_shuffled", "whole_brain_shuffled"}:
        fly = replace(fly, topology=args.condition)
        training = replace(training, condition=args.condition)
    elif args.condition == "shuffled_reward":
        if args.reinforcement_schedule is None:
            raise ValueError("shuffled_reward evaluation requires --reinforcement-schedule")
        if args.action_schedule is None:
            raise ValueError("shuffled_reward evaluation requires --action-schedule")
        training = replace(
            training,
            condition="shuffled_reward",
            reinforcement_mode="shuffled_schedule",
            reinforcement_schedule_path=str(args.reinforcement_schedule),
            action_schedule_path=str(args.action_schedule),
        )
    else:
        training = replace(training, condition="plastic_real")
    updated = replace(config, fly=fly, training=training)
    updated.validate()
    return updated


if __name__ == "__main__":
    raise SystemExit(main())
