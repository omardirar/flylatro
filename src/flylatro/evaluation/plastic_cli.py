"""Evaluate a frozen plastic fly on held-out Balatro seed streams."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import shutil
import time
from typing import Sequence

import numpy as np

from flylatro.evaluation.plastic import PlasticEvaluationResult, evaluate_plastic_fly
from flylatro.learning.checkpoints import load_plastic_checkpoint, load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_environment, build_plastic_stack
from flylatro.seeds import SeedPlan
from flylatro.replay.bundle import ReplayBundleWriter, ReplayIdentity
from flylatro.fly.flywire_artifact import sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--learner-index", type=int,
        help="explicitly select one zero-based trained learner from a multi-fly checkpoint; other flies are not averaged or shared")
    parser.add_argument("--initial-weights", action="store_true",
        help="evaluate verified checkpoint initial efficacies as an unchanged-weight baseline; do not replay training actions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1,
        help="parallel held-out seed lanes using copies of the same frozen checkpoint")
    parser.add_argument("--win-ante", type=int, choices=range(1, 9), default=8,
                        help="Ante whose completion counts as a win (default: 8)")
    parser.add_argument("--progress", action="store_true", help="print each completed held-out episode")
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
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.episodes > 100 and not args.heavy:
        raise ValueError("more than 100 evaluation episodes requires --heavy")
    if args.seed_stream == "final_test" and not args.unlock_final_test:
        raise ValueError("final-test seeds require --unlock-final-test")
    if args.record_neural and args.episodes != 1:
        raise ValueError("detailed neural recording requires exactly one episode")
    if args.record_spikes and (not args.record_neural or args.seed_stream not in {"showcase", "final_test"}):
        raise ValueError("--record-spikes requires --record-neural and showcase or unlocked final-test seeds")
    if args.output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output_dir}")
    config = PlasticExperimentConfig.load(args.config)
    config = _apply_evaluation_condition(config, args)
    config.require_heavy_opt_in(args.heavy)
    if args.learner_index is not None and not 0 <= args.learner_index < config.environment.num_envs:
        raise ValueError("learner index is outside the checkpoint configuration")
    if config.environment.num_envs != 1 and args.learner_index is None:
        raise ValueError(
            "multi-fly frozen evaluation requires an explicit --learner-index"
        )
    stack = build_plastic_stack(config)
    if args.record_spikes:
        if config.fly.backend != "flywire":
            raise ValueError("time-resolved showcase spikes require the real FlyWire backend")
        stack.agent.processor.backend.record_events = True
    checkpoint = load_plastic_checkpoint(
        args.checkpoint,
        stack.trainer,
        expected_components=stack.components,
        require_matching_exposure_budget=False,
    )
    source_weight_sha256 = stack.agent.plasticity.state.weight_sha256
    if args.learner_index is not None:
        stack.agent.plasticity.state = _select_frozen_learner(stack.agent.plasticity.state, args.learner_index)
        stack.env = _frozen_environment(config, 1, args.win_ante, stack.env._beta)
    if args.initial_weights:
        _restore_initial_efficacy(stack.agent.plasticity.state)
    # This describes the training exposure that produced the frozen weights,
    # not the unused training budget in the evaluation configuration.
    stack.components["exposure_budget_decisions"] = checkpoint["components"]["exposure_budget_decisions"]
    stack.env.set_win_ante(args.win_ante)
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
    evaluation_started = time.perf_counter()
    try:
        with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as episode_stream:
            for start in range(0, len(seeds), args.batch_size):
                batch_seeds = seeds[start:start + args.batch_size]
                result = _evaluate_seed_batch(stack, config, batch_seeds, win_ante=args.win_ante,
                    max_vector_steps=args.max_vector_steps, neural_recorder=neural_recorder)
                results.append(result)
                for episode in result.episodes:
                    episode_stream.write(json.dumps(asdict(episode), sort_keys=True) + "\n")
                episode_stream.flush()
                if args.progress:
                    print(json.dumps({"completed_episodes": start + len(batch_seeds), "requested_episodes": args.episodes,
                                      "seeds": list(batch_seeds), "wins": sum(episode.won for episode in result.episodes)}), flush=True)
    finally:
        if neural_recorder is not None:
            neural_recorder.close()
    combined = PlasticEvaluationResult(
        episodes=tuple(episode for result in results for episode in result.episodes),
        transitions=tuple(
            transition for result in results for transition in result.transitions
        ),
        frozen_weight_sha256=results[0].frozen_weight_sha256,
    )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    summary = {
        "version": "frozen-plastic-evaluation-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "plastic-brain-v1",
        "learning_disabled": True,
        "evaluation_weight_source": "initial-efficacy-from-verified-checkpoint" if args.initial_weights else "trained-checkpoint",
        "source_checkpoint_weight_sha256": source_weight_sha256,
        "source_checkpoint_learners": config.environment.num_envs,
        "selected_learner_index": args.learner_index,
        "evaluated_learner_semantics": "one explicitly selected frozen learner" if args.learner_index is not None else "one frozen checkpoint learner",
        "plastic_weight_sha256": combined.frozen_weight_sha256,
        "checkpoint_manifest": checkpoint_manifest,
        "config_sha256": config.sha256,
        "seed_stream": args.seed_stream,
        "evaluation_batch_size": args.batch_size,
        "evaluation_replica_semantics": "independent held-out seeds; fixed copies of one frozen trained fly; no learning or training-trajectory replay",
        "seeds": list(seeds),
        "win_ante": args.win_ante,
        "action_policy": "independent-fixed-motor-choices-no-training-action-replay",
        "training_action_schedule_replayed": False,
        "episodes": [asdict(episode) for episode in combined.episodes],
        "seed_range": [seeds[0], seeds[-1]],
        "metrics": combined.metrics(),
        "runtime": {"evaluation_wall_seconds": evaluation_wall_seconds,
            "completed_environment_decisions": sum(episode.decisions for episode in combined.episodes),
            "completed_environment_decisions_per_second": sum(episode.decisions for episode in combined.episodes) / evaluation_wall_seconds},
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
        # Baseline overrides and selected learner rows cannot be replayed
        # by loading the full source checkpoint alone. Preserve transitions,
        # but do not publish bundles the generic replayer cannot reproduce.
        if args.initial_weights or args.learner_index is not None:
            continue
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
                "win_ante": args.win_ante,
                "shaping_beta": float(stack.env._beta),
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


def _restore_initial_efficacy(state: object) -> None:
    """Set an explicitly labelled baseline before frozen evaluation begins."""
    if getattr(state, "is_torch", False):
        import torch
        with torch.no_grad():
            state.efficacy.copy_(state.initial_efficacy)
    else:
        np.copyto(state.efficacy, state.initial_efficacy)
    # Fast traces never influence frozen action selection, but baseline state
    # should have no residual learning credit from the source checkpoint.
    state.reset_fast_traces(eligibility=True)


def _select_frozen_learner(state: object, index: int) -> object:
    """Copy one verified learner row without changing the source checkpoint."""
    if not 0 <= index < state.learners:
        raise ValueError("learner index is outside the plastic state")
    arrays = {key: value[index:index + 1].copy() for key, value in state.state_dict().items()}
    state_type = type(state)
    return state_type.from_state_dict(arrays, device=str(state.device)) if getattr(state, 'is_torch', False) else state_type.from_state_dict(arrays)


def _frozen_environment(config: PlasticExperimentConfig, width: int, win_ante: int, beta: float):
    replica = replace(config, environment=replace(config.environment, num_envs=width))
    env = build_plastic_environment(replica)
    env.set_win_ante(win_ante)
    env.set_shaping_beta(beta)
    return env


def _evaluate_seed_batch(stack: object, config: PlasticExperimentConfig, seeds: Sequence[int], *,
                         win_ante: int, max_vector_steps: int, neural_recorder: object | None) -> PlasticEvaluationResult:
    if len(seeds) == 1:
        return evaluate_plastic_fly(stack.env, stack.agent, seeds,
            max_vector_steps=max_vector_steps, neural_recorder=neural_recorder)
    if neural_recorder is not None:
        raise ValueError("neural showcase recording requires one seed lane")
    from flylatro.fly.mushroom_body.state import state_numpy

    original = stack.agent.plasticity.state
    original_hash = original.weight_sha256
    arrays = {key: np.repeat(value, len(seeds), axis=0) for key, value in original.state_dict().items()}
    state_type = type(original)
    cloned = state_type.from_state_dict(arrays, device=str(original.device)) if getattr(original, "is_torch", False) else state_type.from_state_dict(arrays)
    if not np.array_equal(state_numpy(cloned.efficacy), arrays["efficacy"]):
        raise ValueError("frozen evaluation replicas differ from their checkpoint")
    env = _frozen_environment(config, len(seeds), win_ante, stack.env._beta)
    stack.agent.plasticity.state = cloned
    try:
        result = evaluate_plastic_fly(env, stack.agent, seeds, max_vector_steps=max_vector_steps)
        # evaluate_plastic_fly checks complete cloned weights and fast traces.
        # The public learned-weight identity remains the original checkpoint's
        # one efficacy vector, rather than a concatenated replica-batch hash.
        if original.weight_sha256 != original_hash:
            raise ValueError("source checkpoint mutated during frozen evaluation")
        order = {int(seed): index for index, seed in enumerate(seeds)}
        return replace(result, episodes=tuple(sorted(result.episodes, key=lambda episode: order[episode.seed])),
            frozen_weight_sha256=original_hash)
    finally:
        stack.agent.plasticity.state = original


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
