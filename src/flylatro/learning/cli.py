"""Train Flylatro by changing sparse KC->MBON synapses inside the fly."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
from typing import Sequence

import numpy as np

from flylatro.learning.checkpoints import (
    load_plastic_checkpoint,
    save_plastic_checkpoint,
)
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.curriculum import (
    PlasticAnteCurriculum,
    PlasticCurriculumConfig,
)
from flylatro.evaluation.plastic import evaluate_plastic_fly
from flylatro.telemetry.metrics import MetricLogger
from flylatro.replay.neural import NeuralEventRecorder
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.fly.flywire_artifact import sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--condition",
        choices=(
            "plastic_real",
            "no_plasticity",
            "kc_mbon_shuffled",
            "whole_brain_shuffled",
            "shuffled_reward",
        ),
    )
    parser.add_argument("--sensory-mapping-seed", type=int)
    parser.add_argument("--output-mode", choices=("mbon_direct", "whole_brain"))
    parser.add_argument(
        "--reinforcement-schedule", "--dopamine-schedule",
        dest="reinforcement_schedule", metavar="REINFORCEMENT_SCHEDULE", type=Path,
    )
    parser.add_argument("--action-schedule", type=Path)
    parser.add_argument("--max-environment-decisions", type=int)
    parser.add_argument(
        "--budget-basis",
        help="measured benchmark or gate name justifying the exposure budget",
    )
    parser.add_argument("--checkpoint-every-decisions", type=int)
    parser.add_argument(
        "--record-plasticity-events",
        action="store_true",
        help="record sparse changed KC->MBON edges for a selected short run",
    )
    parser.add_argument(
        "--curriculum-ladder",
        help="one fixed Ante or increasing Antes ending at 8, e.g. 1 or 1,2,3,5,8",
    )
    parser.add_argument("--curriculum-evaluation-every-decisions", type=int)
    parser.add_argument("--curriculum-evaluation-episodes", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = PlasticExperimentConfig.load(args.config)
    config = _apply_overrides(config, args)
    if config.fly.backend == "flywire" and "PLACEHOLDER" in config.training.budget_basis:
        raise ValueError(
            "real training budget is an explicit placeholder; provide --budget-basis and a measured --max-environment-decisions"
        )
    config.require_heavy_opt_in(args.heavy)
    run_dir = (args.run_dir or _default_run_dir(config)).resolve()
    if run_dir.exists() and args.resume is None and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    stack = build_plastic_stack(config)
    stack.trainer.record_detailed_plasticity = args.record_plasticity_events
    curriculum = (
        PlasticAnteCurriculum(
            PlasticCurriculumConfig(
                ladder=config.curriculum.ladder,
                promotion_win_rate=config.curriculum.promotion_win_rate,
                evaluation_every_decisions=config.curriculum.evaluation_every_decisions,
                evaluation_episodes=config.curriculum.evaluation_episodes,
            )
        )
        if config.curriculum.enabled
        else None
    )
    if args.resume is not None:
        resumed = load_plastic_checkpoint(
            args.resume,
            stack.trainer,
            expected_components=stack.components,
        )
        if curriculum is not None and resumed.get("curriculum_state"):
            curriculum.load_state_dict(resumed["curriculum_state"])
            stack.env.set_win_ante(curriculum.current_ante)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "components": stack.components,
        "authority": "PLAN.MD plastic-brain V1",
        "record_plasticity_events": args.record_plasticity_events,
    }
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training_started = time.perf_counter()
    with MetricLogger(
        run_dir,
        run_id=config.name,
        tensorboard=config.runtime.tensorboard and not args.no_tensorboard,
    ) as metrics:
        dopamine_stream = (run_dir / "synthetic-reinforcement-events.jsonl").open("a", encoding="utf-8")
        action_stream = (run_dir / "training-actions.jsonl").open("a", encoding="utf-8")
        plasticity_stream = (
            (run_dir / "plasticity-events.jsonl").open("a", encoding="utf-8")
            if args.record_plasticity_events
            else None
        )
        plasticity_parquet = (
            NeuralEventRecorder(run_dir / "plasticity-events.parquet")
            if args.record_plasticity_events
            else None
        )
        def on_step(trainer: object, values: dict[str, float]) -> None:
            values = {
                **values,
                **_runtime_metrics(
                    training_started,
                    stack.trainer.state.environment_decisions,
                ),
            }
            ante_used = (
                stack.trainer.last_scheduled_ante
                if stack.trainer.last_scheduled_ante is not None
                else curriculum.current_ante
                if curriculum is not None
                else 8
            )
            if curriculum is not None:
                decision = curriculum.maybe_evaluate(
                    stack.trainer.state.environment_decisions,
                    lambda ante, seeds: _curriculum_win_rate(
                        stack, ante=ante, seeds=seeds
                    ),
                )
                if decision.evaluated:
                    values = {
                        **values,
                        "curriculum/evaluation_win_rate": float(decision.win_rate),
                        "curriculum/current_ante": float(decision.current_ante),
                        "curriculum/promoted": float(decision.promoted),
                    }
                    stack.env.set_win_ante(decision.current_ante)
            metrics.log(stack.trainer.state.environment_decisions, values)
            if stack.trainer.last_learning is not None:
                dopamine_stream.write(
                    json.dumps(
                        {
                            "environment_decisions": stack.trainer.state.environment_decisions,
                            "synthetic_reinforcement_channels": [
                                {
                                    "synthetic_appetitive": pulse.appetitive,
                                    "synthetic_aversive": pulse.aversive,
                                    "events": pulse.events,
                                }
                                for pulse in stack.trainer.last_learning.pulses
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                dopamine_stream.flush()
                if plasticity_stream is not None:
                    topology = stack.agent.plasticity.topology
                    efficacy = state_numpy(stack.agent.plasticity.state.efficacy)
                    for event in stack.trainer.last_learning.events:
                        indices = event.changed_edge_indices
                        plasticity_stream.write(
                            json.dumps(
                                {
                                    "environment_decisions": stack.trainer.state.environment_decisions,
                                    "learner": event.learner,
                                    "appetitive": event.appetitive,
                                    "aversive": event.aversive,
                                    "before_hash": event.before_hash,
                                    "after_hash": event.after_hash,
                                    "changed_edges": [
                                        {
                                            "edge_index": edge,
                                            "pre_root_id": int(topology.pre_root_ids[edge]),
                                            "post_root_id": int(topology.post_root_ids[edge]),
                                            "efficacy_change": change,
                                        }
                                        for edge, change in zip(
                                            indices,
                                            event.efficacy_changes,
                                            strict=True,
                                        )
                                    ],
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        if plasticity_parquet is not None and indices:
                            edge_indices = np.asarray(indices, dtype=np.int64)
                            changes = np.asarray(event.efficacy_changes, dtype=np.float32)
                            new_values = efficacy[event.learner, edge_indices]
                            plasticity_parquet.record(
                                decision_id=stack.trainer.state.vector_steps - 1,
                                times_ms=np.full(len(edge_indices), stack.agent.processor.duration_ms, dtype=np.float32),
                                neuron_ids=topology.post_root_ids[edge_indices],
                                roles=np.full(len(edge_indices), "plasticity", dtype=object),
                                activities=changes,
                                event_kind="plasticity",
                                pre_root_ids=topology.pre_root_ids[edge_indices],
                                post_root_ids=topology.post_root_ids[edge_indices],
                                old_efficacy=new_values - changes,
                                new_efficacy=new_values,
                            )
                    if plasticity_parquet is not None:
                        for pulse in stack.trainer.last_learning.pulses:
                            for root, role, magnitude in (
                                (-1, "synthetic_appetitive", pulse.appetitive),
                                (-2, "synthetic_aversive", pulse.aversive),
                            ):
                                if magnitude:
                                    plasticity_parquet.record(
                                        decision_id=stack.trainer.state.vector_steps - 1,
                                        times_ms=[stack.agent.processor.duration_ms],
                                        neuron_ids=[root],
                                        roles=[role],
                                        activities=[magnitude],
                                        event_kind="synthetic_reinforcement",
                                    )
                    plasticity_stream.flush()
            if stack.trainer.last_executed_actions is not None:
                action_stream.write(
                    json.dumps(
                        {
                            "environment_decisions": stack.trainer.state.environment_decisions,
                            "actions": {
                                key: value.tolist()
                                for key, value in stack.trainer.last_executed_actions.items()
                            },
                            "state_hashes_before": list(
                                stack.trainer.last_state_hashes_before
                            ),
                            "state_hashes_after": list(
                                stack.trainer.last_state_hashes_after
                            ),
                            "curriculum_ante": ante_used,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                action_stream.flush()
            decisions = stack.trainer.state.environment_decisions
            cadence = config.training.checkpoint_every_decisions
            if decisions % cadence == 0:
                save_plastic_checkpoint(
                    run_dir / f"plastic-checkpoint-{decisions:012d}.pkl",
                    stack.trainer,
                    experiment_config=config.to_dict(),
                    component_metadata=stack.components,
                    curriculum_state=(
                        curriculum.state_dict() if curriculum is not None else None
                    ),
                )

        try:
            stack.trainer.train(on_step=on_step)
        finally:
            dopamine_stream.close()
            action_stream.close()
            if plasticity_stream is not None:
                plasticity_stream.close()
            if plasticity_parquet is not None:
                plasticity_parquet.close()
    generated_action_hash = sha256_file(run_dir / "training-actions.jsonl")
    completed_at = datetime.now(timezone.utc).isoformat()
    # One canonical completed identity is written to every final product, so a
    # manifest, a summary and a checkpoint can never disagree merely because
    # one of them was serialized first.
    completed_components = finalize_components(
        stack.components,
        executed_action_schedule_sha256=generated_action_hash,
        reserved_action_legal_observations=stack.agent.motor.reserved_action_legal_count,
        completed_at=completed_at,
    )
    manifest["components"] = completed_components
    manifest["completed_at"] = completed_at
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_path = run_dir / "plastic-checkpoint-final.pkl"
    save_plastic_checkpoint(
        final_path,
        stack.trainer,
        experiment_config=config.to_dict(),
        component_metadata=completed_components,
        curriculum_state=(curriculum.state_dict() if curriculum is not None else None),
    )
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(final_path),
        "completed_at": completed_at,
        "environment_decisions": stack.trainer.state.environment_decisions,
        "episodes": stack.trainer.state.completed_episodes,
        "plasticity_events": stack.trainer.state.plasticity_events,
        "plastic_weight_sha256": stack.agent.plasticity.state.weight_sha256,
        "plastic_weight_audit": stack.agent.plasticity.weight_audit(),
        "external_trainable_parameter_count": 0,
        "runtime": _runtime_metrics(
            training_started, stack.trainer.state.environment_decisions
        ),
        "components": completed_components,
        "record_plasticity_events": args.record_plasticity_events,
    }
    (run_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "run_dir",
                    "checkpoint",
                    "environment_decisions",
                    "episodes",
                    "plasticity_events",
                    "plastic_weight_sha256",
                    "external_trainable_parameter_count",
                    "runtime",
                )
            },
            sort_keys=True,
        )
    )
    return 0


COMPLETED_COMPONENT_IDENTITY_VERSION = "completed-run-component-identity-v1"

#: Fields that must be identical in the run manifest, the run summary and the
#: final checkpoint (and its manifest). Asserted by tests.
COMPLETED_IDENTITY_FIELDS: tuple[str, ...] = (
    "executed_action_schedule_sha256",
    "action_schedule_sha256",
    "state_hash_schedule_sha256",
    "motor_mapping_sha256",
    "sensory_mapping_sha256",
    "plastic_topology_sha256",
    "plasticity_rule_sha256",
    "reinforcement_mapping_sha256",
    "canonical_motor_candidate_set_sha256",
    "fly_dynamics_sha256",
    "artifact_sha256",
    "population_sha256",
    "condition",
    "completed_at",
)


def finalize_components(
    components: dict[str, object],
    *,
    executed_action_schedule_sha256: str,
    reserved_action_legal_observations: int,
    completed_at: str,
) -> dict[str, object]:
    """Build the single immutable completed-run component identity."""

    completed = dict(components)
    declared = completed.get("action_schedule_sha256")
    if declared is None:
        completed["action_schedule_sha256"] = executed_action_schedule_sha256
        completed["state_hash_schedule_sha256"] = executed_action_schedule_sha256
    elif declared != executed_action_schedule_sha256:
        raise RuntimeError(
            "executed matched action schedule bytes differ from the source schedule"
        )
    completed["executed_action_schedule_sha256"] = executed_action_schedule_sha256
    completed["reserved_action_legal_observations"] = int(
        reserved_action_legal_observations
    )
    completed["component_identity_version"] = COMPLETED_COMPONENT_IDENTITY_VERSION
    completed["completed_at"] = completed_at
    return completed


def _default_run_dir(config: PlasticExperimentConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(config.runtime.output_root) / f"plastic-{config.name}-{stamp}"


def _apply_overrides(
    config: PlasticExperimentConfig, args: argparse.Namespace
) -> PlasticExperimentConfig:
    fly = config.fly
    training = config.training
    curriculum = config.curriculum
    if args.sensory_mapping_seed is not None:
        fly = replace(fly, sensory_mapping_seed=args.sensory_mapping_seed)
    if args.output_mode is not None:
        fly = replace(fly, mode=args.output_mode)
    if args.max_environment_decisions is not None:
        training = replace(
            training, max_environment_decisions=args.max_environment_decisions
        )
    if args.budget_basis is not None:
        training = replace(training, budget_basis=args.budget_basis)
    if args.checkpoint_every_decisions is not None:
        training = replace(
            training,
            checkpoint_every_decisions=args.checkpoint_every_decisions,
        )
    if args.curriculum_ladder:
        curriculum = replace(
            curriculum,
            enabled=True,
            ladder=tuple(int(value) for value in args.curriculum_ladder.split(",")),
        )
    if args.curriculum_evaluation_every_decisions is not None:
        curriculum = replace(
            curriculum,
            evaluation_every_decisions=args.curriculum_evaluation_every_decisions,
        )
    if args.curriculum_evaluation_episodes is not None:
        curriculum = replace(
            curriculum,
            evaluation_episodes=args.curriculum_evaluation_episodes,
        )
    condition = args.condition or training.condition
    if condition == "no_plasticity":
        if args.action_schedule is None and not training.action_schedule_path:
            raise ValueError(
                "no_plasticity requires --action-schedule from the matched source run"
            )
        training = replace(
            training,
            condition=condition,
            plasticity_enabled=False,
            action_schedule_path=str(
                args.action_schedule or training.action_schedule_path
            ),
        )
    elif condition in {"kc_mbon_shuffled", "whole_brain_shuffled"}:
        fly = replace(fly, topology=condition)
        training = replace(training, condition=condition)
    elif condition == "shuffled_reward":
        if args.reinforcement_schedule is None and not training.reinforcement_schedule_path:
            raise ValueError("shuffled_reward requires --reinforcement-schedule")
        if args.action_schedule is None and not training.action_schedule_path:
            raise ValueError("shuffled_reward requires --action-schedule")
        training = replace(
            training,
            condition=condition,
            reinforcement_mode="shuffled_schedule",
            reinforcement_schedule_path=str(
                args.reinforcement_schedule or training.reinforcement_schedule_path
            ),
            action_schedule_path=str(
                args.action_schedule or training.action_schedule_path
            ),
        )
    else:
        training = replace(training, condition="plastic_real")
    updated = replace(
        config, fly=fly, training=training, curriculum=curriculum
    )
    updated.validate()
    return updated


def _curriculum_win_rate(
    stack: object, *, ante: int, seeds: Sequence[int]
) -> float:
    """Evaluate held-out seeds and restore the exact in-flight training state."""

    if stack.env.num_envs != 1:
        raise ValueError("curriculum evaluation currently requires one sequential fly")
    snapshot = stack.trainer.state_dict()
    wins = 0
    try:
        stack.env.set_win_ante(ante)
        for seed in seeds:
            result = evaluate_plastic_fly(stack.env, stack.agent, (seed,))
            wins += result.episodes[0].won
    finally:
        stack.trainer.load_state_dict(snapshot)
    return wins / len(seeds)


def _runtime_metrics(started: float, decisions: int) -> dict[str, float]:
    elapsed = max(time.perf_counter() - started, 1e-12)
    metrics = {
        "runtime/wall_seconds": elapsed,
        "runtime/environment_decisions_per_second": decisions / elapsed,
        "runtime/process_peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        / 1024.0,
    }
    try:
        import torch
    except ImportError:
        return metrics
    if torch.cuda.is_available():
        metrics.update(
            {
                "runtime/gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "runtime/gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "runtime/gpu_peak_allocated_mb": torch.cuda.max_memory_allocated()
                / 1024**2,
            }
        )
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
