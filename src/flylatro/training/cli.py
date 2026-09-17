"""Guarded PPO training entry point for development and dedicated machines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from flylatro.evaluation.evaluator import evaluate_policy
from flylatro.seeds import SeedPlan
from flylatro.telemetry.metrics import MetricLogger
from flylatro.training.checkpoints import load_checkpoint, save_checkpoint
from flylatro.training.config import (
    ExperimentConfig,
    build_environment,
    build_training_stack,
)
from flylatro.training.curriculum import AnteCurriculum, CurriculumConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dev-v1.toml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument(
        "--heavy", action="store_true",
        help="explicitly allow real/GPU/long profiles on the dedicated machine",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    max_updates = config.ppo.max_updates if args.max_updates is None else args.max_updates
    if max_updates < 1:
        raise ValueError("max-updates must be positive")
    if max_updates > 10 and not args.heavy:
        raise ValueError("more than 10 updates requires --heavy")
    run_dir = args.run_dir or _default_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    config_path = run_dir / "config.json"
    expected_config = config.payload()
    config_text = json.dumps(expected_config, indent=2, sort_keys=True) + "\n"
    canonical_config = json.loads(config_text)
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != canonical_config:
            raise ValueError("resume config differs from the run directory snapshot")
    else:
        config_path.write_text(config_text, encoding="utf-8")
    stack = build_training_stack(config)
    seed_plan = SeedPlan()
    curriculum = _curriculum(config)
    if args.resume is not None:
        payload = load_checkpoint(args.resume, stack.trainer)
        if curriculum is not None:
            curriculum.load_state_dict(payload["curriculum_state"])
            stack.env.set_win_ante(curriculum.current_ante)
    with MetricLogger(
        run_dir,
        run_id=run_dir.name,
        tensorboard=config.runtime.tensorboard and not args.no_tensorboard,
    ) as logger:
        while stack.trainer.update_index < max_updates:
            metrics = stack.trainer.train_update()
            metrics["model/trainable_parameters"] = float(
                stack.policy.trainable_parameter_count
            )
            if curriculum is not None:
                decision = curriculum.maybe_evaluate(
                    stack.trainer.update_index,
                    lambda ante, seeds: _curriculum_win_rate(
                        config, stack.processor, stack.policy, ante, seeds
                    ),
                )
                if decision.evaluated:
                    metrics["curriculum/evaluation_win_rate"] = float(decision.win_rate)
                    metrics["curriculum/current_ante"] = float(decision.current_ante)
                    metrics["curriculum/promoted"] = float(decision.promoted)
                if decision.promoted:
                    stack.env.set_win_ante(decision.current_ante)
            logger.log(stack.trainer.global_step, metrics)
            if stack.trainer.update_index % config.ppo.checkpoint_every_updates == 0:
                _checkpoint(run_dir, config, stack, curriculum, seed_plan)
    manifest = _checkpoint(run_dir, config, stack, curriculum, seed_plan)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "global_step": stack.trainer.global_step,
                "updates": stack.trainer.update_index,
                "checkpoint_manifest": str(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


def _curriculum(config: ExperimentConfig) -> AnteCurriculum | None:
    if not config.curriculum.enabled:
        return None
    settings = config.curriculum
    return AnteCurriculum(
        CurriculumConfig(
            ladder=settings.ladder,
            promotion_win_rate=settings.promotion_win_rate,
            evaluation_frequency_updates=settings.evaluation_frequency_updates,
            evaluation_episodes=settings.evaluation_episodes,
        )
    )


def _curriculum_win_rate(config, processor, policy, ante, seeds) -> float:
    wins = 0
    count = 0
    width = config.environment.num_envs
    env = build_environment(config, win_ante=ante)
    env.set_shaping_beta(config.reward.shaping_beta)
    for start in range(0, len(seeds), width):
        batch = seeds[start : start + width]
        result = evaluate_policy(
            env,
            processor,
            policy,
            batch,
            device=config.runtime.device,
            deterministic=True,
            max_vector_steps=config.evaluation.max_vector_steps,
        )
        wins += len(result.successful_seeds)
        count += len(result.episodes)
    return wins / count


def _checkpoint(run_dir, config, stack, curriculum, seed_plan):
    checkpoint = run_dir / f"checkpoint-{stack.trainer.global_step:012d}.pt"
    return save_checkpoint(
        checkpoint,
        stack.trainer,
        experiment_config=config.payload(),
        component_metadata=stack.components,
        seed_metadata={"plan_sha256": seed_plan.sha256},
        curriculum_state=curriculum.state_dict() if curriculum else {},
        reward_config={
            "shaping_beta": config.reward.shaping_beta,
            "progress_component": config.reward.progress_component,
            "blind_clear_base": config.reward.blind_clear_base,
            "final_win_bonus": config.reward.final_win_bonus,
        },
    )


def _default_run_dir(config: ExperimentConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(config.runtime.output_root) / f"{config.name}-{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
