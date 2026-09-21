"""Calibrate fixed motor pools from reward-free neural activity only."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from flylatro.evaluation.state_hash import hash_observation_row
from flylatro.interface.motor import MotorMapping
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.seeds import derive_seed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--calibration-seed", type=int, required=True)
    parser.add_argument("--high-rate-hz", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 2:
        raise ValueError("motor calibration requires at least two reward-free states")
    config = PlasticExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    # State sampling must not depend on a previous calibrated motor artifact.
    config = replace(config, fly=replace(config.fly, motor_mapping_path=""))
    stack = build_plastic_stack(config, allow_uncalibrated_motor=True)
    learners = config.environment.num_envs
    rows = []
    state_hashes = []
    initial_seeds = tuple(
        derive_seed("motor-calibration-state", args.calibration_seed, row)
        for row in range(learners)
    )
    observations, masks = stack.env.reset(initial_seeds)
    decisions = 0
    while len(rows) < args.samples:
        fly_seeds = tuple(
            derive_seed("motor-calibration-fly", args.calibration_seed, decisions, row)
            for row in range(learners)
        )
        activity = stack.agent.processor.process(
            observations,
            fly_seeds=fly_seeds,
            efficacy=stack.agent.plasticity.state.efficacy,
        )
        for row in range(learners):
            rows.append(activity.output_activity[row])
            state_hashes.append(hash_observation_row(observations, row))
            if len(rows) >= args.samples:
                break
        actions = stack.agent.motor.decode(
            activity.output_activity,
            masks,
            deterministic=True,
            learner_ids=np.asarray(initial_seeds, dtype=np.int64),
            decision_ids=np.full(learners, decisions, dtype=np.int64),
        )
        step = stack.env.step(actions)
        observations, masks = step.observations, step.masks
        decisions += 1
    mapping = MotorMapping.from_reward_free_calibration(
        stack.agent.processor.output_root_ids,
        np.stack(rows),
        mode=config.fly.mode,
        pool_width=config.fly.motor_pool_width,
        high_rate_hz=args.high_rate_hz,
        exploration_epsilon=config.motor.exploration_epsilon,
        exploration_temperature=config.motor.exploration_temperature,
        exploration_seed=config.motor.exploration_seed,
        calibration_metadata={
            "calibration_seed": args.calibration_seed,
            "initial_environment_seeds": list(initial_seeds),
            "fly_seed_rule": "derive_seed(motor-calibration-fly, calibration_seed, decision, row)",
            "observable_state_hashes": state_hashes,
            "state_sampling": "fixed deterministic bootstrap decoder trajectory; rewards and outcomes ignored",
            "decision_duration_ms": config.fly.duration_ms,
            "reward_or_outcome_observed": False,
        },
    )
    mapping.save(args.output)
    print(json.dumps({"output": str(args.output), "sha256": mapping.sha256, "samples": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
