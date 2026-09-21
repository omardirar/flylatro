"""Measure whether reinforcement-driven change concentrates on the chosen action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from flylatro.analysis.evidence import experiment_identity, write_report
from flylatro.analysis.specificity import (
    SPECIFICITY_REPORT_VERSION,
    ChosenActionSpecificity,
)
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.seeds import SeedPlan, derive_seed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-decision-detail",
        action="store_true",
        help="record the full per-decision attribution table",
    )
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if args.decisions < 1:
        raise ValueError("specificity diagnostics need at least one decision")
    config = PlasticExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    if config.environment.num_envs != 1:
        raise ValueError("specificity diagnostics use the one-sequential-fly form")
    if not config.training.plasticity_enabled:
        raise ValueError("specificity diagnostics require plasticity to be enabled")
    stack = build_plastic_stack(config)
    agent = stack.agent
    agent.motor.record_choices = True
    recorder = ChosenActionSpecificity(agent.plasticity.topology, agent.motor.mapping)
    seeds = SeedPlan().seeds(
        "training", 1, offset=config.training.training_seed_offset
    )
    observations, masks = stack.env.reset(seeds)
    for decision in range(args.decisions):
        fly_seeds = (
            derive_seed("specificity-fly", config.training.base_fly_seed, decision),
        )
        result = agent.act(
            observations,
            masks,
            fly_seeds=fly_seeds,
            deterministic_motor=True,
            motor_learner_ids=seeds,
            motor_decision_ids=(decision,),
        )
        choices = agent.motor.last_choices[0]
        eligibility = state_numpy(agent.plasticity.state.eligibility)[0].copy()
        before = state_numpy(agent.plasticity.state.efficacy)[0].copy()
        step = stack.env.step(result.actions)
        agent.learn(step.infos, plasticity_enabled=True)
        after = state_numpy(agent.plasticity.state.efficacy)[0]
        recorder.record(
            eligibility=eligibility,
            efficacy_delta=np.asarray(after, dtype=np.float64) - before,
            choices=choices,
            keep_detail=args.per_decision_detail,
        )
        observations, masks = step.observations, step.masks
    report = recorder.report()
    report["condition"] = config.training.condition
    report["reserved_action_legal_observations"] = agent.motor.reserved_action_legal_count
    write_report(
        args.output,
        report,
        experiment_identity(
            config,
            stack.components,
            report_kind="chosen_action_specificity",
            report_version=SPECIFICITY_REPORT_VERSION,
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decisions": report["decisions"],
                "chosen_motor_update_fraction": report["chosen_motor_update_fraction"],
                "competing_motor_update_fraction": report["competing_motor_update_fraction"],
                "non_motor_update_fraction": report["non_motor_update_fraction"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
