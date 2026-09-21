"""Run a deliberately short plasticity calibration and write safety gates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.plasticity import (
    PlasticitySafetyThresholds,
    plasticity_calibration_report,
    run_controlled_plasticity_sequence,
)
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("controlled", "integrated"), default="controlled",
        help="controlled uses fixed Hz/pulse sequences; integrated runs the environment stack",
    )
    parser.add_argument("--maximum-modified-fraction", type=float, default=0.80)
    parser.add_argument("--maximum-bound-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-mean-absolute-change", type=float, default=0.50)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if args.decisions < 2:
        raise ValueError("plasticity calibration needs at least two decisions")
    config = PlasticExperimentConfig.load(args.config)
    if args.decisions % config.environment.num_envs:
        raise ValueError("decisions must be divisible by num_envs")
    config = replace(
        config,
        training=replace(
            config.training,
            max_environment_decisions=args.decisions,
            checkpoint_every_decisions=args.decisions,
        ),
    )
    config.require_heavy_opt_in(args.heavy)
    stack = build_plastic_stack(config)
    updates: list[float] = []
    eligible_reinforcement_events = 0
    if args.mode == "controlled":
        updates.extend(
            run_controlled_plasticity_sequence(
                stack.agent.plasticity,
                decisions=args.decisions,
            )
        )
        eligible_reinforcement_events = sum(
            1
            for decision in range(args.decisions)
            if decision % 4 in {1, 2, 3}
        ) * stack.agent.plasticity.state.learners
    else:
        stack.trainer.record_sparse_changes = True

        def collect_updates(trainer: object, values: dict[str, float]) -> None:
            nonlocal eligible_reinforcement_events
            del trainer, values
            learning = stack.trainer.last_learning
            if learning is not None:
                eligible_reinforcement_events += sum(
                    event.eligible_synapses > 0
                    and (event.appetitive > 0 or event.aversive > 0)
                    for event in learning.events
                )
                updates.extend(
                    abs(value)
                    for event in learning.events
                    for value in event.efficacy_changes
                )

        stack.trainer.train(on_step=collect_updates)
    report = plasticity_calibration_report(
        stack.agent.plasticity.state,
        stack.agent.plasticity.config,
        thresholds=PlasticitySafetyThresholds(
            maximum_modified_fraction=args.maximum_modified_fraction,
            maximum_bound_fraction=args.maximum_bound_fraction,
            maximum_mean_absolute_change=args.maximum_mean_absolute_change,
            maximum_eligibility=config.plasticity.max_eligibility,
        ),
        update_magnitudes=updates,
        eligible_reinforcement_events=eligible_reinforcement_events,
    )
    report["calibration"] = {
        "config_sha256": config.sha256,
        "decisions": args.decisions,
        "mode": args.mode,
        "controlled_sequence": (
            "zero/weak/medium/reference-rate coincidence with fixed bidirectional reinforcement"
            if args.mode == "controlled"
            else None
        ),
        "plastic_topology_sha256": stack.components["plastic_topology_sha256"],
        "plasticity_rule_sha256": stack.components["plasticity_rule_sha256"],
        "synthetic_development_only": config.fly.backend == "synthetic",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["gates"]["status"]}))
    return 0 if report["gates"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
