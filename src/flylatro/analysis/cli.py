"""Write inspectable KC->MBON synaptic-change diagnostics for a checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.plasticity import plasticity_diagnostics, synaptic_change_report
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learner", type=int, default=0)
    parser.add_argument("--output-mode", choices=("mbon_direct", "whole_brain"))
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
    parser.add_argument(
        "--reinforcement-schedule", "--dopamine-schedule",
        dest="reinforcement_schedule", metavar="REINFORCEMENT_SCHEDULE", type=Path,
    )
    parser.add_argument("--action-schedule", type=Path)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    config = PlasticExperimentConfig.load(args.config)
    if args.output_mode is not None:
        config = replace(config, fly=replace(config.fly, mode=args.output_mode))
    if args.sensory_mapping_seed is not None:
        config = replace(
            config,
            fly=replace(
                config.fly, sensory_mapping_seed=args.sensory_mapping_seed
            ),
        )
    if args.condition == "no_plasticity":
        if args.action_schedule is None:
            raise ValueError("no_plasticity analysis requires --action-schedule")
        config = replace(
            config,
            training=replace(
                config.training,
                condition="no_plasticity",
                plasticity_enabled=False,
                action_schedule_path=str(args.action_schedule),
            ),
        )
    elif args.condition in {"kc_mbon_shuffled", "whole_brain_shuffled"}:
        config = replace(
            config,
            fly=replace(config.fly, topology=args.condition),
            training=replace(config.training, condition=args.condition),
        )
    elif args.condition == "shuffled_reward":
        if args.reinforcement_schedule is None or args.action_schedule is None:
            raise ValueError(
                "shuffled_reward analysis requires both schedule paths"
            )
        config = replace(
            config,
            training=replace(
                config.training,
                condition="shuffled_reward",
                reinforcement_mode="shuffled_schedule",
                reinforcement_schedule_path=str(args.reinforcement_schedule),
                action_schedule_path=str(args.action_schedule),
            ),
        )
    else:
        config = replace(
            config,
            training=replace(config.training, condition="plastic_real"),
        )
    config.validate()
    config.require_heavy_opt_in(args.heavy)
    stack = build_plastic_stack(config)
    load_plastic_checkpoint(
        args.checkpoint, stack.trainer, expected_components=stack.components
    )
    report = {
        "synaptic_changes": synaptic_change_report(
            stack.agent.plasticity.topology,
            stack.agent.plasticity.state,
            learner=args.learner,
        ),
        "diagnostics": plasticity_diagnostics(
            stack.agent.plasticity.state,
            stack.agent.plasticity.config,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
