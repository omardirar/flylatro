"""Measure naive fly representation health before long plastic training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from flylatro.analysis.representation import representation_diagnostics
from flylatro.evaluation.state_hash import hash_observation_row
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.seeds import derive_seed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="independent neural trials per identical Balatro state",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 2:
        raise ValueError("representation diagnostics require at least two samples")
    if args.repeats < 2:
        raise ValueError("at least two repeats are required for same-state variability")
    if args.samples > 32 and not args.heavy:
        raise ValueError("more than 32 diagnostic samples requires --heavy")
    config = PlasticExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    if config.environment.num_envs != 1:
        raise ValueError("diagnostics require the one-sequential-fly configuration")
    stack = build_plastic_stack(config)
    observations, masks = stack.env.reset((12_000_000,))
    kc = []
    mbon = []
    descending = []
    labels = []
    for decision in range(args.samples):
        output = None
        state_label = int(hash_observation_row(observations, 0)[:16], 16) % (2**63)
        for repeat in range(args.repeats):
            output = stack.agent.act(
                observations,
                masks,
                fly_seeds=(derive_seed("representation", decision, repeat),),
                deterministic_motor=True,
                record_eligibility=False,
            )
            kc.append(output.neural.kc_activity[0])
            mbon.append(output.neural.mbon_activity[0])
            descending.append(output.neural.descending_activity[0])
            labels.append(state_label)
        assert output is not None
        step = stack.env.step(output.actions)
        observations, masks = step.observations, step.masks
    report = representation_diagnostics(
        np.stack(kc),
        np.stack(mbon),
        np.stack(descending),
        state_labels=np.asarray(labels, dtype=np.int64),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
