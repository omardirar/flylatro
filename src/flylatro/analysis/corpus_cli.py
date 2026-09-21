"""Build the frozen reward-free calibration state corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.corpus import build_calibration_corpus
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--environment-seeds",
        required=True,
        help="comma-separated deterministic simulator seeds, e.g. 9000001,9000002",
    )
    parser.add_argument("--states-per-seed", type=int, required=True)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--navigation-seed", type=int, required=True)
    parser.add_argument("--maximum-decisions-per-seed", type=int)
    parser.add_argument(
        "--store-snapshots",
        action="store_true",
        help="store exact environment snapshots where the backend supports bytes",
    )
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    seeds = tuple(
        int(value) for value in args.environment_seeds.split(",") if value.strip()
    )
    if not seeds:
        parser.error("at least one environment seed is required")
    config = PlasticExperimentConfig.load(args.config)
    if config.environment.num_envs != 1:
        raise ValueError("corpus generation uses the one-sequential-environment form")
    config.require_heavy_opt_in(args.heavy)
    env = build_plastic_environment(config)
    corpus = build_calibration_corpus(
        env,
        environment_seeds=seeds,
        states_per_seed=args.states_per_seed,
        sample_every=args.sample_every,
        navigation_seed=args.navigation_seed,
        maximum_decisions_per_seed=args.maximum_decisions_per_seed,
        store_snapshots=args.store_snapshots,
    )
    manifest = corpus.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(manifest),
                "sha256": corpus.sha256,
                "states": len(corpus),
                "phases_observed": corpus.coverage()["phases_observed"],
                "phases_missing": corpus.coverage()["phases_missing"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
