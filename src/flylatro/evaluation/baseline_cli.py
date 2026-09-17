"""Evaluate random-legal or explicitly isolated upstream heuristic baselines."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from flylatro.evaluation.baselines import (
    ExternalHeuristicBaseline,
    RandomLegalPolicy,
    evaluate_action_baseline,
    heuristic_action_source,
)
from flylatro.evaluation.evaluator import EvaluationResult
from flylatro.seeds import SeedPlan, derive_seed
from flylatro.training.config import ExperimentConfig, build_environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=("random_legal", "heuristic_external"), required=True
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    config = ExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    count = args.episodes or config.evaluation.episodes
    if count > 100 and not args.heavy:
        raise ValueError("more than 100 episodes requires --heavy")
    if config.evaluation.stream == "final_test":
        raise ValueError("baseline CLI cannot consume final-test seeds implicitly")
    seeds = SeedPlan().seeds(
        config.evaluation.stream,
        count,
        offset=config.evaluation.seed_offset,
    )
    results = []
    width = config.environment.num_envs
    for start in range(0, count, width):
        batch = seeds[start : min(start + width, count)]
        batch_config = replace(
            config, environment=replace(config.environment, num_envs=len(batch))
        )
        env = build_environment(batch_config, win_ante=8)
        env.set_shaping_beta(config.reward.shaping_beta)
        if args.condition == "random_legal":
            baseline = RandomLegalPolicy(derive_seed("random-baseline", start))
            source = baseline.act
        else:
            baseline = ExternalHeuristicBaseline(env)
            source = heuristic_action_source(baseline)
        results.append(
            evaluate_action_baseline(
                env, batch, source,
                max_vector_steps=config.evaluation.max_vector_steps,
            )
        )
    result = EvaluationResult(
        tuple(episode for item in results for episode in item.episodes),
        tuple(transition for item in results for transition in item.transitions),
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "config_sha256": config.sha256,
        "seed_stream": config.evaluation.stream,
        "metrics": result.metrics(),
        "expert_actions_exposed_to_fly_policy": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
