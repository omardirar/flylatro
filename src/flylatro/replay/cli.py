"""Verify a replay in the simulator, then optionally apply it to live Balatro."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tomllib
from typing import Sequence

from flylatro.replay.bundle import ReplayBundle
from flylatro.replay.live import BalatrobotClient, play_live_replay
from flylatro.replay.verify import verify_replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument(
        "--live", action="store_true",
        help="after simulator verification, mutate a running local Balatro game",
    )
    parser.add_argument("--url", default="http://127.0.0.1:12346")
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--cash-out-delay-seconds", type=float, default=8.0)
    args = parser.parse_args(argv)
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    if "fly" in raw and "training" in raw:
        from flylatro.learning.config import (
            PlasticExperimentConfig,
            build_plastic_environment,
        )

        config = PlasticExperimentConfig.load(args.config)
        config.require_heavy_opt_in(args.heavy)
        one = replace(
            config, environment=replace(config.environment, num_envs=1)
        )
        env = build_plastic_environment(one)
        env.set_win_ante(8)
    else:
        from flylatro.training.config import ExperimentConfig, build_environment

        config = ExperimentConfig.load(args.config)
        config.require_heavy_opt_in(args.heavy)
        one = replace(
            config, environment=replace(config.environment, num_envs=1)
        )
        env = build_environment(one, win_ante=8)
    bundle = ReplayBundle.open(args.bundle)
    divergences = verify_replay(bundle, env)
    payload = {
        "bundle": str(bundle.path),
        "simulator_verified": not divergences,
        "decisions": len(bundle.decisions),
    }
    if args.live:
        report = play_live_replay(
            bundle,
            BalatrobotClient(args.url),
            settle_seconds=args.settle_seconds,
            cash_out_delay_seconds=args.cash_out_delay_seconds,
        )
        payload.update(
            live_applied=True,
            live_final_state=report.final_state.get("state"),
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
