"""Safe command-line entry point for the lightweight vertical slice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from flylatro import __version__
from flylatro.agent import FlyAgent
from flylatro.config import AppConfig, config_payload
from flylatro.env.mock import MockBalatroEnv
from flylatro.fly.backend import TinyGraphFlyBackend
from flylatro.fly.encoder import EncoderSpec, FixedBalatroEncoder
from flylatro.fly.features import FeatureSpec, RateFeatureExtractor
from flylatro.policy.structured import StructuredLinearPolicy
from flylatro.replay.recorder import JsonlTransitionRecorder
from flylatro.runner.episodes import run_episodes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the lightweight Flylatro mock vertical slice."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dev.toml"), help="TOML config"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/smoke"), help="trace root"
    )
    parser.add_argument(
        "--allow-large-dev-run",
        action="store_true",
        help="explicitly exceed conservative mock-run limits",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.load(args.config)
    config.check_development_limits(allow_large=args.allow_large_dev_run)
    run_id = datetime.now(timezone.utc).strftime("smoke-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_root / run_id

    encoder_spec = EncoderSpec.development_default(
        neuron_count=config.fly.neuron_count,
        max_cards=8,
        max_targets=8,
    )
    readout_start = config.fly.neuron_count - config.fly.readout_count
    feature_spec = FeatureSpec(
        version="tiny-output-v1",
        readout_neuron_ids=tuple(range(readout_start, config.fly.neuron_count)),
    )
    encoder = FixedBalatroEncoder(encoder_spec)
    backend = TinyGraphFlyBackend(
        neuron_count=config.fly.neuron_count,
        graph_seed=config.fly.graph_seed,
        edge_probability=config.fly.edge_probability,
    )
    extractor = RateFeatureExtractor(feature_spec)
    policy = StructuredLinearPolicy(
        feature_size=feature_spec.output_size,
        seed=config.policy.seed,
    )
    agent = FlyAgent(
        encoder=encoder,
        fly_backend=backend,
        feature_extractor=extractor,
        policy=policy,
        duration_ms=config.fly.duration_ms,
        microbatch_size=config.fly.microbatch_size,
    )
    env = MockBalatroEnv(
        num_envs=config.environment.num_envs,
        hand_size=config.environment.hand_size,
        blind_target=config.environment.blind_target,
        initial_hands=config.environment.initial_hands,
        initial_discards=config.environment.initial_discards,
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "lightweight-mock-vertical-slice",
        "flylatro_version": __version__,
        "git": _git_metadata(),
        "config": config_payload(config),
        "seeds": list(config.seeds()),
        "components": {
            "balatro_simulator": env.simulator_version,
            "encoder_version": encoder_spec.version,
            "encoder_hash": encoder_spec.sha256,
            "fly_backend": backend.backend_version,
            "connectivity_hash": backend.connectivity_hash,
            "feature_version": feature_spec.version,
            "feature_hash": feature_spec.sha256,
            "policy_version": policy.policy_version,
            "trainable_parameter_count": policy.trainable_parameter_count,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    with JsonlTransitionRecorder(output_dir, manifest) as recorder:
        summary = run_episodes(
            env,
            agent,
            config.seeds(),
            recorder=recorder,
            deterministic_policy=config.policy.deterministic,
            max_decisions=config.run.max_decisions,
        )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episodes": summary.episodes,
                "wins": summary.wins,
                "win_rate": summary.win_rate,
                "decisions": summary.decisions,
                "total_reward": summary.total_reward,
                "trainable_parameters": policy.trainable_parameter_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _git_metadata() -> dict[str, object]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ("git", *args), capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
