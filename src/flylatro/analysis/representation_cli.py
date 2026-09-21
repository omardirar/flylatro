"""Measure naive fly representation health before long plastic training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from flylatro.analysis.representation import RepresentationThresholds, representation_diagnostics
from flylatro.env.upstream_contract import (
    BLIND_REQ_OFF,
    BLIND_SCORED_OFF,
    GLOBAL_ANTE_OFF,
    GLOBAL_HANDS_LEFT,
    GLOBAL_PHASE_OFF,
)
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
    parser.add_argument("--high-rate-hz", type=float, default=200.0)
    parser.add_argument("--silence-hz", type=float, default=0.0)
    parser.add_argument("--maximum-silent-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-high-rate-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-separation-ratio", type=float, default=1.10)
    parser.add_argument("--minimum-motor-dynamic-range-hz", type=float, default=1.0)
    parser.add_argument("--minimum-action-coverage-fraction", type=float, default=0.50)
    parser.add_argument("--duration-ms", type=float)
    args = parser.parse_args(argv)
    if args.samples < 2:
        raise ValueError("representation diagnostics require at least two samples")
    if args.repeats < 2:
        raise ValueError("at least two repeats are required for same-state variability")
    if args.samples > 32 and not args.heavy:
        raise ValueError("more than 32 diagnostic samples requires --heavy")
    config = PlasticExperimentConfig.load(args.config)
    if args.duration_ms is not None:
        config = replace(config, fly=replace(config.fly, duration_ms=args.duration_ms))
    config.require_heavy_opt_in(args.heavy)
    if config.environment.num_envs != 1:
        raise ValueError("diagnostics require the one-sequential-fly configuration")
    stack = build_plastic_stack(config, allow_uncalibrated_motor=True)
    observations, masks = stack.env.reset((12_000_000,))
    kc = []
    mbon = []
    descending = []
    labels = []
    phase_labels = []
    ante_labels = []
    hands_remaining_labels = []
    blind_progress_labels = []
    rank_presence_labels = []
    shop_presence_labels = []
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
            phase_labels.append(int(np.argmax(observations["global"][0, GLOBAL_PHASE_OFF:GLOBAL_PHASE_OFF + 6])))
            ante_labels.append(int(np.argmax(observations["global"][0, GLOBAL_ANTE_OFF:GLOBAL_ANTE_OFF + 9])) + 1)
            hands_remaining_labels.append(
                int(round(float(observations["global"][0, GLOBAL_HANDS_LEFT]) * 1000))
            )
            blind_required = float(np.expm1(observations["blind"][0, BLIND_REQ_OFF]))
            blind_scored = float(np.expm1(observations["blind"][0, BLIND_SCORED_OFF]))
            blind_progress_labels.append(
                int(np.clip(round(10 * blind_scored / max(blind_required, 1e-12)), 0, 10))
            )
            rank_bits = np.any(observations["hand"][0, :, :13] > 0, axis=0)
            rank_presence_labels.append(
                int(sum((1 << index) for index, present in enumerate(rank_bits) if present))
            )
            shop_presence_labels.append(
                int(np.any(observations["shop_feats"][0] != 0))
            )
        assert output is not None
        step = stack.env.step(output.actions)
        observations, masks = step.observations, step.masks
    report = representation_diagnostics(
        np.stack(kc),
        np.stack(mbon),
        np.stack(descending),
        state_labels=np.asarray(labels, dtype=np.int64),
        observable_categories={
            "phase": np.asarray(phase_labels, dtype=np.int64),
            "ante": np.asarray(ante_labels, dtype=np.int64),
            "hands_remaining": np.asarray(hands_remaining_labels, dtype=np.int64),
            "blind_progress_decile": np.asarray(blind_progress_labels, dtype=np.int64),
            "rank_presence_signature": np.asarray(rank_presence_labels, dtype=np.int64),
            "shop_presence": np.asarray(shop_presence_labels, dtype=np.int64),
        },
        motor_pools=stack.agent.motor.mapping.pools,
        thresholds=RepresentationThresholds(
            silence_hz=args.silence_hz,
            high_rate_hz=args.high_rate_hz,
            maximum_silent_fraction=args.maximum_silent_fraction,
            maximum_high_rate_fraction=args.maximum_high_rate_fraction,
            minimum_separation_ratio=args.minimum_separation_ratio,
            minimum_motor_dynamic_range_hz=args.minimum_motor_dynamic_range_hz,
            minimum_action_coverage_fraction=args.minimum_action_coverage_fraction,
        ),
    )
    report["backend_evidence"] = (
        "real_flywire_spike_rate_hz"
        if config.fly.backend == "flywire"
        else "synthetic_rate_hz_proxy_development_only"
    )
    report["decision_duration_ms"] = config.fly.duration_ms
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
