"""Measure fly representation health before and after motor calibration.

``--stage pre`` runs before any motor artifact exists and makes no motor
readiness claim.  ``--stage post`` re-runs with the persisted final motor
mapping and is the only report that may satisfy motor preflight gates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from flylatro.analysis.evidence import (
    experiment_identity,
    load_corpus,
    observable_category_labels,
    record_corpus_activity,
    write_report,
)
from flylatro.analysis.reachability import (
    REACHABILITY_REPORT_VERSION,
    ReachabilityThresholds,
    kc_reachability_report,
)
from flylatro.analysis.representation import (
    MOTOR_POPULATION_BY_MODE,
    REPRESENTATION_REPORT_VERSION,
    RepresentationThresholds,
    representation_diagnostics,
)
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("pre", "post"), default="pre")
    parser.add_argument("--calibration-corpus", type=Path)
    parser.add_argument(
        "--states", type=int, help="limit the corpus states used (default: all)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="independent neural trials per identical Balatro state",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--activity-seed", type=int, default=12_000_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reachability-output",
        type=Path,
        help="also write the KC-subtype reachability report",
    )
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--high-rate-hz", type=float, default=200.0)
    parser.add_argument("--silence-hz", type=float, default=0.0)
    parser.add_argument("--maximum-silent-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-high-rate-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-separation-ratio", type=float, default=1.10)
    parser.add_argument("--minimum-motor-dynamic-range-hz", type=float, default=1.0)
    parser.add_argument("--minimum-action-coverage-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-normalized-option-range", type=float, default=0.25)
    parser.add_argument("--maximum-competing-pool-correlation", type=float, default=0.99)
    parser.add_argument(
        "--minimum-context-states",
        type=int,
        default=4,
        help=(
            "states in which a motor head must actually be read before its "
            "post-motor evidence counts; 0 deliberately opts out"
        ),
    )
    parser.add_argument(
        "--minimum-competing-context-states",
        type=int,
        default=2,
        help="of those, states offering more than one legal option",
    )
    parser.add_argument("--duration-ms", type=float)
    args = parser.parse_args(argv)
    if args.repeats < 2:
        raise ValueError("at least two repeats are required for same-state variability")
    config = PlasticExperimentConfig.load(args.config)
    if args.duration_ms is not None:
        config = replace(config, fly=replace(config.fly, duration_ms=args.duration_ms))
    if args.stage == "pre":
        # The pre stage must not depend on a motor artifact that does not exist.
        config = replace(config, fly=replace(config.fly, motor_mapping_path=""))
    config.require_heavy_opt_in(args.heavy)
    if config.environment.num_envs != 1:
        raise ValueError("diagnostics require the one-sequential-fly configuration")
    corpus = load_corpus(config, args.calibration_corpus)
    if len(corpus) < 2:
        raise ValueError("representation diagnostics need at least two corpus states")
    stack = build_plastic_stack(config, allow_uncalibrated_motor=args.stage == "pre")
    if args.stage == "post" and stack.components["motor_mapping_bootstrap_only"]:
        raise ValueError(
            "post-motor representation requires the persisted reward-free motor "
            "artifact; set [fly].motor_mapping_path to the calibrated file"
        )
    activity = record_corpus_activity(
        stack,
        corpus,
        seed=args.activity_seed,
        repeats=args.repeats,
        batch_size=args.batch_size,
        limit=args.states,
    )
    order = activity.state_labels.tolist()
    report = representation_diagnostics(
        activity.kc,
        activity.mbon,
        activity.descending,
        motor_activity=activity.output,
        motor_activity_population=MOTOR_POPULATION_BY_MODE[config.fly.mode],
        state_labels=activity.state_labels,
        observable_categories=observable_category_labels(corpus, order),
        motor_interface=stack.agent.motor if args.stage == "post" else None,
        motor_contexts=(
            motor_context_windows(corpus.masks, order=order)
            if args.stage == "post"
            else None
        ),
        stage=args.stage,
        thresholds=RepresentationThresholds(
            silence_hz=args.silence_hz,
            high_rate_hz=args.high_rate_hz,
            maximum_silent_fraction=args.maximum_silent_fraction,
            maximum_high_rate_fraction=args.maximum_high_rate_fraction,
            minimum_separation_ratio=args.minimum_separation_ratio,
            minimum_motor_dynamic_range_hz=args.minimum_motor_dynamic_range_hz,
            minimum_action_coverage_fraction=args.minimum_action_coverage_fraction,
            minimum_normalized_option_range=args.minimum_normalized_option_range,
            maximum_competing_pool_correlation=args.maximum_competing_pool_correlation,
            minimum_context_states=args.minimum_context_states,
            minimum_competing_context_states=args.minimum_competing_context_states,
        ),
    )
    report["backend_evidence"] = (
        "real_flywire_spike_rate_hz"
        if config.fly.backend == "flywire"
        else "synthetic_rate_hz_proxy_development_only"
    )
    report["decision_duration_ms"] = config.fly.duration_ms
    report["calibration_corpus_sha256"] = corpus.sha256
    report["corpus_states_used"] = len(set(order))
    report["repeats_per_state"] = args.repeats
    identity = experiment_identity(
        config,
        stack.components,
        report_kind=f"representation_{args.stage}",
        report_version=REPRESENTATION_REPORT_VERSION,
        corpus=corpus,
    )
    write_report(args.output, report, identity)
    outputs = {"output": str(args.output), "status": report["gates"]["status"]}
    if args.reachability_output is not None:
        reachability = kc_reachability_report(
            kc_activity=activity.kc,
            kc_indices=_kc_indices(stack),
            kc_types=_kc_types(stack),
            mbon_activity=activity.mbon,
            mbon_indices=_mbon_indices(stack),
            topology=stack.agent.plasticity.topology,
            plasticity=config.plasticity,
            motor_output_indices=_output_indices(stack),
            motor_activity=activity.output,
            thresholds=ReachabilityThresholds(silence_hz=args.silence_hz),
        )
        reachability["calibration_corpus_sha256"] = corpus.sha256
        reachability["decision_duration_ms"] = config.fly.duration_ms
        write_report(
            args.reachability_output,
            reachability,
            experiment_identity(
                config,
                stack.components,
                report_kind="kc_reachability",
                report_version=REACHABILITY_REPORT_VERSION,
                corpus=corpus,
            ),
        )
        outputs["reachability_output"] = str(args.reachability_output)
        outputs["reachability_status"] = reachability["gates"]["status"]
    print(json.dumps(outputs, sort_keys=True))
    return 0 if report["gates"]["status"] == "PASS" else 2


def _kc_indices(stack: object) -> np.ndarray:
    processor = stack.agent.processor
    indices = getattr(processor, "_kc_indices", None)
    if indices is not None:
        return np.asarray(indices, dtype=np.int64)
    return np.unique(stack.agent.plasticity.topology.pre_indices)


def _mbon_indices(stack: object) -> np.ndarray:
    processor = stack.agent.processor
    indices = getattr(processor, "_mbon_indices", None)
    if indices is not None:
        return np.asarray(indices, dtype=np.int64)
    return np.unique(stack.agent.plasticity.topology.post_indices)


def _output_indices(stack: object) -> np.ndarray:
    processor = stack.agent.processor
    indices = getattr(processor, "output_indices", None)
    if indices is None:
        return np.unique(stack.agent.plasticity.topology.post_indices)
    return np.asarray(indices, dtype=np.int64)


def _kc_types(stack: object) -> np.ndarray:
    processor = stack.agent.processor
    types = getattr(processor, "kc_types", None)
    if types is not None:
        return np.asarray(types, dtype=np.str_)
    topology = stack.agent.plasticity.topology
    indices = _kc_indices(stack)
    lookup: dict[int, str] = {}
    for position, value in enumerate(topology.pre_indices.tolist()):
        lookup.setdefault(int(value), str(topology.kc_types[position]))
    return np.asarray(
        [lookup.get(int(value), "unknown") for value in indices], dtype=np.str_
    )


if __name__ == "__main__":
    raise SystemExit(main())
