"""Calibrate fixed motor pools from reward-free neural activity only.

Candidates come from the canonical plastic-reachable motor universe of the real
unshuffled anatomy, states come from the frozen reward-free calibration corpus,
and nothing in the selection observes reward, score or action quality.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from flylatro.analysis.evidence import experiment_identity, load_corpus, record_corpus_activity
from flylatro.analysis.provenance import EvidenceIdentity
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.interface.motor_calibration import (
    MOTOR_CALIBRATION_VERSION,
    MotorCalibrationError,
    MotorCalibrationThresholds,
    calibrate_reward_free_motor,
)
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-corpus", type=Path)
    parser.add_argument("--states", type=int, help="limit corpus states (default: all)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-seed", type=int, required=True)
    parser.add_argument("--high-rate-hz", type=float, default=200.0)
    parser.add_argument("--silence-hz", type=float, default=0.0)
    parser.add_argument("--minimum-candidate-robust-scale-hz", type=float, default=0.5)
    parser.add_argument("--minimum-scale-hz", type=float, default=0.5)
    parser.add_argument("--maximum-within-group-correlation", type=float, default=0.95)
    parser.add_argument("--minimum-effective-signal-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-normalized-option-range", type=float, default=0.25)
    parser.add_argument("--maximum-pool-silent-fraction", type=float, default=0.90)
    parser.add_argument(
        "--minimum-context-states",
        type=int,
        default=4,
        help=(
            "states in which a motor context must actually be interpreted before "
            "its evidence counts; 0 deliberately opts out (development doubles)"
        ),
    )
    parser.add_argument(
        "--minimum-competing-context-states",
        type=int,
        default=2,
        help="of those, states offering more than one legal option",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="where to write the calibration quality report (default: alongside)",
    )
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    config = PlasticExperimentConfig.load(args.config)
    # State sampling must never depend on a previous calibrated motor artifact.
    config = replace(config, fly=replace(config.fly, motor_mapping_path=""))
    config.require_heavy_opt_in(args.heavy)
    corpus = load_corpus(config, args.calibration_corpus)
    stack = build_plastic_stack(config, allow_uncalibrated_motor=True)
    activity = record_corpus_activity(
        stack,
        corpus,
        seed=args.calibration_seed,
        repeats=args.repeats,
        batch_size=args.batch_size,
        limit=args.states,
    )
    candidate_set = stack.components.get("canonical_motor_candidate_set", {})
    thresholds = MotorCalibrationThresholds(
        silence_hz=args.silence_hz,
        high_rate_hz=args.high_rate_hz,
        minimum_candidate_robust_scale_hz=args.minimum_candidate_robust_scale_hz,
        minimum_scale_hz=args.minimum_scale_hz,
        maximum_within_group_correlation=args.maximum_within_group_correlation,
        minimum_effective_signal_fraction=args.minimum_effective_signal_fraction,
        minimum_normalized_option_range=args.minimum_normalized_option_range,
        maximum_pool_silent_fraction=args.maximum_pool_silent_fraction,
        minimum_pool_width=2,
        minimum_context_states=args.minimum_context_states,
        minimum_competing_context_states=args.minimum_competing_context_states,
    )
    # Candidate quality is judged in the states where each routing group is
    # actually read, using the corpus's own legality masks.
    contexts = motor_context_windows(
        corpus.masks, order=activity.state_labels.tolist()
    )
    details = _candidate_details(stack, config)
    report_path = args.report or args.output.with_name(args.output.stem + "-report.json")
    identity = experiment_identity(
        config,
        stack.components,
        report_kind="motor_calibration",
        report_version=MOTOR_CALIBRATION_VERSION,
        corpus=corpus,
    )
    try:
        result = calibrate_reward_free_motor(
            np.asarray(stack.agent.processor.output_root_ids, dtype=np.int64),
            activity.output,
            mode=config.fly.mode,
            pool_width=config.fly.motor_pool_width,
            thresholds=thresholds,
            contexts=contexts,
            exploration_epsilon=config.motor.exploration_epsilon,
            exploration_temperature=config.motor.exploration_temperature,
            exploration_seed=config.motor.exploration_seed,
            candidate_set_sha256=stack.components.get(
                "canonical_motor_candidate_set_sha256"
            ),
            candidate_details=details,
            calibration_metadata={
                "calibration_seed": args.calibration_seed,
                "calibration_corpus_sha256": corpus.sha256,
                "corpus_states": len(set(activity.state_labels.tolist())),
                "repeats_per_state": args.repeats,
                "motor_context_coverage": {
                    name: window.counts() for name, window in contexts.items()
                },
                "observable_state_hashes": list(dict.fromkeys(activity.state_hashes)),
                "state_sampling": (
                    "frozen reward-free calibration corpus; no bootstrap decoder "
                    "trajectory, no reward and no outcome observed"
                ),
                "decision_duration_ms": config.fly.duration_ms,
                "reward_or_outcome_observed": False,
            },
        )
    except MotorCalibrationError as error:
        failure = {
            "version": MOTOR_CALIBRATION_VERSION,
            "gates": {"status": "FAIL", "checks": {}, "failed": ["calibration_error"]},
            "error": str(error),
            "detail": error.report,
            "motor_context_coverage": {
                name: window.counts() for name, window in contexts.items()
            },
            "candidate_set": candidate_set,
            "evidence_identity": identity.to_dict(),
        }
        _write(report_path, failure)
        print(json.dumps({"report": str(report_path), "status": "FAIL", "error": str(error)}))
        return 2
    report = {**result.report, "candidate_set": candidate_set}
    report["evidence_identity"] = _identity_with_mapping(
        identity, result.mapping.structure_sha256
    )
    _write(report_path, report)
    if result.status != "PASS":
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "status": result.status,
                    "failed": result.report["gates"]["failed"],
                    "artifact_written": False,
                },
                sort_keys=True,
            )
        )
        return 2
    result.mapping.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(report_path),
                "status": result.status,
                "artifact_sha256": result.mapping.sha256,
                "motor_mapping_sha256": result.mapping.structure_sha256,
                "candidate_set_sha256": result.mapping.candidate_set_sha256,
                "states": len(set(activity.state_labels.tolist())),
            },
            sort_keys=True,
        )
    )
    return 0


def _identity_with_mapping(identity: EvidenceIdentity, motor_sha256: str) -> dict[str, object]:
    payload = identity.to_dict()
    payload["motor_mapping_sha256"] = motor_sha256
    return payload


def _candidate_details(stack: object, config: PlasticExperimentConfig) -> list[dict[str, object]] | None:
    if config.fly.backend != "flywire":
        return None
    from flylatro.fly.flywire_artifact import FlyWireArtifact
    from flylatro.interface.motor_candidates import canonical_motor_candidates
    from flylatro.learning.config import resolve_path

    artifact = FlyWireArtifact.load(resolve_path(config, config.fly.artifact_path))
    return canonical_motor_candidates(artifact, mode=config.fly.mode).describe()


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
