"""Report exactly what remains before a real Ante-1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.preflight import PreflightThresholds, load_json
from flylatro.analysis.readiness import EvidencePaths, run_readiness
from flylatro.learning.config import PlasticExperimentConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sensory-health-report", type=Path)
    parser.add_argument("--representation-pre-report", type=Path)
    parser.add_argument("--representation-post-report", type=Path)
    parser.add_argument("--motor-calibration-report", type=Path)
    parser.add_argument("--reachability-report", type=Path)
    parser.add_argument("--plasticity-report", type=Path)
    parser.add_argument("--specificity-report", type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--control-manifest", type=Path, action="append", default=[])
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = PlasticExperimentConfig.load(args.config)
    evidence = EvidencePaths.from_config(config).merge(
        EvidencePaths(
            sensory_health=args.sensory_health_report,
            representation_pre=args.representation_pre_report,
            representation_post=args.representation_post_report,
            motor_calibration=args.motor_calibration_report,
            reachability=args.reachability_report,
            plasticity=args.plasticity_report,
            specificity=args.specificity_report,
            benchmark=args.benchmark_report,
            protocol=args.protocol,
            control_manifests=tuple(args.control_manifest),
        )
    )
    report = run_readiness(
        config,
        evidence,
        thresholds=PreflightThresholds(**dict(load_json(args.thresholds) or {})),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["state"] == "READY_FOR_ANTE1" else 1


if __name__ == "__main__":
    raise SystemExit(main())
