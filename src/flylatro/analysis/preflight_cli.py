"""Evaluate the formal Flylatro pre-training gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.preflight import (
    PREFLIGHT_PROFILES,
    PreflightThresholds,
    load_json,
    run_preflight,
)
from flylatro.analysis.readiness import EvidencePaths
from flylatro.learning.config import PlasticExperimentConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PREFLIGHT_PROFILES),
        default="initial",
        help="initial authorises tiny real gates; ante1 is the strict gate",
    )
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
    parser.add_argument(
        "--thresholds", type=Path, help="JSON overrides for documented preflight thresholds"
    )
    parser.add_argument(
        "--ignore-config-evidence",
        action="store_true",
        help="do not read evidence paths declared in the config's [calibration] section",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    threshold_values = dict(load_json(args.thresholds) or {})
    config = PlasticExperimentConfig.load(args.config)
    explicit = EvidencePaths(
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
    evidence = (
        explicit
        if args.ignore_config_evidence
        else EvidencePaths.from_config(config).merge(explicit)
    )
    report = run_preflight(
        config,
        profile=args.profile,
        thresholds=PreflightThresholds(**threshold_values),
        **evidence.as_reports(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile": report["profile"],
                "status": report["status"],
                "failed": report["failed"],
                "warnings": report["warnings"],
            },
            sort_keys=True,
        )
    )
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
