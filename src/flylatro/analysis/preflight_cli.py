"""Evaluate the formal Flylatro pre-training gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.preflight import PreflightThresholds, load_json, run_preflight
from flylatro.learning.config import PlasticExperimentConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--representation-report", type=Path)
    parser.add_argument("--plasticity-report", type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--control-manifest", type=Path, action="append", default=[])
    parser.add_argument("--thresholds", type=Path, help="JSON overrides for documented preflight thresholds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    threshold_values = dict(load_json(args.thresholds) or {})
    report = run_preflight(
        PlasticExperimentConfig.load(args.config),
        representation_report=load_json(args.representation_report),
        plasticity_report=load_json(args.plasticity_report),
        benchmark_report=load_json(args.benchmark_report),
        control_manifests=tuple(load_json(path) or {} for path in args.control_manifest),
        thresholds=PreflightThresholds(**threshold_values),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
