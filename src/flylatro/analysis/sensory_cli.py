"""Create a deterministic ALPN sensory mapping and audit it on real states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.corpus import CalibrationCorpus
from flylatro.analysis.provenance import EvidenceIdentity
from flylatro.analysis.sensory_health import (
    SENSORY_HEALTH_VERSION,
    SensoryHealthThresholds,
    sensory_health_report,
)
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.plastic_features import channel_manifest
from flylatro.interface.sensory import SensoryMapping


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mapping-seed", type=int, required=True)
    parser.add_argument("--population-width", type=int, default=3)
    parser.add_argument("--max-rate-hz", type=float, default=150.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-corpus",
        type=Path,
        help="frozen reward-free corpus for state-conditioned sensory health",
    )
    parser.add_argument(
        "--health-report",
        type=Path,
        help="where to write the state-conditioned sensory health report",
    )
    parser.add_argument("--maximum-saturated-alpn-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-collision-saturated-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-identical-vector-fraction", type=float, default=0.0)
    parser.add_argument("--minimum-active-alpn-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-state-separation-hz", type=float, default=1.0)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if not args.full:
        parser.error("real FlyWire sensory mapping requires explicit --full")
    artifact = FlyWireArtifact.load(args.artifact)
    mapping = SensoryMapping.from_artifact(
        artifact,
        mapping_seed=args.mapping_seed,
        population_width=args.population_width,
        max_rate_hz=args.max_rate_hz,
    )
    mapping.save(args.output)
    summary = {
        "output": str(args.output),
        "sha256": mapping.sha256,
        "structural_assignment_metrics": mapping.collision_audit(),
    }
    status = 0
    if args.calibration_corpus is not None:
        corpus = CalibrationCorpus.load(args.calibration_corpus)
        report = sensory_health_report(
            mapping,
            dict(corpus.observations),
            state_hashes=corpus.state_hashes,
            thresholds=SensoryHealthThresholds(
                maximum_saturated_alpn_fraction=args.maximum_saturated_alpn_fraction,
                maximum_collision_saturated_fraction=args.maximum_collision_saturated_fraction,
                maximum_identical_vector_fraction=args.maximum_identical_vector_fraction,
                minimum_active_alpn_fraction=args.minimum_active_alpn_fraction,
                minimum_state_separation_hz=args.minimum_state_separation_hz,
            ),
        )
        identity = EvidenceIdentity.from_components(
            {
                "artifact_sha256": artifact.manifest["artifact_sha256"],
                "population_sha256": artifact.population_hash,
                "sensory_mapping_sha256": mapping.sha256,
                "sensory_mapping": {"feature_contract": channel_manifest()},
            },
            report_kind="sensory_health",
            report_version=SENSORY_HEALTH_VERSION,
            calibration_corpus_sha256=corpus.sha256,
        )
        payload = {**report, "evidence_identity": identity.to_dict()}
        destination = args.health_report or args.output.with_name(
            args.output.stem + "-health.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["health_report"] = str(destination)
        summary["health_status"] = report["gates"]["status"]
        summary["failed_checks"] = report["gates"]["failed"]
        status = 0 if report["gates"]["status"] == "PASS" else 2
    print(json.dumps(summary, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
