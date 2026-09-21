"""Validate a matched-control group and explain each arm's matching rule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.provenance import EvidenceIdentity, git_identity
from flylatro.learning.protocol import (
    CONTROL_VALIDATION_VERSION,
    find_reference,
    validate_control_group,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "run-manifest.json of one arm; order is irrelevant. Exactly one "
            "manifest must declare condition=plastic_real: that is the reference"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.manifest
    ]
    try:
        report = validate_control_group(manifests)
    except ValueError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 2
    report["manifest_paths"] = [str(path) for path in args.manifest]
    git = git_identity()
    # The evidence identity describes the reference arm, located by condition
    # rather than by the order the --manifest flags happened to be typed in.
    reference_index, _ = find_reference(manifests)
    reference = manifests[reference_index].get(
        "components", manifests[reference_index]
    )
    report["evidence_identity"] = EvidenceIdentity(
        report_kind="control_validation",
        report_version=CONTROL_VALIDATION_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        artifact_sha256=reference.get("artifact_sha256"),
        population_sha256=reference.get("population_sha256"),
        motor_mapping_sha256=reference.get("motor_mapping_sha256"),
        sensory_mapping_sha256=reference.get("sensory_mapping_sha256"),
        plasticity_rule_sha256=reference.get("plasticity_rule_sha256"),
        reinforcement_mapping_sha256=reference.get("reinforcement_mapping_sha256"),
        motor_candidate_set_sha256=reference.get(
            "canonical_motor_candidate_set_sha256"
        ),
    ).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["gates"]["status"],
                "failed": report["gates"]["failed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["gates"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
