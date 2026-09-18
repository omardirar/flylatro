"""Validate and report the pinned FlyWire mushroom-body population census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.fly.flywire_artifact import FlyWireArtifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if not args.full:
        parser.error("full population validation requires explicit --full")
    artifact = FlyWireArtifact.load(args.artifact)
    artifact.validate()
    payload = {
        "dataset": artifact.manifest["dataset"],
        "version": artifact.manifest["version"],
        "artifact_sha256": artifact.manifest["artifact_sha256"],
        "population_sha256": artifact.population_hash,
        "population_rules": artifact.manifest["mushroom_body_population_rules"],
        "counts": {
            "neurons": artifact.neuron_count,
            "kenyon": len(artifact.kenyon_indices),
            "mbon": len(artifact.mbon_indices),
            "dan": len(artifact.dan_indices),
            "pam": len(artifact.pam_indices),
            "ppl1": len(artifact.ppl1_indices),
            "projection": len(artifact.projection_indices),
            "apl": len(artifact.apl_indices),
            "dpm": len(artifact.dpm_indices),
            "descending": len(artifact.descending_indices),
            "kc_mbon_edges": len(artifact.kc_mbon_edge_indices),
        },
        "root_ids": {
            "kenyon": artifact.root_ids[artifact.kenyon_indices].tolist(),
            "mbon": artifact.root_ids[artifact.mbon_indices].tolist(),
            "dan": artifact.root_ids[artifact.dan_indices].tolist(),
            "pam": artifact.root_ids[artifact.pam_indices].tolist(),
            "ppl1": artifact.root_ids[artifact.ppl1_indices].tolist(),
            "projection": artifact.root_ids[artifact.projection_indices].tolist(),
            "apl": artifact.root_ids[artifact.apl_indices].tolist(),
            "dpm": artifact.root_ids[artifact.dpm_indices].tolist(),
            "descending": artifact.root_ids[artifact.descending_indices].tolist(),
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
