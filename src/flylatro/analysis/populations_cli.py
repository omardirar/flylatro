"""Validate and report the pinned FlyWire mushroom-body population census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.provenance import EvidenceIdentity
from flylatro.fly.flywire_artifact import FlyWireArtifact, population_sha256
from flylatro.fly.mushroom_body.topology import weak_edge_diagnostics


POPULATION_REPORT_VERSION = "population-and-data-quality-v2"


def _git() -> dict[str, object]:
    from flylatro.analysis.provenance import git_identity

    values = git_identity()
    return {"git_commit": values["commit"], "git_dirty": values["dirty"]}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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
    populations = {
        "kenyon": artifact.kenyon_indices,
        "mbon": artifact.mbon_indices,
        "dan": artifact.dan_indices,
        "pam": artifact.pam_indices,
        "ppl1": artifact.ppl1_indices,
        "projection": artifact.projection_indices,
        "apl": artifact.apl_indices,
        "dpm": artifact.dpm_indices,
        "descending": artifact.descending_indices,
    }
    rules = {
        **artifact.manifest["mushroom_body_population_rules"],
        "descending": artifact.manifest.get(
            "readout_population_rule", "super_class == descending"
        ),
    }
    identity = EvidenceIdentity(
        report_kind="population_census",
        report_version=POPULATION_REPORT_VERSION,
        generated_at=_now(),
        artifact_sha256=str(artifact.manifest["artifact_sha256"]),
        population_sha256=artifact.population_hash,
        fly_connectivity_sha256=artifact.connectivity_hash,
        **_git(),
    )
    payload = {
        "version": POPULATION_REPORT_VERSION,
        "evidence_identity": identity.to_dict(),
        "dataset": artifact.manifest["dataset"],
        "version": artifact.manifest["version"],
        "artifact_sha256": artifact.manifest["artifact_sha256"],
        "population_sha256": artifact.population_hash,
        "population_rules": artifact.manifest["mushroom_body_population_rules"],
        "data_quality": {
            key: artifact.manifest.get(key)
            for key in (
                "n_unresolved_nt_neurons",
                "n_connection_pairs_unresolved_sign",
                "n_synapses_unresolved_sign",
                "n_connection_rows_invalid_endpoint",
                "n_synapses_invalid_endpoint",
                "fraction_raw_synapses_dropped",
                "n_missing_coordinates",
            )
        },
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
        "populations": {
            name: {
                "count": int(len(indices)),
                "root_ids": artifact.root_ids[indices].tolist(),
                "classification_rule": rules.get(
                    name, "reported; no hard reference census asserted"
                ),
                "primary_types": sorted(
                    set(artifact.primary_types[indices].tolist())
                ),
                "sha256": population_sha256(
                    {name: artifact.root_ids[indices]}
                ),
            }
            for name, indices in populations.items()
        },
        "weak_edge_diagnostics": weak_edge_diagnostics(artifact),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
