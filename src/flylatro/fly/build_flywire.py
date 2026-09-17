"""Build the full FlyWire FAFB v783 simulation artifact.

This is intentionally a heavy, explicit command. It reads the user-supplied
Codex release and is never invoked by installation, imports, or ordinary tests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from flylatro.fly.flywire_artifact import (
    EXPECTED_CONNECTION_PAIRS,
    EXPECTED_DATASET,
    EXPECTED_NEURONS,
    EXPECTED_VERSION,
    connectivity_sha256,
    sha256_file,
)


BUILD_VERSION = "flywire-v783-builder-v1"
REQUIRED_FILES = (
    "neurons.csv.gz",
    "classification.csv.gz",
    "consolidated_cell_types.csv.gz",
    "connections_princeton.csv.gz",
    "coordinates.csv.gz",
)
REFERENCE_SHA256 = {
    "neurons.csv.gz": "6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719",
    "classification.csv.gz": "e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663",
    "consolidated_cell_types.csv.gz": "8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d",
    "connections_princeton.csv.gz": "445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b",
    "coordinates.csv.gz": "14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05",
}
NT_SIGN = {"ACH": 1, "DA": 1, "OCT": 1, "SER": 1, "GABA": -1, "GLUT": -1}


def build(source_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("install Flylatro's 'flywire' extra") from error
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    sources = {name: source_dir / name for name in REQUIRED_FILES}
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("required FlyWire files are missing:\n" + "\n".join(missing))
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    mismatches = {
        name: {"expected": REFERENCE_SHA256[name], "actual": actual}
        for name, actual in hashes.items()
        if actual != REFERENCE_SHA256[name]
    }
    if mismatches:
        raise ValueError(
            "FlyWire source checksums do not match the documented v783 release: "
            + json.dumps(mismatches, indent=2)
        )

    neurons = pd.read_csv(
        sources["neurons.csv.gz"],
        usecols=["root_id", "nt_type"],
        dtype={"root_id": np.int64, "nt_type": "object"},
    ).sort_values("root_id", kind="mergesort").reset_index(drop=True)
    if len(neurons) != EXPECTED_NEURONS:
        raise ValueError(f"expected {EXPECTED_NEURONS} neurons, got {len(neurons)}")
    neurons.insert(0, "idx", np.arange(len(neurons), dtype=np.int64))
    classification = pd.read_csv(
        sources["classification.csv.gz"],
        usecols=["root_id", "flow", "super_class", "class", "sub_class", "side"],
        dtype={"root_id": np.int64},
    )
    cell_types = pd.read_csv(
        sources["consolidated_cell_types.csv.gz"],
        usecols=["root_id", "primary_type"],
        dtype={"root_id": np.int64},
    )
    coordinates = pd.read_csv(
        sources["coordinates.csv.gz"],
        usecols=["root_id", "position"],
        dtype={"root_id": np.int64, "position": "object"},
    ).drop_duplicates("root_id", keep="first")
    xyz = np.asarray(
        [np.fromstring(str(value).strip("[]"), sep=" ") for value in coordinates.position],
        dtype=np.float32,
    )
    coordinates[["x_nm", "y_nm", "z_nm"]] = xyz
    coordinates = coordinates.drop(columns="position")
    neurons = (
        neurons.merge(classification, on="root_id", how="left")
        .merge(cell_types, on="root_id", how="left")
        .merge(coordinates, on="root_id", how="left")
        .sort_values("root_id", kind="mergesort")
        .reset_index(drop=True)
    )
    neurons["idx"] = np.arange(len(neurons), dtype=np.int64)

    connections = pd.read_csv(
        sources["connections_princeton.csv.gz"],
        usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
        dtype={
            "pre_root_id": np.int64,
            "post_root_id": np.int64,
            "syn_count": np.int32,
            "nt_type": "object",
        },
    )
    root_ids = neurons.root_id.to_numpy(dtype=np.int64)
    pre = np.searchsorted(root_ids, connections.pre_root_id.to_numpy())
    post = np.searchsorted(root_ids, connections.post_root_id.to_numpy())
    valid = (
        (pre < len(root_ids))
        & (post < len(root_ids))
        & (root_ids[np.clip(pre, 0, len(root_ids) - 1)] == connections.pre_root_id.to_numpy())
        & (root_ids[np.clip(post, 0, len(root_ids) - 1)] == connections.post_root_id.to_numpy())
    )
    connections = connections.loc[valid].copy()
    connections["pre_idx"] = pre[valid]
    connections["post_idx"] = post[valid]

    nt = neurons.nt_type.fillna("").astype(str).str.upper()
    unresolved = ~nt.isin(NT_SIGN)
    if unresolved.any():
        votes = (
            connections.assign(nt_type=connections.nt_type.fillna("").str.upper())
            .groupby(["pre_root_id", "nt_type"], observed=True)["syn_count"]
            .sum()
            .reset_index()
            .sort_values("syn_count", ascending=False)
            .drop_duplicates("pre_root_id")
            .set_index("pre_root_id")["nt_type"]
        )
        nt.loc[unresolved] = (
            neurons.loc[unresolved, "root_id"].map(votes).fillna("").to_numpy()
        )
    neurons["nt_resolved"] = nt
    neuron_sign = nt.map(NT_SIGN).fillna(0).to_numpy(dtype=np.int8)
    neurons["sign"] = neuron_sign

    grouped = (
        connections.groupby(["pre_idx", "post_idx"], sort=True, observed=True)[
            "syn_count"
        ]
        .sum()
        .reset_index()
    )
    if len(grouped) != EXPECTED_CONNECTION_PAIRS:
        raise ValueError(
            f"expected {EXPECTED_CONNECTION_PAIRS} neuron pairs, got {len(grouped)}"
        )
    pre_indices = grouped.pre_idx.to_numpy(dtype=np.int64)
    post_indices = grouped.post_idx.to_numpy(dtype=np.int64)
    signed = (
        grouped.syn_count.to_numpy(dtype=np.float32) * neuron_sign[pre_indices]
    ).astype(np.float32)
    nonzero = signed != 0
    pre_indices = pre_indices[nonzero]
    post_indices = post_indices[nonzero]
    signed = signed[nonzero]

    sensory = neurons.loc[
        neurons.super_class.astype(str) == "sensory", "idx"
    ].to_numpy(dtype=np.int64)
    descending = neurons.loc[
        neurons.super_class.astype(str) == "descending", "idx"
    ].to_numpy(dtype=np.int64)
    if len(sensory) != 16_938 or len(descending) != 1_305:
        raise ValueError(
            f"unexpected population census: sensory={len(sensory)}, "
            f"descending={len(descending)}"
        )
    coordinates_nm = neurons[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "flywire_fafb_v783.npz"
    np.savez(
        artifact_path,
        root_ids=root_ids,
        pre_indices=pre_indices,
        post_indices=post_indices,
        signed_synapse_counts=signed,
        sensory_indices=sensory,
        descending_indices=descending,
        coordinates_nm=coordinates_nm,
    )
    neuron_table_path = output_dir / "flywire_fafb_v783_neurons.parquet"
    neurons.to_parquet(neuron_table_path, index=False)
    manifest = {
        "schema_version": 1,
        "builder_version": BUILD_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": EXPECTED_DATASET,
        "version": EXPECTED_VERSION,
        "data_licence": "CC BY-NC-SA 4.0",
        "source_url": "https://codex.flywire.ai/api/download",
        "source_sha256": hashes,
        "artifact_sha256": sha256_file(artifact_path),
        "connectivity_sha256": connectivity_sha256(pre_indices, post_indices, signed),
        "neuron_table_sha256": sha256_file(neuron_table_path),
        "n_neurons": len(root_ids),
        "n_connection_pairs_source": len(grouped),
        "n_connection_entries_nonzero": len(pre_indices),
        "n_synapses_source": int(grouped.syn_count.sum()),
        "n_sensory": len(sensory),
        "n_descending": len(descending),
        "n_missing_coordinates": int(np.isnan(coordinates_nm).any(axis=1).sum()),
        "sign_rule": "ACH/DA/OCT/SER positive; GABA/GLUT negative; outgoing edge-majority fallback; unresolved zero",
        "input_population_rule": "all neurons with super_class == sensory, ascending root_id order",
        "readout_population_rule": "all neurons with super_class == descending, ascending root_id order",
    }
    manifest_path = artifact_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return artifact_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="required acknowledgement that this reads/builds the full connectome",
    )
    args = parser.parse_args(argv)
    if not args.full:
        parser.error("the full FlyWire build requires explicit --full")
    artifact, manifest = build(args.source_dir, args.output_dir)
    print(json.dumps({"artifact": str(artifact), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
