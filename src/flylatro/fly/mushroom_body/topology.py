"""Immutable sparse KC->MBON anatomy shared by independently learning flies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.flywire_artifact import FlyWireArtifact


def _compartment(type_name: str) -> str:
    """Preserve a conservative compartment token from a curated MBON type."""

    value = type_name.strip()
    if not value.upper().startswith("MBON"):
        return "unknown"
    suffix = value[4:].lstrip("-_")
    if not suffix:
        return "unknown"
    # A resolved type may contain a projection (e.g. gamma1pedc>alpha/beta).
    # The first segment describes the dendritic/source compartment.
    return suffix.split(">", 1)[0]


@dataclass(frozen=True, slots=True)
class PlasticEdgeTopology:
    version: str
    edge_indices: NDArray[np.int64]
    pre_indices: NDArray[np.int64]
    post_indices: NDArray[np.int64]
    pre_root_ids: NDArray[np.int64]
    post_root_ids: NDArray[np.int64]
    anatomical_weights: NDArray[np.float32]
    kc_types: NDArray[np.str_]
    mbon_types: NDArray[np.str_]
    compartments: NDArray[np.str_]
    minimum_synapse_count: int = 1

    def __post_init__(self) -> None:
        arrays = (
            self.edge_indices,
            self.pre_indices,
            self.post_indices,
            self.pre_root_ids,
            self.post_root_ids,
            self.anatomical_weights,
            self.kc_types,
            self.mbon_types,
            self.compartments,
        )
        if not arrays[0].size:
            raise ValueError("plastic topology needs at least one KC->MBON edge")
        if any(array.ndim != 1 or len(array) != len(arrays[0]) for array in arrays):
            raise ValueError("plastic edge arrays must be aligned vectors")
        if not np.isfinite(self.anatomical_weights).all():
            raise ValueError("anatomical weights must be finite")
        if np.any(self.anatomical_weights <= 0):
            raise ValueError("KC->MBON anatomical weights must be excitatory")
        if self.minimum_synapse_count < 1:
            raise ValueError("minimum_synapse_count must be at least one")

    @property
    def edge_count(self) -> int:
        return len(self.edge_indices)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.version.encode("utf-8"))
        digest.update(str(self.minimum_synapse_count).encode("ascii"))
        for array in (
            self.edge_indices.astype("<i8", copy=False),
            self.pre_indices.astype("<i8", copy=False),
            self.post_indices.astype("<i8", copy=False),
            self.pre_root_ids.astype("<i8", copy=False),
            self.post_root_ids.astype("<i8", copy=False),
            self.anatomical_weights.astype("<f4", copy=False),
        ):
            digest.update(array.tobytes())
        digest.update(
            json.dumps(
                {
                    "kc_types": self.kc_types.tolist(),
                    "mbon_types": self.mbon_types.tolist(),
                    "compartments": self.compartments.tolist(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    @classmethod
    def from_artifact(
        cls,
        artifact: FlyWireArtifact,
        *,
        post_indices: NDArray[np.int64] | None = None,
        version: str = "flywire-v783-kc-mbon-neuron-pairs-v1",
        minimum_synapse_count: int = 1,
    ) -> "PlasticEdgeTopology":
        edges = artifact.kc_mbon_edges
        posts = (
            edges.post_indices
            if post_indices is None
            else np.asarray(post_indices, dtype=np.int64)[edges.edge_indices]
        )
        weights = edges.signed_synapse_counts.astype(np.float32, copy=True)
        if np.any(weights <= 0):
            raise ValueError(
                "pinned KC->MBON edges include non-excitatory weights; "
                "inspect neurotransmitter resolution before plastic training"
            )
        keep = weights >= minimum_synapse_count
        if not np.any(keep):
            raise ValueError("minimum synapse threshold removed every KC->MBON edge")
        edge_indices = edges.edge_indices[keep]
        pres = edges.pre_indices[keep]
        posts = posts[keep]
        weights = weights[keep]
        kc_types = artifact.primary_types[pres].astype(np.str_, copy=True)
        mbon_types = artifact.primary_types[posts].astype(
            np.str_, copy=True
        )
        return cls(
            version=version,
            edge_indices=edge_indices.copy(),
            pre_indices=pres.copy(),
            post_indices=posts.copy(),
            pre_root_ids=artifact.root_ids[pres].copy(),
            post_root_ids=artifact.root_ids[posts].copy(),
            anatomical_weights=weights,
            kc_types=kc_types,
            mbon_types=mbon_types,
            compartments=np.asarray(
                [_compartment(value) for value in mbon_types], dtype=np.str_
            ),
            minimum_synapse_count=minimum_synapse_count,
        )

    @classmethod
    def synthetic(
        cls,
        pre_indices: NDArray[np.int64],
        post_indices: NDArray[np.int64],
        anatomical_weights: NDArray[np.float32],
    ) -> "PlasticEdgeTopology":
        count = len(pre_indices)
        edge_indices = np.arange(count, dtype=np.int64)
        return cls(
            version="synthetic-kc-mbon-v1",
            edge_indices=edge_indices,
            pre_indices=np.asarray(pre_indices, dtype=np.int64),
            post_indices=np.asarray(post_indices, dtype=np.int64),
            pre_root_ids=np.asarray(pre_indices, dtype=np.int64),
            post_root_ids=np.asarray(post_indices, dtype=np.int64),
            anatomical_weights=np.asarray(anatomical_weights, dtype=np.float32),
            kc_types=np.full(count, "KC-synthetic", dtype=np.str_),
            mbon_types=np.full(count, "MBON-synthetic", dtype=np.str_),
            compartments=np.full(count, "synthetic", dtype=np.str_),
            minimum_synapse_count=1,
        )


def weak_edge_diagnostics(artifact: FlyWireArtifact) -> dict[str, object]:
    """Report calibration-sensitive KC->MBON edge counts at fixed thresholds."""

    weights = artifact.kc_mbon_edges.signed_synapse_counts
    positive = weights[weights > 0]
    total_edges = max(len(positive), 1)
    total_synapses = max(float(positive.sum()), 1.0)
    thresholds = (1, 2, 5, 10)
    return {
        "total_positive_kc_mbon_edges": int(len(positive)),
        "total_positive_kc_mbon_synapses": float(positive.sum()),
        "thresholds": {
            str(threshold): {
                "retained_edges": int(np.count_nonzero(weights >= threshold)),
                "fraction_of_plastic_edge_count": float(np.count_nonzero(positive >= threshold) / total_edges),
                "retained_synapses": float(weights[weights >= threshold].sum()),
                "fraction_of_total_kc_mbon_synapses": float(weights[weights >= threshold].sum() / total_synapses),
            }
            for threshold in thresholds
        },
    }
