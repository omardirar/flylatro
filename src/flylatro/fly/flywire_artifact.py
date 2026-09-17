"""Versioned FlyWire FAFB v783 simulation artifact loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


EXPECTED_DATASET = "FlyWire FAFB"
EXPECTED_VERSION = "783"
EXPECTED_NEURONS = 139_255
EXPECTED_CONNECTION_PAIRS = 3_732_460


@dataclass(frozen=True, slots=True)
class FlyWireArtifact:
    path: Path
    manifest: dict[str, Any]
    root_ids: NDArray[np.int64]
    pre_indices: NDArray[np.int64]
    post_indices: NDArray[np.int64]
    signed_synapse_counts: NDArray[np.float32]
    sensory_indices: NDArray[np.int64]
    descending_indices: NDArray[np.int64]
    coordinates_nm: NDArray[np.float32]

    @classmethod
    def load(cls, path: Path, *, verify_hash: bool = True) -> "FlyWireArtifact":
        path = path.resolve()
        manifest_path = path.with_suffix(".manifest.json")
        if not path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"FlyWire artifact or manifest missing: {path}, {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != EXPECTED_DATASET or str(
            manifest.get("version")
        ) != EXPECTED_VERSION:
            raise ValueError("artifact is not FlyWire FAFB v783")
        if verify_hash:
            actual = sha256_file(path)
            expected = manifest.get("artifact_sha256")
            if actual != expected:
                raise ValueError(
                    f"artifact SHA-256 mismatch: expected {expected}, got {actual}"
                )
        with np.load(path, allow_pickle=False) as data:
            artifact = cls(
                path=path,
                manifest=manifest,
                root_ids=data["root_ids"].astype(np.int64, copy=False),
                pre_indices=data["pre_indices"].astype(np.int64, copy=False),
                post_indices=data["post_indices"].astype(np.int64, copy=False),
                signed_synapse_counts=data["signed_synapse_counts"].astype(
                    np.float32, copy=False
                ),
                sensory_indices=data["sensory_indices"].astype(np.int64, copy=False),
                descending_indices=data["descending_indices"].astype(
                    np.int64, copy=False
                ),
                coordinates_nm=data["coordinates_nm"].astype(np.float32, copy=False),
            )
        artifact.validate()
        return artifact

    @property
    def neuron_count(self) -> int:
        return len(self.root_ids)

    @property
    def connectivity_hash(self) -> str:
        return str(self.manifest["connectivity_sha256"])

    def validate(self) -> None:
        if self.neuron_count != EXPECTED_NEURONS:
            raise ValueError(
                f"expected {EXPECTED_NEURONS} neurons, got {self.neuron_count}"
            )
        edge_count = len(self.pre_indices)
        if not (
            edge_count
            == len(self.post_indices)
            == len(self.signed_synapse_counts)
        ):
            raise ValueError("connectivity arrays have inconsistent lengths")
        if self.coordinates_nm.shape != (self.neuron_count, 3):
            raise ValueError("coordinates must have shape [neurons, 3]")
        for name, indices in (
            ("pre", self.pre_indices),
            ("post", self.post_indices),
            ("sensory", self.sensory_indices),
            ("descending", self.descending_indices),
        ):
            if indices.size and (indices.min() < 0 or indices.max() >= self.neuron_count):
                raise ValueError(f"{name} indices are outside the neuron table")
        if not self.sensory_indices.size or not self.descending_indices.size:
            raise ValueError("artifact must contain sensory and descending populations")

    def edge_arrays(
        self, *, shuffle_seed: int | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], str]:
        if shuffle_seed is None:
            return (
                self.pre_indices,
                self.post_indices,
                self.signed_synapse_counts,
                "real-topology",
            )
        rng = np.random.default_rng(shuffle_seed)
        post_label_permutation = np.arange(self.neuron_count, dtype=np.int64)
        rng.shuffle(post_label_permutation)
        shuffled_post = post_label_permutation[self.post_indices]
        procedure = (
            "postsynaptic-neuron-label-permutation-v2; preserves every edge entry, "
            "exact pre out-degree, post in-degree distribution, signed-weight "
            "distribution and presynaptic identity without sparse-coordinate merging; "
            f"seed={shuffle_seed}"
        )
        return self.pre_indices, shuffled_post, self.signed_synapse_counts, procedure


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def connectivity_sha256(
    pre: NDArray[np.int64],
    post: NDArray[np.int64],
    weights: NDArray[np.float32],
) -> str:
    digest = hashlib.sha256()
    digest.update(pre.astype("<i8", copy=False).tobytes())
    digest.update(post.astype("<i8", copy=False).tobytes())
    digest.update(weights.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()
