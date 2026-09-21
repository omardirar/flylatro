"""Versioned FlyWire FAFB v783 simulation artifact loading."""

from __future__ import annotations

from dataclasses import dataclass, field
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
EXPECTED_KENYON_CELLS = 5_177


def _empty_int64() -> NDArray[np.int64]:
    return np.empty(0, dtype=np.int64)


def _empty_text() -> NDArray[np.str_]:
    return np.empty(0, dtype=np.str_)


@dataclass(frozen=True, slots=True)
class KCMNONEdges:
    """Sparse anatomical KC->MBON edge view into the fixed edge table."""

    edge_indices: NDArray[np.int64]
    pre_indices: NDArray[np.int64]
    post_indices: NDArray[np.int64]
    signed_synapse_counts: NDArray[np.float32]


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
    kenyon_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    mbon_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    dan_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    pam_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    ppl1_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    projection_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    apl_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    dpm_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    kc_mbon_edge_indices: NDArray[np.int64] = field(default_factory=_empty_int64)
    neuron_classes: NDArray[np.str_] = field(default_factory=_empty_text)
    primary_types: NDArray[np.str_] = field(default_factory=_empty_text)

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
            def optional(name: str, dtype: Any) -> Any:
                if name not in data:
                    return np.empty(0, dtype=dtype)
                return data[name].astype(dtype, copy=False)

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
                kenyon_indices=optional("kenyon_indices", np.int64),
                mbon_indices=optional("mbon_indices", np.int64),
                dan_indices=optional("dan_indices", np.int64),
                pam_indices=optional("pam_indices", np.int64),
                ppl1_indices=optional("ppl1_indices", np.int64),
                projection_indices=optional("projection_indices", np.int64),
                apl_indices=optional("apl_indices", np.int64),
                dpm_indices=optional("dpm_indices", np.int64),
                kc_mbon_edge_indices=optional("kc_mbon_edge_indices", np.int64),
                neuron_classes=optional("neuron_classes", np.str_),
                primary_types=optional("primary_types", np.str_),
            )
        artifact.validate()
        return artifact

    @property
    def neuron_count(self) -> int:
        return len(self.root_ids)

    @property
    def connectivity_hash(self) -> str:
        return str(self.manifest["connectivity_sha256"])

    @property
    def population_hash(self) -> str:
        return str(self.manifest.get("population_sha256", "legacy-no-population-hash"))

    @property
    def kc_mbon_edges(self) -> KCMNONEdges:
        edge_indices = self.kc_mbon_edge_indices
        return KCMNONEdges(
            edge_indices=edge_indices,
            pre_indices=self.pre_indices[edge_indices],
            post_indices=self.post_indices[edge_indices],
            signed_synapse_counts=self.signed_synapse_counts[edge_indices],
        )

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
            ("kenyon", self.kenyon_indices),
            ("mbon", self.mbon_indices),
            ("dan", self.dan_indices),
            ("pam", self.pam_indices),
            ("ppl1", self.ppl1_indices),
            ("projection", self.projection_indices),
            ("apl", self.apl_indices),
            ("dpm", self.dpm_indices),
        ):
            if indices.size and (indices.min() < 0 or indices.max() >= self.neuron_count):
                raise ValueError(f"{name} indices are outside the neuron table")
        if not self.sensory_indices.size or not self.descending_indices.size:
            raise ValueError("artifact must contain sensory and descending populations")
        schema_version = int(self.manifest.get("schema_version", 1))
        if schema_version >= 2:
            if len(self.kenyon_indices) != EXPECTED_KENYON_CELLS:
                raise ValueError(
                    f"expected {EXPECTED_KENYON_CELLS} Kenyon cells, "
                    f"got {len(self.kenyon_indices)}"
                )
            for name, indices in (
                ("MBON", self.mbon_indices),
                ("DAN", self.dan_indices),
                ("PAM", self.pam_indices),
                ("PPL1", self.ppl1_indices),
                ("projection", self.projection_indices),
                ("KC->MBON edge", self.kc_mbon_edge_indices),
            ):
                if not indices.size:
                    raise ValueError(f"v2 artifact has no {name} population")
            if self.kc_mbon_edge_indices.max() >= edge_count:
                raise ValueError("KC->MBON edge indices are outside the edge table")
            if self.neuron_classes.shape != (self.neuron_count,):
                raise ValueError("neuron_classes must align with root_ids")
            if self.primary_types.shape != (self.neuron_count,):
                raise ValueError("primary_types must align with root_ids")
            kc_mask = np.zeros(self.neuron_count, dtype=np.bool_)
            mbon_mask = np.zeros(self.neuron_count, dtype=np.bool_)
            kc_mask[self.kenyon_indices] = True
            mbon_mask[self.mbon_indices] = True
            plastic = self.kc_mbon_edges
            if not (
                np.all(kc_mask[plastic.pre_indices])
                and np.all(mbon_mask[plastic.post_indices])
            ):
                raise ValueError("KC->MBON edge table contains an invalid endpoint")

    def edge_arrays(
        self,
        *,
        shuffle_seed: int | None = None,
        preserve_populations: bool = False,
        shuffle_scope: str = "whole_brain",
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], str]:
        if shuffle_seed is None:
            return (
                self.pre_indices,
                self.post_indices,
                self.signed_synapse_counts,
                "real-topology",
            )
        if shuffle_scope not in {"whole_brain", "kc_mbon"}:
            raise ValueError("shuffle_scope must be whole_brain or kc_mbon")
        rng = np.random.default_rng(shuffle_seed)
        if shuffle_scope == "kc_mbon":
            shuffled_post = self.post_indices.copy()
            shuffled_mbons = self.mbon_indices.copy()
            rng.shuffle(shuffled_mbons)
            remap = np.arange(self.neuron_count, dtype=np.int64)
            remap[self.mbon_indices] = shuffled_mbons
            selected = self.kc_mbon_edge_indices
            shuffled_post[selected] = remap[self.post_indices[selected]]
            procedure = (
                "kc-mbon-postsynaptic-label-permutation-v1; changes only "
                "identified KC->MBON pairs; preserves edge entries, KC out-degree, "
                "MBON in-degree distribution, weights and all non-KC->MBON edges; "
                f"seed={shuffle_seed}"
            )
            return self.pre_indices, shuffled_post, self.signed_synapse_counts, procedure
        post_label_permutation = np.arange(self.neuron_count, dtype=np.int64)
        if preserve_populations:
            groups = np.zeros(self.neuron_count, dtype=np.int8)
            # Priority makes the groups disjoint while preserving all named
            # learning and interface populations under the permutation.
            for group, indices in enumerate(
                (
                    self.sensory_indices,
                    self.descending_indices,
                    self.projection_indices,
                    self.dan_indices,
                    self.kenyon_indices,
                    self.mbon_indices,
                ),
                start=1,
            ):
                groups[indices] = group
            for group in np.unique(groups):
                members = np.flatnonzero(groups == group)
                shuffled = members.copy()
                rng.shuffle(shuffled)
                post_label_permutation[members] = shuffled
        else:
            rng.shuffle(post_label_permutation)
        shuffled_post = post_label_permutation[self.post_indices]
        scheme = (
            "population-preserving-postsynaptic-label-permutation-v1"
            if preserve_populations
            else "postsynaptic-neuron-label-permutation-v2"
        )
        procedure = (
            f"{scheme}; preserves every edge entry, "
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


def population_sha256(populations: dict[str, NDArray[Any]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(populations):
        values = np.asarray(populations[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()
