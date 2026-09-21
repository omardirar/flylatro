"""Seeded fixed mapping from observable Balatro features to fly inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import OBS_SPEC, ObsDict, validate_batch
from flylatro.fly.encoder import Stimulus
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.plastic_features import (
    PLASTIC_FEATURE_CHANNELS,
    channel_manifest,
    feature_names,
    observation_features,
)


#: v3 draws each feature's own ALPN population *without replacement*, so a
#: feature can no longer be counted as several independent contributors to one
#: ALPN.  Collisions between different features remain deliberate. The
#: projection stays deterministic, seeded, versioned, hashable and untrained.
SENSORY_MAPPING_VERSION = "plastic-balatro-alpn-random-projection-v3"

#: Mapping versions this build can still read. ``v2`` sampled with replacement
#: and is retained only so an archived artifact can be inspected.
SUPPORTED_SENSORY_MAPPING_VERSIONS: tuple[str, ...] = (
    "plastic-balatro-alpn-random-projection-v2",
    SENSORY_MAPPING_VERSION,
)


def _sample_distinct(
    rng: np.random.Generator, population: int, width: int
) -> NDArray[np.int64]:
    """Deterministic seeded draw of `width` distinct positions."""

    chosen: list[int] = []
    seen: set[int] = set()
    # Rejection sampling keeps the stream cheap when width << population and
    # is exact for any seed; the permutation fallback covers the dense case.
    for _ in range(16 * width):
        if len(chosen) == width:
            break
        candidate = int(rng.integers(0, population))
        if candidate not in seen:
            seen.add(candidate)
            chosen.append(candidate)
    if len(chosen) < width:
        for candidate in rng.permutation(population):
            value = int(candidate)
            if value not in seen:
                seen.add(value)
                chosen.append(value)
            if len(chosen) == width:
                break
    return np.asarray(chosen, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SensoryMapping:
    version: str
    mapping_seed: int
    neuron_count: int
    feature_names: tuple[str, ...]
    population_indices: NDArray[np.int64]
    population_root_ids: NDArray[np.int64]
    available_alpn_indices: NDArray[np.int64]
    available_alpn_root_ids: NDArray[np.int64]
    max_rate_hz: float = 150.0
    input_population_rule: str = "classification.class == ALPN"
    collision_policy: str = "clipped_sum_preserve_sparse_indicators"

    def __post_init__(self) -> None:
        shape = self.population_indices.shape
        if self.population_indices.ndim != 2 or shape[0] != len(self.feature_names):
            raise ValueError("population_indices must have shape [features, width]")
        if self.population_root_ids.shape != shape:
            raise ValueError("population root IDs do not match mapping indices")
        if self.available_alpn_root_ids.ndim != 1 or not len(self.available_alpn_root_ids):
            raise ValueError("available ALPN root IDs must be a non-empty vector")
        if self.available_alpn_indices.shape != self.available_alpn_root_ids.shape:
            raise ValueError("available ALPN indices and root IDs must align")
        if self.population_indices.size and (
            self.population_indices.min() < 0
            or self.population_indices.max() >= self.neuron_count
        ):
            raise ValueError("sensory mapping index is outside the connectome")
        if self.max_rate_hz <= 0:
            raise ValueError("max_rate_hz must be positive")
        if self.version == SENSORY_MAPPING_VERSION and self.population_indices.size:
            width = self.population_indices.shape[1]
            for row in self.population_indices:
                if len(set(row.tolist())) != width:
                    raise ValueError(
                        f"{SENSORY_MAPPING_VERSION} requires distinct ALPNs inside "
                        "one feature population"
                    )

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "version": self.version,
                    "mapping_seed": self.mapping_seed,
                    "neuron_count": self.neuron_count,
                    "feature_names": self.feature_names,
                    "max_rate_hz": self.max_rate_hz,
                    "input_population_rule": self.input_population_rule,
                    "collision_policy": self.collision_policy,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.population_root_ids.astype("<i8", copy=False).tobytes())
        digest.update(self.available_alpn_root_ids.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    def to_manifest(self) -> dict[str, object]:
        """Return the exact fixed mapping needed to audit a learned fly."""

        return {
            "version": self.version,
            "mapping_seed": self.mapping_seed,
            "neuron_count": self.neuron_count,
            "feature_names": list(self.feature_names),
            "population_indices": self.population_indices.tolist(),
            "population_root_ids": self.population_root_ids.tolist(),
            "available_alpn_indices": self.available_alpn_indices.tolist(),
            "available_alpn_root_ids": self.available_alpn_root_ids.tolist(),
            "max_rate_hz": self.max_rate_hz,
            "input_population_rule": self.input_population_rule,
            "collision_policy": self.collision_policy,
            "feature_contract": channel_manifest(),
            "collision_audit": self.collision_audit(),
            "sha256": self.sha256,
        }

    @classmethod
    def from_artifact(
        cls,
        artifact: FlyWireArtifact,
        *,
        mapping_seed: int,
        population_width: int = 3,
        max_rate_hz: float = 150.0,
    ) -> "SensoryMapping":
        if population_width < 1:
            raise ValueError("population_width must be positive")
        inputs = artifact.projection_indices
        rule = "classification.class == ALPN"
        if not len(inputs):
            raise ValueError(
                "plastic-brain sensory mapping requires annotated ALPN inputs"
            )
        names = feature_names()
        if population_width > len(inputs):
            raise ValueError(
                f"population_width={population_width} exceeds the {len(inputs)} "
                "annotated ALPNs available; one feature cannot drive more "
                "distinct input neurons than exist"
            )
        rng = np.random.default_rng(mapping_seed)
        # More Balatro features exist than ALPNs, so *different* features may
        # legitimately collide on one ALPN. Within a single feature, however,
        # the draw is without replacement: population_width must mean
        # population_width distinct neurons, or the collision and contributor
        # diagnostics would count one feature as several contributors.
        positions = np.empty((len(names), population_width), dtype=np.int64)
        for row in range(len(names)):
            positions[row] = _sample_distinct(rng, len(inputs), population_width)
        indices = inputs[positions]
        return cls(
            version=SENSORY_MAPPING_VERSION,
            mapping_seed=mapping_seed,
            neuron_count=artifact.neuron_count,
            feature_names=names,
            population_indices=indices,
            population_root_ids=artifact.root_ids[indices],
            available_alpn_indices=np.asarray(inputs, dtype=np.int64).copy(),
            available_alpn_root_ids=artifact.root_ids[inputs].copy(),
            max_rate_hz=max_rate_hz,
            input_population_rule=rule,
        )

    def collision_audit(self) -> dict[str, object]:
        """Structural assignment counts.

        These are *informational* diagnostics of the fixed random projection.
        A high assignment count says how many Balatro feature channels could in
        principle drive one ALPN, not how many of them are simultaneously
        active in any real state.  Readiness must be judged from the
        state-conditioned metrics in ``flylatro.analysis.sensory_health``.
        """

        distinct_per_feature = [
            len(set(row.tolist())) for row in self.population_root_ids
        ]
        # Count each feature at most once per ALPN, so a (legacy v2) duplicate
        # assignment is never reported as two independent contributors.
        deduplicated = np.concatenate(
            [np.unique(row) for row in self.population_root_ids]
        ) if self.population_root_ids.size else np.zeros(0, dtype=np.int64)
        unique, used_counts = np.unique(deduplicated, return_counts=True)
        lookup = {int(root): int(count) for root, count in zip(unique, used_counts, strict=True)}
        counts = np.asarray(
            [lookup.get(int(root), 0) for root in self.available_alpn_root_ids],
            dtype=np.int64,
        )
        by_class: dict[str, dict[str, object]] = {}
        for semantic_class in sorted({channel.semantic_class for channel in PLASTIC_FEATURE_CHANNELS}):
            rows = np.asarray(
                [index for index, channel in enumerate(PLASTIC_FEATURE_CHANNELS) if channel.semantic_class == semantic_class],
                dtype=np.int64,
            )
            roots = self.population_root_ids[rows].ravel()
            _, class_counts = np.unique(roots, return_counts=True)
            by_class[semantic_class] = {
                "feature_channels": int(len(rows)),
                "assignments": int(len(roots)),
                "unique_alpns": int(len(np.unique(roots))),
                "collision_assignments": int(np.sum(np.maximum(class_counts - 1, 0))),
                "maximum_same_class_assignments_per_alpn": int(class_counts.max(initial=0)),
            }
        percentiles = {
            name: float(np.quantile(counts, quantile))
            for name, quantile in (("min", 0), ("median", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))
        }
        return {
            "metric_kind": "structural_assignment_capacity_informational",
            "interpretation": (
                "structural collisions count feature channels sharing one ALPN "
                "across the whole contract; simultaneous collision load counts "
                "channels that are non-zero in the same observed state and is "
                "measured separately on the calibration corpus"
            ),
            "assignments": int(self.population_root_ids.size),
            "feature_channels": len(self.feature_names),
            "population_width": int(self.population_indices.shape[1]),
            "distinct_alpns_per_feature": {
                "min": int(min(distinct_per_feature, default=0)),
                "max": int(max(distinct_per_feature, default=0)),
            },
            "features_with_duplicate_alpns": int(
                sum(
                    1
                    for count in distinct_per_feature
                    if count != self.population_indices.shape[1]
                )
            ),
            "contributor_counting_rule": (
                "one feature contributes at most one simultaneous drive to an "
                "ALPN; duplicate within-feature assignments would inflate both "
                "structural and simultaneous contributor counts and are refused"
            ),
            "available_alpns": int(len(self.available_alpn_root_ids)),
            "unique_alpns": int(len(unique)),
            "fraction_alpns_used": float(len(unique) / len(self.available_alpn_root_ids)),
            "colliding_alpns": int(np.count_nonzero(counts > 1)),
            "collision_assignments": int(np.sum(counts[counts > 1] - 1)),
            "maximum_features_per_alpn": int(counts.max(initial=0)),
            "assignments_per_alpn": percentiles,
            "features_per_alpn_histogram": {
                str(value): int(np.count_nonzero(counts == value))
                for value in np.unique(counts)
            },
            "collisions_by_feature_class": by_class,
            "isolated_feature_effective_rates_hz": {
                "binary_one": {"min": self.max_rate_hz, "median": self.max_rate_hz, "max": self.max_rate_hz},
                "representative_scalar_0_25": {"min": 0.25 * self.max_rate_hz, "median": 0.25 * self.max_rate_hz, "max": 0.25 * self.max_rate_hz},
                "note": "clipped-sum policy; values describe one active feature in isolation",
            },
            "policy": self.collision_policy,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_manifest(cls, payload: dict[str, object]) -> "SensoryMapping":
        mapping = cls(
            version=str(payload["version"]),
            mapping_seed=int(payload["mapping_seed"]),
            neuron_count=int(payload["neuron_count"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            population_indices=np.asarray(payload["population_indices"], dtype=np.int64),
            population_root_ids=np.asarray(payload["population_root_ids"], dtype=np.int64),
            available_alpn_indices=np.asarray(
                payload["available_alpn_indices"], dtype=np.int64
            ),
            available_alpn_root_ids=np.asarray(
                payload["available_alpn_root_ids"], dtype=np.int64
            ),
            max_rate_hz=float(payload["max_rate_hz"]),
            input_population_rule=str(payload["input_population_rule"]),
            collision_policy=str(payload["collision_policy"]),
        )
        if payload.get("sha256") != mapping.sha256:
            raise ValueError("sensory mapping artifact hash mismatch")
        if mapping.version not in SUPPORTED_SENSORY_MAPPING_VERSIONS:
            raise ValueError(
                f"unsupported sensory mapping version {mapping.version!r}; "
                f"supported: {list(SUPPORTED_SENSORY_MAPPING_VERSIONS)}"
            )
        return mapping

    @classmethod
    def load(cls, path: Path) -> "SensoryMapping":
        return cls.from_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


class FixedPlasticSensoryEncoder:
    """Clipped-sum drive: collisions cannot attenuate an active indicator."""

    trainable_parameter_count = 0

    def __init__(self, mapping: SensoryMapping) -> None:
        self.mapping = mapping

    def encode(self, observations: ObsDict) -> Stimulus:
        if not observations:
            raise ValueError("observations cannot be empty")
        batch_size = next(iter(observations.values())).shape[0]
        validate_batch(OBS_SPEC, observations, batch_size, "observations")
        features = observation_features(observations).astype(np.float32, copy=False)
        if features.shape[1] != len(self.mapping.feature_names):
            raise RuntimeError("sensory feature layout drifted from mapping")
        rates = np.zeros(
            (batch_size, self.mapping.neuron_count), dtype=np.float32
        )
        for width in range(self.mapping.population_indices.shape[1]):
            indices = self.mapping.population_indices[:, width]
            for row in range(batch_size):
                np.add.at(rates[row], indices, features[row])
        rates *= self.mapping.max_rate_hz
        np.clip(rates, 0.0, self.mapping.max_rate_hz, out=rates)
        rates.setflags(write=False)
        return Stimulus(
            rates,
            self.mapping.version,
            self.mapping.sha256,
        )
