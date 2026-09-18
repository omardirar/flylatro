"""Seeded fixed mapping from observable Balatro features to fly inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import OBS_SPEC, ObsDict, validate_batch
from flylatro.fly.encoder import Stimulus
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.upstream_encoder import full_feature_names, observation_features


@dataclass(frozen=True, slots=True)
class SensoryMapping:
    version: str
    mapping_seed: int
    neuron_count: int
    feature_names: tuple[str, ...]
    population_indices: NDArray[np.int64]
    population_root_ids: NDArray[np.int64]
    max_rate_hz: float = 150.0
    input_population_rule: str = "classification.class == ALPN"

    def __post_init__(self) -> None:
        shape = self.population_indices.shape
        if self.population_indices.ndim != 2 or shape[0] != len(self.feature_names):
            raise ValueError("population_indices must have shape [features, width]")
        if self.population_root_ids.shape != shape:
            raise ValueError("population root IDs do not match mapping indices")
        if self.population_indices.size and (
            self.population_indices.min() < 0
            or self.population_indices.max() >= self.neuron_count
        ):
            raise ValueError("sensory mapping index is outside the connectome")
        if self.max_rate_hz <= 0:
            raise ValueError("max_rate_hz must be positive")

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
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(self.population_root_ids.astype("<i8", copy=False).tobytes())
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
            "max_rate_hz": self.max_rate_hz,
            "input_population_rule": self.input_population_rule,
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
        names = full_feature_names()
        rng = np.random.default_rng(mapping_seed)
        # More Balatro features exist than ALPNs. Sampling with replacement is
        # an explicit fixed random projection, not a learned encoder.
        positions = rng.integers(
            0,
            len(inputs),
            size=(len(names), population_width),
            dtype=np.int64,
        )
        indices = inputs[positions]
        return cls(
            version="plastic-balatro-alpn-random-projection-v1",
            mapping_seed=mapping_seed,
            neuron_count=artifact.neuron_count,
            feature_names=names,
            population_indices=indices,
            population_root_ids=artifact.root_ids[indices],
            max_rate_hz=max_rate_hz,
            input_population_rule=rule,
        )


class FixedPlasticSensoryEncoder:
    """Averaging collision-safe synthetic drive over fixed input neurons."""

    trainable_parameter_count = 0

    def __init__(self, mapping: SensoryMapping) -> None:
        self.mapping = mapping
        counts = np.zeros(mapping.neuron_count, dtype=np.float32)
        np.add.at(counts, mapping.population_indices.ravel(), 1.0)
        counts[counts == 0] = 1.0
        self._assignment_counts = counts

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
        rates /= self._assignment_counts[None, :]
        rates *= self.mapping.max_rate_hz
        np.clip(rates, 0.0, self.mapping.max_rate_hz, out=rates)
        rates.setflags(write=False)
        return Stimulus(
            rates,
            self.mapping.version,
            self.mapping.sha256,
        )
