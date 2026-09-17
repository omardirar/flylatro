"""Fixed full-state Balatro encoder for the pinned upstream array contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import (
    CONSUMABLE_SLOTS,
    CONSUMABLE_VOCAB,
    HAND_MAX,
    JOKER_SLOTS,
    JOKER_VOCAB,
    OBS_SPEC,
    SHOP_SLOTS,
    SHOP_VOCAB,
    ObsDict,
    validate_batch,
)
from flylatro.fly.encoder import Stimulus
from flylatro.fly.flywire_artifact import FlyWireArtifact


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FlyWireEncoderSpec:
    version: str
    neuron_count: int
    feature_names: tuple[str, ...]
    population_indices: IntArray
    population_root_ids: IntArray
    min_rate_hz: float = 0.0
    max_rate_hz: float = 150.0

    def __post_init__(self) -> None:
        if self.population_indices.ndim != 2:
            raise ValueError("population_indices must have shape [features, width]")
        if self.population_indices.shape != self.population_root_ids.shape:
            raise ValueError("population index and root-ID layouts differ")
        if self.population_indices.shape[0] != len(self.feature_names):
            raise ValueError("each feature needs one sensory population")
        if self.population_indices.size and (
            self.population_indices.min() < 0
            or self.population_indices.max() >= self.neuron_count
        ):
            raise ValueError("sensory population index is outside the connectome")
        if not 0 <= self.min_rate_hz < self.max_rate_hz:
            raise ValueError("invalid stimulation rate range")

    @property
    def population_width(self) -> int:
        return self.population_indices.shape[1]

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        header = {
            "version": self.version,
            "neuron_count": self.neuron_count,
            "feature_names": self.feature_names,
            "min_rate_hz": self.min_rate_hz,
            "max_rate_hz": self.max_rate_hz,
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(self.population_root_ids.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    @classmethod
    def from_artifact(
        cls,
        artifact: FlyWireArtifact,
        *,
        population_width: int = 3,
        max_rate_hz: float = 150.0,
    ) -> "FlyWireEncoderSpec":
        if population_width < 1:
            raise ValueError("population_width must be positive")
        names = full_feature_names()
        required = len(names) * population_width
        if required > len(artifact.sensory_indices):
            raise ValueError(
                f"encoder needs {required} sensory neurons, artifact has "
                f"{len(artifact.sensory_indices)}"
            )
        indices = artifact.sensory_indices[:required].reshape(
            len(names), population_width
        )
        roots = artifact.root_ids[indices]
        return cls(
            version="flywire-sensory-v1",
            neuron_count=artifact.neuron_count,
            feature_names=names,
            population_indices=indices.copy(),
            population_root_ids=roots.copy(),
            max_rate_hz=max_rate_hz,
        )


class FixedUpstreamBalatroEncoder:
    """Converts every upstream observation field to bounded sensory drive.

    Existing one-hot fields remain one-hot. Entity IDs receive per-slot
    one-hot populations. Potentially signed scalar groups use dual rails. Raw
    deck counts are clipped to a documented eight-copy scale. No parameter is
    learned.
    """

    def __init__(self, spec: FlyWireEncoderSpec) -> None:
        self.spec = spec

    def encode(self, observations: ObsDict) -> Stimulus:
        batch_size = _batch_size(observations)
        validate_batch(OBS_SPEC, observations, batch_size, "observations")
        features = observation_features(observations)
        if features.shape[1] != len(self.spec.feature_names):
            raise RuntimeError("encoder feature layout drifted from its spec")
        rates = np.zeros(
            (batch_size, self.spec.neuron_count), dtype=np.float32
        )
        scaled = self.spec.min_rate_hz + features * (
            self.spec.max_rate_hz - self.spec.min_rate_hz
        )
        for column in range(self.spec.population_width):
            rates[:, self.spec.population_indices[:, column]] = scaled
        rates.setflags(write=False)
        return Stimulus(rates, self.spec.version, self.spec.sha256)


def full_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(_flat_names("hand", OBS_SPEC["hand"][0]))
    names.append("hand_len")
    names.extend(_categorical_names("joker_id", JOKER_SLOTS, JOKER_VOCAB))
    names.extend(_dual_names("joker_feats", OBS_SPEC["joker_feats"][0]))
    names.extend(
        _categorical_names(
            "consumable_id", CONSUMABLE_SLOTS, CONSUMABLE_VOCAB
        )
    )
    names.append("consumables_len")
    names.extend(_categorical_names("shop_id", SHOP_SLOTS, SHOP_VOCAB))
    names.extend(_flat_names("shop_feats", OBS_SPEC["shop_feats"][0]))
    names.extend(_flat_names("blind", OBS_SPEC["blind"][0]))
    names.extend(_dual_names("global", OBS_SPEC["global"][0]))
    names.extend(_flat_names("deck_counts", OBS_SPEC["deck_counts"][0]))
    names.extend(_flat_names("deck_aggregates", OBS_SPEC["deck_aggregates"][0]))
    names.extend(_flat_names("drawpile_counts", OBS_SPEC["drawpile_counts"][0]))
    return tuple(names)


def observation_features(observations: ObsDict) -> FloatArray:
    batch = _batch_size(observations)
    groups = [
        _bounded_flat(observations["hand"]),
        np.clip(observations["hand_len"][:, None] / HAND_MAX, 0.0, 1.0),
        _one_hot_slots(observations["joker_ids"], JOKER_VOCAB),
        _dual_rail(observations["joker_feats"], scale=8.0),
        _one_hot_slots(observations["consumable_ids"], CONSUMABLE_VOCAB),
        np.clip(
            observations["consumables_len"][:, None] / CONSUMABLE_SLOTS,
            0.0,
            1.0,
        ),
        _one_hot_slots(observations["shop_ids"], SHOP_VOCAB),
        _bounded_flat(observations["shop_feats"], scale=8.0),
        _bounded_flat(observations["blind"], scale=16.0),
        _dual_rail(observations["global"], scale=16.0),
        _bounded_flat(observations["deck_counts"], scale=8.0),
        _bounded_flat(observations["deck_aggregates"], scale=8.0),
        _bounded_flat(observations["drawpile_counts"], scale=8.0),
    ]
    result = np.concatenate(groups, axis=1).astype(np.float64, copy=False)
    if result.shape[0] != batch or not np.isfinite(result).all():
        raise ValueError("encoder produced invalid features")
    return result


def _batch_size(observations: ObsDict) -> int:
    if not observations:
        raise ValueError("observations cannot be empty")
    return next(iter(observations.values())).shape[0]


def _bounded_flat(values: np.ndarray, scale: float = 1.0) -> FloatArray:
    return np.clip(values.reshape(values.shape[0], -1) / scale, 0.0, 1.0)


def _dual_rail(values: np.ndarray, scale: float) -> FloatArray:
    flat = values.reshape(values.shape[0], -1) / scale
    return np.concatenate(
        (np.clip(flat, 0.0, 1.0), np.clip(-flat, 0.0, 1.0)), axis=1
    )


def _one_hot_slots(values: np.ndarray, vocabulary: int) -> FloatArray:
    if values.size and (values.min() < 0 or values.max() >= vocabulary):
        raise ValueError(f"categorical ID is outside vocabulary size {vocabulary}")
    eye = np.eye(vocabulary, dtype=np.float64)
    return eye[values].reshape(values.shape[0], -1)


def _flat_names(prefix: str, shape: tuple[int, ...]) -> list[str]:
    return [f"{prefix}:{index}" for index in range(int(np.prod(shape)))]


def _dual_names(prefix: str, shape: tuple[int, ...]) -> list[str]:
    count = int(np.prod(shape))
    return [f"{prefix}:{index}:positive" for index in range(count)] + [
        f"{prefix}:{index}:negative" for index in range(count)
    ]


def _categorical_names(prefix: str, slots: int, vocabulary: int) -> list[str]:
    return [
        f"{prefix}:{slot}:category:{category}"
        for slot in range(slots)
        for category in range(vocabulary)
    ]

