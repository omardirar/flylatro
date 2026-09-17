from __future__ import annotations

from pathlib import Path

import numpy as np

from flylatro.env.upstream_contract import OBS_SPEC
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.upstream_encoder import (
    FixedUpstreamBalatroEncoder,
    FlyWireEncoderSpec,
    full_feature_names,
    observation_features,
)


def observations(count: int = 2):
    batch = {
        key: np.zeros((count, *shape), dtype=dtype)
        for key, (shape, dtype) in OBS_SPEC.items()
    }
    batch["hand_len"][:] = 5
    batch["hand"][:, 0, 0] = 1.0
    batch["joker_ids"][:, 0] = 4
    batch["blind"][:, 0] = 1.0
    return batch


def artifact_for_encoder(tmp_path: Path) -> FlyWireArtifact:
    feature_count = len(full_feature_names())
    neuron_count = feature_count + 4
    return FlyWireArtifact(
        path=tmp_path / "encoder-artifact.npz",
        manifest={"connectivity_sha256": "test"},
        root_ids=np.arange(10_000, 10_000 + neuron_count, dtype=np.int64),
        pre_indices=np.asarray([0], dtype=np.int64),
        post_indices=np.asarray([1], dtype=np.int64),
        signed_synapse_counts=np.asarray([1], dtype=np.float32),
        sensory_indices=np.arange(feature_count, dtype=np.int64),
        descending_indices=np.asarray([feature_count], dtype=np.int64),
        coordinates_nm=np.zeros((neuron_count, 3), dtype=np.float32),
    )


def test_full_encoder_is_deterministic_bounded_and_hashed(tmp_path: Path) -> None:
    artifact = artifact_for_encoder(tmp_path)
    spec = FlyWireEncoderSpec.from_artifact(artifact, population_width=1)
    encoder = FixedUpstreamBalatroEncoder(spec)

    first = encoder.encode(observations())
    second = encoder.encode(observations())

    np.testing.assert_array_equal(first.rates_hz, second.rates_hz)
    assert first.rates_hz.shape == (2, artifact.neuron_count)
    assert first.rates_hz.min() >= 0
    assert first.rates_hz.max() <= 150
    assert first.encoder_hash == spec.sha256
    assert len(first.encoder_hash) == 64


def test_every_full_contract_feature_is_bounded() -> None:
    batch = observations(1)
    batch["global"][0, 0] = -100
    batch["global"][0, 1] = 100
    batch["deck_counts"][0, 0] = 99

    features = observation_features(batch)

    assert features.shape == (1, len(full_feature_names()))
    assert features.min() >= 0
    assert features.max() <= 1


def test_encoder_hash_covers_exact_sensory_root_ids(tmp_path: Path) -> None:
    artifact = artifact_for_encoder(tmp_path)
    original = FlyWireEncoderSpec.from_artifact(artifact, population_width=1)
    remapped = FlyWireEncoderSpec(
        version=original.version,
        neuron_count=original.neuron_count,
        feature_names=original.feature_names,
        population_indices=original.population_indices,
        population_root_ids=original.population_root_ids + 1,
        min_rate_hz=original.min_rate_hz,
        max_rate_hz=original.max_rate_hz,
    )

    assert original.sha256 != remapped.sha256

