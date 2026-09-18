from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.env.upstream_contract import (
    MASK_SPEC,
    N_ACTION_TYPES,
    UpstreamActionType,
)
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.interface.motor import HEAD_SIZES, FixedMotorInterface, MotorMapping
from flylatro.interface.sensory import FixedPlasticSensoryEncoder, SensoryMapping


def sensory_artifact(tmp_path: Path) -> FlyWireArtifact:
    count = 96
    return FlyWireArtifact(
        path=tmp_path / "tiny.npz",
        manifest={"connectivity_sha256": "test"},
        root_ids=np.arange(1000, 1000 + count, dtype=np.int64),
        pre_indices=np.asarray([0], dtype=np.int64),
        post_indices=np.asarray([1], dtype=np.int64),
        signed_synapse_counts=np.asarray([1.0], dtype=np.float32),
        sensory_indices=np.arange(16, dtype=np.int64),
        descending_indices=np.arange(16, 80, dtype=np.int64),
        coordinates_nm=np.zeros((count, 3), dtype=np.float32),
        projection_indices=np.arange(8, 16, dtype=np.int64),
    )


def test_seeded_sensory_mapping_is_fixed_non_trainable_and_replicable(
    tmp_path: Path,
) -> None:
    artifact = sensory_artifact(tmp_path)
    first = SensoryMapping.from_artifact(artifact, mapping_seed=4)
    same = SensoryMapping.from_artifact(artifact, mapping_seed=4)
    different = SensoryMapping.from_artifact(artifact, mapping_seed=5)
    assert first.sha256 == same.sha256
    assert first.sha256 != different.sha256

    env = MockArrayBalatroEnv(2)
    observations, _ = env.reset((10, 11))
    encoder = FixedPlasticSensoryEncoder(first)
    stimulus_a = encoder.encode(observations)
    stimulus_b = encoder.encode(observations)
    np.testing.assert_array_equal(stimulus_a.rates_hz, stimulus_b.rates_hz)
    assert stimulus_a.rates_hz.max() <= first.max_rate_hz
    assert encoder.trainable_parameter_count == 0


def motor() -> tuple[FixedMotorInterface, MotorMapping]:
    mapping = MotorMapping.round_robin(
        np.arange(sum(HEAD_SIZES.values()), dtype=np.int64),
        mode="mbon_direct",
    )
    return FixedMotorInterface(mapping), mapping


def mock_masks() -> dict[str, np.ndarray]:
    masks = {
        key: np.zeros((1, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    masks["action_type_mask"][0, int(UpstreamActionType.PLAY_HAND)] = True
    masks["action_type_mask"][0, int(UpstreamActionType.DISCARD)] = True
    masks["card_select_mask"][0, :5] = True
    return masks


def test_motor_is_deterministic_legal_and_predictable() -> None:
    interface, mapping = motor()
    activity = np.zeros((1, len(mapping.output_root_ids)), dtype=np.float32)
    discard_pool = mapping.pools["action_type"][int(UpstreamActionType.DISCARD)]
    activity[0, discard_pool] = 4.0
    activity[0, mapping.pools["card_count"][2]] = 3.0
    activity[0, mapping.pools["card"][2]] = 5.0
    activity[0, mapping.pools["card"][4]] = 4.0
    masks = mock_masks()

    first = interface.decode(activity, masks)
    second = interface.decode(activity, masks)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["action_type"][0] == int(UpstreamActionType.DISCARD)
    assert first["n_cards"][0] == 2
    np.testing.assert_array_equal(first["cards"][0, :2], [2, 4])

    masks["action_type_mask"][0, int(UpstreamActionType.DISCARD)] = False
    legal = interface.decode(activity, masks)
    assert legal["action_type"][0] == int(UpstreamActionType.PLAY_HAND)
    assert interface.trainable_parameter_count == 0


def test_motor_supports_optional_consumable_card_targets() -> None:
    interface, mapping = motor()
    activity = np.zeros((1, len(mapping.output_root_ids)), dtype=np.float32)
    activity[
        0, mapping.pools["action_type"][int(UpstreamActionType.USE_CONSUMABLE)]
    ] = 5.0
    activity[0, mapping.pools["consumable"][1]] = 4.0
    activity[0, mapping.pools["card_count"][2]] = 3.0
    activity[0, mapping.pools["card"][0]] = 2.0
    activity[0, mapping.pools["card"][3]] = 1.0
    masks = mock_masks()
    masks["action_type_mask"][:] = False
    masks["action_type_mask"][0, int(UpstreamActionType.USE_CONSUMABLE)] = True
    masks["consumable_target_mask"][0, 1] = True

    action = interface.decode(activity, masks)

    assert action["n_cards"][0] == 2
    np.testing.assert_array_equal(action["cards"][0, :2], [0, 3])
    assert action["consumable_target"][0] == 1


def test_motor_hash_covers_mapping_and_mode() -> None:
    _, mapping = motor()
    same = MotorMapping.round_robin(
        mapping.output_root_ids, mode="mbon_direct"
    )
    changed = replace(mapping, mode="whole_brain")
    assert mapping.sha256 == same.sha256
    assert mapping.sha256 != changed.sha256
    assert len(mapping.sha256) == 64
