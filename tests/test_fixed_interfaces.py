from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from flylatro.env.array_mock import MockArrayBalatroEnv
import pytest

from flylatro.env.upstream_contract import (
    MASK_SPEC,
    N_ACTION_TYPES,
    UpstreamActionType,
)
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.interface.motor import (
    HEAD_SIZES,
    MOTOR_POOL_COUNT,
    RESERVED_ACTION_TYPES,
    SUPPORTED_ACTION_TYPES,
    FixedMotorInterface,
    MotorMapping,
)
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


def motor(pool_width: int = 2) -> tuple[FixedMotorInterface, MotorMapping]:
    mapping = MotorMapping.contiguous_pools(
        np.arange(MOTOR_POOL_COUNT * pool_width, dtype=np.int64),
        mode="mbon_direct",
        pool_width=pool_width,
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
    same = MotorMapping.contiguous_pools(
        mapping.output_root_ids, mode="mbon_direct", pool_width=2
    )
    changed = replace(mapping, mode="whole_brain")
    assert mapping.sha256 == same.sha256
    assert mapping.sha256 != changed.sha256
    assert len(mapping.sha256) == 64


def test_reserved_action_slots_own_no_population_and_cannot_be_selected() -> None:
    interface, mapping = motor()
    assert tuple(SUPPORTED_ACTION_TYPES) == tuple(range(13))
    assert set(RESERVED_ACTION_TYPES) == set(range(13, N_ACTION_TYPES))
    for reserved in RESERVED_ACTION_TYPES:
        assert mapping.pools["action_type"][reserved] == ()
        assert mapping.routing.head_routes["action_type"][reserved] == -1
    activity = np.zeros((1, len(mapping.output_root_ids)), dtype=np.float32)
    masks = mock_masks()
    # A reserved slot that the contract reports as legal is excluded, counted,
    # and never chosen while a represented alternative exists.
    masks["action_type_mask"][0, RESERVED_ACTION_TYPES[0]] = True
    action = interface.decode(activity, masks)
    assert action["action_type"][0] in SUPPORTED_ACTION_TYPES
    assert interface.reserved_action_legal_count == 1

    masks["action_type_mask"][:] = False
    masks["action_type_mask"][0, RESERVED_ACTION_TYPES[0]] = True
    with pytest.raises(ValueError, match="no scientifically represented action"):
        interface.decode(activity, masks)


def test_contextual_heads_reuse_pools_but_simultaneous_heads_do_not() -> None:
    _, mapping = motor()
    routing = mapping.routing
    # Target heads are mutually exclusive by action type, so they share pools.
    assert routing.head_routes["shop"][2] == routing.head_routes["pack"][2]
    assert routing.head_routes["joker"][2] == routing.head_routes["consumable"][2]
    # Card slots are read at the same time as a consumable target, and every
    # head that competes inside one comparison keeps distinct populations.
    assert set(routing.head_routes["card"]).isdisjoint(routing.head_routes["consumable"])
    assert set(routing.head_routes["card"]).isdisjoint(routing.head_routes["card_count"])
    represented = [value for value in routing.head_routes["action_type"] if value >= 0]
    assert len(set(represented)) == len(represented)
    # The naive sum of head widths would need far more distinct neurons.
    assert routing.pool_count < sum(HEAD_SIZES.values())
