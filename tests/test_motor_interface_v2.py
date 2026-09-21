from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flylatro.env.upstream_contract import MASK_SPEC, N_ACTION_TYPES, UpstreamActionType
from flylatro.interface.motor import (
    CONTEXTUAL_ROUTING,
    HEAD_SIZES,
    MOTOR_POOL_COUNT,
    RESERVED_ACTION_TYPES,
    SUPPORTED_ACTION_TYPES,
    FixedMotorInterface,
    MotorMapping,
    MotorNormalization,
)
from flylatro.interface.motor_calibration import (
    MotorCalibrationError,
    MotorCalibrationThresholds,
    calibrate_reward_free_motor,
)
from flylatro.interface.motor_candidates import (
    CANONICAL_MINIMUM_SYNAPSE_COUNT,
    canonical_motor_candidates,
)
from helpers import tiny_artifact


def _activity(count: int, samples: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(1.0, 80.0, count)
    return np.clip(baseline[None, :] + rng.normal(0.0, 6.0, (samples, count)), 0.0, None)


def test_routing_needs_far_fewer_neurons_than_summed_head_widths() -> None:
    naive = sum(HEAD_SIZES.values())
    assert CONTEXTUAL_ROUTING.pool_count == MOTOR_POOL_COUNT < naive
    # 13 action types + 6 card counts + 10 card slots + 6 shared context slots.
    assert CONTEXTUAL_ROUTING.pool_count == 35
    shared = CONTEXTUAL_ROUTING.reuse_summary()["shared_pools"]
    assert shared, "contextual heads must reuse pools"


def test_reserved_slots_are_never_allocated_or_selectable() -> None:
    roots = np.arange(MOTOR_POOL_COUNT * 2, dtype=np.int64)
    mapping = MotorMapping.contiguous_pools(roots, mode="mbon_direct", pool_width=2)
    assert set(RESERVED_ACTION_TYPES) == set(range(13, N_ACTION_TYPES))
    allocated = {
        index for pool in mapping.pool_indices for index in pool
    }
    assert len(allocated) == MOTOR_POOL_COUNT * 2
    interface = FixedMotorInterface(mapping)
    masks = {
        key: np.zeros((1, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    masks["action_type_mask"][0, int(UpstreamActionType.SELECT_BLIND)] = True
    for reserved in RESERVED_ACTION_TYPES:
        masks["action_type_mask"][0, reserved] = True
    activity = np.zeros((1, len(roots)))
    # Give every reserved slot the strongest possible drive: it still loses.
    scores = interface.head_scores(activity)["action_type"]
    assert np.all(np.isneginf(scores[0, list(RESERVED_ACTION_TYPES)]))
    action = interface.decode(activity, masks)
    assert int(action["action_type"][0]) == int(UpstreamActionType.SELECT_BLIND)
    assert interface.reserved_action_legal_count == len(RESERVED_ACTION_TYPES)


def test_canonical_candidates_restrict_to_plastic_reachable_mbons() -> None:
    artifact = tiny_artifact()
    candidates = canonical_motor_candidates(artifact, mode="mbon_direct")
    # MBON index 7 is annotated but is not postsynaptic in any plastic edge.
    assert candidates.root_ids.tolist() == [1004, 1005, 1006]
    assert 1007 in artifact.root_ids[artifact.mbon_indices].tolist()
    assert candidates.minimum_synapse_count == CANONICAL_MINIMUM_SYNAPSE_COUNT
    details = {entry["root_id"]: entry for entry in candidates.describe()}
    assert details[1005]["kc_plastic_input_edges"] == 2
    assert details[1005]["kc_mbon_anatomical_synapse_weight"] == pytest.approx(3.0)
    assert details[1004]["compartment"] == "gamma1pedc"
    assert details[1004]["neuron_type"] == "MBON-gamma1pedc>a/b"


def test_candidate_universe_is_invariant_across_topology_controls() -> None:
    artifact = tiny_artifact()
    reference = canonical_motor_candidates(artifact, mode="mbon_direct")
    for scope in ("kc_mbon", "whole_brain"):
        _, shuffled_post, _, _ = artifact.edge_arrays(
            shuffle_seed=5, preserve_populations=True, shuffle_scope=scope
        )
        assert shuffled_post is not None
        # A shuffled topology must not move the motor universe.
        assert canonical_motor_candidates(artifact, mode="mbon_direct").sha256 == reference.sha256
    assert canonical_motor_candidates(artifact, mode="whole_brain").sha256 != reference.sha256


def test_fixed_normalization_compares_offset_pools_fairly() -> None:
    samples = 40
    rng = np.random.default_rng(3)
    modulation = rng.normal(0.0, 1.0, samples)
    count = MOTOR_POOL_COUNT * 2
    activity = np.tile(np.linspace(1.0, 60.0, count)[None, :], (samples, 1))
    # Two pools with very different baselines but identical relative modulation.
    low = np.asarray([0, 1])
    high = np.asarray([2, 3])
    activity[:, low] = 5.0 + 2.0 * modulation[:, None]
    activity[:, high] = 90.0 + 2.0 * modulation[:, None]
    mapping = MotorMapping.contiguous_pools(
        np.arange(count, dtype=np.int64), mode="mbon_direct", pool_width=2
    )
    normalization = MotorNormalization.from_pool_activity(
        np.stack(
            [activity[:, list(pool)].mean(axis=1) for pool in mapping.pool_indices],
            axis=1,
        )
    )
    calibrated = MotorMapping(
        version="test",
        mode="mbon_direct",
        output_root_ids=mapping.output_root_ids,
        pool_indices=mapping.pool_indices,
        normalization=normalization,
    )
    interface = FixedMotorInterface(calibrated)
    normalized = normalization.apply(interface.pool_activity(activity))
    np.testing.assert_allclose(normalized[:, 0], normalized[:, 1], atol=1e-9)
    # Without normalization the high-baseline pool would always win.
    raw = interface.pool_activity(activity)
    assert raw[:, 1].min() > raw[:, 0].max()


def test_normalization_floor_is_deterministic_and_hash_stable() -> None:
    values = np.tile(np.asarray([[1.0, 5.0]]), (8, 1))
    first = MotorNormalization.from_pool_activity(values, minimum_scale_hz=0.5)
    second = MotorNormalization.from_pool_activity(values, minimum_scale_hz=0.5)
    assert first.sha256 == second.sha256
    assert float(first.scale.min()) == pytest.approx(0.5)
    assert all(entry["scale_floored"] for entry in first.statistics)
    wider = MotorNormalization.from_pool_activity(values, minimum_scale_hz=2.0)
    assert wider.sha256 != first.sha256


def test_reward_free_calibration_is_deterministic_and_persists(tmp_path: Path) -> None:
    roots = np.arange(2_000, 2_000 + MOTOR_POOL_COUNT * 4, dtype=np.int64)
    activity = _activity(len(roots))
    result = calibrate_reward_free_motor(
        roots, activity, mode="mbon_direct", pool_width=2, candidate_set_sha256="cand"
    )
    again = calibrate_reward_free_motor(
        roots, activity, mode="mbon_direct", pool_width=2, candidate_set_sha256="cand"
    )
    assert result.mapping.sha256 == again.mapping.sha256
    assert result.report["calibration"]["reward_used"] is False
    assert result.report["calibration"]["outcome_information_used"] is False
    path = tmp_path / "motor.json"
    result.mapping.save(path)
    loaded = MotorMapping.load(path)
    assert loaded.sha256 == result.mapping.sha256
    assert loaded.candidate_set_sha256 == "cand"
    assert loaded.normalization is not None
    np.testing.assert_allclose(
        loaded.normalization.baseline, result.mapping.normalization.baseline
    )


def test_calibration_reports_diversity_and_refuses_insufficient_candidates() -> None:
    roots = np.arange(MOTOR_POOL_COUNT * 2, dtype=np.int64)
    activity = _activity(len(roots), seed=11)
    report = calibrate_reward_free_motor(
        roots, activity, mode="mbon_direct", pool_width=2
    ).report
    quality = report["calibration"]["quality"]["by_group"]
    assert set(quality) == {"action_type", "card_count", "card_slot", "context_slot"}
    for group in quality.values():
        assert 0.0 <= group["effective_signal_fraction"] <= 1.0
        assert "competing_pool_correlation" in group
        assert "baseline_hz" in group and "variance_hz2" in group
    assert report["calibration"]["candidate_statistics"]["eligible"] >= len(roots) // 2

    with pytest.raises(MotorCalibrationError, match="usable motor candidates"):
        calibrate_reward_free_motor(
            roots[:10], _activity(10), mode="mbon_direct", pool_width=2
        )


def test_calibration_fails_when_a_head_has_no_usable_diversity() -> None:
    roots = np.arange(MOTOR_POOL_COUNT * 2, dtype=np.int64)
    shared = np.linspace(0.0, 40.0, 30)
    # Every candidate carries exactly the same signal: no distinct alternatives.
    activity = np.tile(shared[:, None], (1, len(roots))) + np.arange(len(roots))[None, :]
    result = calibrate_reward_free_motor(
        roots,
        activity,
        mode="mbon_direct",
        pool_width=2,
        thresholds=MotorCalibrationThresholds(minimum_effective_signal_fraction=0.5),
    )
    assert result.status == "FAIL"
    assert "group_signal_diversity" in result.report["gates"]["failed"]


def test_decode_uses_calibrated_normalization_before_comparing() -> None:
    count = MOTOR_POOL_COUNT * 2
    mapping = MotorMapping.contiguous_pools(
        np.arange(count, dtype=np.int64), mode="mbon_direct", pool_width=2
    )
    play_pool = mapping.routing.head_routes["action_type"][int(UpstreamActionType.PLAY_HAND)]
    discard_pool = mapping.routing.head_routes["action_type"][int(UpstreamActionType.DISCARD)]
    baseline = np.zeros(mapping.routing.pool_count)
    baseline[discard_pool] = 100.0
    scale = np.ones(mapping.routing.pool_count)
    normalized = MotorMapping(
        version="test",
        mode="mbon_direct",
        output_root_ids=mapping.output_root_ids,
        pool_indices=mapping.pool_indices,
        normalization=MotorNormalization(
            version="test-v1", baseline=baseline, scale=scale, minimum_scale_hz=0.5
        ),
    )
    activity = np.zeros((1, count))
    activity[0, list(mapping.pool_indices[play_pool])] = 10.0
    activity[0, list(mapping.pool_indices[discard_pool])] = 50.0
    masks = {
        key: np.zeros((1, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    masks["action_type_mask"][0, int(UpstreamActionType.PLAY_HAND)] = True
    masks["action_type_mask"][0, int(UpstreamActionType.DISCARD)] = True
    masks["card_select_mask"][0, :5] = True
    raw_choice = FixedMotorInterface(mapping).decode(activity, masks)
    calibrated_choice = FixedMotorInterface(normalized).decode(activity, masks)
    assert int(raw_choice["action_type"][0]) == int(UpstreamActionType.DISCARD)
    # After subtracting the high baseline the weaker absolute pool wins.
    assert int(calibrated_choice["action_type"][0]) == int(UpstreamActionType.PLAY_HAND)


def test_competing_pools_inside_one_group_never_share_a_neuron() -> None:
    roots = np.arange(MOTOR_POOL_COUNT * 3, dtype=np.int64)
    mapping = calibrate_reward_free_motor(
        roots, _activity(len(roots), seed=5), mode="mbon_direct", pool_width=2
    ).mapping
    for group in mapping.routing.group_names:
        used: list[int] = []
        for pool_id in mapping.routing.group_pool_ids(group):
            used.extend(mapping.pool_indices[pool_id])
        assert len(set(used)) == len(used)
    with pytest.raises(ValueError, match="distinct neurons"):
        MotorMapping(
            version="bad",
            mode="mbon_direct",
            output_root_ids=roots,
            pool_indices=tuple(((0, 1),) * MOTOR_POOL_COUNT),
        )
