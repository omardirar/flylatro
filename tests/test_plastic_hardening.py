from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.preflight import preflight_status
from flylatro.analysis.plasticity import (
    plasticity_calibration_report,
    run_controlled_plasticity_sequence,
)
from flylatro.analysis.representation import RepresentationThresholds, representation_diagnostics
from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.env.upstream_contract import GLOBAL_ANTE_OFF, GLOBAL_PHASE_OFF, MASK_SPEC
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology, weak_edge_diagnostics
from flylatro.fly.plastic_features import feature_names, observation_features
from flylatro.interface.motor import (
    MOTOR_POOL_COUNT,
    FixedMotorInterface,
    MotorMapping,
)
from flylatro.interface.motor_calibration import (
    MotorCalibrationThresholds,
    calibrate_reward_free_motor,
)
from flylatro.interface.sensory import FixedPlasticSensoryEncoder, SensoryMapping
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.protocol import ExperimentProtocol, assert_matched_control_manifests
from flylatro.learning.reinforcement import ReinforcementPulse


def _observations() -> dict[str, np.ndarray]:
    env = MockArrayBalatroEnv(1)
    observations, _ = env.reset((1,))
    for value in observations.values():
        value[...] = 0
    return observations


def test_field_aware_encoder_preserves_categorical_and_boolean_amplitude() -> None:
    observations = _observations()
    observations["global"][0, GLOBAL_PHASE_OFF + 1] = 1
    observations["global"][0, GLOBAL_ANTE_OFF + 2] = 1
    observations["hand"][0, 0, 12] = 1
    observations["shop_feats"][0, 0, 0] = 1
    observations["blind"][0, 3 + 7] = 1
    values = observation_features(observations)[0]
    names = feature_names()
    for name in (
        "global:14", "global:5", "hand:0:12", "shop_feats:0:0", "blind:10"
    ):
        assert values[names.index(name)] == 1.0


def _sensory_artifact() -> FlyWireArtifact:
    count = 32
    return FlyWireArtifact(
        path=Path("tiny.npz"), manifest={"connectivity_sha256": "x"},
        root_ids=np.arange(100, 100 + count, dtype=np.int64),
        pre_indices=np.asarray([0], dtype=np.int64),
        post_indices=np.asarray([1], dtype=np.int64),
        signed_synapse_counts=np.asarray([1], dtype=np.float32),
        sensory_indices=np.arange(8, dtype=np.int64),
        descending_indices=np.arange(8, 16, dtype=np.int64),
        coordinates_nm=np.zeros((count, 3), dtype=np.float32),
        projection_indices=np.arange(2, 6, dtype=np.int64),
    )


def test_alpn_collision_audit_and_clipped_sum_do_not_attenuate_binary_signal() -> None:
    mapping = SensoryMapping.from_artifact(
        _sensory_artifact(), mapping_seed=7, population_width=3, max_rate_hz=120
    )
    audit = mapping.collision_audit()
    assert audit["assignments_per_alpn"]["max"] > 1
    assert audit["fraction_alpns_used"] <= 1
    assert audit["collisions_by_feature_class"]
    observations = _observations()
    observations["global"][0, GLOBAL_PHASE_OFF + 1] = 1
    rates = FixedPlasticSensoryEncoder(mapping).encode(observations).rates_hz
    assert rates.max() == 120.0
    assert SensoryMapping.from_artifact(_sensory_artifact(), mapping_seed=7).sha256 == SensoryMapping.from_artifact(_sensory_artifact(), mapping_seed=7).sha256


def _rule(*, learners: int = 1, maximum: float = 2.0) -> ThreeFactorPlasticity:
    topology = PlasticEdgeTopology.synthetic(
        np.asarray([0, 1]), np.asarray([2, 3]), np.ones(2, dtype=np.float32)
    )
    return ThreeFactorPlasticity(
        topology,
        PlasticEdgeState.initialize(2, learners=learners),
        PlasticityConfig(
            learning_rate=0.1, eligibility_decay=0.5,
            kc_reference_hz=20, mbon_reference_hz=10,
            max_eligibility=maximum,
        ),
    )


def test_fixed_reference_eligibility_keeps_weak_activity_weak_and_clips() -> None:
    weak = _rule(maximum=0.5)
    weak.record_activity(np.asarray([[0.02, 0.0]]), np.asarray([[0.01, 1.0]]))
    assert weak.state.eligibility[0, 0] == pytest.approx(1e-6)
    assert weak.state.eligibility[0, 1] == 0
    weak.record_activity(np.full((1, 2), 200.0), np.full((1, 2), 200.0))
    np.testing.assert_allclose(weak.state.eligibility, 0.5)
    weak.record_activity(np.zeros((1, 2)), np.zeros((1, 2)))
    np.testing.assert_allclose(weak.state.eligibility, 0.25)


def test_controlled_plasticity_calibration_measures_actual_updates() -> None:
    rule = _rule()
    updates = run_controlled_plasticity_sequence(rule, decisions=8)
    report = plasticity_calibration_report(
        rule.state,
        rule.config,
        update_magnitudes=updates,
        eligible_reinforcement_events=6,
    )
    assert report["measured_update_count"] == len(updates) > 0
    assert report["max_update_magnitude"] >= report["mean_update_magnitude"] > 0
    assert report["diagnostics"]["positive_changes"] > 0
    assert not report["safety_flags"]["zero_learning_despite_eligible_reinforcement"]

    stalled = _rule()
    stalled.config = replace(stalled.config, learning_rate=0.0)
    stalled.record_activity(np.full((1, 2), 20.0), np.full((1, 2), 10.0))
    stalled.apply_dopamine(1.0, 0.0)
    stalled_report = plasticity_calibration_report(
        stalled.state,
        stalled.config,
        eligible_reinforcement_events=1,
    )
    assert stalled_report["safety_flags"][
        "zero_learning_despite_eligible_reinforcement"
    ]


def test_terminal_reset_is_per_learner_after_update_and_efficacy_persists() -> None:
    config = PlasticExperimentConfig.load(Path("configs/plastic-smoke.toml"))
    config = replace(config, environment=replace(config.environment, num_envs=2))
    stack = build_plastic_stack(config)
    state = stack.agent.plasticity.state
    state.eligibility[:] = 0.5
    before = state.efficacy.copy()
    stack.agent.learn(
        (
            {"episode": {"won": False, "ante": 1}, "reward_components": {}},
            {"reward_components": {"blind_progress": 0.5}},
        ),
        plasticity_enabled=True,
    )
    assert np.all(state.eligibility[0] == 0) and np.all(state.dopamine[0] == 0)
    assert np.all(state.eligibility[1] == 0.5)
    assert np.any(state.efficacy[0] != before[0])
    assert np.all(state.efficacy[0] >= stack.agent.plasticity.config.min_efficacy)
    after_terminal = state.efficacy[0].copy()
    stack.agent.learn(
        ({"reward_components": {"blind_progress": 1.0}}, {"reward_components": {}}),
        plasticity_enabled=True,
    )
    np.testing.assert_array_equal(state.efficacy[0], after_terminal)


def test_representation_units_and_thresholds_are_explicit() -> None:
    mbon = np.asarray([[0, 50], [0, 250]], dtype=float)
    report = representation_diagnostics(
        np.asarray([[0, 10], [0, 20]], dtype=float),
        mbon,
        np.asarray([[1, 2], [3, 8]], dtype=float),
        motor_activity=mbon,
        motor_activity_population="mbon",
        state_labels=np.asarray([0, 1]),
        thresholds=RepresentationThresholds(high_rate_hz=200),
    )
    assert report["activity_unit"] == "spikes_per_second_hz"
    assert report["populations"]["mbon"]["p95_hz"] > 0
    assert report["populations"]["mbon"]["high_rate_sample_fraction"] == 0.25


def test_motor_calibration_and_per_learner_rng_are_order_independent() -> None:
    roots = np.arange(MOTOR_POOL_COUNT * 3, dtype=np.int64)
    activity = np.stack([(roots + 1) * scale for scale in (0.0, 0.4, 0.8, 1.2, 1.6)])
    thresholds = MotorCalibrationThresholds(high_rate_hz=1_000)
    first = calibrate_reward_free_motor(
        roots, activity, mode="mbon_direct", pool_width=2,
        thresholds=thresholds, exploration_epsilon=1.0, exploration_seed=44,
    ).mapping
    second = calibrate_reward_free_motor(
        roots, activity, mode="mbon_direct", pool_width=2,
        thresholds=thresholds, exploration_epsilon=1.0, exploration_seed=44,
    ).mapping
    assert first.sha256 == second.sha256
    assert first.pool_width == 2
    interface = FixedMotorInterface(first)
    values = np.ones((2, len(roots)), dtype=np.float32)
    masks = {key: np.ones((2, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}
    together = interface.decode(values, masks, deterministic=False, learner_ids=np.asarray([10, 20]), decision_ids=np.asarray([3, 3]))
    reordered = interface.decode(values[::-1], {key: value[::-1] for key, value in masks.items()}, deterministic=False, learner_ids=np.asarray([20, 10]), decision_ids=np.asarray([3, 3]))
    for key in together:
        np.testing.assert_array_equal(together[key][0], reordered[key][1])
        np.testing.assert_array_equal(together[key][1], reordered[key][0])
    assert interface.trainable_parameter_count == 0


def _topology_artifact() -> FlyWireArtifact:
    roots = np.arange(100, 112, dtype=np.int64)
    pre = np.asarray([0, 1, 0, 1, 8, 9, 10, 11], dtype=np.int64)
    post = np.asarray([4, 5, 6, 7, 9, 10, 11, 8], dtype=np.int64)
    return FlyWireArtifact(
        path=Path("topology.npz"), manifest={"connectivity_sha256": "x"},
        root_ids=roots, pre_indices=pre, post_indices=post,
        signed_synapse_counts=np.asarray([1, 2, 5, 10, 3, -2, 4, 6], dtype=np.float32),
        sensory_indices=np.asarray([8]), descending_indices=np.asarray([9]),
        coordinates_nm=np.zeros((12, 3), dtype=np.float32),
        kenyon_indices=np.asarray([0, 1]), mbon_indices=np.asarray([4, 5, 6, 7]),
        dan_indices=np.asarray([10]), projection_indices=np.asarray([8]),
        kc_mbon_edge_indices=np.asarray([0, 1, 2, 3]),
        primary_types=np.asarray(["KC"] * 4 + ["MBON-a", "MBON-b", "MBON-c", "MBON-d"] + ["ALPN", "DN", "DAN", "other"]),
    )


def test_topology_controls_and_weak_edge_thresholds_preserve_declared_invariants() -> None:
    artifact = _topology_artifact()
    pre, kc_post, weights, _ = artifact.edge_arrays(shuffle_seed=3, shuffle_scope="kc_mbon")
    np.testing.assert_array_equal(pre, artifact.pre_indices)
    np.testing.assert_array_equal(weights, artifact.signed_synapse_counts)
    np.testing.assert_array_equal(kc_post[4:], artifact.post_indices[4:])
    assert sorted(kc_post[:4]) == sorted(artifact.post_indices[:4])
    _, whole_post, _, _ = artifact.edge_arrays(shuffle_seed=3, preserve_populations=True, shuffle_scope="whole_brain")
    assert len(whole_post) == len(artifact.post_indices)
    diagnostics = weak_edge_diagnostics(artifact)
    assert diagnostics["thresholds"]["5"]["retained_edges"] == 2
    topology = PlasticEdgeTopology.from_artifact(artifact, minimum_synapse_count=5)
    assert topology.edge_count == 2 and topology.minimum_synapse_count == 5
    assert topology.sha256 != PlasticEdgeTopology.from_artifact(artifact, minimum_synapse_count=1).sha256


def test_preflight_status_reduction() -> None:
    assert preflight_status([{"status": "PASS"}]) == "PASS"
    assert preflight_status([{"status": "PASS"}, {"status": "WARN"}]) == "WARN"
    assert preflight_status([{"status": "WARN"}, {"status": "FAIL"}]) == "FAIL"


def test_matched_control_assertions_require_same_canonical_motor_and_budget() -> None:
    shared = {
        "action_schedule_sha256": "a",
        "state_hash_schedule_sha256": "s",
        "curriculum_ladder": [1],
        "exposure_budget_decisions": 20,
        "training_seeds": [1],
        "sensory_mapping_sha256": "x",
        "motor_mapping_sha256": "m",
        "canonical_motor_root_ids_sha256": "r",
        "canonical_motor_candidate_set_sha256": "candidates",
        "reinforcement_mapping_sha256": "reinforcement",
        "reinforcement_mode": "outcome",
        "plasticity_rule_sha256": "rule",
        "fly_dynamics_sha256": "dynamics",
        "artifact_sha256": "artifact",
        "population_sha256": "population",
        "plastic_topology_sha256": "real-topology",
    }
    assert_matched_control_manifests(
        [
            {**shared, "condition": "plastic_real", "topology_condition": "real"},
            {
                **shared,
                "condition": "kc_mbon_shuffled",
                "topology_condition": "kc_mbon_shuffled",
                "plastic_topology_sha256": "shuffled-topology",
                # A topology control cannot replay the real fly's behaviour, so
                # its own actions and states are expected to differ.
                "action_schedule_sha256": "independent-actions",
                "state_hash_schedule_sha256": "independent-states",
            },
        ]
    )
    with pytest.raises(ValueError, match="motor_mapping_sha256"):
        assert_matched_control_manifests(
            [
                {**shared, "condition": "plastic_real", "topology_condition": "real"},
                {
                    **shared,
                    "condition": "kc_mbon_shuffled",
                    "topology_condition": "kc_mbon_shuffled",
                    "plastic_topology_sha256": "shuffled-topology",
                    "motor_mapping_sha256": "different",
                },
            ]
        )
    with pytest.raises(ValueError, match="action_schedule_sha256"):
        assert_matched_control_manifests(
            [
                {**shared, "condition": "plastic_real", "topology_condition": "real"},
                {
                    **shared,
                    "condition": "no_plasticity",
                    "topology_condition": "real",
                    "action_schedule_sha256": "different",
                },
            ]
        )


def test_replicate_protocol_expands_paired_condition_arms() -> None:
    protocol = ExperimentProtocol.create(
        name="paired",
        base_seed=7,
        replicate_count=2,
        conditions=("plastic_real", "no_plasticity", "kc_mbon_shuffled"),
        exposure_budget_decisions=100,
        curriculum_ladder=(1,),
        motor_mapping_id="motor-hash",
    )
    assert len(protocol.arms) == 6
    first = [
        arm
        for arm in protocol.arms
        if arm.replicate_id == "mapping-000-replicate-000"
    ]
    assert {arm.condition for arm in first} == {
        "plastic_real", "no_plasticity", "kc_mbon_shuffled"
    }
    assert len({arm.environment_seed_stream for arm in first}) == 1
    assert len({arm.motor_mapping_id for arm in protocol.arms}) == 1


def test_torch_plastic_state_round_trip_stays_manual_and_device_resident() -> None:
    torch = pytest.importorskip("torch")
    from flylatro.fly.mushroom_body.state import TorchPlasticEdgeState

    topology = PlasticEdgeTopology.synthetic(
        np.asarray([0, 1]), np.asarray([2, 3]), np.ones(2, dtype=np.float32)
    )
    state = TorchPlasticEdgeState.initialize(2, learners=2, device="cpu")
    rule = ThreeFactorPlasticity(topology, state, PlasticityConfig())
    rule.record_activity(torch.ones((2, 2)), torch.ones((2, 2)))
    rule.apply_dopamine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert state.efficacy.device.type == "cpu"
    assert not state.efficacy.requires_grad
    restored = TorchPlasticEdgeState.from_state_dict(state.state_dict())
    assert restored.weight_sha256 == state.weight_sha256
