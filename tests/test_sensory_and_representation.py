from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.representation import (
    MOTOR_POPULATION_BY_MODE,
    RepresentationThresholds,
    representation_diagnostics,
)
from flylatro.analysis.sensory_health import (
    SensoryHealthThresholds,
    sensory_health_report,
)
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.interface.motor import MOTOR_POOL_COUNT, FixedMotorInterface, MotorMapping
from flylatro.interface.sensory import SensoryMapping
from helpers import build_mock_corpus


def _sensory_artifact(alpns: int = 120, neurons: int | None = None) -> FlyWireArtifact:
    neurons = neurons if neurons is not None else alpns + 32
    return FlyWireArtifact(
        path=Path("tiny.npz"),
        manifest={"connectivity_sha256": "x", "artifact_sha256": "y"},
        root_ids=np.arange(1_000, 1_000 + neurons, dtype=np.int64),
        pre_indices=np.asarray([0], dtype=np.int64),
        post_indices=np.asarray([1], dtype=np.int64),
        signed_synapse_counts=np.asarray([1.0], dtype=np.float32),
        sensory_indices=np.arange(8, dtype=np.int64),
        descending_indices=np.arange(neurons - 16, neurons, dtype=np.int64),
        coordinates_nm=np.zeros((neurons, 3), dtype=np.float32),
        projection_indices=np.arange(8, 8 + alpns, dtype=np.int64),
    )


def test_structural_collisions_are_not_simultaneous_collisions(tmp_path: Path) -> None:
    mapping = SensoryMapping.from_artifact(
        _sensory_artifact(), mapping_seed=3, population_width=3, max_rate_hz=150.0
    )
    audit = mapping.collision_audit()
    assert audit["metric_kind"] == "structural_assignment_capacity_informational"
    corpus = build_mock_corpus(tmp_path / "corpus.npz")
    report = sensory_health_report(
        mapping, dict(corpus.observations), state_hashes=corpus.state_hashes
    )
    structural = audit["assignments_per_alpn"]["median"]
    simultaneous = report["simultaneous_collision_load"][
        "active_contributors_per_active_alpn"
    ]["median"]
    # Thousands of channels share each ALPN structurally, but only the ones
    # that are non-zero in an observed state can actually sum together.
    assert structural > simultaneous
    assert report["structural_assignment_metrics"]["assignments"] == audit["assignments"]


def test_state_conditioned_sensory_metrics_are_reported(tmp_path: Path) -> None:
    mapping = SensoryMapping.from_artifact(
        _sensory_artifact(alpns=900), mapping_seed=1, population_width=3
    )
    corpus = build_mock_corpus(tmp_path / "corpus.npz", seeds=(1, 2, 3, 4))
    report = sensory_health_report(
        mapping, dict(corpus.observations), state_hashes=corpus.state_hashes
    )
    state = report["state_conditioned"]
    for key in (
        "fraction_alpns_silent",
        "fraction_alpns_ever_active",
        "fraction_state_alpn_at_max_rate",
        "mean_alpn_rate_hz",
        "median_active_alpn_rate_hz",
        "p95_active_alpn_rate_hz",
    ):
        assert key in state
    load = report["simultaneous_collision_load"]
    for key in ("median", "p90", "p95", "p99", "max"):
        assert key in load["active_contributors_per_active_alpn"]
    assert "fraction_state_alpn_saturated_by_collision" in load
    separation = report["state_separation"]
    assert separation["same_state_vectors_identical"] is True
    assert report["determinism"]["same_input_produces_identical_vector"] is True
    assert "distinct_state_pairs_with_identical_vector_fraction" in separation


def test_saturating_mapping_fails_the_state_conditioned_gate(tmp_path: Path) -> None:
    # A tiny ALPN population makes every state saturate the same neurons, which
    # destroys observable-state information even though it is "only" collisions.
    mapping = SensoryMapping.from_artifact(
        _sensory_artifact(alpns=12), mapping_seed=2, population_width=3
    )
    corpus = build_mock_corpus(tmp_path / "corpus.npz", seeds=(1, 2, 3, 4))
    report = sensory_health_report(
        mapping, dict(corpus.observations), state_hashes=corpus.state_hashes
    )
    assert report["gates"]["status"] == "FAIL"
    assert "saturation_bounded" in report["gates"]["failed"]
    relaxed = sensory_health_report(
        mapping,
        dict(corpus.observations),
        state_hashes=corpus.state_hashes,
        thresholds=SensoryHealthThresholds(
            maximum_saturated_alpn_fraction=1.0,
            maximum_collision_saturated_fraction=1.0,
            maximum_identical_vector_fraction=1.0,
            minimum_state_separation_hz=0.0,
        ),
    )
    assert relaxed["gates"]["status"] == "PASS"


def _matrices(samples: int, mbon: int, descending: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kc = rng.uniform(0.0, 20.0, (samples, 16))
    return (
        kc,
        rng.uniform(0.0, 30.0, (samples, mbon)),
        rng.uniform(0.0, 5.0, (samples, descending)),
    )


def test_mbon_direct_diagnostics_measure_the_mbon_output_population() -> None:
    kc, mbon, descending = _matrices(20, mbon=9, descending=31)
    assert mbon.shape[1] != descending.shape[1]
    assert not np.isclose(mbon.mean(), descending.mean())
    report = representation_diagnostics(
        kc,
        mbon,
        descending,
        motor_activity=mbon,
        motor_activity_population=MOTOR_POPULATION_BY_MODE["mbon_direct"],
    )
    assert report["motor_activity_population"] == "mbon"
    assert report["motor_activity_neurons"] == 9
    assert report["motor_activity_matches_named_population"] is True
    assert report["motor_population"]["neurons"] == mbon.shape[1]
    assert report["motor_population"]["mean_dynamic_range_hz"] == pytest.approx(
        float(np.ptp(mbon, axis=0).mean())
    )
    # The old defect: measuring descending activity in mbon_direct mode.
    assert report["motor_population"]["mean_dynamic_range_hz"] != pytest.approx(
        float(np.ptp(descending, axis=0).mean())
    )


def test_whole_brain_diagnostics_measure_the_descending_output_population() -> None:
    kc, mbon, descending = _matrices(20, mbon=9, descending=31, seed=4)
    report = representation_diagnostics(
        kc,
        mbon,
        descending,
        motor_activity=descending,
        motor_activity_population=MOTOR_POPULATION_BY_MODE["whole_brain"],
    )
    assert report["motor_activity_population"] == "descending"
    assert report["motor_activity_neurons"] == 31
    assert report["motor_population"]["mean_dynamic_range_hz"] == pytest.approx(
        float(np.ptp(descending, axis=0).mean())
    )


def test_pre_stage_makes_no_motor_claim_and_post_stage_requires_the_mapping() -> None:
    kc, mbon, descending = _matrices(24, mbon=MOTOR_POOL_COUNT * 2, descending=12, seed=7)
    pre = representation_diagnostics(
        kc, mbon, descending, motor_activity=mbon, motor_activity_population="mbon"
    )
    assert pre["stage"] == "pre"
    assert pre["final_motor_interface_evaluated"] is False
    assert pre["motor_interface"]["evaluated"] is False
    assert "motor_option_dynamic_range" not in pre["gates"]["checks"]

    with pytest.raises(ValueError, match="persisted final motor mapping"):
        representation_diagnostics(
            kc, mbon, descending, motor_activity=mbon,
            motor_activity_population="mbon", stage="post",
        )

    mapping = MotorMapping.contiguous_pools(
        np.arange(mbon.shape[1], dtype=np.int64), mode="mbon_direct", pool_width=2
    )
    post = representation_diagnostics(
        kc,
        mbon,
        descending,
        motor_activity=mbon,
        motor_activity_population="mbon",
        motor_interface=FixedMotorInterface(mapping),
        stage="post",
        thresholds=RepresentationThresholds(minimum_normalized_option_range=0.0),
    )
    assert post["final_motor_interface_evaluated"] is True
    assert post["motor_interface"]["motor_mapping_sha256"] == mapping.structure_sha256
    assert post["motor_interface"]["motor_artifact_sha256"] == mapping.sha256
    assert set(post["motor_interface"]["by_head"]) == {
        "action_type", "card", "card_count", "joker", "consumable", "shop", "pack",
    }
    assert post["motor_interface"]["by_head"]["action_type"]["represented_options"] == 13
    assert post["motor_interface"]["by_head"]["action_type"]["contract_options"] == 25
    assert "motor_option_dynamic_range" in post["gates"]["checks"]
    assert "action_option_coverage" in post["gates"]["checks"]


def test_centroid_probe_excludes_singleton_classes() -> None:
    values = np.zeros((6, 3))
    values[:, 0] = [0.0, 0.1, 5.0, 5.1, 50.0, 0.05]
    labels = np.asarray([0, 0, 1, 1, 2, 0])
    report = representation_diagnostics(
        np.abs(values),
        np.abs(values),
        np.abs(values),
        motor_activity=np.abs(values),
        motor_activity_population="mbon",
        observable_categories={"phase": labels},
    )["observable_category_probes"]["phase"]
    assert report["classes"] == 3
    assert report["usable_classes"] == 2
    assert report["excluded_singleton_classes"] == [2]
    assert report["excluded_singleton_samples"] == 1
    assert report["evaluated_samples"] == 5
    assert report["sample_support"] == {0: 3, 1: 2, 2: 1}
    assert report["nearest_centroid_leave_one_out_accuracy"] == pytest.approx(1.0)
    assert report["majority_class_reference_accuracy"] == pytest.approx(0.6)
    assert report["uniform_chance_accuracy"] == pytest.approx(0.5)
    assert report["diagnostic_only_not_a_policy_input"] is True


def test_probe_reports_nothing_when_every_class_is_a_singleton() -> None:
    values = np.abs(np.arange(6, dtype=float).reshape(3, 2))
    report = representation_diagnostics(
        values, values, values,
        motor_activity=values, motor_activity_population="mbon",
        observable_categories={"ante": np.asarray([1, 2, 3])},
    )["observable_category_probes"]["ante"]
    assert report["usable_classes"] == 0
    assert report["nearest_centroid_leave_one_out_accuracy"] is None
    assert "support >= 2" in report["reason"]
