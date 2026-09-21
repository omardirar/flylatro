from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol import (
    CONDITION_SEED_USAGE,
    EXPERIMENT_PROTOCOL_VERSION,
    SEED_EFFECTS,
    ExperimentProtocol,
    assert_configuration_matches_arm,
    control_class,
    validate_control_group,
)
from flylatro.learning.protocol_materialize import materialize_protocol
from helpers import build_mock_corpus, write_config


def _protocol(**overrides) -> ExperimentProtocol:
    values = dict(
        name="unit",
        base_seed=20260921,
        replicate_count=2,
        conditions=("plastic_real", "no_plasticity"),
        exposure_budget_decisions=16,
        curriculum_ladder=(1,),
        motor_mapping_id="motor-sha",
    )
    values.update(overrides)
    return ExperimentProtocol.create(**values)


def test_every_stored_seed_has_a_declared_effect() -> None:
    protocol = _protocol()
    assert protocol.version == EXPERIMENT_PROTOCOL_VERSION
    replicate_fields = {
        field.name for field in dataclasses.fields(protocol.replicates[0])
    } - {"replicate_id"}
    assert replicate_fields == set(SEED_EFFECTS)
    for name, description in SEED_EFFECTS.items():
        assert description and len(description) > 20
    # plasticity_seed controlled no independent randomness and was removed.
    assert "plasticity_seed" not in SEED_EFFECTS
    assert not hasattr(protocol.replicates[0], "plasticity_seed")
    assert not hasattr(protocol.arms[0], "plasticity_seed")
    for condition, seeds in CONDITION_SEED_USAGE.items():
        assert set(seeds) <= set(SEED_EFFECTS), condition
    assert "topology_seed" not in CONDITION_SEED_USAGE["plastic_real"]
    assert "reward_seed" in CONDITION_SEED_USAGE["shuffled_reward"]
    assert protocol.arms[0].active_seeds == CONDITION_SEED_USAGE["plastic_real"]


def test_protocol_round_trip_and_hash_stability(tmp_path: Path) -> None:
    protocol = _protocol()
    path = protocol.save(tmp_path / "protocol.json")
    loaded = ExperimentProtocol.load(path)
    assert loaded.sha256 == protocol.sha256
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["exposure_budget_decisions"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ExperimentProtocol.load(path)


def test_protocol_refuses_placeholder_motor_identity() -> None:
    with pytest.raises(ValueError, match="motor mapping SHA-256"):
        _protocol(motor_mapping_id="REPLACE_WITH_MOTOR_MAPPING_SHA256")


def test_materialized_arms_are_exact_runnable_configurations(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1, 2), states_per_seed=2)
    base_path = write_config(tmp_path, corpus_path=corpus, name="base")
    base = PlasticExperimentConfig.load(base_path)
    protocol = _protocol()
    protocol_path = protocol.save(tmp_path / "protocol.json")
    plan = materialize_protocol(
        protocol,
        base,
        protocol_path=protocol_path,
        output_dir=tmp_path / "arms",
        run_root=str(tmp_path / "runs"),
        budget_basis="unit-measured",
        checkpoint_every_decisions=8,
        heavy=False,
    )
    assert len(plan["arms"]) == len(protocol.arms) == 4
    for entry in plan["arms"]:
        config = PlasticExperimentConfig.load(Path(entry["config"]))
        arm = protocol.arm(entry["arm_id"])
        assert config.protocol.protocol_sha256 == protocol.sha256
        assert config.protocol.arm_id == arm.arm_id
        assert config.protocol.replicate_id == arm.replicate_id
        assert config.training.condition == arm.condition
        assert config.training.max_environment_decisions == arm.exposure_budget_decisions
        assert config.fly.sensory_mapping_seed == arm.sensory_mapping_seed
        assert config.training.base_fly_seed == arm.fly_seed
        assert config.motor.exploration_seed == arm.motor_seed
        assert config.training.training_seed_offset == arm.training_seed_offset
        assert tuple(config.curriculum.ladder) == arm.curriculum_ladder
        assert config.reinforcement.condition == arm.reinforcement_condition
        assert "--budget-basis" in entry["command"]
        if arm.condition == "no_plasticity":
            assert config.training.plasticity_enabled is False
            assert entry["depends_on"] == [f"{arm.replicate_id}:plastic_real"]
            assert "--action-schedule" in entry["command"]


def test_materialization_is_reproducible(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1,), states_per_seed=2)
    base = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus, name="base")
    )
    protocol = _protocol()
    protocol_path = protocol.save(tmp_path / "protocol.json")
    first = materialize_protocol(
        protocol, base, protocol_path=protocol_path, output_dir=tmp_path / "a",
        run_root="runs", budget_basis="unit", checkpoint_every_decisions=8,
    )
    second = materialize_protocol(
        protocol, base, protocol_path=protocol_path, output_dir=tmp_path / "b",
        run_root="runs", budget_basis="unit", checkpoint_every_decisions=8,
    )
    for left, right in zip(first["arms"], second["arms"], strict=True):
        assert Path(left["config"]).read_text() == Path(right["config"]).read_text()
    assert first["protocol_sha256"] == second["protocol_sha256"]


def test_a_run_validates_itself_against_its_protocol_arm(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1,), states_per_seed=2)
    base = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus, name="base")
    )
    protocol = _protocol()
    protocol_path = protocol.save(tmp_path / "protocol.json")
    materialize_protocol(
        protocol, base, protocol_path=protocol_path, output_dir=tmp_path / "arms",
        run_root="runs", budget_basis="unit", checkpoint_every_decisions=8,
    )
    arm_path = tmp_path / "arms" / "replicate-000-plastic_real.toml"
    config = PlasticExperimentConfig.load(arm_path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    identity = assert_configuration_matches_arm(payload, config)
    assert identity["arm_id"] == "replicate-000:plastic_real"
    assert identity["motor_mapping_id"] == "motor-sha"

    from dataclasses import replace

    drifted = replace(
        config, fly=replace(config.fly, sensory_mapping_seed=config.fly.sensory_mapping_seed + 1)
    )
    with pytest.raises(ValueError, match="sensory_mapping_seed"):
        assert_configuration_matches_arm(payload, drifted)


def _manifest(condition: str, **overrides) -> dict[str, object]:
    base = {
        "condition": condition,
        "topology_condition": (
            condition if condition.endswith("shuffled") and condition != "shuffled_reward" else "real"
        ),
        "curriculum_ladder": [1],
        "exposure_budget_decisions": 16,
        "training_seeds": [0],
        "sensory_mapping_sha256": "sensory",
        "motor_mapping_sha256": "motor",
        "canonical_motor_root_ids_sha256": "roots",
        "canonical_motor_candidate_set_sha256": "candidates",
        "reinforcement_mapping_sha256": "reinforcement",
        "reinforcement_mode": "outcome",
        "plasticity_rule_sha256": "rule",
        "fly_dynamics_sha256": "dynamics",
        "artifact_sha256": "artifact",
        "population_sha256": "population",
        "plastic_topology_sha256": "real-topology",
        "action_schedule_sha256": "actions",
        "state_hash_schedule_sha256": "states",
    }
    base.update(overrides)
    return base


def test_control_classes_are_explicit() -> None:
    assert control_class("no_plasticity") == "exact_action_and_state_matched"
    assert control_class("shuffled_reward") == "exact_action_and_state_matched"
    assert control_class("kc_mbon_shuffled") == "behaviourally_independent_topology_control"
    assert control_class("whole_brain_shuffled") == "behaviourally_independent_topology_control"
    assert control_class("plastic_real") == "reference"


def test_exact_action_controls_must_replay_the_real_trajectory() -> None:
    good = validate_control_group(
        [_manifest("plastic_real"), _manifest("no_plasticity")]
    )
    assert good["gates"]["status"] == "PASS"
    bad = validate_control_group(
        [
            _manifest("plastic_real"),
            _manifest("no_plasticity", action_schedule_sha256="different"),
        ]
    )
    assert bad["gates"]["status"] == "FAIL"
    assert "exact_action_controls_match" in bad["gates"]["failed"]


def test_topology_controls_are_matched_on_protocol_not_on_actions() -> None:
    report = validate_control_group(
        [
            _manifest("plastic_real"),
            _manifest(
                "kc_mbon_shuffled",
                plastic_topology_sha256="shuffled-topology",
                action_schedule_sha256="its-own-actions",
                state_hash_schedule_sha256="its-own-states",
            ),
        ]
    )
    assert report["gates"]["status"] == "PASS"
    arm = report["arms"][1]
    assert arm["control_class"] == "behaviourally_independent_topology_control"
    assert "not applicable" in arm["action_schedule_matching"]
    assert "artifact_sha256" in arm["matched_fields"]
    assert "action_schedule_sha256" not in arm["matched_fields"]


def test_a_topology_control_that_did_not_change_topology_fails() -> None:
    report = validate_control_group(
        [_manifest("plastic_real"), _manifest("whole_brain_shuffled")]
    )
    assert report["gates"]["status"] == "FAIL"
    assert "topology_controls_actually_shuffled" in report["gates"]["failed"]


def test_topology_controls_must_reuse_the_canonical_motor_map() -> None:
    report = validate_control_group(
        [
            _manifest("plastic_real"),
            _manifest(
                "kc_mbon_shuffled",
                plastic_topology_sha256="shuffled-topology",
                motor_mapping_sha256="a-different-motor-map",
            ),
        ]
    )
    assert report["gates"]["status"] == "FAIL"
    assert "common_fields_match" in report["gates"]["failed"]


def test_a_run_refuses_a_motor_artifact_the_protocol_did_not_authorize(
    tmp_path: Path,
) -> None:
    from flylatro.interface.motor import MOTOR_POOL_COUNT, MotorMapping
    from flylatro.interface.motor_calibration import calibrate_reward_free_motor
    from flylatro.learning.config import build_plastic_stack
    import numpy as np

    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1,), states_per_seed=2)
    roots = np.arange(9_000_000, 9_000_000 + MOTOR_POOL_COUNT * 2 + 16, dtype=np.int64)
    rng = np.random.default_rng(0)
    activity = np.clip(
        rng.uniform(1.0, 40.0, len(roots))[None, :]
        + rng.normal(0.0, 5.0, (32, len(roots))),
        0.0,
        None,
    )
    mapping = calibrate_reward_free_motor(
        roots, activity, mode="mbon_direct", pool_width=2
    ).mapping
    motor_path = tmp_path / "motor.json"
    mapping.save(motor_path)
    protocol = _protocol(motor_mapping_id="an-unrelated-motor-interface")
    protocol_path = protocol.save(tmp_path / "protocol.json")
    base = PlasticExperimentConfig.load(
        write_config(
            tmp_path,
            corpus_path=corpus,
            name="base",
            overrides={"fly": {"motor_mapping_path": str(motor_path.resolve())}},
        )
    )
    materialize_protocol(
        protocol, base, protocol_path=protocol_path, output_dir=tmp_path / "arms",
        run_root="runs", budget_basis="unit", checkpoint_every_decisions=8,
    )
    arm = PlasticExperimentConfig.load(
        tmp_path / "arms" / "replicate-000-plastic_real.toml"
    )
    with pytest.raises(ValueError, match="requires motor mapping"):
        build_plastic_stack(arm)

    authorized = _protocol(motor_mapping_id=mapping.structure_sha256)
    authorized_path = authorized.save(tmp_path / "authorized.json")
    materialize_protocol(
        authorized, base, protocol_path=authorized_path,
        output_dir=tmp_path / "ok-arms", run_root="runs",
        budget_basis="unit", checkpoint_every_decisions=8,
    )
    good = PlasticExperimentConfig.load(
        tmp_path / "ok-arms" / "replicate-000-plastic_real.toml"
    )
    stack = build_plastic_stack(good)
    assert stack.components["protocol_sha256"] == authorized.sha256
    assert stack.components["protocol_arm_id"] == "replicate-000:plastic_real"
    assert stack.components["motor_mapping_sha256"] == mapping.structure_sha256
