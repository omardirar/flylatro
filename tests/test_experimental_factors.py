"""Experimental-control hardening: factors, references and fixed interfaces.

* ordinary stochastic replicates must not silently redraw the sensory mapping;
* a sensory-mapping change must be an explicit block with its own calibration;
* matched-control validation must find `plastic_real` by condition, not by the
  order the manifests were supplied in;
* a feature must not map twice to the same ALPN inside its own population;
* motor exploration identity must stay deterministic without reading the
  device-resident plastic state.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.preflight import run_preflight
from flylatro.interface.sensory import (
    SENSORY_MAPPING_VERSION,
    SUPPORTED_SENSORY_MAPPING_VERSIONS,
    SensoryMapping,
)
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.protocol import (
    BLOCK_LEVEL_FACTORS,
    CONDITION_STOCHASTIC_SEEDS,
    EXPERIMENT_PROTOCOL_VERSION,
    SEED_EFFECTS,
    STOCHASTIC_REPLICATE_SEEDS,
    ExperimentProtocol,
    SensoryMappingVariant,
    find_reference,
    validate_control_group,
)
from flylatro.learning.protocol_cli import main as protocol_main
from flylatro.learning.protocol_materialize import materialize_protocol
from helpers import build_mock_corpus, tiny_artifact, write_config


def _protocol(**overrides) -> ExperimentProtocol:
    values = dict(
        name="factors",
        base_seed=99,
        replicate_count=3,
        conditions=("plastic_real", "no_plasticity", "shuffled_reward"),
        exposure_budget_decisions=16,
        curriculum_ladder=(1,),
        motor_mapping_id="motor-primary",
    )
    values.update(overrides)
    return ExperimentProtocol.create(**values)


# --------------------------------------------------------------------------
# sensory mapping as an explicit experimental factor
# --------------------------------------------------------------------------


def test_ordinary_replicates_share_one_fixed_sensory_mapping() -> None:
    protocol = _protocol()
    assert protocol.version == EXPERIMENT_PROTOCOL_VERSION
    assert protocol.mapping_replicate_count == 1
    assert len(protocol.replicates) == 3
    # The interface is fixed across ordinary replicates...
    assert len({item.sensory_mapping_seed for item in protocol.replicates}) == 1
    assert len({arm.motor_mapping_id for arm in protocol.arms}) == 1
    # ...while the stochastic mechanisms genuinely vary.
    for name in ("fly_seed", "motor_seed", "training_seed_offset"):
        assert len({getattr(item, name) for item in protocol.replicates}) == 3
    audit = protocol.seed_audit()
    assert audit["ordinary_replicates_share_one_sensory_mapping"] is True
    assert audit["distinct_values_across_replicates"]["sensory_mapping_seed"] == 1
    assert audit["distinct_values_across_replicates"]["fly_seed"] == 3
    assert set(STOCHASTIC_REPLICATE_SEEDS) <= set(SEED_EFFECTS)
    assert "sensory_mapping_seed" not in STOCHASTIC_REPLICATE_SEEDS
    assert "sensory_mapping_seed" in BLOCK_LEVEL_FACTORS
    assert CONDITION_STOCHASTIC_SEEDS["plastic_real"] == (
        "training_seed_offset",
        "fly_seed",
        "motor_seed",
    )


def test_a_mapping_replicate_is_an_explicit_separate_block() -> None:
    protocol = _protocol(
        sensory_mapping_seed=0,
        mapping_sensitivity_variants=(
            {"sensory_mapping_seed": 7, "motor_mapping_id": "motor-seed-7"},
        ),
    )
    assert protocol.mapping_replicate_count == 2
    assert [item.role for item in protocol.sensory_mappings] == [
        "primary",
        "mapping_sensitivity",
    ]
    assert len(protocol.replicates) == 6
    blocks = {"mapping-000": set(), "mapping-001": set()}
    for replicate in protocol.replicates:
        blocks[replicate.mapping_id].add(replicate.sensory_mapping_seed)
    assert blocks == {"mapping-000": {0}, "mapping-001": {7}}
    for arm in protocol.arms:
        expected = protocol.mapping(arm.mapping_id).motor_mapping_id
        assert arm.motor_mapping_id == expected
    # Seed streams never collide between blocks.
    offsets = [item.training_seed_offset for item in protocol.replicates]
    assert len(set(offsets)) == len(offsets)
    assert protocol.seed_audit()["ordinary_replicates_share_one_sensory_mapping"]


def test_a_mapping_replicate_must_regenerate_its_motor_calibration() -> None:
    with pytest.raises(ValueError, match="own motor mapping SHA-256"):
        _protocol(
            mapping_sensitivity_variants=(
                {"sensory_mapping_seed": 7, "motor_mapping_id": "motor-primary"},
            )
        )
    with pytest.raises(ValueError, match="distinct sensory_mapping_seed"):
        _protocol(
            sensory_mapping_seed=3,
            mapping_sensitivity_variants=(
                {"sensory_mapping_seed": 3, "motor_mapping_id": "motor-other"},
            ),
        )
    with pytest.raises(ValueError, match="measured motor mapping SHA-256"):
        SensoryMappingVariant("mapping-001", 7, "REPLACE_WITH_MOTOR_SHA")


def test_a_pre_v3_protocol_is_refused_rather_than_reinterpreted() -> None:
    payload = _protocol().to_payload()
    payload["version"] = "plastic-replicate-protocol-v2"
    with pytest.raises(ValueError, match="unsupported experiment protocol version"):
        ExperimentProtocol.from_payload(payload)


def test_the_protocol_cli_exposes_both_factors(tmp_path: Path) -> None:
    output = tmp_path / "protocol.json"
    assert (
        protocol_main(
            [
                "--name", "cli-factors",
                "--base-seed", "13",
                "--replicates", "2",
                "--conditions", "plastic_real,no_plasticity",
                "--exposure-budget-decisions", "16",
                "--curriculum-ladder", "1",
                "--motor-mapping-id", "motor-a",
                "--sensory-mapping-seed", "0",
                "--mapping-sensitivity", "5:motor-b",
                "--output", str(output),
            ]
        )
        == 0
    )
    protocol = ExperimentProtocol.load(output)
    assert protocol.mapping_replicate_count == 2
    assert protocol.mapping("mapping-001").sensory_mapping_seed == 5
    assert protocol.mapping("mapping-001").motor_mapping_id == "motor-b"
    assert len(protocol.arms) == 8


def test_materialization_keeps_each_block_on_its_own_mapping(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1,), states_per_seed=2)
    base = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus, name="base")
    )
    protocol = _protocol(
        conditions=("plastic_real",),
        replicate_count=2,
        mapping_sensitivity_variants=(
            {"sensory_mapping_seed": 7, "motor_mapping_id": "motor-seed-7"},
        ),
    )
    protocol_path = protocol.save(tmp_path / "protocol.json")
    plan = materialize_protocol(
        protocol,
        base,
        protocol_path=protocol_path,
        output_dir=tmp_path / "arms",
        run_root="runs",
        budget_basis="unit",
        checkpoint_every_decisions=8,
    )
    seeds: dict[str, set[int]] = {}
    for entry in plan["arms"]:
        config = PlasticExperimentConfig.load(Path(entry["config"]))
        block = entry["arm_id"].split("-replicate-", 1)[0]
        seeds.setdefault(block, set()).add(config.fly.sensory_mapping_seed)
    assert seeds == {"mapping-000": {0}, "mapping-001": {7}}
    assert [item["mapping_id"] for item in plan["sensory_mappings"]] == [
        "mapping-000",
        "mapping-001",
    ]


# --------------------------------------------------------------------------
# matched-control reference selection
# --------------------------------------------------------------------------


def _manifest(condition: str, **overrides) -> dict[str, object]:
    base = {
        "condition": condition,
        "topology_condition": (
            condition
            if condition.endswith("shuffled") and condition != "shuffled_reward"
            else "real"
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
        "plastic_topology_sha256": (
            "shuffled-topology" if condition.endswith("_shuffled") else "real-topology"
        ),
        "action_schedule_sha256": "actions",
        "state_hash_schedule_sha256": "states",
    }
    base.update(overrides)
    return base


def test_control_validation_is_independent_of_manifest_order() -> None:
    group = [
        _manifest("plastic_real"),
        _manifest("no_plasticity"),
        _manifest("kc_mbon_shuffled"),
    ]
    reports = []
    for permutation in itertools.permutations(group):
        report = validate_control_group(list(permutation))
        assert report["gates"]["status"] == "PASS"
        assert report["reference_condition"] == "plastic_real"
        reports.append(json.dumps(report, sort_keys=True, default=str))
    assert len(set(reports)) == 1, "the report depended on the manifest order"


def test_a_control_first_group_is_still_validated_against_the_real_arm() -> None:
    # Before the fix, manifests[0] was the reference, so a control-first group
    # silently compared every arm against a control.
    diverged = _manifest("no_plasticity", action_schedule_sha256="other-actions")
    report = validate_control_group([diverged, _manifest("plastic_real")])
    assert report["gates"]["status"] == "FAIL"
    assert "action_schedule_sha256" in report["differences"]
    index, reference = find_reference([diverged, _manifest("plastic_real")])
    assert index == 1
    assert reference["condition"] == "plastic_real"


def test_a_group_with_no_real_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly one plastic_real reference"):
        validate_control_group([_manifest("no_plasticity"), _manifest("kc_mbon_shuffled")])


def test_a_group_with_several_real_references_is_refused() -> None:
    with pytest.raises(ValueError, match="2 were supplied"):
        validate_control_group([_manifest("plastic_real"), _manifest("plastic_real")])


# --------------------------------------------------------------------------
# sensory mapping: distinct ALPNs inside one feature population
# --------------------------------------------------------------------------


def _mapping(seed: int, width: int = 3, alpns: int = 6) -> SensoryMapping:
    artifact = tiny_artifact(neurons=24 + alpns)
    projection = np.arange(8, 8 + alpns, dtype=np.int64)
    artifact = type(artifact)(
        **{
            **{
                field: getattr(artifact, field)
                for field in artifact.__dataclass_fields__
            },
            "projection_indices": projection,
        }
    )
    return SensoryMapping.from_artifact(
        artifact, mapping_seed=seed, population_width=width
    )


def test_no_feature_maps_twice_to_the_same_alpn() -> None:
    mapping = _mapping(0)
    assert mapping.version == SENSORY_MAPPING_VERSION
    for row in mapping.population_indices:
        assert len(set(row.tolist())) == len(row)
    for row in mapping.population_root_ids:
        assert len(set(row.tolist())) == len(row)
    audit = mapping.collision_audit()
    assert audit["features_with_duplicate_alpns"] == 0
    assert audit["distinct_alpns_per_feature"] == {"min": 3, "max": 3}


def test_different_features_may_still_collide() -> None:
    # Far more feature channels than ALPNs, so between-feature collisions are
    # unavoidable and deliberate.
    mapping = _mapping(0, width=2, alpns=4)
    audit = mapping.collision_audit()
    assert audit["colliding_alpns"] > 0
    assert audit["maximum_features_per_alpn"] > 1
    assert audit["features_with_duplicate_alpns"] == 0


def test_the_mapping_is_deterministic_by_seed() -> None:
    first = _mapping(11)
    again = _mapping(11)
    other = _mapping(12)
    np.testing.assert_array_equal(first.population_indices, again.population_indices)
    assert first.sha256 == again.sha256
    assert first.sha256 != other.sha256
    assert not np.array_equal(first.population_indices, other.population_indices)


def test_population_width_cannot_exceed_the_available_alpns() -> None:
    with pytest.raises(ValueError, match="exceeds the 4 annotated ALPNs"):
        _mapping(0, width=5, alpns=4)


def test_a_duplicate_population_row_is_refused_on_construction() -> None:
    mapping = _mapping(0)
    duplicated = mapping.population_indices.copy()
    duplicated[0, 1] = duplicated[0, 0]
    with pytest.raises(ValueError, match="distinct ALPNs inside"):
        type(mapping)(
            version=SENSORY_MAPPING_VERSION,
            mapping_seed=mapping.mapping_seed,
            neuron_count=mapping.neuron_count,
            feature_names=mapping.feature_names,
            population_indices=duplicated,
            population_root_ids=mapping.population_root_ids,
            available_alpn_indices=mapping.available_alpn_indices,
            available_alpn_root_ids=mapping.available_alpn_root_ids,
        )


def test_the_mapping_version_was_incremented_and_v2_is_readable() -> None:
    assert SENSORY_MAPPING_VERSION.endswith("-v3")
    assert "plastic-balatro-alpn-random-projection-v2" in SUPPORTED_SENSORY_MAPPING_VERSIONS


# --------------------------------------------------------------------------
# motor decision identity without a device round trip
# --------------------------------------------------------------------------


def _smoke_stack(tmp_path: Path, name: str, **training):
    corpus = tmp_path / f"{name}-corpus.npz"
    build_mock_corpus(corpus, seeds=(1, 2), states_per_seed=2)
    config = PlasticExperimentConfig.load(
        write_config(
            tmp_path,
            corpus_path=corpus,
            name=name,
            overrides={"training": {"max_environment_decisions": 8, **training}},
        )
    )
    return build_plastic_stack(config)


def test_motor_decision_ids_never_read_the_plastic_state_tensor(
    tmp_path: Path, monkeypatch
) -> None:
    """The exploration RNG identifier must be CPU-side, not a device scalar."""

    stack = _smoke_stack(tmp_path, "cpu-ids")
    state = stack.agent.plasticity.state
    observed: list[str] = []

    class Tripwire:
        """Any read of `decision_count` in the decision path is a CUDA sync risk."""

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __iter__(self):
            observed.append("iterated")
            return iter(self._wrapped)

        def __getitem__(self, index):
            observed.append("indexed")
            return self._wrapped[index]

        def __int__(self):
            observed.append("int")
            return int(self._wrapped)

        def __iadd__(self, other):
            # Plasticity's own bookkeeping increment is device-local and fine.
            return Tripwire(self._wrapped + other)

        def add_(self, other):
            self._wrapped.add_(other)
            return self

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    original = state.decision_count
    monkeypatch.setattr(state, "decision_count", Tripwire(original), raising=False)
    try:
        stack.trainer.step()
        stack.trainer.step()
    finally:
        monkeypatch.setattr(state, "decision_count", original, raising=False)
    assert not observed, f"the decision path read decision_count: {observed}"


def test_exploration_is_identical_across_replays_and_row_order(tmp_path: Path) -> None:
    first = _smoke_stack(tmp_path, "explore-a")
    second = _smoke_stack(tmp_path, "explore-b")
    left, right = [], []
    for _ in range(4):
        first.trainer.step()
        second.trainer.step()
        left.append(
            {k: v.copy() for k, v in first.trainer.last_executed_actions.items()}
        )
        right.append(
            {k: v.copy() for k, v in second.trainer.last_executed_actions.items()}
        )
    for one, other in zip(left, right, strict=True):
        for key in one:
            np.testing.assert_array_equal(one[key], other[key])
    # The decision identifier is the trainer's own vector step, so it advances
    # deterministically and is part of the checkpointed training state.
    assert first.trainer.state.vector_steps == 4


def test_exploration_resumes_identically_from_a_checkpoint(tmp_path: Path) -> None:
    stack = _smoke_stack(tmp_path, "resume")
    for _ in range(3):
        stack.trainer.step()
    snapshot = stack.trainer.state_dict()
    baseline = []
    for _ in range(3):
        stack.trainer.step()
        baseline.append(
            {k: v.copy() for k, v in stack.trainer.last_executed_actions.items()}
        )
    stack.trainer.load_state_dict(snapshot)
    assert stack.trainer.state.vector_steps == 3
    replayed = []
    for _ in range(3):
        stack.trainer.step()
        replayed.append(
            {k: v.copy() for k, v in stack.trainer.last_executed_actions.items()}
        )
    for one, other in zip(baseline, replayed, strict=True):
        for key in one:
            np.testing.assert_array_equal(one[key], other[key])


# --------------------------------------------------------------------------
# preflight strictness
# --------------------------------------------------------------------------


def _config(tmp_path: Path, name: str = "strict") -> PlasticExperimentConfig:
    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1, 2), states_per_seed=3)
    return PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus, name=name)
    )


def _gate(report: dict, name: str) -> dict:
    return next(item for item in report["gates"] if item["name"] == name)


def _reachability(config: PlasticExperimentConfig, status: str) -> dict:
    from flylatro.analysis.reachability import REACHABILITY_REPORT_VERSION
    from flylatro.analysis.evidence import experiment_identity

    stack = build_plastic_stack(config)
    identity = experiment_identity(
        config,
        stack.components,
        report_kind="kc_reachability",
        report_version=REACHABILITY_REPORT_VERSION,
    )
    return {
        "version": REACHABILITY_REPORT_VERSION,
        "gates": {"status": status, "checks": {}, "failed": []},
        "evidence_identity": identity.to_dict(),
    }


def test_failed_reachability_warns_at_initial_and_fails_at_ante1(tmp_path: Path) -> None:
    config = _config(tmp_path)
    failing = _reachability(config, "FAIL")
    initial = run_preflight(config, profile="initial", reachability_report=failing)
    strict = run_preflight(config, profile="ante1", reachability_report=failing)
    assert _gate(initial, "kc_subtype_reachability")["status"] == "WARN"
    assert _gate(strict, "kc_subtype_reachability")["status"] == "FAIL"
    assert "kc_subtype_reachability" in strict["failed"]
    assert "kc_subtype_reachability" not in initial["failed"]


def test_passing_reachability_passes_in_both_profiles(tmp_path: Path) -> None:
    config = _config(tmp_path)
    passing = _reachability(config, "PASS")
    for profile in ("initial", "ante1"):
        report = run_preflight(config, profile=profile, reachability_report=passing)
        assert _gate(report, "kc_subtype_reachability")["status"] == "PASS"


def test_a_report_of_the_wrong_kind_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    impostor = _reachability(config, "PASS")
    report = run_preflight(config, profile="initial", motor_calibration_report=impostor)
    gate = _gate(report, "motor_calibration")
    assert gate["status"] == "FAIL"
    assert "expects a 'motor_calibration' report" in gate["reason"]
    assert gate["evidence"]["report_kind"] == "kc_reachability"


def test_a_report_of_an_incompatible_version_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stale = _reachability(config, "PASS")
    stale["evidence_identity"]["report_version"] = "kc-subtype-reachability-v0"
    report = run_preflight(config, profile="initial", reachability_report=stale)
    gate = _gate(report, "kc_subtype_reachability")
    assert gate["status"] == "FAIL"
    assert "not compatible with this build" in gate["reason"]
    assert "kc-subtype-reachability-v1" in gate["evidence"]["compatible_versions"]


def test_a_pre_stage_report_cannot_satisfy_the_post_stage_gate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    from flylatro.analysis.evidence import experiment_identity
    from flylatro.analysis.representation import REPRESENTATION_REPORT_VERSION

    stack = build_plastic_stack(config)
    identity = experiment_identity(
        config,
        stack.components,
        report_kind="representation_pre",
        report_version=REPRESENTATION_REPORT_VERSION,
    )
    pre = {
        "stage": "pre",
        "gates": {"status": "PASS", "checks": {}, "failed": []},
        "evidence_identity": identity.to_dict(),
    }
    report = run_preflight(config, profile="initial", representation_post_report=pre)
    gate = _gate(report, "representation_post")
    assert gate["status"] == "FAIL"
    assert "expects a 'representation_post' report" in gate["reason"]
