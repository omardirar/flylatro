"""Corpus coverage gates, corpus provenance, occupancy and run identity.

Everything here is an engineering gate on the frozen calibration corpus.  None
of it is a biological claim, and none of it inspects reward, actions or strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.corpus import (
    CALIBRATION_CORPUS_VERSION,
    SNAPSHOT_IDENTITY_POLICY,
    CalibrationCorpus,
    build_calibration_corpus,
    scripted_navigation_actions,
)
from flylatro.analysis.coverage import (
    CALIBRATION_COVERAGE_POLICY_VERSION,
    CoveragePolicy,
    evaluate_coverage,
)
from flylatro.analysis.preflight import run_preflight
from flylatro.env.upstream_contract import (
    CONSUMABLE_SLOTS,
    JOKER_SLOTS,
    MASK_SPEC,
    OBS_SPEC,
    SHOP_SLOTS,
    validate_strict_action_batch,
)
from flylatro.learning.config import PlasticExperimentConfig
from helpers import (
    ScriptedPhaseEnv,
    build_mock_corpus,
    build_phase_corpus,
    write_config,
)


# --------------------------------------------------------------------------
# occupancy: empty vocabulary slots are 0, not -1
# --------------------------------------------------------------------------


def _corpus_with(observations: dict[str, np.ndarray], states: int) -> CalibrationCorpus:
    full = {
        key: np.zeros((states, *shape), dtype=dtype)
        for key, (shape, dtype) in OBS_SPEC.items()
    }
    full.update(observations)
    masks = {
        key: np.zeros((states, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    masks["action_type_mask"][:, 2] = True
    return CalibrationCorpus(
        version=CALIBRATION_CORPUS_VERSION,
        simulator_version="unit",
        environment_backend="unit",
        generation_method="unit fixture",
        navigation_rule="unit",
        navigation_seed=0,
        environment_seeds=(0,),
        sample_every=1,
        observations=full,
        masks=masks,
        state_hashes=tuple(f"hash-{index}" for index in range(states)),
        phase_labels=tuple(0 for _ in range(states)),
        source_seed=tuple(0 for _ in range(states)),
        source_decision=tuple(range(states)),
        source_run_seed=tuple(f"RUN{index}" for index in range(states)),
        source_episode=tuple(0 for _ in range(states)),
        source_collection_index=tuple(range(states)),
    )


def test_empty_joker_shop_and_consumable_slots_are_not_occupied() -> None:
    jokers = np.zeros((2, JOKER_SLOTS), dtype=np.int64)
    shop = np.zeros((2, SHOP_SLOTS), dtype=np.int64)
    consumables = np.zeros((2, CONSUMABLE_SLOTS), dtype=np.int64)
    # Row 0: every slot empty. Row 1: two jokers, one shop item, one consumable.
    jokers[1, :2] = [7, 9]
    shop[1, 0] = 42
    consumables[1, 0] = 3
    coverage = _corpus_with(
        {"joker_ids": jokers, "shop_ids": shop, "consumable_ids": consumables}, 2
    ).coverage()
    assert coverage["joker_occupancy"]["min"] == 0.0
    assert coverage["joker_occupancy"]["max"] == 2.0
    assert coverage["shop_occupancy"]["min"] == 0.0
    assert coverage["shop_occupancy"]["max"] == 1.0
    assert coverage["consumable_occupancy"]["min"] == 0.0
    assert coverage["consumable_occupancy"]["max"] == 1.0


# --------------------------------------------------------------------------
# per-state run provenance across auto-reset
# --------------------------------------------------------------------------


def test_every_state_records_the_actual_run_it_came_from(tmp_path: Path) -> None:
    env = ScriptedPhaseEnv(1)
    corpus = build_calibration_corpus(
        env,
        environment_seeds=(101, 202),
        states_per_seed=9,
        navigation_seed=31,
        maximum_decisions_per_seed=40,
    )
    assert corpus.records_run_provenance
    assert len(corpus.source_run_seed) == len(corpus)
    # The environment auto-resets, so one root seed spans several real runs.
    for seed in (101, 202):
        rows = [index for index, value in enumerate(corpus.source_seed) if value == seed]
        runs = {corpus.source_run_seed[index] for index in rows}
        episodes = {corpus.source_episode[index] for index in rows}
        assert len(runs) > 1, "auto-reset produced a new run that was not recorded"
        assert len(episodes) == len(runs)
        assert max(episodes) > 0
    assert list(corpus.source_collection_index) == list(range(len(corpus)))
    # The provenance is part of corpus identity and survives a round trip.
    corpus.save(tmp_path / "c.npz")
    loaded = CalibrationCorpus.load(tmp_path / "c.npz")
    assert loaded.source_run_seed == corpus.source_run_seed
    assert loaded.source_episode == corpus.source_episode
    assert loaded.sha256 == corpus.sha256


def test_run_provenance_is_part_of_corpus_identity() -> None:
    env = ScriptedPhaseEnv(1)
    corpus = build_calibration_corpus(
        env,
        environment_seeds=(7,),
        states_per_seed=6,
        navigation_seed=3,
        maximum_decisions_per_seed=30,
    )
    from dataclasses import replace

    relabelled = replace(
        corpus,
        source_run_seed=tuple("RUNXXXXXXXX" for _ in range(len(corpus))),
    )
    assert relabelled.sha256 != corpus.sha256


# --------------------------------------------------------------------------
# snapshot identity semantics
# --------------------------------------------------------------------------


def test_snapshots_are_hashed_into_corpus_identity(tmp_path: Path) -> None:
    corpus = build_phase_corpus(
        tmp_path / "snap.npz", seeds=(5,), states_per_seed=6, store_snapshots=True
    )
    manifest = json.loads(
        (tmp_path / "snap.npz.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["snapshot_identity_policy"] == SNAPSHOT_IDENTITY_POLICY
    assert manifest["stores_environment_snapshots"] is True
    assert len(manifest["snapshot_sha256"]) == len(corpus)
    assert manifest["snapshot_sha256"] == list(corpus.snapshot_sha256)
    from dataclasses import replace

    tampered = replace(
        corpus, snapshots=tuple(b"different" for _ in range(len(corpus)))
    )
    assert tampered.sha256 != corpus.sha256
    assert list(tampered.snapshot_sha256) != list(corpus.snapshot_sha256)


def test_a_corpus_without_snapshots_declares_that_explicitly(tmp_path: Path) -> None:
    build_phase_corpus(tmp_path / "plain.npz", seeds=(5,), states_per_seed=6)
    manifest = json.loads(
        (tmp_path / "plain.npz.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stores_environment_snapshots"] is False
    assert manifest["snapshot_sha256"] == []
    assert manifest["snapshot_identity_policy"] == SNAPSHOT_IDENTITY_POLICY


# --------------------------------------------------------------------------
# coverage policy
# --------------------------------------------------------------------------


def test_a_playing_only_corpus_fails_the_motor_context_coverage_gate(
    tmp_path: Path,
) -> None:
    corpus = build_mock_corpus(tmp_path / "playing.npz", seeds=(1, 2, 3), states_per_seed=4)
    result = evaluate_coverage(corpus.coverage())
    assert result["policy_version"] == CALIBRATION_COVERAGE_POLICY_VERSION
    assert result["satisfied"] is False
    assert result["motor_context_evidence_satisfied"] is False
    summary = result["deficiency_summary"]
    # Exact, greppable deficiencies, not a bare boolean.
    assert "SHOP: 0 states, required >= 4" in summary
    assert "PACK: 0 states, required >= 4" in summary
    assert any(item.startswith("shop_target:") for item in summary)
    assert any(item.startswith("pack_target:") for item in summary)


def test_a_phase_covering_corpus_satisfies_the_policy(tmp_path: Path) -> None:
    corpus = build_phase_corpus(tmp_path / "phases.npz")
    result = evaluate_coverage(corpus.coverage())
    assert result["satisfied"] is True, result["deficiency_summary"]
    assert result["observed"]["records_run_provenance"] is True
    contexts = corpus.coverage()["motor_context_coverage"]
    for name in ("shop_target", "pack_target", "joker_target", "consumable_target"):
        assert contexts[name]["relevant_states"] > 0
        assert contexts[name]["active_option_count"] >= 2


def test_the_coverage_policy_is_versioned_configurable_and_hashed(tmp_path: Path) -> None:
    policy = CoveragePolicy(minimum_states=4, minimum_unique_state_hashes=2)
    path = policy.save(tmp_path / "policy.json")
    loaded = CoveragePolicy.load(path)
    assert loaded == policy
    assert loaded.sha256 == policy.sha256
    assert CoveragePolicy().sha256 != policy.sha256
    with pytest.raises(ValueError, match="unknown coverage policy fields"):
        CoveragePolicy.from_payload({"nonsense": 1})
    with pytest.raises(ValueError, match="unknown motor contexts"):
        CoveragePolicy(required_motor_contexts=("not_a_context",))


def test_coverage_warns_at_initial_and_fails_at_ante1(tmp_path: Path) -> None:
    corpus_path = tmp_path / "playing.npz"
    build_mock_corpus(corpus_path, seeds=(1, 2, 3), states_per_seed=4)
    config = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus_path, name="cov")
    )
    statuses = {}
    for profile in ("initial", "ante1"):
        report = run_preflight(config, profile=profile)
        statuses[profile] = {
            item["name"]: item["status"] for item in report["gates"]
        }["calibration_corpus_coverage"]
    assert statuses == {"initial": "WARN", "ante1": "FAIL"}


def test_a_configured_coverage_policy_overrides_the_defaults(tmp_path: Path) -> None:
    corpus_path = tmp_path / "playing.npz"
    build_mock_corpus(corpus_path, seeds=(1, 2, 3), states_per_seed=4)
    policy = CoveragePolicy(
        minimum_states=4,
        minimum_unique_state_hashes=2,
        minimum_distinct_source_runs=1,
        required_phases=("PLAYING",),
        minimum_states_per_required_phase=2,
        minimum_unique_states_per_required_phase=2,
        required_motor_contexts=("action_type", "card_slot"),
        minimum_states_per_motor_context=2,
        minimum_competing_states_per_motor_context=1,
    )
    policy_path = policy.save(tmp_path / "policy.json")
    config = PlasticExperimentConfig.load(
        write_config(
            tmp_path,
            corpus_path=corpus_path,
            name="narrow",
            overrides={"calibration": {"coverage_policy_path": str(policy_path)}},
        )
    )
    report = run_preflight(config, profile="ante1")
    gate = next(
        item for item in report["gates"] if item["name"] == "calibration_corpus_coverage"
    )
    assert gate["status"] == "PASS"
    assert gate["evidence"]["policy_sha256"] == policy.sha256


# --------------------------------------------------------------------------
# corpus environment provenance
# --------------------------------------------------------------------------


def _real_experiment_config(tmp_path: Path, corpus_path: Path, name: str) -> Path:
    """A configuration that declares a real experiment, without a real artifact."""

    return write_config(
        tmp_path,
        corpus_path=corpus_path,
        name=name,
        overrides={"environment": {"backend": "balatro_sim"}},
    )


def test_a_real_experiment_rejects_a_mock_generated_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "mock.npz"
    build_mock_corpus(corpus_path, seeds=(1, 2, 3), states_per_seed=4)
    config = PlasticExperimentConfig.load(
        _real_experiment_config(tmp_path, corpus_path, "real-with-mock-corpus")
    )
    report = run_preflight(config, profile="initial")
    gate = next(
        item
        for item in report["gates"]
        if item["name"] == "calibration_corpus_provenance"
    )
    assert gate["status"] == "FAIL"
    assert gate["evidence"]["required_environment_backend"] == "BalatroSimAdapter"
    assert gate["evidence"]["corpus_environment_backend"] == "MockArrayBalatroEnv"
    assert "MockArrayBalatroEnv" in gate["reason"]


def test_a_real_experiment_accepts_a_matching_real_corpus_fixture(
    tmp_path: Path,
) -> None:
    """The same corpus relabelled with the real adapter identity is accepted.

    Only the *declared* provenance changes here; this is a metadata fixture, not
    a claim that the states came from real Balatro.
    """

    from dataclasses import replace

    from flylatro.env.balatro_sim import BalatroSimAdapter

    corpus_path = tmp_path / "real.npz"
    corpus = build_mock_corpus(corpus_path, seeds=(1, 2, 3), states_per_seed=4)
    relabelled = replace(
        corpus,
        environment_backend=BalatroSimAdapter.__name__,
        simulator_version=BalatroSimAdapter.simulator_version,
    )
    relabelled.save(corpus_path)
    config = PlasticExperimentConfig.load(
        _real_experiment_config(tmp_path, corpus_path, "real-with-real-corpus")
    )
    report = run_preflight(config, profile="initial")
    gate = next(
        item
        for item in report["gates"]
        if item["name"] == "calibration_corpus_provenance"
    )
    assert gate["status"] == "PASS", gate["reason"]
    identity = next(
        item for item in report["gates"] if item["name"] == "calibration_corpus_identity"
    )
    assert identity["status"] == "PASS", identity["reason"]


def test_a_wrong_simulator_version_corpus_is_rejected(tmp_path: Path) -> None:
    from dataclasses import replace

    corpus_path = tmp_path / "stale.npz"
    corpus = build_mock_corpus(corpus_path, seeds=(1, 2), states_per_seed=3)
    replace(corpus, simulator_version="balatroagent-000000000000").save(corpus_path)
    config = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus_path, name="stale-sim")
    )
    report = run_preflight(config, profile="initial")
    gate = next(
        item
        for item in report["gates"]
        if item["name"] == "calibration_corpus_provenance"
    )
    assert gate["status"] == "FAIL"
    assert gate["evidence"]["corpus_simulator_version"] == "balatroagent-000000000000"


def test_an_edited_manifest_cannot_authorize_a_run(tmp_path: Path) -> None:
    corpus_path = tmp_path / "edited.npz"
    build_mock_corpus(corpus_path, seeds=(1, 2), states_per_seed=3)
    manifest_path = tmp_path / "edited.npz.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"]["phases_observed"]["SHOP"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus_path, name="edited")
    )
    report = run_preflight(config, profile="ante1")
    gate = next(
        item for item in report["gates"] if item["name"] == "calibration_corpus_coverage"
    )
    # Coverage comes from the re-loaded arrays, not the edited manifest.
    assert gate["status"] == "FAIL"
    assert "SHOP: 0 states, required >= 4" in gate["evidence"]["deficiency_summary"]


# --------------------------------------------------------------------------
# scripted navigation stays strictly legal
# --------------------------------------------------------------------------


def test_scripted_navigation_satisfies_the_pinned_strict_referee() -> None:
    """Only the pointer the chosen type consumes may be set."""

    env = ScriptedPhaseEnv(1)
    _, masks = env.reset((17,))
    rng = np.random.default_rng(5)
    for _ in range(40):
        actions = scripted_navigation_actions(masks, rng)
        validate_strict_action_batch(actions, masks)
        step = env.step(actions)
        masks = step.masks
