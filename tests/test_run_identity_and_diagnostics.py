from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.evidence import record_corpus_activity
from flylatro.analysis.reachability import (
    ReachabilityThresholds,
    kc_reachability_report,
)
from flylatro.analysis.readiness import EvidencePaths, READINESS_STATES, run_readiness
from flylatro.analysis.specificity import ChosenActionSpecificity
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.cli import (
    COMPLETED_IDENTITY_FIELDS,
    finalize_components,
    main as train_main,
)
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from helpers import build_mock_corpus, write_config


def test_final_metadata_products_share_one_component_identity(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1,), states_per_seed=2)
    config = write_config(
        tmp_path,
        corpus_path=corpus,
        overrides={"training": {"max_environment_decisions": 4, "checkpoint_every_decisions": 4}},
    )
    run_dir = tmp_path / "run"
    assert train_main(
        ["--config", str(config), "--run-dir", str(run_dir), "--no-tensorboard"]
    ) == 0
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run-summary.json").read_text(encoding="utf-8"))
    checkpoint = load_plastic_manifest(run_dir / "plastic-checkpoint-final.pkl")
    products = {
        "run-manifest": manifest["components"],
        "run-summary": summary["components"],
        "checkpoint-manifest": checkpoint["components"],
    }
    for field in COMPLETED_IDENTITY_FIELDS:
        values = {name: product.get(field) for name, product in products.items()}
        assert len(set(map(str, values.values()))) == 1, (field, values)
    assert manifest["completed_at"] == summary["completed_at"]
    for product in products.values():
        assert product["component_identity_version"] == "completed-run-component-identity-v1"
        assert product["external_trainable_parameter_count"] == 0
    assert summary["external_trainable_parameter_count"] == 0
    assert summary["plastic_weight_sha256"] == summary["plastic_weight_audit"][
        "plastic_weight_sha256"
    ]


def test_finalize_components_refuses_a_diverged_action_schedule() -> None:
    components = {"action_schedule_sha256": "declared"}
    with pytest.raises(RuntimeError, match="differ from the source schedule"):
        finalize_components(
            components,
            executed_action_schedule_sha256="executed",
            reserved_action_legal_observations=0,
            completed_at="now",
        )
    filled = finalize_components(
        {},
        executed_action_schedule_sha256="executed",
        reserved_action_legal_observations=3,
        completed_at="now",
    )
    assert filled["action_schedule_sha256"] == "executed"
    assert filled["state_hash_schedule_sha256"] == "executed"
    assert filled["reserved_action_legal_observations"] == 3


def _stack(tmp_path: Path):
    corpus_path = tmp_path / "corpus.npz"
    corpus = build_mock_corpus(corpus_path, seeds=(1, 2), states_per_seed=2)
    config = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus_path)
    )
    return config, corpus, build_plastic_stack(config, allow_uncalibrated_motor=True)


def test_kc_reachability_reports_subtype_coverage(tmp_path: Path) -> None:
    config, corpus, stack = _stack(tmp_path)
    activity = record_corpus_activity(stack, corpus, seed=5, repeats=2)
    processor = stack.agent.processor
    report = kc_reachability_report(
        kc_activity=activity.kc,
        kc_indices=processor.kc_indices,
        kc_types=processor.kc_types,
        mbon_activity=activity.mbon,
        mbon_indices=processor.mbon_indices,
        topology=stack.agent.plasticity.topology,
        plasticity=config.plasticity,
        motor_output_indices=processor.output_indices,
        motor_activity=activity.output,
    )
    assert report["kc_population"]["neurons"] == len(processor.kc_indices)
    subtype = report["by_kc_subtype"]["KC-synthetic"]
    for key in (
        "kc_neurons",
        "fraction_active_samples",
        "fraction_ever_active",
        "mean_hz",
        "median_hz",
        "p95_hz",
        "plastic_edges",
        "plastic_edges_with_active_presynaptic_kc",
        "plastic_edges_ever_eligible",
        "eligibility_mass",
        "eligibility_mass_fraction",
    ):
        assert key in subtype
    assert 0.0 <= report["plastic_edges"]["fraction_ever_eligible"] <= 1.0
    assert 0.0 <= report["mbon_coverage"]["fraction_plastic_mbons_active"] <= 1.0
    assert report["motor_coverage"]["evaluated"] is True
    assert "gates" in report


def test_reachability_detects_an_unreachable_plastic_subpopulation() -> None:
    from flylatro.fly.mushroom_body.plasticity import PlasticityConfig
    from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology

    topology = PlasticEdgeTopology.synthetic(
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.asarray([4, 5, 4, 5], dtype=np.int64),
        np.ones(4, dtype=np.float32),
    )
    kc = np.zeros((4, 4))
    kc[:, 0] = 10.0  # only one Kenyon cell ever fires
    mbon = np.full((4, 2), 5.0)
    report = kc_reachability_report(
        kc_activity=kc,
        kc_indices=np.asarray([0, 1, 2, 3]),
        kc_types=np.asarray(["KCa", "KCb", "KCb", "KCg"], dtype=np.str_),
        mbon_activity=mbon,
        mbon_indices=np.asarray([4, 5]),
        topology=topology,
        plasticity=PlasticityConfig(kc_reference_hz=10.0, mbon_reference_hz=5.0),
        thresholds=ReachabilityThresholds(minimum_reachable_edge_fraction=0.5),
    )
    assert report["plastic_edges"]["fraction_with_ever_active_presynaptic_kc"] == 0.25
    assert report["by_kc_subtype"]["KCb"]["fraction_ever_active"] == 0.0
    assert report["by_kc_subtype"]["KCb"]["eligibility_mass"] == 0.0
    assert report["by_kc_subtype"]["KCa"]["eligibility_mass_fraction"] == 1.0
    assert report["gates"]["status"] == "FAIL"
    assert "plastic_edges_reachable" in report["gates"]["failed"]


def test_chosen_action_specificity_attributes_updates(tmp_path: Path) -> None:
    from flylatro.fly.mushroom_body.state import state_numpy

    _, corpus, stack = _stack(tmp_path)
    agent = stack.agent
    agent.motor.record_choices = True
    recorder = ChosenActionSpecificity(agent.plasticity.topology, agent.motor.mapping)
    assert 0.0 < recorder.motor_edge_fraction <= 1.0
    observations, masks = stack.env.reset((11,))
    for decision in range(4):
        result = agent.act(
            observations, masks, fly_seeds=(decision,), deterministic_motor=True
        )
        choices = agent.motor.last_choices[0]
        eligibility = state_numpy(agent.plasticity.state.eligibility)[0].copy()
        before = state_numpy(agent.plasticity.state.efficacy)[0].copy()
        step = stack.env.step(result.actions)
        agent.learn(step.infos, plasticity_enabled=True)
        after = state_numpy(agent.plasticity.state.efficacy)[0]
        values = recorder.record(
            eligibility=eligibility,
            efficacy_delta=np.asarray(after, dtype=np.float64) - before,
            choices=choices,
        )
        assert values["total_eligibility"] >= values["chosen_eligibility"]
        observations, masks = step.observations, step.masks
    report = recorder.report()
    assert report["decisions"] == 4
    assert report["diagnostic_only_learning_rule_unchanged"] is True
    assert report["plasticity_rule"] == "three-factor-global-v1"
    fractions = (
        report["chosen_motor_update_fraction"],
        report["competing_motor_update_fraction"],
        report["unconsulted_motor_update_fraction"],
        report["non_motor_update_fraction"],
    )
    assert all(0.0 <= value <= 1.0 for value in fractions)
    assert sum(fractions) == pytest.approx(1.0, abs=1e-6)
    eligibility_fractions = (
        report["chosen_motor_eligibility_fraction"],
        report["competing_motor_eligibility_fraction"],
        report["unconsulted_motor_eligibility_fraction"],
        report["non_motor_eligibility_fraction"],
    )
    assert sum(eligibility_fractions) == pytest.approx(1.0, abs=1e-6)


def test_readiness_never_claims_ante1_from_synthetic_evidence(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1,), states_per_seed=2)
    config = PlasticExperimentConfig.load(
        write_config(tmp_path, corpus_path=corpus)
    )
    report = run_readiness(config, EvidencePaths.from_config(config))
    assert report["state"] in READINESS_STATES
    assert report["state"] not in {"READY_FOR_TINY_REAL_RUN", "READY_FOR_ANTE1"}
    assert report["real_experiment_configuration"] is False
    assert [stage["stage"] for stage in report["stages"]] == [
        "software_and_configuration",
        "flywire_artifact",
        "calibration_evidence",
        "measured_evidence_and_controls",
    ]
    assert report["remaining"]
    assert report["preflight_ante1"]["failed"]


def test_readiness_reports_a_missing_corpus_as_the_blocking_stage(tmp_path: Path) -> None:
    config = PlasticExperimentConfig.load(write_config(tmp_path))
    report = run_readiness(config, EvidencePaths.from_config(config))
    blocked = {stage["stage"]: stage for stage in report["stages"]}
    assert blocked["calibration_evidence"]["satisfied"] is False
    assert any(
        "calibration corpus" in reason
        for reason in blocked["calibration_evidence"]["blocked_by"]
    )
