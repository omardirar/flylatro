from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from flylatro.analysis.preflight import (
    PREFLIGHT_PROFILES,
    PreflightThresholds,
    run_preflight,
)
from flylatro.analysis.provenance import EvidenceIdentity
from flylatro.learning.config import PlasticExperimentConfig
from helpers import build_mock_corpus, write_config


def _gate(report: dict[str, Any], name: str) -> dict[str, Any]:
    for item in report["gates"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"preflight has no gate {name}")


@pytest.fixture()
def evidence(tmp_path: Path):
    """A synthetic stack with provenance-matched representation evidence."""

    from flylatro.analysis.evidence import experiment_identity
    from flylatro.analysis.representation import REPRESENTATION_REPORT_VERSION
    from flylatro.learning.config import build_plastic_stack

    corpus_path = tmp_path / "corpus.npz"
    corpus = build_mock_corpus(corpus_path, seeds=(1, 2), states_per_seed=2)
    config_path = write_config(tmp_path, corpus_path=corpus_path)
    config = PlasticExperimentConfig.load(config_path)
    stack = build_plastic_stack(config, allow_uncalibrated_motor=True)
    identity = experiment_identity(
        config,
        stack.components,
        report_kind="representation_pre",
        report_version=REPRESENTATION_REPORT_VERSION,
        corpus=corpus,
    )
    report = {
        "stage": "pre",
        "motor_activity_population": "mbon",
        "final_motor_interface_evaluated": False,
        "gates": {"status": "PASS", "checks": {}, "failed": []},
        "evidence_identity": identity.to_dict(),
    }
    return config, report


def test_matching_evidence_passes_and_records_its_identity(evidence) -> None:
    config, report = evidence
    result = run_preflight(config, profile="initial", representation_pre_report=report)
    gate = _gate(result, "representation_pre")
    assert gate["status"] == "PASS"
    assert "provenance verified" in gate["reason"]
    assert result["expected_identity"]["calibration_corpus_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision_duration_ms", 987.0),
        ("sensory_mapping_sha256", "stale-sensory-mapping"),
        ("artifact_sha256", "stale-artifact"),
        ("plastic_topology_sha256", "stale-topology"),
        ("minimum_synapse_count", 7),
        ("calibration_corpus_sha256", "stale-corpus"),
        ("fly_dynamics_sha256", "stale-dynamics"),
        ("output_mode", "whole_brain"),
    ],
)
def test_preflight_rejects_mismatched_evidence(evidence, field, value) -> None:
    config, report = evidence
    stale = copy.deepcopy(report)
    stale["evidence_identity"][field] = value
    result = run_preflight(config, profile="initial", representation_pre_report=stale)
    gate = _gate(result, "representation_pre")
    assert gate["status"] == "FAIL"
    assert field in gate["reason"]
    assert "different configuration" in gate["reason"]


def test_preflight_rejects_evidence_with_no_identity(evidence) -> None:
    config, report = evidence
    anonymous = {key: value for key, value in report.items() if key != "evidence_identity"}
    result = run_preflight(config, profile="initial", representation_pre_report=anonymous)
    gate = _gate(result, "representation_pre")
    assert gate["status"] == "FAIL"
    assert "no evidence identity" in gate["reason"]


def test_preflight_rejects_a_passing_report_from_the_wrong_stage(evidence) -> None:
    config, report = evidence
    wrong = copy.deepcopy(report)
    wrong["evidence_identity"]["report_kind"] = "representation_post"
    result = run_preflight(config, profile="initial", representation_post_report=wrong)
    gate = _gate(result, "representation_post")
    assert gate["status"] == "FAIL"
    assert "'pre' stage report" in gate["reason"]


def test_plasticity_evidence_must_match_the_current_rule(evidence) -> None:
    config, _ = evidence
    from dataclasses import replace

    from flylatro.analysis.evidence import experiment_identity
    from flylatro.learning.config import build_plastic_stack

    stack = build_plastic_stack(config, allow_uncalibrated_motor=True)
    identity = experiment_identity(
        config,
        stack.components,
        report_kind="plasticity_calibration",
        report_version="plasticity-calibration-v2",
    )
    report = {
        "gates": {"status": "PASS", "checks": {}, "failed": []},
        "diagnostics": {"lower_bound_fraction": 0.0, "upper_bound_fraction": 0.0},
        "evidence_identity": identity.to_dict(),
    }
    assert _gate(
        run_preflight(config, profile="initial", plasticity_report=report),
        "plasticity_calibration",
    )["status"] == "PASS"

    changed = replace(
        config, plasticity=replace(config.plasticity, learning_rate=0.5)
    )
    gate = _gate(
        run_preflight(changed, profile="initial", plasticity_report=report),
        "plasticity_calibration",
    )
    assert gate["status"] == "FAIL"
    assert "plasticity_rule_sha256" in gate["reason"]


def test_bound_occupancy_fails_even_when_the_report_says_pass(evidence) -> None:
    config, _ = evidence
    from flylatro.analysis.evidence import experiment_identity
    from flylatro.learning.config import build_plastic_stack

    stack = build_plastic_stack(config, allow_uncalibrated_motor=True)
    report = {
        "gates": {"status": "PASS", "checks": {}, "failed": []},
        "diagnostics": {"lower_bound_fraction": 0.4, "upper_bound_fraction": 0.4},
        "evidence_identity": experiment_identity(
            config,
            stack.components,
            report_kind="plasticity_calibration",
            report_version="plasticity-calibration-v2",
        ).to_dict(),
    }
    gate = _gate(
        run_preflight(config, profile="initial", plasticity_report=report),
        "plasticity_calibration",
    )
    assert gate["status"] == "FAIL"
    assert "weight-bound occupancy" in gate["reason"]


def test_initial_and_strict_profiles_differ_in_what_they_demand(evidence) -> None:
    config, report = evidence
    initial = run_preflight(config, profile="initial", representation_pre_report=report)
    strict = run_preflight(config, profile="ante1", representation_pre_report=report)
    assert initial["profile"] == "initial" and strict["profile"] == "ante1"
    for name in (
        "throughput_and_memory_benchmark",
        "matched_control_readiness",
        "tiny_real_plastic_run",
        "protocol_identity",
        "chosen_action_specificity",
    ):
        assert _gate(initial, name)["status"] == "WARN"
        assert _gate(strict, name)["status"] == "FAIL"
    assert set(PREFLIGHT_PROFILES) == {"initial", "ante1"}


def test_threshold_overrides_can_re_enable_the_structural_collision_gate(evidence) -> None:
    config, _ = evidence
    default = run_preflight(config, profile="initial")
    assert default["thresholds"]["maximum_alpn_assignment_p99"] is None
    tightened = run_preflight(
        config,
        profile="initial",
        thresholds=PreflightThresholds(maximum_alpn_assignment_p99=1.0),
    )
    assert tightened["thresholds"]["maximum_alpn_assignment_p99"] == 1.0


def test_identity_mismatch_helper_requires_claimed_fields() -> None:
    expected = EvidenceIdentity(
        report_kind="x", report_version="1", artifact_sha256="a", decision_duration_ms=50.0
    )
    silent = EvidenceIdentity(report_kind="x", report_version="1")
    mismatches = silent.mismatches(
        expected, required_fields=("artifact_sha256", "decision_duration_ms")
    )
    assert set(mismatches) == {"artifact_sha256", "decision_duration_ms"}
    assert mismatches["artifact_sha256"]["report"] is None
