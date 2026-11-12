from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flylatro.analysis.provenance import EvidenceIdentity
from flylatro.analysis.reachability import REACHABILITY_REPORT_VERSION
from flylatro.analysis.representation import REPRESENTATION_REPORT_VERSION
from flylatro.learning.checkpoints import load_plastic_manifest, save_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.protocol import ExperimentProtocol
from flylatro.learning.protocol_materialize import materialize_protocol
from flylatro.learning import staged_cli
from helpers import build_mock_corpus, write_config


@pytest.fixture
def checkpoint_health(tmp_path):
    config = PlasticExperimentConfig.load(write_config(tmp_path))
    stack = build_plastic_stack(config)
    stack.trainer.train()
    checkpoint = tmp_path / "trained.pkl"
    save_plastic_checkpoint(checkpoint, stack.trainer, experiment_config=config.to_dict(), component_metadata=stack.components)
    manifest = load_plastic_manifest(checkpoint)
    expected = EvidenceIdentity.from_components(stack.components, report_kind="expected", report_version="test", calibration_corpus_sha256="corpus", include_git=False).to_dict()
    reports = []
    for kind, version in (("representation_checkpoint", REPRESENTATION_REPORT_VERSION), ("kc_reachability_checkpoint", REACHABILITY_REPORT_VERSION)):
        reports.append({
            "evidence_identity": {**expected, "report_kind": kind, "report_version": version, "plastic_weight_sha256": manifest["plastic_weight_sha256"], "checkpoint_sha256": manifest["checkpoint_sha256"]},
            "checkpoint": {"path": str(checkpoint)},
            "efficacy_policy": "frozen-checkpoint-efficacy", "gates": {"status": "PASS", "failed": []},
            "corpus_states_used": 4, "repeats_per_state": 3,
        })
    return reports, expected, checkpoint


def test_checkpoint_health_accepts_bound_frozen_reports(checkpoint_health):
    reports, expected, checkpoint = checkpoint_health
    staged_cli.validate_checkpoint_health(*reports, expected, corpus_states=4, checkpoint=checkpoint)


@pytest.mark.parametrize("report_index,field,value", [
    (0, "report_kind", "representation_post"),
    (1, "report_kind", "kc_reachability"),
    (0, "report_version", "old"),
    (1, "report_version", "old"),
    (0, "plastic_weight_sha256", "different"),
    (1, "checkpoint_sha256", "different"),
    (0, "sensory_mapping_sha256", "different"),
    (1, "motor_mapping_sha256", "different"),
])
def test_checkpoint_health_refuses_initial_stale_or_different_weight_reports(checkpoint_health, report_index, field, value):
    reports, expected, checkpoint = checkpoint_health
    reports[report_index]["evidence_identity"][field] = value
    with pytest.raises(ValueError):
        staged_cli.validate_checkpoint_health(*reports, expected, corpus_states=4, checkpoint=checkpoint)


@pytest.mark.parametrize("change", ["failed", "partial_corpus", "one_repeat", "initial_weights", "missing_checkpoint", "corrupt_checkpoint"])
def test_checkpoint_health_refuses_unsafe_or_incomplete_evidence(checkpoint_health, change):
    reports, expected, checkpoint = checkpoint_health
    if change == "failed":
        reports[1]["gates"]["status"] = "FAIL"
    elif change == "partial_corpus":
        reports[0]["corpus_states_used"] = 3
    elif change == "one_repeat":
        reports[0]["repeats_per_state"] = 1
    elif change == "initial_weights":
        reports[0]["efficacy_policy"] = "frozen-initial-efficacy"
    elif change == "missing_checkpoint":
        reports[0]["checkpoint"] = None
    else:
        checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        staged_cli.validate_checkpoint_health(*reports, expected, corpus_states=4, checkpoint=checkpoint)


@pytest.fixture
def staged_plan(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.npz"
    build_mock_corpus(corpus_path, seeds=(1, 2), states_per_seed=2)
    base = PlasticExperimentConfig.load(write_config(tmp_path, corpus_path=corpus_path))
    protocol = ExperimentProtocol.create(name="staged-test", base_seed=700, replicate_count=2, conditions=("plastic_real", "no_plasticity"), exposure_budget_decisions=16, curriculum_ladder=(1,), motor_mapping_id="motor-test")
    protocol_path = protocol.save(tmp_path / "protocol.json")
    plan = materialize_protocol(protocol, base, protocol_path=protocol_path, output_dir=tmp_path / "arms", run_root=str(tmp_path / "runs"), budget_basis="measured-test", checkpoint_every_decisions=4, heavy=False)
    health = tmp_path / "health.json"
    health.write_text("{}")
    audits = []
    monkeypatch.setattr(staged_cli, "run_preflight", lambda *args, **kwargs: {"status": "PASS", "failed": [], "expected_identity": {"report_kind": "expected", "report_version": "test"}})
    monkeypatch.setattr(staged_cli, "validate_checkpoint_health", lambda *args, **kwargs: audits.append(kwargs))
    arguments = ["--plan", str(tmp_path / "arms" / "protocol-plan.json"), "--health-report", str(health), "--reachability-report", str(health), "--control-manifest", str(health), "--stages", "4,16", "--output", str(tmp_path / "execution.json")]
    return plan, arguments, audits


def test_dry_plan_preserves_declared_budget_and_exact_stage_resume(staged_plan):
    plan, arguments, audits = staged_plan
    assert staged_cli.main(arguments) == 0
    report = json.loads(Path(arguments[-1]).read_text())
    assert report["status"] == "planned" and len(audits) == 2
    assert len(report["arms"]) == 4 and not report["completed_stages"]
    for arm in report["arms"]:
        first, final = arm["commands"]
        assert "--resume" not in first["training"]
        assert final["training"][final["training"].index("--resume") + 1] == first["checkpoint"]
        assert first["checkpoint"].endswith("plastic-checkpoint-stage-000000000004.pkl")
        assert final["checkpoint"].endswith("plastic-checkpoint-final.pkl")
        assert "--max-environment-decisions" not in first["training"]
        assert PlasticExperimentConfig.load(Path(arm["config"])).training.max_environment_decisions == 16
        evaluation = arm["evaluation_command"]
        assert evaluation[evaluation.index("--win-ante") + 1] == "1"
        assert evaluation[evaluation.index("--episodes") + 1] == "32"
    assert all(not Path(entry["run_dir"]).exists() for entry in plan["arms"])


@pytest.mark.parametrize("failed_module", ["flylatro.learning.cli", "flylatro.analysis.representation_cli"])
def test_failed_stage_halts_every_later_stage_and_arm(staged_plan, monkeypatch, failed_module):
    _, arguments, _ = staged_plan
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=2 if failed_module in command else 0)
    monkeypatch.setattr(staged_cli.subprocess, "run", execute)
    assert staged_cli.main([*arguments, "--execute", "--heavy"]) == 2
    report = json.loads(Path(arguments[-1]).read_text())
    assert report["status"] == "halted" and report["failed"]
    assert report["completed_stages"] == []
    assert len(commands) == (1 if failed_module == "flylatro.learning.cli" else 2)


def test_successful_runner_audits_stages_and_compares_paired_evaluations(staged_plan, monkeypatch):
    _, arguments, audits = staged_plan
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        if "flylatro.learning.cli" in command:
            run_dir = Path(command[command.index("--run-dir") + 1])
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run-manifest.json").write_text('{"status": "completed"}')
        if "flylatro.analysis.representation_cli" in command:
            for option in ("--output", "--reachability-output"):
                path = Path(command[command.index(option) + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(staged_cli.subprocess, "run", execute)
    monkeypatch.setattr(staged_cli, "validate_control_group", lambda manifests: {"gates": {"status": "PASS", "failed": []}})
    assert staged_cli.main([*arguments, "--execute", "--heavy"]) == 0
    report = json.loads(Path(arguments[-1]).read_text())
    assert report["status"] == "completed" and len(report["completed_stages"]) == 8
    assert len(audits) == 6 and len(report["control_validations"]) == 2
    assert sum("flylatro.evaluation.compare" in command for command in commands) == 2
    assert sum("flylatro.evaluation.plastic_cli" in command for command in commands) == 4


def test_invalid_dependencies_are_refused_before_execution(staged_plan):
    _, arguments, _ = staged_plan
    plan_path = Path(arguments[arguments.index("--plan") + 1])
    plan = json.loads(plan_path.read_text())
    plan["arms"][1]["depends_on"] = ["unrelated-arm"]
    plan_path.write_text(json.dumps(plan))
    assert staged_cli.main(arguments) == 2
    report = json.loads(Path(arguments[-1]).read_text())
    assert report["status"] == "halted" and "dependencies" in report["failed"][0]


def test_existing_execution_report_is_never_overwritten(staged_plan):
    _, arguments, _ = staged_plan
    output = Path(arguments[-1])
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        staged_cli.main(arguments)
    assert output.read_text() == "existing"
