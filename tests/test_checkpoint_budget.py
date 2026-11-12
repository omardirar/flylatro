from __future__ import annotations

import json
from pathlib import Path

import pytest

from flylatro.analysis.cli import main as analyze_main
from flylatro.evaluation.plastic_cli import main as evaluate_main
from flylatro.learning.checkpoints import load_plastic_checkpoint, load_plastic_manifest
from flylatro.learning.cli import main as train_main
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack


def test_cli_budget_override_supports_frozen_evaluation_and_analysis(tmp_path: Path) -> None:
    config_path = Path("configs/plastic-smoke.toml")
    run = tmp_path / "run"
    assert train_main([
        "--config", str(config_path), "--run-dir", str(run),
        "--max-environment-decisions", "4", "--checkpoint-every-decisions", "4",
        "--no-tensorboard",
    ]) == 0
    checkpoint = run / "plastic-checkpoint-final.pkl"
    manifest = load_plastic_manifest(checkpoint)
    assert manifest["components"]["exposure_budget_decisions"] == 4
    output = tmp_path / "evaluation"
    assert evaluate_main([
        "--config", str(config_path), "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--episodes", "2",
    ]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["learning_disabled"] is True
    assert summary["plastic_weight_sha256"] == manifest["plastic_weight_sha256"]
    assert summary["components"]["exposure_budget_decisions"] == 4
    analysis_path = tmp_path / "synapses.json"
    assert analyze_main([
        "--config", str(config_path), "--checkpoint", str(checkpoint),
        "--output", str(analysis_path),
    ]) == 0
    assert json.loads(analysis_path.read_text())["synaptic_changes"]


def test_resume_budget_and_frozen_model_identity_checks_remain_strict(tmp_path: Path) -> None:
    config_path = Path("configs/plastic-smoke.toml")
    run = tmp_path / "run"
    assert train_main([
        "--config", str(config_path), "--run-dir", str(run),
        "--max-environment-decisions", "4", "--checkpoint-every-decisions", "4",
        "--no-tensorboard",
    ]) == 0
    checkpoint = run / "plastic-checkpoint-final.pkl"
    stack = build_plastic_stack(PlasticExperimentConfig.load(config_path))
    with pytest.raises(ValueError, match="exposure_budget_decisions"):
        load_plastic_checkpoint(checkpoint, stack.trainer, expected_components=stack.components)
    assert stack.trainer.state.environment_decisions == 0
    mismatched = {**stack.components, "sensory_mapping_sha256": "different-mapping"}
    with pytest.raises(ValueError, match="sensory_mapping_sha256"):
        load_plastic_checkpoint(
            checkpoint, stack.trainer, expected_components=mismatched,
            require_matching_exposure_budget=False,
        )
    assert stack.trainer.state.environment_decisions == 0
