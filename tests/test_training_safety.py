from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from flylatro.learning.cli import main as train_main
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.safety import TrainingSafetyConfig, TrainingSafetyMonitor, TrainingSafetyStop
from helpers import write_config


def healthy():
    return {
        "plasticity/finite_state": 1.0, "plasticity/efficacy_outside_bounds": 0.0,
        "plasticity/max_learner_bound_fraction": 0.0,
        "plasticity/max_learner_mean_absolute_drift": 0.0, "plasticity/eligibility_max": 1.0,
        "neural/finite_activity": 1.0, "neural/min_learner_kc_active_fraction": 0.3,
        "neural/min_learner_motor_active_fraction": 0.3,
    }


@pytest.mark.parametrize("field,value,reason", [
    ("plasticity/finite_state", 0.0, "non_finite_state"),
    ("plasticity/eligibility_max", float("nan"), "non_finite_state"),
    ("plasticity/max_learner_bound_fraction", 0.11, "bound_saturation"),
    ("plasticity/max_learner_mean_absolute_drift", 0.51, "efficacy_drift"),
    ("plasticity/efficacy_outside_bounds", 1.0, "efficacy_outside_bounds"),
    ("plasticity/eligibility_max", 4.1, "eligibility_outside_bounds"),
    ("neural/finite_activity", 0.0, "non_finite_neural_activity"),
])
def test_safety_stops_on_unhealthy_scalar_telemetry(field, value, reason):
    monitor = TrainingSafetyMonitor(TrainingSafetyConfig())
    failure = monitor.observe({**healthy(), field: value})
    assert reason in failure["failed"]
    # The forensic report itself remains valid JSON, even for NaN telemetry.
    json.dumps(failure, allow_nan=False)


def test_silence_patience_resets_and_resumes_exactly():
    config = TrainingSafetyConfig(silence_patience_steps=3)
    monitor = TrainingSafetyMonitor(config)
    silent = {**healthy(), "neural/min_learner_motor_active_fraction": 0.0}
    assert monitor.observe(silent) is None
    assert monitor.observe(healthy()) is None
    assert monitor.silent_steps["motor"] == 0
    assert monitor.observe(silent) is None
    assert monitor.observe(silent) is None
    restored = TrainingSafetyMonitor(config)
    restored.load_state_dict(monitor.state_dict())
    assert restored.observe(silent) == monitor.observe(silent)
    assert restored.failure["failed"] == ["motor_activity_collapse"]


def test_exact_safety_limits_pass():
    metrics = {**healthy(), "plasticity/max_learner_bound_fraction": 0.10,
               "plasticity/max_learner_mean_absolute_drift": 0.50, "plasticity/eligibility_max": 4.0}
    assert TrainingSafetyMonitor(TrainingSafetyConfig()).observe(metrics) is None


@pytest.mark.parametrize("settings", [
    {"maximum_bound_fraction": -0.1}, {"maximum_mean_absolute_drift": float("inf")},
    {"minimum_motor_active_fraction": 1.1}, {"silence_patience_steps": 0},
    {"silence_patience_steps": 1.5}, {"enabled": "yes"},
])
def test_invalid_safety_configuration_is_refused(settings):
    with pytest.raises(ValueError):
        TrainingSafetyConfig(**settings)


def test_cli_stops_with_forensic_checkpoint_and_non_success_status(tmp_path):
    config_path = write_config(tmp_path, overrides={"safety": {"maximum_mean_absolute_drift": 1e-8}})
    run_dir = tmp_path / "stopped"
    assert train_main(["--config", str(config_path), "--run-dir", str(run_dir), "--no-tensorboard"]) == 2
    report = json.loads((run_dir / "safety-stop.json").read_text())
    assert "efficacy_drift" in report["failed"]
    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    assert manifest["status"] == "safety_stopped"
    assert "completed_at" not in manifest
    assert not (run_dir / "plastic-checkpoint-final.pkl").exists()
    assert (run_dir / "plastic-checkpoint-safety-stop.pkl").exists()
    assert report["environment_decisions"] < report["requested_environment_decisions"]


def test_a_single_bad_learner_cannot_hide_in_the_batch_average(tmp_path):
    path = write_config(tmp_path, overrides={"environment": {"num_envs": 16}, "training": {"max_environment_decisions": 16}})
    stack = build_plastic_stack(PlasticExperimentConfig.load(path))
    stack.agent.plasticity.state.efficacy[0] = stack.agent.plasticity.config.min_efficacy
    metrics = stack.agent.plasticity.routine_metrics()
    assert metrics["lower_bound_fraction"] < 0.1
    assert metrics["max_learner_bound_fraction"] == 1.0
    with pytest.raises(TrainingSafetyStop, match="bound_saturation"):
        stack.trainer.train()
    assert stack.trainer.state.environment_decisions == 0


def test_staged_budget_and_exact_resume_match_uninterrupted_training(tmp_path):
    from flylatro.learning.checkpoints import load_plastic_checkpoint, save_plastic_checkpoint

    config = PlasticExperimentConfig.load(write_config(tmp_path))
    reference = build_plastic_stack(config)
    staged = build_plastic_stack(config)
    reference.trainer.train()
    staged.trainer.train(stop_after_environment_decisions=4)
    checkpoint = tmp_path / "stage.pkl"
    save_plastic_checkpoint(checkpoint, staged.trainer, experiment_config=config.to_dict(), component_metadata=staged.components)
    restored = build_plastic_stack(config)
    load_plastic_checkpoint(checkpoint, restored.trainer, expected_components=restored.components)
    restored.trainer.train()
    np.testing.assert_array_equal(reference.agent.plasticity.state.efficacy, restored.agent.plasticity.state.efficacy)
    assert reference.trainer.state == restored.trainer.state
    assert reference.trainer.safety.state_dict() == restored.trainer.safety.state_dict()


@pytest.mark.parametrize("strict_resume", [True, False])
def test_resume_safety_policy_is_bound_but_frozen_analysis_can_load(tmp_path, strict_resume):
    from flylatro.learning.checkpoints import load_plastic_checkpoint, save_plastic_checkpoint

    config = PlasticExperimentConfig.load(write_config(tmp_path))
    stack = build_plastic_stack(config)
    stack.trainer.train(stop_after_environment_decisions=4)
    checkpoint = tmp_path / "stage.pkl"
    save_plastic_checkpoint(checkpoint, stack.trainer, experiment_config=config.to_dict(), component_metadata=stack.components)
    relaxed = replace(config, safety=replace(config.safety, maximum_bound_fraction=0.50))
    restored = build_plastic_stack(relaxed)
    if strict_resume:
        with pytest.raises(ValueError, match="safety policy differs"):
            load_plastic_checkpoint(checkpoint, restored.trainer, expected_components=restored.components)
        assert restored.trainer.state.environment_decisions == 0
    else:
        load_plastic_checkpoint(checkpoint, restored.trainer, expected_components=restored.components, require_matching_exposure_budget=False)
        assert restored.trainer.state.environment_decisions == 4


def test_cli_stage_checkpoint_is_not_marked_final_and_can_resume(tmp_path):
    config_path = write_config(tmp_path)
    run = tmp_path / "staged"
    common = ["--config", str(config_path), "--run-dir", str(run), "--no-tensorboard"]
    assert train_main([*common, "--stop-after-decisions", "4"]) == 0
    checkpoint = run / "plastic-checkpoint-stage-000000000004.pkl"
    assert checkpoint.exists() and not (run / "plastic-checkpoint-final.pkl").exists()
    summary = json.loads((run / "run-summary.json").read_text())
    assert summary["status"] == "stage_complete"
    assert summary["components"]["exposure_budget_decisions"] == 16
    assert summary["runtime"]["runtime/environment_decisions_this_invocation"] == 4
    assert train_main([*common, "--resume", str(checkpoint)]) == 0
    resumed_summary = json.loads((run / "run-summary.json").read_text())
    assert resumed_summary["status"] == "completed"
    assert resumed_summary["runtime"]["runtime/environment_decisions_this_invocation"] == 12
    assert (run / "plastic-checkpoint-final.pkl").exists()
