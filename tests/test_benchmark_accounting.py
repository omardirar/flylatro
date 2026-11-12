from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from flylatro.benchmarks.cli import plastic_end_to_end_main
import flylatro.learning.config as configuration
from flylatro.learning.protocol_materialize import render_toml


def test_component_totals_cover_every_decision_and_exclude_comparison(tmp_path, monkeypatch):
    config = configuration.PlasticExperimentConfig.load(Path("configs/plastic-smoke.toml"))
    config = replace(config, environment=replace(config.environment, num_envs=2))
    payload = asdict(config); payload.pop("source_path")
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(render_toml(payload))
    real_build = configuration.build_plastic_stack

    def instrument(config):
        stack = real_build(config)
        backend = SimpleNamespace(component_seconds={}, poisson_method="instrumented")
        stack.agent.processor.backend = backend
        process = stack.agent.processor.process

        def replaced_process(*args, **kwargs):
            result = process(*args, **kwargs)
            if len(kwargs["fly_seeds"]) > 1:
                result = replace(result, output_activity=result.output_activity + 1)
            backend.component_seconds = {"recurrent": 999.0}
            return result

        stack.agent.processor.process = replaced_process
        step = stack.trainer.step
        calls = 0

        def measured_step():
            nonlocal calls
            result = step(); calls += 1
            backend.component_seconds = {"recurrent": float(calls)}
            return result

        stack.trainer.step = measured_step
        return stack

    monkeypatch.setattr(configuration, "build_plastic_stack", instrument)
    output = tmp_path / "benchmark.json"
    assert plastic_end_to_end_main(["--config", str(config_path), "--steps", "3",
        "--profile-components", "--compare-sequential", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["component_seconds"]["recurrent"] == 6.0
    assert report["evidence_identity"]["report_version"] == "plastic-end-to-end-benchmark-v3"
    assert report["neural_execution_comparison"]["batch_size"] == 2
    assert not report["neural_execution_comparison"]["exact_activity_agreement"]
    assert report["neural_execution_comparison"]["maximum_activity_difference_hz"] == pytest.approx(1.0, abs=1e-5)
