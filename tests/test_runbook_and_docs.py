"""The runbook must stay executable: no stale flags, no stale report names."""

from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
import sys

import pytest

RUNBOOK = pathlib.Path("docs/GPU_RUNBOOK.md")
REPO_COMMANDS = {
    "flylatro-build-calibration-corpus": "flylatro.analysis.corpus_cli",
    "flylatro-create-sensory-mapping": "flylatro.analysis.sensory_cli",
    "flylatro-diagnose-representation": "flylatro.analysis.representation_cli",
    "flylatro-calibrate-motor": "flylatro.analysis.motor_calibration_cli",
    "flylatro-calibrate-plasticity": "flylatro.analysis.calibration_cli",
    "flylatro-diagnose-specificity": "flylatro.analysis.specificity_cli",
    "flylatro-preflight": "flylatro.analysis.preflight_cli",
    "flylatro-readiness": "flylatro.analysis.readiness_cli",
    "flylatro-validate-controls": "flylatro.analysis.control_validation_cli",
    "flylatro-create-protocol": "flylatro.learning.protocol_cli",
    "flylatro-materialize-protocol": "flylatro.learning.protocol_materialize_cli",
    "flylatro-verify-populations": "flylatro.analysis.populations_cli",
    "flylatro-train": "flylatro.learning.cli",
    "flylatro-evaluate": "flylatro.evaluation.plastic_cli",
    "flylatro-analyze-synapses": "flylatro.analysis.cli",
    "flylatro-shuffle-reward": "flylatro.learning.reward_schedule_cli",
    "bench-fly": "flylatro.benchmarks.cli:fly_main",
    "bench-plastic-end-to-end": "flylatro.benchmarks.cli:plastic_end_to_end_main",
}


def _commands() -> list[list[str]]:
    text = RUNBOOK.read_text(encoding="utf-8")
    parsed: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            line = line.strip().split("|")[0]
            if not line.startswith((".venv/bin/", "python ")):
                continue
            parsed.append(shlex.split(line))
    return parsed


def _help(parts: list[str]) -> str:
    if parts[0].endswith("python") or parts[0].endswith("python3"):
        target = parts[parts.index("-m") + 1]
    else:
        name = parts[0].rsplit("/", 1)[-1]
        target = REPO_COMMANDS.get(name)
        if target is None:
            pytest.skip(f"unmapped runbook command: {name}")
    if ":" in target:
        module, function = target.split(":", 1)
        # Several console scripts share one module, so ask the exact entry point.
        command = [
            sys.executable,
            "-c",
            f"from {module} import {function} as run; run(['--help'])",
        ]
    else:
        command = [sys.executable, "-m", target, "--help"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_the_runbook_has_commands_and_they_parse() -> None:
    commands = _commands()
    assert len(commands) >= 20, "the runbook lost its executable commands"


@pytest.mark.parametrize("parts", _commands(), ids=lambda parts: parts[0].rsplit("/", 1)[-1])
def test_every_runbook_flag_exists_in_its_cli(parts: list[str]) -> None:
    flags = sorted({item for item in parts if item.startswith("--")})
    if not flags:
        return
    help_text = _help(parts)
    missing = [flag for flag in flags if flag not in help_text]
    assert not missing, f"{parts[0]} no longer accepts {missing}"


def test_the_runbook_uses_the_corrected_dependency_order() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    stages = [
        "Build and verify the v783 artifact",
        "Inspect the population census",
        "Generate the frozen reward-free calibration state corpus",
        "Create the field-aware sensory mapping",
        "Audit the sensory mapping on the actual calibration corpus",
        "Run neural representation PRE diagnostics",
        "Choose the neural duration",
        "Calibrate canonical reward-free motor candidates",
        "Confirm the persisted motor normalization",
        "Run representation POST diagnostics",
        "Calibrate eligibility and plasticity stability",
        "Run the initial preflight",
        "Run a tiny real plastic experiment",
        "Run the exact matched no-plasticity control",
        "Validate the control manifests",
        "Benchmark throughput, memory and synchronization cost",
        "Tune implementation performance only where measured necessary",
        "Choose exposure, checkpoint and evaluation budgets",
        "Create the paired replicate protocol",
        "Materialize the protocol arms",
        "Run the strict Ante-1 preflight",
        "Run the paired Ante-1 experiments",
        "Analyze behaviour, plasticity specificity and stability",
        "Only then decide whether to proceed to curriculum",
    ]
    positions = [text.find(stage) for stage in stages]
    assert all(position >= 0 for position in positions), [
        stage for stage, position in zip(stages, positions, strict=True) if position < 0
    ]
    assert positions == sorted(positions), "runbook stages are out of dependency order"
    assert "## Stop conditions" in text
