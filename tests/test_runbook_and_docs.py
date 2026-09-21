"""The runbook must stay executable: no stale flags, no missing required ones.

The previous version of this test only asked whether each documented ``--flag``
appeared *somewhere* in the CLI's help text.  That cannot catch a documented
command that omits a **required** option, which is exactly how the shuffled-
reward example lost ``--source-checkpoint``.  Every documented Flylatro command
is now checked against a schema derived from its real ``argparse`` parser:

* every documented flag must exist in the parser;
* every option the parser marks required must be documented;
* every documented command name must be a live console script.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
import sys
import tomllib

import pytest

RUNBOOK = pathlib.Path("docs/GPU_RUNBOOK.md")
PYPROJECT = pathlib.Path("pyproject.toml")

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

#: Console scripts whose parser builds `--help` around a required positional or
#: an interactive step, so the schema check is not meaningful for them.
UNCHECKED_COMMANDS: frozenset[str] = frozenset()


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


def _module_target(parts: list[str]) -> str:
    if parts[0].endswith(("python", "python3")):
        return parts[parts.index("-m") + 1]
    name = parts[0].rsplit("/", 1)[-1]
    target = REPO_COMMANDS.get(name)
    assert target is not None, (
        f"runbook documents {name!r} but the test has no mapping for it; either "
        "the command was renamed or the mapping is stale"
    )
    return target


def _help(target: str) -> str:
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


def parser_schema(help_text: str) -> dict[str, set[str]]:
    """Derive ``{"known", "required"}`` option sets from an argparse help page.

    argparse prints required options bare in the usage line and optional ones in
    square brackets, so bracket depth is an exact reading of the real parser
    rather than a guess from the prose below it.
    """

    lines = help_text.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.lower().startswith("usage:")
    )
    usage: list[str] = []
    for line in lines[start:]:
        if usage and (not line.strip() or not line.startswith(" ")):
            break
        usage.append(line)
    usage_text = " ".join(usage)
    usage_text = usage_text.split(":", 1)[1] if ":" in usage_text else usage_text
    required: set[str] = set()
    depth = 0
    token = ""

    def flush(current_depth: int) -> None:
        nonlocal token
        if token.startswith("--") and current_depth == 0:
            required.add(token)
        token = ""

    for character in usage_text + " ":
        if character == "[":
            flush(depth)
            depth += 1
            continue
        if character == "]":
            # The token ends *inside* the bracket it is closing.
            flush(depth)
            depth = max(depth - 1, 0)
            continue
        if character.isspace() or character in "|{}(),":
            flush(depth)
            continue
        token += character
    known = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", help_text))
    return {"known": known, "required": required}


def test_the_runbook_has_commands_and_they_parse() -> None:
    commands = _commands()
    assert len(commands) >= 20, "the runbook lost its executable commands"


def test_every_documented_command_is_a_live_console_script() -> None:
    scripts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["scripts"]
    for name, target in REPO_COMMANDS.items():
        if ":" in target and not target.startswith("flylatro.benchmarks"):
            continue
        assert name in scripts, f"{name} is no longer a declared console script"
        declared = scripts[name]
        module = declared.split(":", 1)[0]
        expected = target.split(":", 1)[0]
        assert module == expected, f"{name} now points at {declared}, not {target}"


@pytest.mark.parametrize("parts", _commands(), ids=lambda parts: parts[0].rsplit("/", 1)[-1])
def test_every_runbook_command_matches_its_real_parser(parts: list[str]) -> None:
    name = parts[0].rsplit("/", 1)[-1]
    if name in UNCHECKED_COMMANDS:
        pytest.skip(f"{name} has no checkable option schema")
    documented = {item for item in parts if item.startswith("--")}
    schema = parser_schema(_help(_module_target(parts)))
    nonexistent = sorted(documented - schema["known"])
    assert not nonexistent, f"{name} no longer accepts {nonexistent}"
    missing_required = sorted(schema["required"] - documented)
    assert not missing_required, (
        f"the documented {name} command omits required option(s) "
        f"{missing_required}; the runbook is not executable as written"
    )


def test_the_schema_reader_detects_missing_required_options() -> None:
    """The check itself must fail on a command that drops a required flag."""

    help_text = _help(REPO_COMMANDS["flylatro-shuffle-reward"])
    schema = parser_schema(help_text)
    assert {"--events", "--source-checkpoint", "--seed", "--output"} <= schema["required"]
    assert "--source-run-manifest" in schema["known"]
    assert "--source-run-manifest" not in schema["required"]
    # A command missing a required option is detected, not silently accepted.
    incomplete = {"--events", "--seed", "--output"}
    assert sorted(schema["required"] - incomplete) == ["--source-checkpoint"]


def test_the_runbook_documents_the_shuffled_reward_source_binding() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    block = text.split("## 22.", 1)[1].split("## 23.", 1)[0]
    for flag in ("--events", "--source-checkpoint", "--seed", "--output"):
        assert flag in block, f"step 22 lost {flag}"
    assert "reward_seed" in block


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
