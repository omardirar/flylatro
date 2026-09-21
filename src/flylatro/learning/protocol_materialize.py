"""Turn a protocol manifest into exact, validated, runnable arm configurations.

A protocol that only *describes* seeds still requires a human to translate JSON
into command-line flags, which is precisely where a matched comparison silently
breaks.  Materialization emits one complete configuration file per arm plus the
exact command for it, and every generated run re-validates itself against the
protocol before it starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol import (
    ConditionArm,
    EXACT_ACTION_MATCHED_CONDITIONS,
    ExperimentProtocol,
    INDEPENDENT_TOPOLOGY_CONDITIONS,
)
from flylatro.learning.reinforcement import SENSITIVITY_CONDITIONS


MATERIALIZED_PLAN_VERSION = "materialized-protocol-plan-v1"


@dataclass(frozen=True, slots=True)
class MaterializedArm:
    arm_id: str
    replicate_id: str
    condition: str
    config_path: Path
    run_dir: str
    command: tuple[str, ...]
    depends_on: tuple[str, ...]
    source_arm_id: str = ""
    source_run_dir: str = ""
    prerequisite_commands: tuple[tuple[str, ...], ...] = ()
    reward_seed: int | None = None


def arm_directory(protocol_name: str, arm: ConditionArm, root: str) -> str:
    return f"{root}/{protocol_name}/{arm.arm_id.replace(':', '-')}"


def materialize_protocol(
    protocol: ExperimentProtocol,
    base_config: PlasticExperimentConfig,
    *,
    protocol_path: Path,
    output_dir: Path,
    run_root: str = "runs",
    budget_basis: str,
    checkpoint_every_decisions: int,
    heavy: bool = True,
) -> dict[str, Any]:
    if not budget_basis or "PLACEHOLDER" in budget_basis:
        raise ValueError("materialization needs a measured --budget-basis")
    if checkpoint_every_decisions < 1:
        raise ValueError("checkpoint cadence must be positive")
    payload = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if payload.get("sha256") != protocol.sha256:
        raise ValueError("protocol file hash does not match the loaded protocol")
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {item.arm_id: item for item in protocol.arms}
    arms: list[MaterializedArm] = []
    for arm in protocol.arms:
        real_arm = arm.reference_arm_id
        if arm.condition in EXACT_ACTION_MATCHED_CONDITIONS and real_arm not in by_id:
            raise ValueError(
                f"arm {arm.arm_id} is exact-action matched but its reference "
                f"{real_arm} is not in this protocol; include plastic_real"
            )
        real_dir = arm_directory(
            protocol.name, by_id.get(real_arm, arm), run_root
        )
        run_dir = arm_directory(protocol.name, arm, run_root)
        document = _arm_document(
            protocol,
            arm,
            base_config,
            protocol_path=protocol_path,
            checkpoint_every_decisions=checkpoint_every_decisions,
            budget_basis=budget_basis,
            matched_source_dir=real_dir,
        )
        config_path = output_dir / f"{arm.arm_id.replace(':', '-')}.toml"
        config_path.write_text(render_toml(document), encoding="utf-8")
        # Parsing here is the validation: a materialized arm that cannot load
        # and validate is a protocol bug, not a runtime surprise.
        materialized = PlasticExperimentConfig.load(config_path)
        if materialized.protocol.arm_id != arm.arm_id:
            raise ValueError("materialized configuration lost its protocol identity")
        command = _command(materialized, config_path, run_dir, arm, heavy=heavy)
        depends: tuple[str, ...] = ()
        if arm.condition in EXACT_ACTION_MATCHED_CONDITIONS:
            depends = (real_arm,)
        # The shuffled-reward dependency is generated, never transcribed: the
        # schedule is produced with exactly `arm.reward_seed` and bound to the
        # source run, and the arm refuses any other schedule.
        prerequisites = (
            _shuffled_reward_commands(arm, real_dir, run_dir)
            if arm.condition == "shuffled_reward"
            else ()
        )
        arms.append(
            MaterializedArm(
                arm_id=arm.arm_id,
                replicate_id=arm.replicate_id,
                condition=arm.condition,
                config_path=config_path,
                run_dir=run_dir,
                command=command,
                depends_on=depends,
                source_arm_id=real_arm if depends else "",
                source_run_dir=real_dir if depends else "",
                prerequisite_commands=prerequisites,
                reward_seed=arm.reward_seed if arm.condition == "shuffled_reward" else None,
            )
        )
    plan = {
        "version": MATERIALIZED_PLAN_VERSION,
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "protocol_path": str(protocol_path),
        "base_config": str(base_config.source_path),
        "base_config_sha256": base_config.sha256,
        "motor_mapping_id": protocol.motor_mapping_id,
        "sensory_mappings": [
            {
                "mapping_id": variant.mapping_id,
                "sensory_mapping_seed": variant.sensory_mapping_seed,
                "motor_mapping_id": variant.motor_mapping_id,
                "role": variant.role,
            }
            for variant in protocol.sensory_mappings
        ],
        "run_root": run_root,
        "arms": [
            {
                "arm_id": item.arm_id,
                "replicate_id": item.replicate_id,
                "condition": item.condition,
                "config": str(item.config_path),
                "run_dir": item.run_dir,
                "command": list(item.command),
                "depends_on": list(item.depends_on),
                "source_arm_id": item.source_arm_id or None,
                "source_run_dir": item.source_run_dir or None,
                "reward_seed": item.reward_seed,
                "prerequisite_commands": [
                    list(command) for command in item.prerequisite_commands
                ],
            }
            for item in arms
        ],
        "execution_notes": [
            "run every plastic_real arm before its matched controls",
            "no_plasticity replays the real arm's executed action schedule",
            "shuffled_reward's prerequisite_commands generate its schedule with "
            "the protocol arm's own reward_seed; the run refuses a schedule "
            "shuffled with any other seed or derived from another source run",
            "ordinary replicates inside one mapping block share the sensory "
            "mapping and the calibrated motor mapping; a different "
            "sensory_mapping_seed is a separate mapping block",
            "topology controls are behaviourally independent and deliberately "
            "do not replay the real arm's actions",
        ],
    }
    (output_dir / "protocol-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def _shuffled_reward_commands(
    arm: ConditionArm, source_dir: str, run_dir: str
) -> tuple[tuple[str, ...], ...]:
    """The exact schedule-generation command for one shuffled-reward arm."""

    return (
        (
            "flylatro-shuffle-reward",
            "--events",
            f"{source_dir}/synthetic-reinforcement-events.jsonl",
            "--source-checkpoint",
            f"{source_dir}/plastic-checkpoint-final.pkl",
            "--source-run-manifest",
            f"{source_dir}/run-manifest.json",
            "--source-arm-id",
            arm.reference_arm_id,
            "--target-arm-id",
            arm.arm_id,
            "--seed",
            str(arm.reward_seed),
            "--output",
            f"{source_dir}/shuffled-reinforcement.jsonl",
        ),
    )


def _arm_document(
    protocol: ExperimentProtocol,
    arm: ConditionArm,
    base: PlasticExperimentConfig,
    *,
    protocol_path: Path,
    checkpoint_every_decisions: int,
    budget_basis: str,
    matched_source_dir: str,
) -> dict[str, Any]:
    document = json.loads(json.dumps(base.to_dict()))
    document["name"] = f"{protocol.name}-{arm.arm_id.replace(':', '-')}"
    fly = document["fly"]
    fly["sensory_mapping_seed"] = arm.sensory_mapping_seed
    fly["topology"] = (
        arm.condition if arm.condition in INDEPENDENT_TOPOLOGY_CONDITIONS else "real"
    )
    fly["shuffle_seed"] = arm.topology_seed
    document["motor"]["exploration_seed"] = arm.motor_seed
    reinforcement = document["reinforcement"]
    reinforcement["condition"] = arm.reinforcement_condition
    reinforcement.update(SENSITIVITY_CONDITIONS[arm.reinforcement_condition])
    training = document["training"]
    training.update(
        condition=arm.condition,
        training_seed_offset=arm.training_seed_offset,
        base_fly_seed=arm.fly_seed,
        max_environment_decisions=arm.exposure_budget_decisions,
        checkpoint_every_decisions=checkpoint_every_decisions,
        budget_basis=budget_basis,
        plasticity_enabled=arm.condition != "no_plasticity",
        reinforcement_mode=(
            "shuffled_schedule" if arm.condition == "shuffled_reward" else "outcome"
        ),
        reinforcement_schedule_path=(
            f"{matched_source_dir}/shuffled-reinforcement.jsonl"
            if arm.condition == "shuffled_reward"
            else ""
        ),
        action_schedule_path=(
            f"{matched_source_dir}/training-actions.jsonl"
            if arm.condition in EXACT_ACTION_MATCHED_CONDITIONS
            else ""
        ),
        matched_source_run_dir=(
            matched_source_dir
            if arm.condition in EXACT_ACTION_MATCHED_CONDITIONS
            else ""
        ),
    )
    curriculum = document["curriculum"]
    curriculum["enabled"] = True
    curriculum["ladder"] = list(arm.curriculum_ladder)
    document["protocol"] = {
        "protocol_path": _relative(protocol_path, base.source_path),
        "protocol_sha256": protocol.sha256,
        "protocol_name": protocol.name,
        "replicate_id": arm.replicate_id,
        "arm_id": arm.arm_id,
    }
    return document


def _relative(path: Path, config_source: Path) -> str:
    root = config_source.parent.parent
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _command(
    config: PlasticExperimentConfig,
    config_path: Path,
    run_dir: str,
    arm: ConditionArm,
    *,
    heavy: bool,
) -> tuple[str, ...]:
    command = [
        "flylatro-train",
        "--config",
        str(config_path),
        "--run-dir",
        run_dir,
        "--condition",
        arm.condition,
        "--max-environment-decisions",
        str(arm.exposure_budget_decisions),
        "--checkpoint-every-decisions",
        str(config.training.checkpoint_every_decisions),
        "--budget-basis",
        config.training.budget_basis,
    ]
    if arm.condition in {"no_plasticity", "shuffled_reward"}:
        command += ["--action-schedule", config.training.action_schedule_path]
    if arm.condition == "shuffled_reward":
        command += [
            "--reinforcement-schedule",
            config.training.reinforcement_schedule_path,
        ]
    if heavy:
        command.append("--heavy")
    return tuple(command)


def render_toml(document: Mapping[str, Any]) -> str:
    """Minimal deterministic TOML writer for the strict configuration schema."""

    lines = ["[experiment]", f"name = {json.dumps(document['name'])}", ""]
    for section in (
        "environment",
        "fly",
        "motor",
        "plasticity",
        "reinforcement",
        "training",
        "curriculum",
        "runtime",
        "calibration",
        "protocol",
    ):
        values = document.get(section)
        if values is None:
            continue
        lines.append(f"[{section}]")
        for key in sorted(values):
            lines.append(f"{key} = {_toml_value(values[key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value)!r}")
