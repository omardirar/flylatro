"""The shuffled-reward control must be bound to its own reference run.

Covers the whole dependency: the current reinforcement-event log shuffles at
all, the protocol's ``reward_seed`` is the seed that is actually used, and a
schedule of exactly the right length taken from another source run, another
events file, another action schedule or another seed is refused *before*
training starts.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from flylatro.learning.cli import main as train_main
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.protocol import ExperimentProtocol
from flylatro.learning.protocol_materialize import materialize_protocol
from flylatro.learning.reward_schedule import (
    REINFORCEMENT_SCHEDULE_VERSION,
    ReinforcementSchedule,
    ReinforcementScheduleMismatch,
    ReinforcementScheduleSource,
    assert_schedule_matches_arm,
)
from flylatro.learning.reward_schedule_cli import (
    main as shuffle_main,
    read_reinforcement_events,
)
from flylatro.learning.reinforcement import ReinforcementPulse
from helpers import build_mock_corpus, write_config

BUDGET = 12


def _current_format_log(path: Path, rows: int = 6) -> Path:
    """Exactly what `flylatro-train` writes today."""

    lines = []
    for index in range(rows):
        lines.append(
            json.dumps(
                {
                    "environment_decisions": index + 1,
                    "synthetic_reinforcement_channels": [
                        {
                            "synthetic_appetitive": 0.1 * index,
                            "synthetic_aversive": 0.0 if index % 2 else 0.25,
                            "events": ["progress"] if index % 2 else ["run_failure"],
                        }
                    ],
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _legacy_format_log(path: Path, rows: int = 4) -> Path:
    """The retained pre-rename layout, for an archived reference run."""

    lines = [
        json.dumps(
            {
                "environment_decisions": index + 1,
                "pulses": [
                    {"appetitive": 0.5 * index, "aversive": 0.0, "events": ["progress"]}
                ],
            },
            sort_keys=True,
        )
        for index in range(rows)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_current_event_log_format_shuffles_successfully(tmp_path: Path) -> None:
    pulses = list(_current_format_log_pulses(tmp_path))
    assert len(pulses) == 6
    schedule = ReinforcementSchedule.shuffled(
        pulses, source_weight_hash="w" * 64, seed=4242
    )
    assert schedule.version == REINFORCEMENT_SCHEDULE_VERSION
    # The marginal stream survives; only the temporal order is broken.
    assert sorted(p.appetitive for p in schedule.pulses) == sorted(
        p.appetitive for p in pulses
    )
    assert sorted(p.aversive for p in schedule.pulses) == sorted(
        p.aversive for p in pulses
    )
    assert tuple(schedule.pulses) != tuple(pulses)
    assert {tuple(p.events) for p in schedule.pulses} == {
        ("progress",),
        ("run_failure",),
    }


def _current_format_log_pulses(tmp_path: Path):
    return read_reinforcement_events(_current_format_log(tmp_path / "events.jsonl"))


def test_the_legacy_event_log_format_is_still_readable(tmp_path: Path) -> None:
    pulses = list(read_reinforcement_events(_legacy_format_log(tmp_path / "old.jsonl")))
    assert [p.appetitive for p in pulses] == [0.0, 0.5, 1.0, 1.5]
    assert all(p.events == ("progress",) for p in pulses)


def test_an_unrecognized_event_log_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"rewards": [1.0]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic-reinforcement event log"):
        list(read_reinforcement_events(path))


# --------------------------------------------------------------------------
# full protocol-bound dependency
# --------------------------------------------------------------------------


@pytest.fixture()
def paired(tmp_path: Path):
    """A materialized plastic_real + shuffled_reward pair with a real source run."""

    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(1, 2), states_per_seed=3)
    base_path = write_config(
        tmp_path,
        corpus_path=corpus,
        name="base",
        overrides={"training": {"max_environment_decisions": BUDGET}},
    )
    base = PlasticExperimentConfig.load(base_path)
    motor_id = build_plastic_stack(base).components["motor_mapping_sha256"]
    protocol = ExperimentProtocol.create(
        name="pairing",
        base_seed=5150,
        replicate_count=1,
        conditions=("plastic_real", "shuffled_reward"),
        exposure_budget_decisions=BUDGET,
        curriculum_ladder=(1,),
        motor_mapping_id=motor_id,
    )
    protocol_path = protocol.save(tmp_path / "protocol.json")
    plan = materialize_protocol(
        protocol,
        base,
        protocol_path=protocol_path,
        output_dir=tmp_path / "arms",
        run_root=str(tmp_path / "runs"),
        budget_basis="unit-measured",
        checkpoint_every_decisions=BUDGET,
        heavy=False,
    )
    arms = {entry["arm_id"]: entry for entry in plan["arms"]}
    real = arms["mapping-000-replicate-000:plastic_real"]
    shuffled = arms["mapping-000-replicate-000:shuffled_reward"]
    assert train_main(
        ["--config", real["config"], "--run-dir", real["run_dir"], "--no-tensorboard"]
    ) == 0
    return protocol, plan, real, shuffled


def _shuffle(entry, *, seed: int, **overrides) -> int:
    command = list(entry["prerequisite_commands"][0][1:])
    arguments = dict(zip(command[::2], command[1::2], strict=True))
    arguments["--seed"] = str(seed)
    arguments.update({key: str(value) for key, value in overrides.items()})
    flat: list[str] = []
    for key, value in arguments.items():
        flat += [key, value]
    return shuffle_main(flat)


def test_the_plan_generates_the_schedule_with_the_arms_own_reward_seed(paired) -> None:
    protocol, _, _, shuffled = paired
    arm = protocol.arm(shuffled["arm_id"])
    command = shuffled["prerequisite_commands"][0]
    assert command[0] == "flylatro-shuffle-reward"
    assert "--source-checkpoint" in command
    assert "--source-run-manifest" in command
    # No seed transcription: the plan already carries the arm's own reward_seed.
    assert command[command.index("--seed") + 1] == str(arm.reward_seed)
    assert shuffled["reward_seed"] == arm.reward_seed
    assert shuffled["source_arm_id"] == arm.reference_arm_id


def test_the_correct_source_is_accepted(paired) -> None:
    protocol, _, real, shuffled = paired
    arm = protocol.arm(shuffled["arm_id"])
    assert _shuffle(shuffled, seed=arm.reward_seed) == 0
    config = PlasticExperimentConfig.load(Path(shuffled["config"]))
    stack = build_plastic_stack(config)
    binding = stack.components["synthetic_reinforcement_schedule_binding"]
    assert binding["shuffle_seed"] == arm.reward_seed
    assert binding["source"]["arm_id"] == arm.reference_arm_id
    manifest = json.loads(
        (Path(real["run_dir"]) / "run-manifest.json").read_text(encoding="utf-8")
    )
    components = manifest["components"]
    assert (
        binding["source"]["event_log_sha256"]
        == components["synthetic_reinforcement_event_log_sha256"]
    )
    assert (
        binding["source"]["action_schedule_sha256"]
        == components["executed_action_schedule_sha256"]
    )


def test_a_mismatched_reward_seed_is_rejected(paired) -> None:
    protocol, _, _, shuffled = paired
    arm = protocol.arm(shuffled["arm_id"])
    assert _shuffle(shuffled, seed=arm.reward_seed + 1) == 0
    config = PlasticExperimentConfig.load(Path(shuffled["config"]))
    with pytest.raises(ReinforcementScheduleMismatch, match="shuffle_seed"):
        build_plastic_stack(config)


def test_a_schedule_from_another_source_arm_is_rejected(paired) -> None:
    protocol, _, _, shuffled = paired
    arm = protocol.arm(shuffled["arm_id"])
    assert (
        _shuffle(
            shuffled,
            seed=arm.reward_seed,
            **{"--source-arm-id": "mapping-000-replicate-009:plastic_real"},
        )
        == 0
    )
    config = PlasticExperimentConfig.load(Path(shuffled["config"]))
    with pytest.raises(ReinforcementScheduleMismatch, match="source_arm_id"):
        build_plastic_stack(config)


def test_a_schedule_from_another_events_file_is_rejected(paired) -> None:
    protocol, _, real, shuffled = paired
    arm = protocol.arm(shuffled["arm_id"])
    foreign = Path(real["run_dir"]).parent / "foreign-events.jsonl"
    _current_format_log(foreign, rows=BUDGET)
    # The CLI itself refuses an events file the source run did not record.
    with pytest.raises(SystemExit):
        _shuffle(shuffled, seed=arm.reward_seed, **{"--events": str(foreign)})
    # And a schedule built without that cross-check is refused at run time.
    assert _shuffle(shuffled, seed=arm.reward_seed) == 0
    schedule_path = Path(
        shuffled["prerequisite_commands"][0][
            list(shuffled["prerequisite_commands"][0]).index("--output") + 1
        ]
    )
    schedule = ReinforcementSchedule.load(schedule_path)
    forged = ReinforcementSchedule.shuffled(
        list(schedule.pulses),
        source_weight_hash=schedule.source_weight_hash,
        seed=arm.reward_seed,
        source=replace(schedule.source, event_log_sha256="f" * 64),
        target_arm_id=schedule.target_arm_id,
    )
    forged.save(schedule_path)
    config = PlasticExperimentConfig.load(Path(shuffled["config"]))
    with pytest.raises(ReinforcementScheduleMismatch, match="source_event_log_sha256"):
        build_plastic_stack(config)


def test_a_schedule_bound_to_another_action_schedule_is_rejected(paired) -> None:
    protocol, _, _, shuffled = paired
    arm = protocol.arm(shuffled["arm_id"])
    assert _shuffle(shuffled, seed=arm.reward_seed) == 0
    schedule_path = Path(
        shuffled["prerequisite_commands"][0][
            list(shuffled["prerequisite_commands"][0]).index("--output") + 1
        ]
    )
    schedule = ReinforcementSchedule.load(schedule_path)
    forged = ReinforcementSchedule.shuffled(
        list(schedule.pulses),
        source_weight_hash=schedule.source_weight_hash,
        seed=arm.reward_seed,
        source=replace(schedule.source, action_schedule_sha256="a" * 64),
        target_arm_id=schedule.target_arm_id,
    )
    forged.save(schedule_path)
    config = PlasticExperimentConfig.load(Path(shuffled["config"]))
    with pytest.raises(
        ReinforcementScheduleMismatch, match="source_action_schedule_sha256"
    ):
        build_plastic_stack(config)


def test_a_pre_provenance_schedule_version_is_refused(tmp_path: Path) -> None:
    payload = {
        "version": "deterministic-temporal-synthetic-reinforcement-shuffle-v2",
        "source_weight_hash": "w" * 64,
        "shuffle_seed": 7,
        "pulses": [
            {"appetitive": 1.0, "aversive": 0.0, "events": []},
            {"appetitive": 0.0, "aversive": 1.0, "events": []},
        ],
        "schedule_sha256": "ignored",
    }
    path = tmp_path / "old-schedule.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReinforcementScheduleMismatch, match="unsupported"):
        ReinforcementSchedule.load(path)


def test_the_binding_check_accepts_an_unbound_manual_run() -> None:
    """Without a protocol there is nothing to compare against, and no lie."""

    schedule = ReinforcementSchedule.shuffled(
        [ReinforcementPulse(1.0, 0.0), ReinforcementPulse(0.0, 1.0)],
        source_weight_hash="w" * 64,
        seed=11,
        source=ReinforcementScheduleSource(arm_id="manual"),
    )
    summary = assert_schedule_matches_arm(
        schedule,
        expected_seed=None,
        expected_source_arm_id=None,
        action_schedule_sha256=None,
        source_manifest=None,
    )
    assert summary["shuffle_seed"] == 11
    assert summary["source"]["arm_id"] == "manual"
