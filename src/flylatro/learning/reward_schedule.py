"""Persisted synthetic-reinforcement schedules for matched controls."""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, overload

import numpy as np

from flylatro.learning.reinforcement import ReinforcementPulse
from flylatro.env.upstream_contract import ACTION_SPEC, ActionDict, validate_batch


@dataclass(frozen=True, slots=True)
class MatchedActionStep:
    """One externally matched experience step for a causal control."""

    actions: ActionDict
    state_hashes_before: tuple[str, ...] = ()
    state_hashes_after: tuple[str, ...] = ()
    curriculum_ante: int | None = None


class MatchedActionSchedule(Sequence[MatchedActionStep]):
    """Validated JSONL schedule with byte-offset lazy random access.

    Million-decision causal controls retain only an integer offset per step;
    action arrays are parsed on demand and checkpoint resume can seek directly
    to the saved vector-step index.
    """

    def __init__(self, path: Path, offsets: Sequence[int], sha256: str) -> None:
        self.path = path.resolve()
        self._offsets = tuple(int(value) for value in offsets)
        self.sha256 = sha256

    def __len__(self) -> int:
        return len(self._offsets)

    @overload
    def __getitem__(self, index: int) -> MatchedActionStep: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[MatchedActionStep, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> MatchedActionStep | tuple[MatchedActionStep, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self.path.open("rb") as stream:
            stream.seek(self._offsets[index])
            return _parse_action_step(json.loads(stream.readline()))


REINFORCEMENT_SCHEDULE_VERSION = (
    "deterministic-temporal-synthetic-reinforcement-shuffle-v3"
)


@dataclass(frozen=True, slots=True)
class ReinforcementScheduleSource:
    """Exactly which reference run a shuffled schedule was derived from.

    Length alone is not identity: two replicates with the same exposure budget
    produce equally long event logs.  A shuffled-reward arm must be able to
    prove that the reinforcement stream it replays is the marginal stream of
    *its own* paired ``plastic_real`` run, with only the temporal order broken.
    """

    arm_id: str = ""
    replicate_id: str = ""
    condition: str = ""
    protocol_sha256: str = ""
    run_manifest_sha256: str = ""
    checkpoint_sha256: str = ""
    plastic_weight_sha256: str = ""
    event_log_sha256: str = ""
    action_schedule_sha256: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "replicate_id": self.replicate_id,
            "condition": self.condition,
            "protocol_sha256": self.protocol_sha256,
            "run_manifest_sha256": self.run_manifest_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "plastic_weight_sha256": self.plastic_weight_sha256,
            "event_log_sha256": self.event_log_sha256,
            "action_schedule_sha256": self.action_schedule_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ReinforcementScheduleSource":
        values = dict(payload or {})
        known = {field.name for field in dataclass_fields(cls)}
        return cls(**{key: str(value or "") for key, value in values.items() if key in known})


class ReinforcementScheduleMismatch(ValueError):
    """A shuffled schedule does not belong to the arm about to consume it."""


@dataclass(frozen=True, slots=True)
class ReinforcementSchedule:
    version: str
    source_weight_hash: str
    shuffle_seed: int
    pulses: tuple[ReinforcementPulse, ...]
    source: ReinforcementScheduleSource = ReinforcementScheduleSource()
    target_arm_id: str = ""

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_weight_hash": self.source_weight_hash,
            "shuffle_seed": self.shuffle_seed,
            "source": self.source.to_payload(),
            "target_arm_id": self.target_arm_id,
            "pulses": [
                {
                    "appetitive": pulse.appetitive,
                    "aversive": pulse.aversive,
                    "events": list(pulse.events),
                }
                for pulse in self.pulses
            ],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def save(self, path: Path) -> Path:
        payload = {**self.payload, "schedule_sha256": self.sha256}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ReinforcementSchedule":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = str(payload["version"])
        if version != REINFORCEMENT_SCHEDULE_VERSION:
            raise ReinforcementScheduleMismatch(
                f"unsupported synthetic reinforcement schedule version {version!r}; "
                f"this build requires {REINFORCEMENT_SCHEDULE_VERSION!r}. Regenerate "
                "it with flylatro-shuffle-reward: older schedules are not bound to "
                "their source run"
            )
        schedule = cls(
            version=version,
            source_weight_hash=str(payload["source_weight_hash"]),
            shuffle_seed=int(payload["shuffle_seed"]),
            pulses=tuple(
                ReinforcementPulse(
                    appetitive=float(item["appetitive"]),
                    aversive=float(item["aversive"]),
                    events=tuple(item.get("events", ())),
                )
                for item in payload["pulses"]
            ),
            source=ReinforcementScheduleSource.from_payload(payload.get("source")),
            target_arm_id=str(payload.get("target_arm_id", "")),
        )
        if schedule.sha256 != payload["schedule_sha256"]:
            raise ValueError("synthetic reinforcement schedule hash mismatch")
        return schedule

    @classmethod
    def shuffled(
        cls,
        pulses: Sequence[ReinforcementPulse],
        *,
        source_weight_hash: str,
        seed: int,
        source: ReinforcementScheduleSource | None = None,
        target_arm_id: str = "",
    ) -> "ReinforcementSchedule":
        if len(pulses) < 2:
            raise ValueError("at least two synthetic reinforcement events are required")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pulses))
        if np.array_equal(order, np.arange(len(pulses))):
            order = np.roll(order, 1)
        identity = source or ReinforcementScheduleSource()
        if identity.plastic_weight_sha256 and (
            identity.plastic_weight_sha256 != source_weight_hash
        ):
            raise ValueError(
                "source identity plastic weight hash differs from the checkpoint"
            )
        if not identity.plastic_weight_sha256:
            identity = replace(identity, plastic_weight_sha256=source_weight_hash)
        return cls(
            version=REINFORCEMENT_SCHEDULE_VERSION,
            source_weight_hash=source_weight_hash,
            shuffle_seed=seed,
            pulses=tuple(pulses[int(index)] for index in order),
            source=identity,
            target_arm_id=target_arm_id,
        )


def assert_schedule_matches_arm(
    schedule: ReinforcementSchedule,
    *,
    expected_seed: int | None,
    expected_source_arm_id: str | None,
    expected_target_arm_id: str | None = None,
    action_schedule_sha256: str | None = None,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse a shuffled schedule that does not belong to this exact arm.

    Checks, in order: the protocol's own ``reward_seed``; the source reference
    arm; the executed action schedule the control is about to replay; and, when
    the source run's manifest is available, its recorded reinforcement
    event-log and action-schedule hashes.
    """

    differences: dict[str, dict[str, Any]] = {}

    def compare(name: str, actual: Any, wanted: Any) -> None:
        if wanted in (None, "") or actual in (None, ""):
            return
        if actual != wanted:
            differences[name] = {"schedule": actual, "expected": wanted}

    compare("shuffle_seed", schedule.shuffle_seed, expected_seed)
    compare("source_arm_id", schedule.source.arm_id, expected_source_arm_id)
    compare("target_arm_id", schedule.target_arm_id, expected_target_arm_id)
    compare(
        "source_action_schedule_sha256",
        schedule.source.action_schedule_sha256,
        action_schedule_sha256,
    )
    if source_manifest is not None:
        components = source_manifest.get("components", source_manifest)
        compare(
            "source_event_log_sha256",
            schedule.source.event_log_sha256,
            components.get("synthetic_reinforcement_event_log_sha256"),
        )
        compare(
            "source_executed_action_schedule_sha256",
            schedule.source.action_schedule_sha256,
            components.get("executed_action_schedule_sha256"),
        )
        compare(
            "source_condition",
            schedule.source.condition,
            components.get("condition"),
        )
    if differences:
        raise ReinforcementScheduleMismatch(
            "shuffled synthetic reinforcement schedule does not belong to this "
            "arm: " + json.dumps(differences, sort_keys=True, default=str)
        )
    return {
        "schedule_sha256": schedule.sha256,
        "shuffle_seed": schedule.shuffle_seed,
        "source": schedule.source.to_payload(),
        "target_arm_id": schedule.target_arm_id,
        "pulses": len(schedule.pulses),
    }


DopamineSchedule = ReinforcementSchedule


def load_action_schedule(path: Path) -> tuple[MatchedActionSchedule, str]:
    """Load exact source-run actions used to preserve shuffled-control experience."""

    path = path.resolve()
    offsets: list[int] = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            digest.update(line)
            if not line.strip():
                continue
            _parse_action_step(json.loads(line))
            offsets.append(offset)
    if not offsets:
        raise ValueError("action schedule is empty")
    sha256 = digest.hexdigest()
    return MatchedActionSchedule(path, offsets, sha256), sha256


def _parse_action_step(payload: dict[str, Any]) -> MatchedActionStep:
    actions = {
        key: np.asarray(value, dtype=ACTION_SPEC[key][1])
        for key, value in payload["actions"].items()
    }
    batch = actions["action_type"].shape[0]
    validate_batch(ACTION_SPEC, actions, batch, "action schedule")
    before = tuple(str(value) for value in payload.get("state_hashes_before", ()))
    after = tuple(str(value) for value in payload.get("state_hashes_after", ()))
    if len(before) != batch:
        raise ValueError("action schedule before-state hashes do not match batch")
    if len(after) != batch:
        raise ValueError("action schedule after-state hashes do not match batch")
    if payload.get("curriculum_ante") is None:
        raise ValueError("action schedule is missing the source curriculum Ante")
    return MatchedActionStep(
        actions=actions,
        state_hashes_before=before,
        state_hashes_after=after,
        curriculum_ante=int(payload["curriculum_ante"]),
    )
