"""Persisted synthetic-reinforcement schedules for matched controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence, overload

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


@dataclass(frozen=True, slots=True)
class ReinforcementSchedule:
    version: str
    source_weight_hash: str
    shuffle_seed: int
    pulses: tuple[ReinforcementPulse, ...]

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_weight_hash": self.source_weight_hash,
            "shuffle_seed": self.shuffle_seed,
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        schedule = cls(
            version=str(payload["version"]),
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
    ) -> "ReinforcementSchedule":
        if len(pulses) < 2:
            raise ValueError("at least two synthetic reinforcement events are required")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pulses))
        if np.array_equal(order, np.arange(len(pulses))):
            order = np.roll(order, 1)
        return cls(
            version="deterministic-temporal-synthetic-reinforcement-shuffle-v2",
            source_weight_hash=source_weight_hash,
            shuffle_seed=seed,
            pulses=tuple(pulses[int(index)] for index in order),
        )


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
