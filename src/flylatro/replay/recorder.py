"""Replay-oriented manifest and transition persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from flylatro.env.types import CompositeAction


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    env_index: int
    episode_id: str
    balatro_seed: int
    decision_id: int
    ante: int
    round: int
    game_state: str
    action: CompositeAction
    reward: float
    reward_components: Mapping[str, float]
    action_probability: float
    action_type_probabilities: Mapping[str, float]
    value: float
    state_hash_before: str
    state_hash_after: str
    terminated: bool
    truncated: bool
    fly_total_spikes: int
    fly_active_neurons: int

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.to_payload()
        return payload


class JsonlTransitionRecorder:
    """Writes a self-describing manifest plus complete composite actions."""

    def __init__(self, output_dir: Path, manifest: Mapping[str, Any]) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.output_dir / "manifest.json"
        self.decisions_path = self.output_dir / "decisions.jsonl"
        self.manifest_path.write_text(
            json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._stream = self.decisions_path.open("x", encoding="utf-8")

    def record(self, transition: TransitionRecord) -> None:
        self._stream.write(
            json.dumps(
                transition.to_payload(), sort_keys=True, separators=(",", ":")
            )
            + "\n"
        )
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "JsonlTransitionRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_records(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open(encoding="utf-8") as stream:
        return tuple(json.loads(line) for line in stream if line.strip())

