"""Versioned, self-verifying episode replay bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPLAY_FORMAT = "flylatro-replay"
REPLAY_FORMAT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayIdentity:
    episode_id: str
    balatro_seed: int | str
    checkpoint_id: str
    checkpoint_sha256: str | None
    simulator_version: str
    encoder_hash: str
    feature_extractor_hash: str
    connectome_hash: str
    policy_version: str
    simulator_seed: int | None = None
    fly_backend_version: str | None = None
    fly_dynamics_hash: str | None = None


class ReplayBundleWriter:
    """Write one episode and publish its manifest only after completion."""

    def __init__(
        self,
        output_dir: Path,
        identity: ReplayIdentity,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.identity = identity
        self.metadata = dict(metadata or {})
        self.decisions_path = self.output_dir / "decisions.jsonl"
        self._stream = self.decisions_path.open("x", encoding="utf-8")
        self._count = 0
        self._last_decision = -1
        self._closed = False
        self._neural_file: str | None = None

    def record(self, decision: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("replay bundle is already closed")
        payload = dict(decision)
        required = {
            "decision_id", "action", "reward", "state_hash_before",
            "state_hash_after", "done",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"decision is missing fields: {sorted(missing)}")
        decision_id = int(payload["decision_id"])
        if decision_id != self._last_decision + 1:
            raise ValueError(
                f"decision IDs must be contiguous from zero; got {decision_id}"
            )
        if not isinstance(payload["action"], Mapping) or "type" not in payload["action"]:
            raise ValueError("decision action must be a complete composite payload")
        self._stream.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        self._stream.flush()
        self._last_decision = decision_id
        self._count += 1

    def attach_neural_file(self, path: Path) -> None:
        if path.resolve().parent != self.output_dir:
            raise ValueError("neural recording must live inside the replay bundle")
        if not path.is_file():
            raise FileNotFoundError(path)
        self._neural_file = path.name

    def close(self) -> Path:
        if self._closed:
            return self.output_dir / "manifest.json"
        self._stream.close()
        files: dict[str, dict[str, Any]] = {
            self.decisions_path.name: {
                "sha256": sha256_file(self.decisions_path),
                "records": self._count,
            }
        }
        if self._neural_file is not None:
            neural_path = self.output_dir / self._neural_file
            files[self._neural_file] = {"sha256": sha256_file(neural_path)}
        manifest = {
            "format": REPLAY_FORMAT,
            "format_version": REPLAY_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": asdict(self.identity),
            "metadata": self.metadata,
            "files": files,
        }
        path = self.output_dir / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self._closed = True
        return path

    def __enter__(self) -> "ReplayBundleWriter":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._stream.close()


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    path: Path
    manifest: dict[str, Any]
    decisions: tuple[dict[str, Any], ...]

    @classmethod
    def open(cls, path: Path, *, verify_hashes: bool = True) -> "ReplayBundle":
        path = path.resolve()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != REPLAY_FORMAT:
            raise ValueError("not a Flylatro replay bundle")
        if manifest.get("format_version") != REPLAY_FORMAT_VERSION:
            raise ValueError("unsupported replay bundle version")
        files = manifest.get("files", {})
        if "decisions.jsonl" not in files:
            raise ValueError("replay manifest has no decision stream")
        for name, description in files.items():
            candidate = (path / name).resolve()
            if candidate.parent != path:
                raise ValueError("replay manifest contains an unsafe file path")
            if verify_hashes and sha256_file(candidate) != description["sha256"]:
                raise ValueError(f"replay file hash mismatch: {name}")
        decisions = tuple(_read_jsonl(path / "decisions.jsonl"))
        expected = int(files["decisions.jsonl"].get("records", len(decisions)))
        if len(decisions) != expected:
            raise ValueError("replay decision count does not match manifest")
        for expected_id, decision in enumerate(decisions):
            if int(decision.get("decision_id", -1)) != expected_id:
                raise ValueError("replay decision IDs are not contiguous")
        return cls(path, manifest, decisions)

    @property
    def identity(self) -> ReplayIdentity:
        return ReplayIdentity(**self.manifest["identity"])

    @property
    def neural_path(self) -> Path | None:
        for name in self.manifest["files"]:
            if name.endswith(".parquet"):
                return self.path / name
        return None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
