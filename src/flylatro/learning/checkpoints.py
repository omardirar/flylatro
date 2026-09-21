"""Atomic, hash-verified checkpoints for learned fly synaptic state."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
from typing import Any, Mapping

import numpy as np

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.trainer import PlasticTrainer


PLASTIC_CHECKPOINT_FORMAT = 1

#: Fields a completed run fills in at the end. A final checkpoint carries the
#: canonical completed identity, so these are present there but absent from a
#: freshly built stack. They are compared only when the current configuration
#: actually declares a value — which is exactly the matched-control case, where
#: the run is bound to a source action schedule that must still agree.
COMPLETION_DERIVED_COMPONENT_FIELDS: frozenset[str] = frozenset(
    {
        "action_schedule_sha256",
        "state_hash_schedule_sha256",
        "executed_action_schedule_sha256",
        "reserved_action_legal_observations",
        "component_identity_version",
        "completed_at",
    }
)


def save_plastic_checkpoint(
    path: Path,
    trainer: PlasticTrainer,
    *,
    experiment_config: Mapping[str, Any],
    component_metadata: Mapping[str, Any],
    curriculum_state: Mapping[str, Any] | None = None,
) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": PLASTIC_CHECKPOINT_FORMAT,
        "kind": "flylatro-plastic-brain",
        "trainer": trainer.state_dict(),
        "experiment_config": dict(experiment_config),
        "components": dict(component_metadata),
        "curriculum_state": dict(curriculum_state or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
    temporary.replace(path)
    plasticity = trainer.agent.plasticity
    manifest = {
        "format_version": PLASTIC_CHECKPOINT_FORMAT,
        "kind": payload["kind"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": path.name,
        "checkpoint_sha256": sha256_file(path),
        "plastic_weight_sha256": plasticity.state.weight_sha256,
        "plastic_parameter_count": trainer.agent.plastic_parameter_count,
        "external_trainable_parameter_count": trainer.agent.trainable_parameter_count,
        "environment_decisions": trainer.state.environment_decisions,
        "episodes": trainer.state.completed_episodes,
        "topology_sha256": plasticity.topology.sha256,
        "plasticity_rule_sha256": plasticity.config.sha256,
        "components": dict(component_metadata),
        "experiment_config": dict(experiment_config),
        "curriculum_state": dict(curriculum_state or {}),
        "git": git_metadata(),
        "software": software_metadata(),
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)
    return manifest_path


def load_plastic_checkpoint(
    path: Path,
    trainer: PlasticTrainer,
    *,
    expected_components: Mapping[str, Any] | None = None,
    verify_hash: bool = True,
) -> dict[str, Any]:
    path = path.resolve()
    manifest = load_plastic_manifest(path, verify_hash=verify_hash)
    with path.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - trusted local experiment state
    if payload.get("format_version") != PLASTIC_CHECKPOINT_FORMAT:
        raise ValueError("unsupported plastic checkpoint format")
    if payload.get("kind") != "flylatro-plastic-brain":
        raise ValueError("checkpoint is not a plastic-brain checkpoint")
    if expected_components is not None:
        differences = {
            key: {"checkpoint": payload["components"].get(key), "current": value}
            for key, value in expected_components.items()
            if payload["components"].get(key) != value
            and not (value is None and key in COMPLETION_DERIVED_COMPONENT_FIELDS)
        }
        if differences:
            raise ValueError(
                "checkpoint components differ: "
                + json.dumps(differences, sort_keys=True)
            )
    trainer.load_state_dict(payload["trainer"])
    if trainer.agent.plasticity.state.weight_sha256 != manifest["plastic_weight_sha256"]:
        raise ValueError("restored plastic weights differ from manifest")
    return payload


def load_plastic_manifest(path: Path, *, verify_hash: bool = True) -> dict[str, Any]:
    path = path.resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify_hash and sha256_file(path) != manifest["checkpoint_sha256"]:
        raise ValueError("plastic checkpoint SHA-256 does not match manifest")
    return manifest


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ("git", *args), capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def software_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    try:
        import torch
    except ImportError:
        metadata.update(torch=None, cuda_available=False)
    else:
        metadata.update(
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            device_count=torch.cuda.device_count(),
            cuda_devices=[
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        )
    return metadata
