"""Atomic PPO checkpoints with complete reproducibility metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import os
from typing import Any, Mapping

import numpy as np
import torch

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.training.ppo import PPOTrainer


CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: Path,
    trainer: PPOTrainer,
    *,
    experiment_config: Mapping[str, Any],
    component_metadata: Mapping[str, Any],
    seed_metadata: Mapping[str, Any],
    curriculum_state: Mapping[str, Any] | None = None,
    reward_config: Mapping[str, Any] | None = None,
) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(trainer.env, "snapshot_all"):
        raise TypeError("environment does not support checkpoint snapshots")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "policy": trainer.policy.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "global_step": trainer.global_step,
        "update_index": trainer.update_index,
        "decision_index": trainer.decision_index,
        "observations": trainer.observations,
        "masks": trainer.masks,
        "environment_snapshots": trainer.env.snapshot_all(),
        "episode_metrics": trainer.episode_metrics,
        "curriculum_state": dict(curriculum_state or {}),
        "reward_config": dict(reward_config or {}),
        "experiment_config": dict(experiment_config),
        "component_metadata": dict(component_metadata),
        "seed_metadata": dict(seed_metadata),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "experiment_id": path.parent.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": path.name,
        "checkpoint_sha256": sha256_file(path),
        "global_step": trainer.global_step,
        "update_index": trainer.update_index,
        "trainable_parameter_count": trainer.policy.trainable_parameter_count,
        "experiment_config": dict(experiment_config),
        "components": dict(component_metadata),
        "seeds": dict(seed_metadata),
        "curriculum_state": dict(curriculum_state or {}),
        "reward_config": dict(reward_config or {}),
        "git": git_metadata(),
        "software": software_metadata(),
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest_path


def load_checkpoint(
    path: Path,
    trainer: PPOTrainer,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    path = path.resolve()
    manifest = load_checkpoint_manifest(path, verify_hash=verify_hash)
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    trainer.policy.load_state_dict(payload["policy"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.global_step = int(payload["global_step"])
    trainer.update_index = int(payload["update_index"])
    trainer.decision_index = int(payload["decision_index"])
    trainer.observations = payload["observations"]
    trainer.masks = payload["masks"]
    trainer.episode_metrics = list(payload["episode_metrics"])
    if not hasattr(trainer.env, "restore_all"):
        raise TypeError("environment does not support checkpoint restoration")
    # Real PyO3 slots must exist before their serialized state can be restored.
    trainer.env.reset(trainer.training_seeds)
    trainer.env.restore_all(payload["environment_snapshots"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    if torch.cuda.is_available() and payload["rng"]["torch_cuda"]:
        torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
    return payload


def load_checkpoint_manifest(
    path: Path, *, verify_hash: bool = True
) -> dict[str, Any]:
    path = path.resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify_hash and sha256_file(path) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match its manifest")
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
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
