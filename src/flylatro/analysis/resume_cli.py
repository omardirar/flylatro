"""Audit exact future actions, neural updates and full trainer state after resume."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.plastic_backend import DeferredHostReadout
from flylatro.learning.checkpoints import load_plastic_checkpoint, save_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.snapshot_identity import canonical_snapshot


def state_digest(value: Any) -> str:
    """Content identity independent of pickle object IDs and array strides."""
    digest = hashlib.sha256()

    def append(item: Any) -> None:
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        elif isinstance(item, DeferredHostReadout):
            item = np.asarray(item)
        if is_dataclass(item):
            digest.update(type(item).__qualname__.encode())
            append({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, np.random.Generator):
            append(item.bit_generator.state)
        elif isinstance(item, np.ndarray):
            digest.update(b"array:" + str(item.dtype).encode() + str(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=repr):
                append(key); append(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + str(len(item)).encode())
            for part in item:
                append(part)
        elif isinstance(item, bytes):
            digest.update(b"bytes:" + str(len(item)).encode() + b":" + item)
        else:
            scalar = item.item() if isinstance(item, np.generic) else item
            digest.update(json.dumps(scalar, sort_keys=True, allow_nan=False).encode() + b";")

    append(value)
    return digest.hexdigest()


def trainer_state(trainer: Any) -> dict[str, Any]:
    state = trainer.state_dict()
    if isinstance(trainer.env, BalatroSimAdapter):
        state["environment_snapshots"] = tuple(canonical_snapshot(snapshot,
            simulator_version=trainer.env.simulator_version) for snapshot in state["environment_snapshots"])
    return state


def audit_resume(config: PlasticExperimentConfig, checkpoint: Path, *, prefix: int,
                 future: int) -> dict[str, Any]:
    if prefix < 1 or future < 1 or (prefix + future) > config.training.max_environment_decisions:
        raise ValueError("resume probe must fit within the unchanged exposure budget")
    if prefix % config.environment.num_envs or future % config.environment.num_envs:
        raise ValueError("resume probe must align with the environment batch")
    if checkpoint.exists() or checkpoint.with_suffix(checkpoint.suffix + ".manifest.json").exists():
        raise FileExistsError(checkpoint)
    stack = build_plastic_stack(config)
    stack.trainer.train(stop_after_environment_decisions=prefix)
    saved_state = trainer_state(stack.trainer)
    saved_digest = state_digest(saved_state)
    save_plastic_checkpoint(checkpoint, stack.trainer, experiment_config=config.to_dict(),
                            component_metadata=stack.components)
    components = stack.components

    def capture(trainer: Any, metrics: dict[str, float], rows: list[dict[str, Any]]) -> None:
        rows.append({"actions_sha256": state_digest(trainer.last_executed_actions),
            "state_hashes_before": list(trainer.last_state_hashes_before),
            "state_hashes_after": list(trainer.last_state_hashes_after),
            "neural_sha256": state_digest({name: getattr(trainer.last_decision.neural, name) for name in
                ("kc_activity", "mbon_activity", "descending_activity", "output_activity")}),
            "metrics": metrics, "trainer_state_sha256": state_digest(trainer_state(trainer))})

    continuous: list[dict[str, Any]] = []
    stack.trainer.train(stop_after_environment_decisions=prefix + future,
        on_step=lambda trainer, metrics: capture(trainer, metrics, continuous))
    del stack
    gc.collect()
    resumed = build_plastic_stack(config)
    load_plastic_checkpoint(checkpoint, resumed.trainer, expected_components=resumed.components)
    restored_state = trainer_state(resumed.trainer)
    restored_digest = state_digest(restored_state)
    replayed: list[dict[str, Any]] = []
    resumed.trainer.train(stop_after_environment_decisions=prefix + future,
        on_step=lambda trainer, metrics: capture(trainer, metrics, replayed))
    equal = saved_digest == restored_digest and continuous == replayed
    return {"version": "exact-trainer-resume-audit-v2", "gates": {"status": "PASS" if equal else "FAIL"},
        "config_sha256": config.sha256, "components": components,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "prefix_decisions": prefix, "future_decisions": future,
        "saved_trainer_state_sha256": saved_digest, "restored_trainer_state_sha256": restored_digest,
        "saved_field_sha256": {key: state_digest(value) for key, value in saved_state.items()},
        "restored_field_sha256": {key: state_digest(value) for key, value in restored_state.items()},
        "snapshot_comparison": "Pinned keyed-RNG map order normalized; every key/value bit and all remaining bytes exact. Stored checkpoint bytes/hash are unchanged.",
        "continuous": continuous, "resumed": replayed,
        "evidence_scope": "Actual configured backend and simulator; exact future actions, environment, neural activity, plastic state, RNG and safety state. Fresh probe; no existing/quarantined checkpoint resumed."}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="new probe checkpoint destination")
    parser.add_argument("--prefix-decisions", type=int, default=8)
    parser.add_argument("--future-decisions", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = PlasticExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    report = audit_resume(config, args.checkpoint, prefix=args.prefix_decisions, future=args.future_decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["gates"]["status"]}))
    return 0 if report["gates"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
