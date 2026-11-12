"""Exact trajectory qualification of frozen checkpoint replicas."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from flylatro.evaluation.compare import compare_frozen_evaluations
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import SeedPlan


def _transitions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keys = [(row["seed"], row["decision_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate seed/decision in frozen evaluation")
    return sorted(rows, key=lambda row: (row["seed"], row["decision_id"]))


def compare_batch_trajectories(reference: Path, batched: Path, *, batch_size: int) -> dict[str, Any]:
    left = json.loads((reference / "summary.json").read_text())
    right = json.loads((batched / "summary.json").read_text())
    compare_frozen_evaluations(left, right)
    if left.get("evaluation_batch_size", 1) != 1 or right.get("evaluation_batch_size") != batch_size:
        raise ValueError("qualification requires sequential reference and declared batch size")
    if left["seed_stream"] != "validation" or left["seeds"] != list(SeedPlan().seeds("validation", 32)):
        raise ValueError("qualification requires all 32 declared validation seeds")
    for key in ("config_sha256", "plastic_weight_sha256", "episodes", "win_ante"):
        if left[key] != right[key]:
            raise ValueError(f"frozen batch {key} differs from sequential evaluation")
    checkpoint_hash = left["checkpoint_manifest"]["checkpoint_sha256"]
    if checkpoint_hash != right["checkpoint_manifest"]["checkpoint_sha256"]:
        raise ValueError("qualification checkpoints differ")
    first, second = _transitions(reference / "transitions.jsonl"), _transitions(batched / "transitions.jsonl")
    if first != second:
        raise ValueError("frozen batch action/reward/state trajectory differs from sequential evaluation")
    by_seed = {seed: [] for seed in left["seeds"]}
    for row in first:
        if row["seed"] not in by_seed:
            raise ValueError("transition seed is outside validation declaration")
        by_seed[row["seed"]].append(row)
    for episode in left["episodes"]:
        rows = by_seed[episode["seed"]]
        if len(rows) != episode["decisions"] or [row["decision_id"] for row in rows] != list(range(len(rows))):
            raise ValueError("qualification trajectory is incomplete")
        if not rows or not rows[-1]["done"] or any(row["done"] for row in rows[:-1]):
            raise ValueError("qualification trajectory lacks exactly one terminal step")
    return {"status": "PASS", "batch_size": batch_size, "episodes": len(left["episodes"]),
        "transitions": len(first), "checkpoint_sha256": checkpoint_hash,
        "plastic_weight_sha256": left["plastic_weight_sha256"], "config_sha256": left["config_sha256"],
        "reference_dir": str(reference.resolve()), "batched_dir": str(batched.resolve()),
        "evidence_sha256": {str((directory / name).resolve()): sha256_file(directory / name)
            for directory in (reference, batched) for name in ("summary.json", "transitions.jsonl")}}


def qualified_batch_size(certificate: Path, pairs: Sequence[Mapping[str, Any]]) -> int:
    """Recheck file, code and trajectory identities before locking final execution."""
    report = json.loads(certificate.read_text())
    if report.get("version") != "frozen-batch-qualification-v1" or report.get("status") != "passed":
        raise ValueError("frozen batch qualification has not passed")
    for path, digest in report["implementation_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("frozen batch implementation differs from qualified code")
    declaration_path = Path(report["predeclaration"])
    if sha256_file(declaration_path) != report["predeclaration_sha256"]:
        raise ValueError("frozen batch predeclaration changed")
    declared = json.loads(declaration_path.read_text())
    width = report["batch_size"]
    if width != declared["execution_batch_size"] or width < 2:
        raise ValueError("frozen batch size differs from prospective declaration")
    rows = {row["checkpoint_sha256"]: row for row in report["comparisons"]}
    for pair in pairs:
        for role in ("trained", "control"):
            row = rows.get(pair[role + "_checkpoint_sha256"])
            if row is None or row["plastic_weight_sha256"] != pair[role + "_weight_sha256"]:
                raise ValueError("locked checkpoint lacks frozen batch qualification")
            if row["config_sha256"] != pair[role + "_config_sha256"]:
                raise ValueError("frozen batch qualification configuration differs")
            for path, digest in row["evidence_sha256"].items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError("frozen batch qualification evidence changed")
            current = compare_batch_trajectories(Path(row["reference_dir"]), Path(row["batched_dir"]), batch_size=width)
            if current != row:
                raise ValueError("frozen batch qualification no longer matches its trajectories")
    return width
