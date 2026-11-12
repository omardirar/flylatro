import json
from pathlib import Path

import pytest

from flylatro.evaluation.batch_validation import compare_batch_trajectories, qualified_batch_size
from flylatro.evaluation.compare import MATCHED_FIELDS
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import SeedPlan


def fixture(tmp_path):
    """Synthetic serialization fixture, never evidence of scientific outcomes."""
    seeds = list(SeedPlan().seeds("validation", 32))
    base = {"version": "frozen-plastic-evaluation-v2", "learning_disabled": True,
        "training_action_schedule_replayed": False,
        "action_policy": "independent-fixed-motor-choices-no-training-action-replay",
        "win_ante": 1, "seed_stream": "validation", "seeds": seeds,
        "components": {key: "synthetic-fixed-model" for key in MATCHED_FIELDS},
        "plastic_weight_sha256": "synthetic-weights", "config_sha256": "synthetic-config",
        "checkpoint_manifest": {"checkpoint_sha256": "synthetic-checkpoint"},
        "episodes": [{"seed": seed, "balatro_seed": str(seed), "won": False,
            "ante": 1, "decisions": 2, "episode_return": -1.0, "score": None} for seed in seeds]}
    rows = [{"seed": seed, "decision_id": decision, "done": decision == 1,
        "action": {"action_type": 2}, "reward": -float(decision),
        "state_hash_before": f"{seed}-{decision}", "state_hash_after": f"{seed}-{decision + 1}"}
        for seed in seeds for decision in range(2)]
    directories = [tmp_path / "sequential", tmp_path / "batched"]
    for directory, width, transitions in zip(directories, (1, 4), (rows, list(reversed(rows))), strict=True):
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({**base, "evaluation_batch_size": width}))
        (directory / "transitions.jsonl").write_text("\n".join(json.dumps(row) for row in transitions) + "\n")
    return directories


def test_batch_qualification_compares_seed_trajectories_independent_of_completion_order(tmp_path):
    reference, batched = fixture(tmp_path)
    report = compare_batch_trajectories(reference, batched, batch_size=4)
    assert report["status"] == "PASS"
    assert report["episodes"] == 32 and report["transitions"] == 64


@pytest.mark.parametrize("change", ["action", "reward", "state_hash_after", "missing", "duplicate"])
def test_batch_qualification_refuses_changed_or_incomplete_trajectories(tmp_path, change):
    reference, batched = fixture(tmp_path)
    path = batched / "transitions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[-1])
    else:
        rows[-1][change] = {"action_type": 1} if change == "action" else 99
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        compare_batch_trajectories(reference, batched, batch_size=4)


def test_final_lock_rechecks_batch_certificate_code_files_and_checkpoint(tmp_path):
    reference, batched = fixture(tmp_path)
    comparison = compare_batch_trajectories(reference, batched, batch_size=4)
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"execution_batch_size": 4}))
    certificate = tmp_path / "certificate.json"
    code = Path(__file__)
    certificate.write_text(json.dumps({"version": "frozen-batch-qualification-v1", "status": "passed",
        "batch_size": 4, "predeclaration": str(declaration), "predeclaration_sha256": sha256_file(declaration),
        "implementation_sha256": {str(code): sha256_file(code)}, "comparisons": [comparison]}))
    pair = {role + "_" + key: value for role in ("trained", "control") for key, value in (
        ("checkpoint_sha256", "synthetic-checkpoint"), ("weight_sha256", "synthetic-weights"),
        ("config_sha256", "synthetic-config"))}
    assert qualified_batch_size(certificate, [pair]) == 4
    changed = {**pair, "trained_checkpoint_sha256": "another-trained-fly"}
    with pytest.raises(ValueError, match="checkpoint lacks"):
        qualified_batch_size(certificate, [changed])
    path = batched / "transitions.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        qualified_batch_size(certificate, [pair])
