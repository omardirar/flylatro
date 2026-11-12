import importlib
import json
from pathlib import Path

import pytest

from flylatro.evaluation.compare import MATCHED_FIELDS
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import SeedPlan


def inputs(tmp_path):
    seeds = list(SeedPlan().seeds("final_test", 128))
    model = {key: "synthetic-development-fixture" for key in MATCHED_FIELDS}
    lock = {"final_seeds": seeds, "model_identity": model, "pairs": []}
    for index in range(3):
        config = tmp_path / f"config-{index}.toml"; config.write_text("synthetic configuration")
        checkpoint = tmp_path / f"checkpoint-{index}.pkl"; checkpoint.write_bytes(b"synthetic checkpoint")
        summary_path = tmp_path / f"summary-{index}.json"
        summary = {"seed_stream": "final_test", "seeds": seeds, "win_ante": 1,
            "learning_disabled": True, "training_action_schedule_replayed": False,
            "action_policy": "independent-fixed-motor-choices-no-training-action-replay",
            "components": model, "plastic_weight_sha256": f"synthetic-weights-{index}",
            "checkpoint_manifest": {"checkpoint_sha256": sha256_file(checkpoint)},
            "config_sha256": "synthetic-config",
            "episodes": [{"seed": seed, "won": False} for seed in seeds]}
        summary_path.write_text(json.dumps(summary))
        lock["pairs"].append({"trained_summary": str(summary_path), "trained_checkpoint": str(checkpoint),
            "trained_config": str(config), "trained_config_file_sha256": sha256_file(config),
            "trained_config_sha256": "synthetic-config", "trained_checkpoint_sha256": sha256_file(checkpoint),
            "trained_weight_sha256": f"synthetic-weights-{index}"})
    return lock


def selection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("produce_heldout_win").select_win


def test_showcase_selection_never_invents_a_win_and_uses_declared_order(tmp_path, monkeypatch):
    select = selection(monkeypatch)
    lock = inputs(tmp_path)
    assert select(lock) is None
    path = Path(lock["pairs"][1]["trained_summary"])
    summary = json.loads(path.read_text())
    summary["episodes"][7]["won"] = True
    summary["episodes"][9]["won"] = True
    path.write_text(json.dumps(summary))
    pair, episode, source = select(lock)
    assert pair is lock["pairs"][1] and source == path and episode["seed"] == lock["final_seeds"][7]


def test_showcase_selection_rejects_postlock_file_and_seed_order_changes(tmp_path, monkeypatch):
    select = selection(monkeypatch)
    lock = inputs(tmp_path)
    config = Path(lock["pairs"][0]["trained_config"])
    original = config.read_bytes(); config.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file differs"):
        select(lock)
    config.write_bytes(original)
    path = Path(lock["pairs"][0]["trained_summary"])
    summary = json.loads(path.read_text()); summary["episodes"].reverse()
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="seed order"):
        select(lock)
