from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flylatro.training.checkpoints import load_checkpoint, save_checkpoint
from test_ppo import make_trainer


def test_checkpoint_resume_reproduces_the_next_update(tmp_path: Path) -> None:
    original = make_trainer()
    original.train_update()
    checkpoint = tmp_path / "checkpoint.pt"
    manifest_path = save_checkpoint(
        checkpoint,
        original,
        experiment_config={"name": "test"},
        component_metadata={
            "connectome_hash": "tiny",
            "encoder_hash": "tiny",
            "feature_hash": "tiny",
            "balatro_simulator": original.env.simulator_version,
            "fly_backend": original.processor.version,
        },
        seed_metadata={"training": [100, 101], "seed_plan_hash": "test"},
        curriculum_state={"ante": 1},
        reward_config={"shaping_beta": 1.0},
    )
    original.train_update()
    expected = {
        key: value.detach().clone() for key, value in original.policy.state_dict().items()
    }

    resumed = make_trainer()
    payload = load_checkpoint(checkpoint, resumed)
    assert resumed.update_index == 1
    assert payload["curriculum_state"] == {"ante": 1}
    resumed.train_update()

    for key, expected_value in expected.items():
        torch.testing.assert_close(resumed.policy.state_dict()[key], expected_value)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["checkpoint_sha256"]
    assert manifest["trainable_parameter_count"] > 0
    assert manifest["git"]["commit"]


def test_checkpoint_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    trainer = make_trainer()
    trainer.reset()
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        trainer,
        experiment_config={},
        component_metadata={},
        seed_metadata={},
    )
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(checkpoint, make_trainer())

