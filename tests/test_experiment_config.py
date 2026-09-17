from __future__ import annotations

from pathlib import Path

import pytest

from flylatro.training.config import ExperimentConfig


ROOT = Path(__file__).parents[1]


def test_safe_v1_profiles_load_and_heavy_profile_is_guarded() -> None:
    dev = ExperimentConfig.load(ROOT / "configs/dev-v1.toml")
    smoke = ExperimentConfig.load(ROOT / "configs/smoke-v1.toml")

    dev.require_heavy_opt_in(False)
    smoke.require_heavy_opt_in(False)
    assert dev.sha256 == dev.sha256
    assert len(dev.sha256) == 64

    heavy = ExperimentConfig.load(ROOT / "configs/curriculum.toml")
    with pytest.raises(ValueError, match="--heavy"):
        heavy.require_heavy_opt_in(False)
    heavy.require_heavy_opt_in(True)


def test_all_named_v1_profiles_are_valid() -> None:
    for name in (
        "benchmark.toml", "ante1.toml", "curriculum.toml",
        "full-training.toml", "final-eval.toml",
        "control-shuffled.toml", "control-conventional.toml",
        "benchmark-gpu.toml", "showcase.toml",
        "heldout-eval.toml",
    ):
        ExperimentConfig.load(ROOT / "configs" / name)
