from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from flylatro.config import AppConfig


def test_smoke_config_loads_and_has_distinct_seeds() -> None:
    config = AppConfig.load(Path("configs/smoke.toml"))

    assert len(config.seeds()) == config.environment.num_envs
    assert len(set(config.seeds())) == config.environment.num_envs
    config.check_development_limits()


def test_large_mock_run_requires_explicit_opt_in() -> None:
    config = AppConfig.load(Path("configs/smoke.toml"))
    config = replace(
        config,
        environment=replace(config.environment, num_envs=65),
        run=replace(config.run, max_decisions=65),
    )

    with pytest.raises(ValueError, match="safety limit"):
        config.check_development_limits()
    config.check_development_limits(allow_large=True)

