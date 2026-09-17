from __future__ import annotations

import pytest

from flylatro.seeds import SeedPlan, SeedRange, derive_seed


def test_default_seed_streams_are_disjoint_and_hashable() -> None:
    plan = SeedPlan()
    streams = [
        set(plan.seeds(name, 100))
        for name in (
            "training",
            "validation",
            "curriculum",
            "final_test",
            "showcase",
        )
    ]

    for index, left in enumerate(streams):
        assert all(left.isdisjoint(right) for right in streams[index + 1 :])
    assert len(plan.sha256) == 64


def test_overlapping_seed_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SeedPlan(
            training=SeedRange(0, 10),
            validation=SeedRange(9, 10),
        )


def test_derived_seed_is_stable_and_context_sensitive() -> None:
    assert derive_seed("fly", 1, 2, 3) == derive_seed("fly", 1, 2, 3)
    assert derive_seed("fly", 1, 2, 3) != derive_seed("fly", 1, 2, 4)

