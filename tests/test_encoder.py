from __future__ import annotations

from dataclasses import replace

import numpy as np

from flylatro.env.types import (
    BalatroObservation,
    CardObservation,
    GamePhase,
    Suit,
)
from flylatro.fly.encoder import EncoderSpec, FixedBalatroEncoder


def observation() -> BalatroObservation:
    return BalatroObservation(
        episode_id="episode-a",
        seed=42,
        decision_id=3,
        ante=2,
        round=4,
        phase=GamePhase.SELECTING_HAND,
        hand=(
            CardObservation(rank=14, suit=Suit.SPADES),
            CardObservation(rank=2, suit=Suit.HEARTS),
        ),
        money=25,
        score=50,
        blind_target=100,
        hands_remaining=2,
        discards_remaining=1,
        shop_slots=2,
    )


def test_encoder_is_bounded_deterministic_and_versioned() -> None:
    spec = EncoderSpec.development_default()
    encoder = FixedBalatroEncoder(spec)

    first = encoder.encode([observation()])
    second = encoder.encode([observation()])

    np.testing.assert_array_equal(first.rates_hz, second.rates_hz)
    assert first.rates_hz.shape == (1, spec.neuron_count)
    assert first.rates_hz.min() >= spec.min_rate_hz
    assert first.rates_hz.max() <= spec.max_rate_hz
    assert first.encoder_hash == second.encoder_hash == spec.sha256
    assert len(spec.sha256) == 64
    assert not first.rates_hz.flags.writeable


def test_state_hash_ignores_recorder_episode_id_but_covers_state() -> None:
    original = observation()
    other_slot = replace(original, episode_id="episode-b")
    changed = replace(original, score=51)

    assert original.state_hash() == other_slot.state_hash()
    assert original.state_hash() != changed.state_hash()
