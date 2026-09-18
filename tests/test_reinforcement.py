from __future__ import annotations

from flylatro.learning.reinforcement import (
    DopaminePulse,
    ReinforcementMapper,
    shuffled_pulse_schedule,
)
from flylatro.learning.reward_schedule import DopamineSchedule
from flylatro.learning.curriculum import (
    PlasticAnteCurriculum,
    PlasticCurriculumConfig,
)


def test_reinforcement_maps_outcomes_not_strategy() -> None:
    mapper = ReinforcementMapper()
    pulse = mapper.map(
        {"blind_progress": 0.5, "blind_clear": 1.0, "pair": 999.0},
        {"ante_cleared": True},
    )
    assert pulse.appetitive > 0
    assert pulse.aversive == 0
    assert "pair" not in pulse.events
    assert set(pulse.events) == {"progress", "blind_clear", "ante_clear"}


def test_failure_and_ante8_have_distinct_channels() -> None:
    mapper = ReinforcementMapper()
    failed = mapper.map({}, {"episode": {"won": False, "ante": 3}})
    success = mapper.map({"win": 1.0}, {"episode": {"won": True, "ante": 8}})
    assert failed.aversive > 0 and failed.appetitive == 0
    assert success.appetitive > 0 and success.aversive == 0


def test_shuffled_reward_preserves_multiset_and_is_deterministic() -> None:
    pulses = tuple(DopaminePulse(float(index), 0.0) for index in range(5))
    first = shuffled_pulse_schedule(pulses, seed=7)
    second = shuffled_pulse_schedule(pulses, seed=7)
    assert first == second
    assert first != pulses
    assert sorted(pulse.appetitive for pulse in first) == list(range(5))


def test_persisted_shuffled_schedule_is_hash_verified(tmp_path) -> None:
    pulses = tuple(DopaminePulse(float(index), 0.0) for index in range(5))
    schedule = DopamineSchedule.shuffled(
        pulses, source_weight_hash="a" * 64, seed=19
    )
    path = schedule.save(tmp_path / "schedule.json")
    loaded = DopamineSchedule.load(path)
    assert loaded == schedule
    assert loaded.sha256 == schedule.sha256
    assert sorted(pulse.appetitive for pulse in loaded.pulses) == list(range(5))


def test_plastic_curriculum_uses_heldout_exposure_cadence() -> None:
    curriculum = PlasticAnteCurriculum(
        PlasticCurriculumConfig(
            ladder=(1, 2, 8),
            promotion_win_rate=0.5,
            evaluation_every_decisions=10,
            evaluation_episodes=4,
        )
    )
    assert not curriculum.maybe_evaluate(9, lambda *_: 1.0).evaluated
    seen = []

    def evaluate(ante, seeds):
        seen.append((ante, seeds))
        return 0.75

    result = curriculum.maybe_evaluate(10, evaluate)
    assert result.promoted and result.current_ante == 2
    assert len(seen[0][1]) == 4
    restored = PlasticAnteCurriculum(curriculum.config)
    restored.load_state_dict(curriculum.state_dict())
    assert restored.current_ante == 2
