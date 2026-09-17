from __future__ import annotations

from flylatro.seeds import SeedPlan
from flylatro.training.curriculum import AnteCurriculum, CurriculumConfig


def test_curriculum_promotes_only_from_held_out_curriculum_seeds() -> None:
    plan = SeedPlan()
    curriculum = AnteCurriculum(
        CurriculumConfig(
            evaluation_frequency_updates=2,
            evaluation_episodes=4,
            promotion_win_rate=0.75,
        ),
        plan,
    )
    calls = []

    def evaluator(ante, seeds):
        calls.append((ante, seeds))
        return 0.75

    decision = curriculum.maybe_evaluate(0, evaluator)

    assert decision.promoted
    assert decision.previous_ante == 1
    assert decision.current_ante == 2
    assert calls[0][1] == plan.seeds("curriculum", 4)
    assert set(calls[0][1]).isdisjoint(plan.seeds("final_test", 4))


def test_curriculum_state_round_trips_and_skips_non_eval_updates() -> None:
    config = CurriculumConfig(evaluation_frequency_updates=5, evaluation_episodes=2)
    curriculum = AnteCurriculum(config)
    assert not curriculum.maybe_evaluate(1, lambda *_: 1.0).evaluated
    curriculum.maybe_evaluate(5, lambda *_: 1.0)
    state = curriculum.state_dict()
    restored = AnteCurriculum(config)

    restored.load_state_dict(state)

    assert restored.state_dict() == state

