"""Milestones use observable transitions, including unshaped/auto-reset runs."""
from types import SimpleNamespace
import numpy as np
import pytest

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.env.balatro_sim import ArrayStep
from flylatro.env.upstream_contract import GLOBAL_PHASE_OFF, GLOBAL_ROUND
from flylatro.evaluation.plastic import evaluate_plastic_fly
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.learning.reinforcement import ReinforcementMapper


@pytest.mark.parametrize('won,expected_clears', [(False, 1), (True, 2)])
def test_frozen_milestones_separate_skips_clears_and_terminal_reset(won, expected_clears):
    class ScriptedEnv(MockArrayBalatroEnv):
        def reset(self, seeds):
            self.index = 0
            self.obs, self.masks = super().reset(seeds)
            self.phase(0, 0)
            return self.obs, self.masks

        def phase(self, phase, round_number):
            self.obs['global'][0, GLOBAL_PHASE_OFF:GLOBAL_PHASE_OFF + 5] = 0
            self.obs['global'][0, GLOBAL_PHASE_OFF + phase] = 1
            self.obs['global'][0, GLOBAL_ROUND] = round_number / 24

        def step(self, actions):
            phase, round_number = [(0,0), (0,0), (1,1), (2,1), (0,1), (1,2), (0,0)][self.index]
            self.phase(phase, round_number)  # Deliberate in-place mutation.
            self.index += 1
            done = self.index == 7
            info = {'episode': {'won': won, 'ante': 1, 'r': 0}} if done else {}
            return ArrayStep(self.obs, self.masks, np.zeros(1, np.float32), np.asarray([done]), (info,))

    class ScriptedAgent:
        def __init__(self):
            self.index = 0
            self.plasticity = SimpleNamespace(state=PlasticEdgeState.initialize(2))
            self.reinforcement = ReinforcementMapper()

        def act(self, *args, **kwargs):
            kind = [3,3,2,0,2,2,0][self.index]
            self.index += 1
            actions = {key: np.asarray([-1], dtype=np.int64) for key in
                ('joker_target','consumable_target','shop_target','pack_target')}
            actions.update(action_type=np.asarray([kind]), n_cards=np.asarray([0]),
                cards=np.full((1,5), -1, dtype=np.int64))
            return SimpleNamespace(actions=actions)

    result = evaluate_plastic_fly(ScriptedEnv(1), ScriptedAgent(), [7])
    episode = result.episodes[0]
    assert episode.skipped_blinds == 2
    assert episode.blinds_beaten == expected_clears
    assert episode.highest_played_round == 2
    assert episode.won == won
