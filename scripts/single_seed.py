"""Repeat native games without changing terminal outcomes or learner identities."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from card_learning.experiment import build_condition_stack
from flylatro.env.balatro_sim import ArrayStep, BalatroSimAdapter
from flylatro.evaluation.plastic import evaluate_plastic_fly
from flylatro.evaluation.plastic_cli import _select_frozen_learner
from flylatro.evaluation.state_hash import hash_observation_row
from flylatro.fly.flywire_artifact import sha256_file
from normalisation_stack import install_normalisation


VERSION = 'repeated-native-game-v1'


class RepeatedSeedEnvironment:
    """Only terminal rows restart; other learners keep their in-flight games.

    Terminal rewards/infos/dones come from the original adapter before restore.
    Training IDs remain distinct; only the native game seed is shared.
    """
    def __init__(self, base, seed):
        self.base, self.seed = base, int(seed)
        self.num_envs = base.num_envs
        self.openings = ()
        self.opening_hashes = ()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError('one independent learner ID per environment required')
        obs, masks = self.base.reset([self.seed] * self.num_envs)
        self.openings = self.base.snapshot_all()
        self.opening_hashes = tuple(hash_observation_row(obs, i) for i in range(self.num_envs))
        return obs, masks

    def step(self, actions):
        step = self.base.step(actions)
        done_rows = np.flatnonzero(step.dones)
        if not len(done_rows):
            return step
        infos = [dict(info) for info in step.infos]
        for row in done_rows:
            self.base.restore(int(row), self.openings[int(row)])
            infos[int(row)]['repeated_game_seed'] = self.seed
        obs, masks = self.base.observe()
        for row in done_rows:
            if hash_observation_row(obs, int(row)) != self.opening_hashes[int(row)]:
                raise RuntimeError('terminal reset did not restore the declared opening')
        return ArrayStep(obs, masks, step.rewards, step.dones, tuple(infos))


def sources():
    root = Path(__file__).parent
    names = ('single_seed.py', 'run_single_seed.py', 'normalisation_stack.py', 'local_normalisation.py')
    return {name: sha256_file(root / name) for name in names}


def build_stack(config, condition, **kwargs):
    stack = build_condition_stack(config, condition, **kwargs)
    if condition['single_seed_arm'] == 'normalised':
        install_normalisation(stack)
    elif condition['single_seed_arm'] != 'original':
        raise ValueError('unknown single-seed treatment')
    stack.env = RepeatedSeedEnvironment(stack.env, condition['single_game_seed'])
    stack.trainer.env = stack.env
    stack.trainer.observations, stack.trainer.masks = stack.env.reset(stack.trainer.training_seeds)
    stack.components['single_seed_experiment'] = {
        'version': VERSION, 'game_seed': condition['single_game_seed'],
        'arm': condition['single_seed_arm'], 'research_sources': sources(),
        'reset': 'restore opening snapshot on terminal rows; preserve outcomes and independent learner IDs',
    }
    return stack


def evaluate(stack, seeds, *, max_vector_steps=1000):
    """Frozen complete Ante-1 games on explicit seeds; restore exact training state."""
    saved = stack.trainer.state_dict()
    # Snapshots own their tensors/arrays so restoration also verifies no hidden
    # RNG/trace/environment changes leak back into training.
    rows, transitions = [], []
    weight_hash = stack.agent.plasticity.state.weight_sha256
    original_state = stack.agent.plasticity.state
    try:
        env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
        env.set_shaping_beta(stack.env._beta)
        for learner in range(stack.env.num_envs):
            stack.agent.plasticity.state = _select_frozen_learner(original_state, learner)
            for seed in seeds:
                # Separate native evaluation adapter: explicit unseen seeds,
                # with one labelled frozen learner, never the training wrapper.
                result = evaluate_plastic_fly(env, stack.agent,
                    [int(seed)], max_vector_steps=max_vector_steps)
                for episode in result.episodes:
                    rows.append({'learner': learner, **asdict(episode)})
                transitions.append({'learner': learner, 'seed': int(seed),
                    'actions': [asdict(t) for t in result.transitions]})
    finally:
        stack.agent.plasticity.state = original_state
        stack.trainer.load_state_dict(saved)
    if stack.agent.plasticity.state.weight_sha256 != weight_hash:
        raise RuntimeError('frozen evaluation changed training weights')
    return {'learning_disabled': True, 'seeds': list(seeds), 'episodes': rows,
        'transitions': transitions, 'weight_sha256': weight_hash,
        'semantics': 'one greedy trajectory per seed and learner; shared seeds are not independent learners',
        'win_rate': float(np.mean([r['won'] for r in rows])),
        'mean_blinds_beaten': float(np.mean([r['blinds_beaten'] for r in rows])),
        'first_blind_clear_fraction': float(np.mean([r['blinds_beaten'] > 0 for r in rows]))}
