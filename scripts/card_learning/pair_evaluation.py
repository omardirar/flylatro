"""Frozen pair selection benchmark on validation snapshots; no learning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flylatro.analysis.card_selection import permute_hand, plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.seeds import derive_seed

from .common import write_json
from .experiment import build_condition_stack, load_condition
from .motor import CardSubsetMotor
from .pairs import pair_slots


def evaluate(condition_path: Path, checkpoint: Path, output: Path, *, learner=0, repeats=4, heavy=False):
    if repeats < 1:
        raise ValueError('pair evaluation repeats must be positive')
    if output.exists():
        raise FileExistsError(output)
    condition, corpus = load_condition(condition_path)
    if corpus is None:
        raise ValueError('pair benchmark needs a condition with a declared pair corpus')
    config = PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(heavy)
    if not 0 <= learner < config.environment.num_envs:
        raise ValueError('learner index outside checkpoint')
    stack = build_condition_stack(config, condition, corpus)
    load_plastic_checkpoint(checkpoint, stack.trainer, expected_components=stack.components)
    before = plastic_hashes(stack.agent.plasticity.state)
    state = stack.agent.plasticity.state
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    env.reset([0])
    motor = CardSubsetMotor(stack.agent.motor.base, stack.agent.motor.exploration, pair_task=True)
    rows = []
    for index in corpus.indices('validation'):
        corpus.restore(env, 0, index)
        original_obs, original_masks = env.observe()
        original_masks['action_type_mask'][:] = False
        original_masks['action_type_mask'][:, 0] = True
        for repeat in range(repeats):
            rng = np.random.default_rng(derive_seed('pair-frozen-permutation-v1', index, repeat))
            order = np.r_[rng.permutation(8), 8, 9]
            obs, masks = permute_hand(original_obs, original_masks, order)
            target = set(pair_slots(obs))
            neural_seed = derive_seed('pair-frozen-neural-v1', corpus.cases[index]['seed'], repeat)
            for label, efficacy in [('initial', state.initial_efficacy), ('trained', state.efficacy)]:
                neural = stack.agent.processor.process(obs, fly_seeds=[neural_seed], efficacy=efficacy[learner:learner+1])
                selected = motor.decode(np.asarray(neural.output_activity), masks, deterministic=True)
                cards = selected['cards'][0, :2]
                action = empty_action_batch(1); action['action_type'][:] = 0; action['n_cards'][:] = 2
                action['cards'][0, :2] = order[cards]
                corpus.restore(env, 0, index)
                scored = env.step(action)
                rows.append({'weights': label, 'case_index': index, 'seed': corpus.cases[index]['seed'],
                    'repeat': repeat, 'permutation': order.tolist(), 'selected_slots': cards.tolist(),
                    'pair_selected': set(cards.tolist()) == target, 'native_reward': float(scored.rewards[0])})
    if plastic_hashes(state) != before:
        raise ValueError('frozen pair benchmark changed plastic state')
    report = {'version': 'frozen-pair-selection-v1', 'learning_disabled': True,
        'scope': 'validation snapshots only; same permutations and neural seeds for own initial/trained weights',
        'chance_pair_selection': 1 / 28, 'independent_validation_hands': len(corpus.indices('validation')),
        'repeats_per_hand': repeats, 'learner': learner, 'corpus_sha256': corpus.sha256,
        'plastic_state_unchanged': True, 'final_test_seeds_used': False,
        'summary': {label: {'pair_selection_fraction': float(np.mean([r['pair_selected'] for r in rows if r['weights'] == label])),
            'mean_native_reward': float(np.mean([r['native_reward'] for r in rows if r['weights'] == label]))}
            for label in ('initial', 'trained')}, 'trials': rows}
    write_json(output, report)
    print(json.dumps({'output': str(output), 'summary': report['summary']}))
    return report
