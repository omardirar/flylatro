"""Bounded fixed-hand memorization probe with a matched frozen learner copy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import array, efficacy_like, isolated_rule, _process
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from card_learning.pairs import pair_slots
from flylatro.analysis.card_selection import permute_hand, plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.plasticity import ThreeFactorPlasticity
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed


VERSION = 'fixed-pair-memorization-probe-v1'


def frozen_scores(agent, obs, masks, efficacy, case_index, repeats):
    correct = np.zeros(2, dtype=int)
    choices = []
    for repeat in range(repeats):
        seed = derive_seed(VERSION, case_index, 'evaluation', repeat)
        neural = agent.processor.process(obs, fly_seeds=[seed, seed], efficacy=efficacy)
        actions = agent.motor.decode(neural.output_activity, masks, deterministic=True)
        row = []
        for learner in range(2):
            cards = actions['cards'][learner, :2]
            hit = set(cards.tolist()) == set(pair_slots(obs, learner))
            correct[learner] += hit
            row.append(cards.tolist())
        choices.append(row)
    return {'correct': correct.tolist(), 'trials_per_learner': repeats,
            'pair_fraction': (correct / repeats).tolist(), 'choices': choices}


def probe(agent, env, corpus, *, learner=0, case_index=None, steps=5000, evaluation_repeats=128, progress=None):
    if steps < 1 or evaluation_repeats < 1 or not 0 <= learner < agent.plasticity.state.learners:
        raise ValueError('positive probe budget and valid learner required')
    if env.num_envs != 2:
        raise ValueError('fixed-hand probe needs one plastic and one frozen lane')
    if case_index is None:
        case_index = corpus.indices('training')[0]
    if case_index not in corpus.indices('training'):
        raise ValueError('fixed-hand probe is training-corpus only')
    before = plastic_hashes(agent.plasticity.state)
    env_before = env.snapshot_all()
    source = agent.plasticity.state
    initial = source.initial_efficacy[learner:learner+1]
    single = isolated_rule(agent, initial)
    values = {k: np.repeat(v, 2, axis=0) for k, v in single.state.state_dict().items()}
    factory = type(single.state)
    state = (factory.from_state_dict(values, device=str(single.state.device))
             if getattr(single.state, 'is_torch', False) else factory.from_state_dict(values))
    local = ThreeFactorPlasticity(agent.plasticity.topology, state, agent.plasticity.config)
    credit = SelectedMotorCredit(agent.motor, local)
    rng = np.random.default_rng(derive_seed(VERSION, case_index, 'display'))
    order = np.r_[rng.permutation(8), 8, 9]
    correct, pulse_sum = np.zeros(2, dtype=int), np.zeros(2)
    blocks = []
    block_correct, block_pulses = np.zeros(2, dtype=int), np.zeros(2)
    try:
        for row in range(2):
            corpus.restore(env, row, case_index)
        raw_obs, raw_masks = env.observe()
        raw_masks['action_type_mask'][:] = False; raw_masks['action_type_mask'][:, 0] = True
        obs, masks = permute_hand(raw_obs, raw_masks, order)
        initial_scores = frozen_scores(agent, obs, masks, local.state.efficacy, case_index, evaluation_repeats)
        if initial_scores['choices'] and any(pair[0] != pair[1] for pair in initial_scores['choices']):
            raise ValueError('identical initial learner lanes produced different choices')
        for step in range(steps):
            seed = derive_seed(VERSION, case_index, 'training', step)
            neural = agent.processor.process(obs, fly_seeds=[seed, seed], efficacy=local.state.efficacy)
            actions = agent.motor.decode(neural.output_activity, masks, deterministic=False,
                learner_ids=[learner, learner], decision_ids=[step, step])
            native = {k: v.copy() for k, v in actions.items()}
            for row in range(2):
                corpus.restore(env, row, case_index)
                native['cards'][row, :2] = order[actions['cards'][row, :2]]
            result = env.step(native)
            if np.any(result.dones):
                raise ValueError('fixed-hand native scoring unexpectedly ended a run')
            pulses = [agent.reinforcement.map(info['reward_components'], info) for info in result.infos]
            if any(p.aversive > 0 or p.appetitive <= 0 for p in pulses):
                raise ValueError('fixed-hand probe expects positive native progress only')
            gate = credit.mask(actions, masks)
            local.record_activity(neural.edge_pre_activity, neural.edge_post_activity, edge_credit=gate)
            # The second lane is a no-update control; its observed reward is
            # logged but never applied to weights. Both lanes reset traces.
            local.apply_dopamine(np.asarray([pulses[0].appetitive, 0], dtype=np.float32), 0)
            local.state.reset_fast_traces(eligibility=True)
            for row in range(2):
                hit = set(actions['cards'][row, :2].tolist()) == set(pair_slots(obs, row))
                correct[row] += hit; block_correct[row] += hit
                pulse_sum[row] += pulses[row].appetitive; block_pulses[row] += pulses[row].appetitive
            if (step+1) % 250 == 0 or step+1 == steps:
                n = 250 if (step+1) % 250 == 0 else (step+1) % 250
                block = {'through_step': step+1, 'trials': n,
                    'pair_fraction': (block_correct/n).tolist(), 'pulse_sum': block_pulses.tolist()}
                blocks.append(block)
                if progress:
                    progress(block)
                block_correct.fill(0); block_pulses.fill(0)
        final_scores = frozen_scores(agent, obs, masks, local.state.efficacy, case_index, evaluation_repeats)
        if final_scores['choices'] and any(before_pair[1] != after_pair[1] for before_pair, after_pair in zip(initial_scores['choices'], final_scores['choices'], strict=True)):
            raise ValueError('frozen lane choices changed')
        if not np.array_equal(array(local.state.efficacy)[1], array(initial)[0]):
            raise ValueError('frozen lane weights changed')
        delta = array(local.state.efficacy) - np.repeat(array(initial), 2, axis=0)
        report = {'version': VERSION, 'case_index': case_index, 'case_seed': corpus.cases[case_index]['seed'],
            'pair_rank': corpus.cases[case_index]['pair_rank'], 'display_order': order.tolist(),
            'steps_per_lane': steps, 'lane_order': ['plastic', 'frozen'],
            'matched': 'same hand, display, initial weights, neural seeds and exploration IDs; source 5% sampler unchanged',
            'learning_signal': 'native positive progress; no pair labels or run-failure pulses',
            'training_pair_fraction': (correct/steps).tolist(), 'pulse_sum': pulse_sum.tolist(),
            'initial_evaluation': initial_scores, 'final_evaluation': final_scores,
            'per_lane_mean_absolute_drift': np.abs(delta).mean(axis=1).tolist(),
            'per_lane_max_absolute_drift': np.abs(delta).max(axis=1).tolist(),
            'blocks': blocks, 'source_state_unchanged': True,
            'interpretation': 'one fixed training hand; a learning capacity probe, not held-out generalization or a Balatro improvement claim'}
    finally:
        env.restore_all(env_before)
        if plastic_hashes(source) != before:
            raise ValueError('fixed-hand probe changed source checkpoint state')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--learner-index', type=int, default=0)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--evaluation-repeats', type=int, default=128)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    condition, corpus = load_condition(args.condition)
    if corpus is None:
        raise ValueError('use a declared pair condition')
    config = PlasticExperimentConfig.load(Path(condition['config'])); config.require_heavy_opt_in(args.heavy)
    stack = build_condition_stack(config, condition, corpus)
    load_plastic_checkpoint(args.checkpoint, stack.trainer, expected_components=stack.components)
    env = BalatroSimAdapter(2, win_ante=1, allow_skip_blinds=False); env.reset([0, 0])
    hashes = {p.name: sha256_file(p) for p in (args.condition, args.checkpoint, Path(__file__), Path(__file__).with_name('audit_card_credit_direction.py'))}
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    def progress(value):
        print(json.dumps(value), flush=True)
        write_json(args.output_dir/'status.json', {'status': 'running', 'source_hashes': hashes, **value})
    progress({'through_step': 0})
    try:
        report = probe(stack.agent, env, corpus, learner=args.learner_index, case_index=args.case_index,
            steps=args.steps, evaluation_repeats=args.evaluation_repeats, progress=progress)
        if sha256_file(args.checkpoint) != hashes[args.checkpoint.name]:
            raise ValueError('source checkpoint file changed')
        report.update(source_hashes=hashes, status='completed', wall_seconds=time.perf_counter()-started)
        write_json(args.output_dir/'report.json', report)
        write_json(args.output_dir/'status.json', {'status': 'completed', 'source_hashes': hashes, 'wall_seconds': report['wall_seconds']})
    except BaseException as error:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'source_hashes': hashes, 'error': f'{type(error).__name__}: {error}'})
        raise
    print(json.dumps({'report': str(args.output_dir/'report.json'), 'status': 'completed'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
