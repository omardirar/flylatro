"""Native two-decision discard-credit capacity probe with a frozen PLAY policy.

Only DISCARD card/count efficacy can change. The subsequent PLAY is selected
using the source initial weights, so improved outcomes cannot be attributed to
learning a different PLAY policy. Pair labels remain diagnostic only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import array, isolated_rule
from evaluate_motor_sampling import repeat_efficacy
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from card_learning.pairs import PairCorpus, build_corpus, pair_slots
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.plasticity import ThreeFactorPlasticity
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed


VERSION = 'native-discard-then-frozen-play-v1'


def local_rule(agent, replicates):
    single = isolated_rule(agent, agent.plasticity.state.initial_efficacy[:1])
    values = {key: np.repeat(value, 2*replicates, axis=0) for key, value in single.state.state_dict().items()}
    factory = type(single.state)
    state = (factory.from_state_dict(values, device=str(single.state.device))
             if getattr(single.state, 'is_torch', False) else factory.from_state_dict(values))
    return ThreeFactorPlasticity(single.topology, state, single.config)


def permute_rows(obs, masks, orders):
    observations = {key: value.copy() for key, value in obs.items()}
    legal = {key: value.copy() for key, value in masks.items()}
    for row, order in enumerate(orders):
        n = int(obs['hand_len'][row])
        if not np.array_equal(np.sort(order[:n]), np.arange(n)) or not np.array_equal(order[n:], np.arange(n, 10)):
            raise ValueError('display permutation must preserve padding')
        observations['hand'][row] = obs['hand'][row, order]
        legal['card_select_mask'][row] = masks['card_select_mask'][row, order]
    return observations, legal


def forced(masks, action_type):
    if not masks['action_type_mask'][:, action_type].all():
        raise ValueError('curriculum action is not natively legal')
    result = {key: value.copy() for key, value in masks.items()}
    result['action_type_mask'][:] = False; result['action_type_mask'][:, action_type] = True
    return result


def translate(actions, orders):
    result = {key: value.copy() for key, value in actions.items()}
    for row, order in enumerate(orders):
        count = int(actions['n_cards'][row])
        result['cards'][row, :count] = order[actions['cards'][row, :count]]
    return result


def zero_gate(activity):
    if hasattr(activity, 'detach'):
        import torch
        return torch.zeros_like(activity, dtype=torch.bool)
    return np.zeros(np.shape(activity), dtype=bool)


def followup(agent, env, initial_efficacy, orders, seeds):
    obs, masks = permute_rows(*env.observe(), orders)
    masks = forced(masks, 0)
    neural = agent.processor.process(obs, fly_seeds=seeds, efficacy=initial_efficacy)
    if not np.isfinite(neural.output_activity).all():
        raise ValueError('nonfinite fixed PLAY activity')
    action = agent.motor.decode(neural.output_activity, masks, deterministic=True)
    result = env.step(translate(action, orders))
    if np.any(result.dones):
        raise ValueError('one opening PLAY unexpectedly ended a native run')
    pulses = [agent.reinforcement.map(info['reward_components'], info) for info in result.infos]
    if any(p.aversive != 0 for p in pulses):
        raise ValueError('artificial task boundary must not produce failure punishment')
    return neural, action, result, pulses


def trial(agent, env, corpus, indices, *, efficacy, initial_efficacy, replicate_ids,
          key, sampled, learning=None, credit=None, plastic_rows=None,
          cut_trace=False, random_control=False):
    batch = len(indices)
    for row, index in enumerate(indices):
        corpus.restore(env, row, index)
    obs, native_masks = env.observe()
    orders, next_orders, discard_seeds, play_seeds, decision_ids = [], [], [], [], []
    for row, (index, replicate) in enumerate(zip(indices, replicate_ids, strict=True)):
        rng = np.random.default_rng(derive_seed(VERSION, key, replicate, index, 'display'))
        orders.append(np.r_[rng.permutation(8), 8, 9]); next_orders.append(np.r_[rng.permutation(8), 8, 9])
        discard_seeds.append(derive_seed(VERSION, key, replicate, index, 'discard-neural'))
        play_seeds.append(derive_seed(VERSION, key, replicate, index, 'fixed-play-neural'))
        decision_ids.append(derive_seed(VERSION, key, index, 'discard-exploration'))
    visible, legal = permute_rows(obs, native_masks, orders)
    masks = forced(legal, 1)
    neural = agent.processor.process(visible, fly_seeds=discard_seeds, efficacy=efficacy)
    if not np.isfinite(neural.output_activity).all():
        raise ValueError('nonfinite DISCARD activity')
    free = agent.motor.decode(neural.output_activity, legal, deterministic=True)
    action = agent.motor.decode(neural.output_activity, masks, deterministic=not sampled,
        learner_ids=replicate_ids, decision_ids=decision_ids)
    # Pair membership is measured only after the native action is chosen.
    pair_kept = [not (set(action['cards'][row, :int(action['n_cards'][row])]) & set(pair_slots(visible, row))) for row in range(batch)]
    if learning is not None:
        if credit is None or plastic_rows is None:
            raise ValueError('learning needs explicit credit and plastic-lane mask')
        gate = credit.mask(action, masks)
        learning.record_activity(neural.edge_pre_activity, neural.edge_post_activity, edge_credit=gate)
    result = env.step(translate(action, orders))
    discard_pulses = [agent.reinforcement.map(info['reward_components'], info) for info in result.infos]
    if np.any(result.dones) or any(p.appetitive != 0 or p.aversive != 0 for p in discard_pulses):
        raise ValueError('DISCARD must be a legal nonterminal zero-immediate-reward action')
    if learning is not None and cut_trace:
        learning.state.reset_fast_traces(eligibility=True)
    # Trace is retained across the two native decisions. PLAY generates no
    # fresh eligibility: it is a fixed follow-up policy, not another learner.
    play_neural, play_action, result, pulses = followup(agent, env, initial_efficacy, next_orders, play_seeds)
    update_l1 = np.zeros(batch)
    if learning is not None:
        learning.record_activity(play_neural.edge_pre_activity, play_neural.edge_post_activity,
                                 edge_credit=zero_gate(play_neural.edge_pre_activity))
        positive = np.asarray([p.appetitive for p in pulses], dtype=np.float32)*np.asarray(plastic_rows, dtype=np.float32)
        events = learning.apply_dopamine(positive, 0)
        if any(e.lower_bound_hits or e.upper_bound_hits for e in events):
            raise ValueError('discard capacity probe reached an efficacy bound')
        update_l1 = np.asarray([e.absolute_change for e in events])
        learning.state.reset_fast_traces(eligibility=True)
    progress = np.asarray([info['reward_components'].get('blind_progress', 0.) for info in result.infos])
    clears = np.asarray([info['reward_components'].get('blind_clear', 0.) > 0 for info in result.infos])
    output = {'case_indices': list(indices), 'discard_counts': action['n_cards'].tolist(),
        'discard_cards': action['cards'].tolist(), 'play_cards': play_action['cards'].tolist(),
        'progress': progress.tolist(), 'clears': clears.tolist(), 'pair_kept': pair_kept,
        'free_action_types': free['action_type'].tolist(), 'pulse': [p.appetitive for p in pulses],
        'update_l1': update_l1.tolist(), 'fixed_play_uses_initial_weights': True}
    if random_control:
        random_action = empty_action_batch(batch); random_action['action_type'][:] = 1
        random_action['n_cards'][:] = action['n_cards']
        for row, (index, replicate) in enumerate(zip(indices, replicate_ids, strict=True)):
            corpus.restore(env, row, index)
            rng = np.random.default_rng(derive_seed(VERSION, key, replicate, index, 'random-same-count'))
            count = int(action['n_cards'][row])
            random_action['cards'][row, :count] = np.sort(rng.choice(8, count, replace=False))
        env.step(translate(random_action, orders))
        _, _, random_result, _ = followup(agent, env, initial_efficacy, next_orders, play_seeds)
        output['random_same_count_progress'] = [info['reward_components'].get('blind_progress', 0.) for info in random_result.infos]
        output['random_discard_counts'] = random_action['n_cards'].tolist()
    return output


def evaluate(agent, env, corpus, local, *, replicates, repeats, policy):
    before = plastic_hashes(local.state)
    ids = [i for i in range(replicates) for _ in range(2)]
    rows = []
    for index in corpus.indices('validation'):
        for repeat in range(repeats):
            rows.append(trial(agent, env, corpus, [index]*(2*replicates), efficacy=local.state.efficacy,
                initial_efficacy=local.state.initial_efficacy, replicate_ids=ids,
                key=('evaluation', repeat, policy), sampled=policy == 'training-sampled', random_control=True))
    if plastic_hashes(local.state) != before:
        raise ValueError('curriculum evaluation changed learned state')
    summary = {}
    for key in ('progress', 'clears', 'pair_kept', 'discard_counts', 'random_same_count_progress'):
        summary[key] = np.mean([row[key] for row in rows], axis=0).tolist()
    summary['free_discard_fraction'] = np.mean([np.asarray(row['free_action_types']) == 1 for row in rows], axis=0).tolist()
    return {'policy': policy, 'trials_per_lane': len(rows), 'summary': summary, 'trials': rows}


def probe(agent, corpus, *, trials=5000, replicates=3, evaluation_repeats=4,
          progress=None, trace_cut=False, log_trial=None):
    if trials < 1 or replicates < 1 or evaluation_repeats < 1:
        raise ValueError('positive curriculum budgets required')
    if not agent.motor.mapping.routing.action_conditioned or agent.plasticity.config.dopamine_decay != 0:
        raise ValueError('use action-conditioned cards and decision-local dopamine')
    source_before = plastic_hashes(agent.plasticity.state)
    local = local_rule(agent, replicates)
    credit = SelectedMotorCredit(agent.motor, local)
    batch = 2*replicates; ids = [i for i in range(replicates) for _ in range(2)]
    plastic = np.arange(batch) % 2 == 0
    env = BalatroSimAdapter(batch, win_ante=1, allow_skip_blinds=False); env.reset([0]*batch)
    before_evaluation, after_evaluation = {}, {}
    for policy in ('greedy', 'training-sampled'):
        if progress:
            progress({'phase': 'initial-evaluation', 'policy': policy})
        before_evaluation[policy] = evaluate(agent, env, corpus, local,
            replicates=replicates, repeats=evaluation_repeats, policy=policy)
    training = corpus.indices('training')
    schedules = [np.random.default_rng(derive_seed(VERSION, i, 'training-order')).permutation(training).tolist() for i in range(replicates)]
    total_progress, total_l1, total_pair_kept = np.zeros(batch), np.zeros(batch), np.zeros(batch)
    blocks = []
    for step in range(trials):
        indices = [schedules[i][step % len(training)] for i in range(replicates) for _ in range(2)]
        row = trial(agent, env, corpus, indices, efficacy=local.state.efficacy,
            initial_efficacy=local.state.initial_efficacy, replicate_ids=ids, key=('training', step),
            sampled=True, learning=local, credit=credit, plastic_rows=plastic, cut_trace=trace_cut)
        total_progress += row['progress']; total_l1 += row['update_l1']; total_pair_kept += row['pair_kept']
        if log_trial:
            log_trial({'trial': step, **row})
        if (step+1) % 250 == 0 or step+1 == trials:
            block = {'phase': 'training', 'trials_per_lane': step+1,
                'mean_progress': (total_progress/(step+1)).tolist(), 'actual_update_l1': total_l1.tolist(),
                'pair_kept_fraction': (total_pair_kept/(step+1)).tolist()}
            blocks.append(block)
            if progress:
                progress(block)
    for policy in ('greedy', 'training-sampled'):
        if progress:
            progress({'phase': 'final-evaluation', 'policy': policy})
        after_evaluation[policy] = evaluate(agent, env, corpus, local,
            replicates=replicates, repeats=evaluation_repeats, policy=policy)
        for before, after in zip(before_evaluation[policy]['trials'], after_evaluation[policy]['trials'], strict=True):
            for key in ('progress', 'clears', 'discard_cards', 'play_cards', 'pair_kept'):
                if any(before[key][i] != after[key][i] for i in range(1, batch, 2)):
                    raise ValueError('frozen lane behavior changed')
    delta = array(local.state.efficacy)-array(local.state.initial_efficacy)
    discard_pools = [pool for head in ('discard_card', 'discard_card_count') for pool in agent.motor.mapping.routing.head_routes[head]]
    roots = [int(agent.motor.mapping.output_root_ids[i]) for pool in discard_pools for i in agent.motor.mapping.pool_indices[pool]]
    allowed = np.isin(local.topology.post_root_ids, roots)
    if np.any(delta[:, ~allowed]) or np.any(delta[~plastic]):
        raise ValueError('learning changed a non-DISCARD edge or a frozen lane')
    if plastic_hashes(agent.plasticity.state) != source_before:
        raise ValueError('curriculum changed source checkpoint state')
    report = {'version': VERSION, 'lane_order': [f'{kind}-{i}' for i in range(replicates) for kind in ('plastic', 'frozen')],
        'trials_per_lane': trials, 'replicates': replicates, 'native_decisions_per_lane': 2*trials,
        'before': before_evaluation, 'after': after_evaluation, 'blocks': blocks,
        'actual_update_l1': total_l1.tolist(), 'mean_absolute_drift': np.abs(delta).mean(axis=1).tolist(),
        'max_absolute_drift': np.abs(delta).max(axis=1).tolist(),
        'source_state_unchanged': True, 'frozen_choices_and_weights_exact': True,
        'non_discard_efficacy_unchanged': True, 'trace_cut': trace_cut,
        'learning_rule': asdict(local.config),
        'interpretation': 'forced DISCARD membership/count capacity with a fixed initial PLAY policy; no action-timing or whole-game learning claim'}
    return report, local.state.state_dict()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--evaluation-repeats', type=int, default=4)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    condition, _ = load_condition(args.condition)
    config = PlasticExperimentConfig.load(Path(condition['config'])); config.require_heavy_opt_in(args.heavy)
    args.output_dir.mkdir(parents=True)
    sources = {str(p.resolve()): sha256_file(p) for p in (args.condition, args.checkpoint, Path(__file__),
        Path(__file__).with_name('audit_card_credit_direction.py'), Path(__file__).with_name('evaluate_motor_sampling.py'))}
    write_json(args.output_dir/'design.json', {'version': VERSION, 'source_hashes': sources,
        'trials_per_lane': args.trials, 'replicates': args.replicates, 'evaluation_repeats': args.evaluation_repeats,
        'training_root_offset': 9000, 'validation_root_offset': 160000,
        'learning': 'fresh unit efficacy; forced DISCARD; native reward from fixed initial-policy PLAY; preserve trace across both decisions; only DISCARD posts update',
        'controls': 'paired frozen lanes; common source states/neural/exploration seeds; same-count random discard diagnostics at evaluation only',
        'reward_and_reference_changes': False, 'pair_labels_in_feedback': False, 'final_test_used': False})
    started = time.perf_counter()
    def progress(row):
        print(json.dumps(row), flush=True); write_json(args.output_dir/'status.json', {'status': 'running', **row})
    try:
        build_corpus(args.output_dir/'pairs.json', train_per_rank=8, validation_per_rank=4,
                     training_offset=9000, validation_offset=160000, max_attempts=5000)
        corpus = PairCorpus(args.output_dir/'pairs.json')
        stack = build_condition_stack(config, condition)
        load_plastic_checkpoint(args.checkpoint, stack.trainer, expected_components=stack.components)
        with (args.output_dir/'training-trials.jsonl').open('w') as stream:
            def log_trial(row):
                stream.write(json.dumps(row, sort_keys=True)+'\n')
            report, state = probe(stack.agent, corpus, trials=args.trials, replicates=args.replicates,
                evaluation_repeats=args.evaluation_repeats, progress=progress, log_trial=log_trial)
        if any(sha256_file(Path(p)) != h for p, h in sources.items()):
            raise ValueError('source identity changed during curriculum')
        np.savez_compressed(args.output_dir/'diagnostic-final-state.npz', **state)
        report.update(status='completed', source_hashes=sources, corpus_sha256=corpus.sha256,
                      wall_seconds=time.perf_counter()-started)
        write_json(args.output_dir/'report.json', report)
        write_json(args.output_dir/'status.json', {'status': 'completed'})
    except BaseException as error:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
