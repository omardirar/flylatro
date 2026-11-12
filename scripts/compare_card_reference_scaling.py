"""Reward-free reference calibration, then a matched three-lane capacity probe.

Standalone exploratory code: never changes production sources, legacy condition
identities, checkpoint state, reward mapping, or the motor's exploration rule.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import array, isolated_rule
from probe_fixed_pair_learning import VERSION as SEED_VERSION
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from card_learning.pairs import pair_slots
from flylatro.analysis.card_selection import permute_hand, plastic_hashes
from flylatro.analysis.corpus import CalibrationCorpus
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.plasticity import ThreeFactorPlasticity
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed


VERSION = 'matched-card-reference-scaling-v1'
LANES = ('baseline', 'rescaled', 'frozen')


def uniform_pair_edge_probability(credit, masks):
    """Exact expected credit mask over uniform legal pairs, including overlaps."""
    legal = np.flatnonzero(masks['card_select_mask'][0])
    if len(legal) < 3 or not masks['action_type_mask'][0, 0]:
        raise ValueError('calibration requires legal PLAY with competing pairs')
    probabilities = np.zeros(len(credit.edge_posts), dtype=np.float64)
    pairs = list(combinations(legal, 2))
    for cards in pairs:
        action = empty_action_batch(1)
        action['action_type'][:] = 0; action['n_cards'][:] = 2
        action['cards'][0, :2] = cards
        pools = credit.motor.selected_pool_mask(action, masks)
        posts = np.any(pools[:, :, None] & credit.pool_posts[None, :, :], axis=1)
        probabilities += posts[0, credit.edge_posts]
    return probabilities / len(pairs)


def fit_references(samples, pre_ids, post_ids, config, *, quantile=.95):
    """Match expected first-update L1 per unit pulse, before float32 rounding.

    Percentiles count each connected neuron once per observation, rather than
    weighting it by edge degree. Matching uses the actual expected edge gate.
    Inputs are frozen neural rates and masks only; no outcomes or pair labels.
    """
    pre_ids, post_ids = np.asarray(pre_ids), np.asarray(post_ids)
    if not .5 < quantile < 1 or not samples or pre_ids.shape != post_ids.shape:
        raise ValueError('aligned calibration samples and quantile in (.5, 1) required')
    unique_pre, unique_post = [], []
    cleaned = []
    for sample in samples:
        pre, post, probability = (np.asarray(sample[key], dtype=np.float64).ravel()
                                  for key in ('pre', 'post', 'probability'))
        if any(x.shape != pre_ids.shape for x in (pre, post, probability)):
            raise ValueError('calibration arrays must be edge aligned')
        if any(not np.isfinite(x).all() for x in (pre, post, probability)):
            raise ValueError('finite calibration activity and probabilities required')
        if np.any(pre < 0) or np.any(post < 0) or np.any((probability < 0) | (probability > 1)):
            raise ValueError('nonnegative rates and probability in [0, 1] required')
        eligible = np.flatnonzero(probability > 0)
        if not len(eligible):
            raise ValueError('calibration has no selectable edges')
        for roots, rates, output in ((pre_ids, pre, unique_pre), (post_ids, post, unique_post)):
            _, first = np.unique(roots[eligible], return_index=True)
            values = rates[eligible[first]]
            output.extend(values[values > 0].tolist())
        cleaned.append((pre, post, probability))
    if not unique_pre or not unique_post:
        raise ValueError('calibration requires positive connected KC and MBON activity')
    kc = max(config.kc_reference_hz, float(np.quantile(unique_pre, quantile)))
    mbon = max(config.mbon_reference_hz, float(np.quantile(unique_post, quantile)))
    total = sum(probability.sum() for _, _, probability in cleaned)
    def statistics(kc_ref, mbon_ref):
        coincidence = pre_clip = post_clip = 0.
        for pre, post, probability in cleaned:
            coincidence += np.sum(probability * np.clip(pre/kc_ref, 0, 1) * np.clip(post/mbon_ref, 0, 1))
            pre_clip += np.sum(probability * (pre >= kc_ref))
            post_clip += np.sum(probability * (post >= mbon_ref))
        return {'mean_coincidence': float(coincidence/total),
                'pre_clipped_fraction': float(pre_clip/total),
                'post_clipped_fraction': float(post_clip/total)}
    baseline = statistics(config.kc_reference_hz, config.mbon_reference_hz)
    candidate = statistics(kc, mbon)
    if min(baseline['mean_coincidence'], candidate['mean_coincidence']) <= 0:
        raise ValueError('calibration requires nonzero coincident activity')
    lr = config.learning_rate * baseline['mean_coincidence'] / candidate['mean_coincidence']
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('finite positive matched learning rate required')
    scaled = replace(config, kc_reference_hz=kc, mbon_reference_hz=mbon, learning_rate=lr)
    old_size = config.learning_rate * baseline['mean_coincidence']
    new_size = scaled.learning_rate * candidate['mean_coincidence']
    return scaled, {'quantile': quantile, 'samples': len(samples),
        'positive_unique_neuron_observations': {'KC': len(unique_pre), 'MBON': len(unique_post)},
        'baseline_config': asdict(config), 'rescaled_config': asdict(scaled),
        'baseline': baseline, 'rescaled': candidate,
        'baseline_expected_update_per_selected_edge_per_unit_pulse': old_size,
        'rescaled_expected_update_per_selected_edge_per_unit_pulse': new_size,
        'relative_matching_error': abs(new_size-old_size)/old_size,
        'matching_scope': 'frozen calibration, zero traces, uniform legal pairs, unit pulse; pre-rounding L1, not trajectory matching'}


def calibrate(agent, corpus, *, learner=0, repeats=2, quantile=.95):
    if repeats < 1 or not 0 <= learner < agent.plasticity.state.learners:
        raise ValueError('positive repeats and valid learner required')
    before = plastic_hashes(agent.plasticity.state)
    source = agent.plasticity.state
    efficacy = source.initial_efficacy[learner:learner+1]
    credit = SelectedMotorCredit(agent.motor, isolated_rule(agent, efficacy))
    indices = np.flatnonzero(corpus.masks['action_type_mask'][:, 0] &
                             (corpus.masks['card_select_mask'].sum(axis=1) >= 3)).tolist()
    samples = []
    for index in indices:
        obs, masks = corpus.row(index)
        probability = uniform_pair_edge_probability(credit, masks)
        for repeat in range(repeats):
            seed = derive_seed(VERSION, 'reward-free-calibration', index, repeat)
            neural = agent.processor.process(obs, fly_seeds=[seed], efficacy=efficacy)
            samples.append({'pre': array(neural.edge_pre_activity).ravel(),
                            'post': array(neural.edge_post_activity).ravel(), 'probability': probability})
    scaled, report = fit_references(samples, agent.plasticity.topology.pre_root_ids,
                                   agent.plasticity.topology.post_root_ids, agent.plasticity.config,
                                   quantile=quantile)
    if plastic_hashes(source) != before:
        raise ValueError('calibration changed source plastic state')
    report.update(corpus_sha256=corpus.sha256, state_indices=indices,
                  repeats=repeats, source_state_unchanged=True,
                  selection='all legal PLAY states with at least three cards; no environment steps, rewards or pair labels',
                  reference_policy='95th percentile of positive unique connected-neuron rates, floored at baseline references'
                                   if quantile == .95 else 'declared positive-rate quantile, floored at baseline references')
    return scaled, report


def batched_efficacy(source, rules):
    if getattr(source, 'is_torch', False):
        import torch
        return torch.cat([rule.state.efficacy for rule in rules], dim=0)
    return np.concatenate([rule.state.efficacy for rule in rules], axis=0)


def evaluate(agent, obs, masks, efficacy, case_index, repeats):
    choices, correct = [], np.zeros(3, dtype=int)
    for repeat in range(repeats):
        seed = derive_seed(SEED_VERSION, case_index, 'evaluation', repeat)
        neural = agent.processor.process(obs, fly_seeds=[seed]*3, efficacy=efficacy)
        if not np.isfinite(neural.output_activity).all():
            raise ValueError('nonfinite evaluation activity')
        actions = agent.motor.decode(neural.output_activity, masks, deterministic=True)
        row_choices = actions['cards'][:, :2].tolist()
        choices.append(row_choices)
        for row in range(3):
            correct[row] += set(row_choices[row]) == set(pair_slots(obs, row))
    return {'correct': correct.tolist(), 'trials_per_lane': repeats,
            'pair_fraction': (correct/repeats).tolist(), 'choices': choices}


def compare(agent, env, corpus, scaled_config, *, learner=0, case_index=None,
            steps=5000, evaluation_repeats=128, progress=None):
    if min(steps, evaluation_repeats) < 1 or not 0 <= learner < agent.plasticity.state.learners:
        raise ValueError('positive budgets and valid learner required')
    if env.num_envs != 3:
        raise ValueError('comparison requires baseline, rescaled and frozen lanes')
    case_index = corpus.indices('training')[0] if case_index is None else case_index
    if case_index not in corpus.indices('training'):
        raise ValueError('capacity comparison is training-corpus only')
    baseline_config = agent.plasticity.config
    allowed = {'kc_reference_hz', 'mbon_reference_hz', 'learning_rate'}
    if any(v != asdict(scaled_config)[k] for k, v in asdict(baseline_config).items() if k not in allowed):
        raise ValueError('rescaled condition may change only references and matched learning rate')
    if baseline_config.dopamine_decay != 0:
        raise ValueError('comparison requires decision-local dopamine')
    source = agent.plasticity.state
    before, env_before = plastic_hashes(source), env.snapshot_all()
    initial = source.initial_efficacy[learner:learner+1]
    rules = [isolated_rule(agent, initial) for _ in LANES]
    rules[1] = ThreeFactorPlasticity(agent.plasticity.topology, rules[1].state, scaled_config)
    credits = [SelectedMotorCredit(agent.motor, rule) for rule in rules]
    rng = np.random.default_rng(derive_seed(SEED_VERSION, case_index, 'display'))
    order = np.r_[rng.permutation(8), 8, 9]
    correct, pulse_sum, update_l1 = np.zeros(3, dtype=int), np.zeros(3), np.zeros(3)
    block_correct, blocks, clipping = np.zeros(3, dtype=int), [], []
    try:
        for row in range(3):
            corpus.restore(env, row, case_index)
        raw_obs, raw_masks = env.observe()
        raw_masks['action_type_mask'][:] = False; raw_masks['action_type_mask'][:, 0] = True
        obs, masks = permute_hand(raw_obs, raw_masks, order)
        initial_scores = evaluate(agent, obs, masks, batched_efficacy(source, rules), case_index, evaluation_repeats)
        if any(not (row[0] == row[1] == row[2]) for row in initial_scores['choices']):
            raise ValueError('identical initial lanes produced different choices')
        for step in range(steps):
            seed = derive_seed(SEED_VERSION, case_index, 'training', step)
            neural = agent.processor.process(obs, fly_seeds=[seed]*3, efficacy=batched_efficacy(source, rules))
            if not np.isfinite(neural.output_activity).all():
                raise ValueError('nonfinite training activity')
            actions = agent.motor.decode(neural.output_activity, masks, deterministic=False,
                learner_ids=[learner]*3, decision_ids=[step]*3)
            native = {k: v.copy() for k, v in actions.items()}
            for row in range(3):
                corpus.restore(env, row, case_index)
                native['cards'][row, :2] = order[actions['cards'][row, :2]]
            result = env.step(native)
            if np.any(result.dones):
                raise ValueError('native scoring unexpectedly ended the run')
            pulses = [agent.reinforcement.map(info['reward_components'], info) for info in result.infos]
            if any(p.aversive != 0 or p.appetitive <= 0 for p in pulses):
                raise ValueError('expected unchanged positive native progress feedback')
            for row, (rule, credit, pulse) in enumerate(zip(rules, credits, pulses, strict=True)):
                action = {k: v[row:row+1] for k, v in actions.items()}
                mask = {k: v[row:row+1] for k, v in masks.items()}
                gate = credit.mask(action, mask)
                pre, post = neural.edge_pre_activity[row:row+1], neural.edge_post_activity[row:row+1]
                if step < 32:
                    from audit_card_credit_direction import clipping_summary
                    clipping.append({'step': step, 'lane': LANES[row], **clipping_summary(pre, post, gate, rule.config)})
                rule.record_activity(pre, post, edge_credit=gate)
                event = rule.apply_dopamine(pulse.appetitive, 0, plasticity_enabled=row != 2)[0]
                if event.lower_bound_hits or event.upper_bound_hits:
                    raise ValueError('comparison reached an efficacy bound')
                update_l1[row] += event.absolute_change
                rule.state.reset_fast_traces(eligibility=True)
                hit = set(actions['cards'][row, :2].tolist()) == set(pair_slots(obs, row))
                correct[row] += hit; block_correct[row] += hit
                pulse_sum[row] += pulse.appetitive
            if (step+1) % 250 == 0 or step+1 == steps:
                n = 250 if (step+1) % 250 == 0 else (step+1) % 250
                block = {'through_step': step+1, 'trials': n, 'pair_fraction': (block_correct/n).tolist(),
                         'pulse_sum': pulse_sum.tolist(), 'cumulative_actual_update_l1': update_l1.tolist()}
                blocks.append(block)
                if progress:
                    progress(block)
                block_correct.fill(0)
        final_scores = evaluate(agent, obs, masks, batched_efficacy(source, rules), case_index, evaluation_repeats)
        if any(a[2] != b[2] for a, b in zip(initial_scores['choices'], final_scores['choices'], strict=True)):
            raise ValueError('frozen individual evaluation choices changed')
        delta = np.concatenate([array(rule.state.efficacy)-array(initial) for rule in rules], axis=0)
        if np.any(delta[2]) or update_l1[2] != 0:
            raise ValueError('frozen control weights changed')
        return {'version': VERSION, 'lane_order': list(LANES), 'learner': learner,
            'case_index': case_index, 'case_seed': corpus.cases[case_index]['seed'],
            'pair_rank': corpus.cases[case_index]['pair_rank'], 'display_order': order.tolist(),
            'steps_per_lane': steps, 'initial_evaluation': initial_scores, 'final_evaluation': final_scores,
            'training_pair_plays': correct.tolist(), 'training_pair_fraction': (correct/steps).tolist(),
            'pulse_sum': pulse_sum.tolist(), 'actual_update_l1': update_l1.tolist(),
            'mean_absolute_drift': np.abs(delta).mean(axis=1).tolist(),
            'max_absolute_drift': np.abs(delta).max(axis=1).tolist(),
            'blocks': blocks, 'first_32_steps_clipping': clipping,
            'source_state_unchanged': True, 'frozen_weights_and_individual_choices_unchanged': True,
            'learning_signal': 'unchanged native progress, no pair teacher or artificial failure pulse',
            'matching': 'same initial weights, hand, display, spike seeds and exploration IDs; reference scaling matched on separate reward-free corpus',
            'interpretation': 'one fixed training hand, one weight initialization; trajectory update sizes may differ; no generalization claim'}
    finally:
        env.restore_all(env_before)
        if plastic_hashes(source) != before:
            raise ValueError('comparison changed source checkpoint state')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('condition', 'checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--learner-index', type=int, default=0)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--evaluation-repeats', type=int, default=128)
    parser.add_argument('--calibration-repeats', type=int, default=2)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    condition, corpus = load_condition(args.condition)
    if corpus is None:
        raise ValueError('a declared pair condition is required')
    config = PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy)
    calibration_path = resolve_path(config, config.calibration.corpus_path)
    calibration = CalibrationCorpus.load(calibration_path)
    if set(calibration.state_hashes) & {case['state_hash'] for case in corpus.cases}:
        raise ValueError('calibration and pair-task states overlap')
    stack = build_condition_stack(config, condition, corpus)
    load_plastic_checkpoint(args.checkpoint, stack.trainer, expected_components=stack.components)
    env = BalatroSimAdapter(3, win_ante=1, allow_skip_blinds=False); env.reset([0]*3)
    paths = (args.condition, args.checkpoint, calibration_path, Path(__file__),
             Path(__file__).with_name('audit_card_credit_direction.py'),
             Path(__file__).with_name('probe_fixed_pair_learning.py'))
    hashes = {str(p.resolve()): sha256_file(p) for p in paths}
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    def progress(value):
        print(json.dumps(value), flush=True)
        write_json(args.output_dir/'status.json', {'status': 'running', 'source_hashes': hashes, **value})
    # Freeze the design before observing any learning outcomes.
    write_json(args.output_dir/'design.json', {'version': VERSION, 'source_hashes': hashes,
        'quantile': .95, 'calibration_repeats': args.calibration_repeats,
        'steps_per_lane': args.steps, 'evaluation_repeats': args.evaluation_repeats,
        'learner': args.learner_index, 'case_index': args.case_index,
        'lane_order': list(LANES), 'feedback': 'unchanged native progress',
        'seed_recipe': SEED_VERSION})
    try:
        progress({'phase': 'reward-free-calibration'})
        scaled, fit = calibrate(stack.agent, calibration, learner=args.learner_index, repeats=args.calibration_repeats)
        write_json(args.output_dir/'calibration.json', fit)
        progress({'phase': 'fixed-hand-comparison', 'through_step': 0,
                  'rescaled_config': fit['rescaled_config']})
        report = compare(stack.agent, env, corpus, scaled, learner=args.learner_index,
            case_index=args.case_index, steps=args.steps, evaluation_repeats=args.evaluation_repeats, progress=progress)
        if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError('source file identity changed during comparison')
        report.update(source_hashes=hashes, calibration=fit, pair_corpus_sha256=condition['pair_corpus_sha256'],
                      status='completed', wall_seconds=time.perf_counter()-started)
        write_json(args.output_dir/'report.json', report)
        write_json(args.output_dir/'status.json', {'status': 'completed', 'wall_seconds': report['wall_seconds']})
    except BaseException as error:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'source_hashes': hashes,
                    'error': f'{type(error).__name__}: {error}'})
        raise
    print(json.dumps({'report': str(args.output_dir/'report.json'), 'status': 'completed'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
