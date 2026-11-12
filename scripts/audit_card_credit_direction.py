"""Causal card-credit/clipping diagnostics using disposable efficacy copies.

Source checkpoints and their legacy harness identity remain unchanged. Oracle
pair choices are labelled diagnostic interventions, never new runtime features.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from card_learning.common import digest, write_json
from card_learning.experiment import build_condition_stack, load_condition
from card_learning.pairs import pair_slots
from flylatro.analysis.card_selection import permute_hand, plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch, validate_strict_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.plasticity import ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed


VERSION = 'card-credit-direction-audit-v1'


def array(value):
    return np.asarray(state_numpy(value)).copy()


def efficacy_like(state, values):
    if getattr(state, 'is_torch', False):
        import torch
        return torch.as_tensor(values, dtype=torch.float32, device=state.device).clone()
    return np.asarray(values, dtype=np.float32).copy()


def isolated_rule(agent, efficacy):
    state = agent.plasticity.state
    values = {key: value[:1].copy() for key, value in state.state_dict().items()}
    values['efficacy'] = array(efficacy)
    values['initial_efficacy'] = array(efficacy)
    for key in values.keys() - {'efficacy', 'initial_efficacy'}:
        values[key].fill(0)
    factory = type(state)
    local = (factory.from_state_dict(values, device=str(state.device))
             if getattr(state, 'is_torch', False) else factory.from_state_dict(values))
    return ThreeFactorPlasticity(agent.plasticity.topology, local, agent.plasticity.config)


def clipping_summary(pre, post, gate, config):
    pre, post, gate = array(pre), array(post), array(gate).astype(bool)
    if pre.shape != post.shape or pre.shape != gate.shape or not gate.any():
        raise ValueError('clipping audit needs aligned selected edges')
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError('nonfinite edge activity')
    a, b = pre[gate], post[gate]
    coincidence = np.clip(a / config.kc_reference_hz, 0, 1) * np.clip(b / config.mbon_reference_hz, 0, 1)
    return {'selected_edges': int(gate.sum()),
        'pre_clipped_fraction': float(np.mean(a >= config.kc_reference_hz)),
        'post_clipped_fraction': float(np.mean(b >= config.mbon_reference_hz)),
        'both_clipped_fraction': float(np.mean((a >= config.kc_reference_hz) & (b >= config.mbon_reference_hz))),
        'zero_coincidence_fraction': float(np.mean(coincidence == 0)),
        'unit_coincidence_fraction': float(np.mean(coincidence == 1)),
        'mean_coincidence': float(np.mean(coincidence)),
        'mean_pre_hz': float(a.mean()), 'mean_post_hz': float(b.mean())}


def one_update(agent, neural, action, masks, efficacy, pulse, sign):
    if sign not in (-1, 1) or not np.isfinite(pulse) or pulse < 0:
        raise ValueError('finite nonnegative pulse and explicit sign required')
    local = isolated_rule(agent, efficacy)
    credit = SelectedMotorCredit(agent.motor, local)
    gate = credit.mask(action, masks)
    stats = clipping_summary(neural.edge_pre_activity, neural.edge_post_activity, gate, local.config)
    local.record_activity(neural.edge_pre_activity, neural.edge_post_activity, edge_credit=gate)
    local.apply_dopamine(pulse if sign > 0 else 0, pulse if sign < 0 else 0)
    updated = array(local.state.efficacy)
    delta = updated - array(efficacy)
    if np.any(delta[~array(gate).astype(bool)] != 0):
        raise ValueError('reward changed an unselected edge from zero eligibility')
    if sign > 0 and np.any(delta < 0) or sign < 0 and np.any(delta > 0):
        raise ValueError('local efficacy update sign is inconsistent')
    return delta, stats


def metrics(motor, output, masks, target):
    target = np.asarray(target, dtype=int)
    legal = np.flatnonzero(masks['card_select_mask'][0])
    competitors = np.setdiff1d(legal, target)
    if len(target) != 2 or len(set(target.tolist())) != 2 or not np.isin(target, legal).all() or not len(competitors):
        raise ValueError('need two distinct legal target cards and competitors')
    scores = motor.head_scores(output)['card'][0]
    rates = motor.pool_activity(output)[0, list(motor.mapping.routing.head_routes['card'])]
    chosen = legal[np.argsort(-scores[legal], kind='stable')[:2]]
    return {'mean_score_gap': float(scores[target].mean() - scores[competitors].mean()),
        'topk_margin': float(scores[target].min() - scores[competitors].max()),
        'mean_rate_gap_hz': float(rates[target].mean() - rates[competitors].mean()),
        'target_membership_fraction': len(set(chosen.tolist()) & set(target.tolist())) / 2,
        'target_pair_selected': set(chosen.tolist()) == set(target.tolist()),
        'chosen_slots': chosen.tolist()}


def _process(agent, obs, seed, efficacy):
    neural = agent.processor.process(obs, fly_seeds=[seed], efficacy=efficacy)
    if not np.isfinite(neural.output_activity).all():
        raise ValueError('nonfinite motor output')
    return neural


def _equal_neural(left, right):
    for key in ('output_activity', 'edge_pre_activity', 'edge_post_activity'):
        if not np.array_equal(array(getattr(left, key)), array(getattr(right, key))):
            raise ValueError(f'exact neural repeat failed: {key}')


def summarize(rows, clips):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['weights'], row['action'], row['probe'], row['sign'], row['multiplier'])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        delta = np.asarray([row['delta_mean_score_gap'] for row in group])
        output.append(dict(zip(('weights', 'action', 'probe', 'sign', 'multiplier'), key),
            trials=len(group), mean_gap_change=float(delta.mean()),
            improved_fraction=float(np.mean(delta > 1e-9)), worsened_fraction=float(np.mean(delta < -1e-9)),
            unchanged_fraction=float(np.mean(np.abs(delta) <= 1e-9)),
            mean_max_edge_change=float(np.mean([r['max_edge_change'] for r in group])),
            mean_bound_fraction=float(np.mean([r['bound_fraction'] for r in group]))))
    clip_groups = defaultdict(list)
    for row in clips:
        clip_groups[(row['weights'], row['action'])].append(row)
    clipping = [dict(weights=key[0], action=key[1], trials=len(group),
        **{name: float(np.mean([row[name] for row in group])) for name in (
            'pre_clipped_fraction', 'post_clipped_fraction', 'both_clipped_fraction',
            'zero_coincidence_fraction', 'unit_coincidence_fraction', 'mean_coincidence')})
        for key, group in sorted(clip_groups.items())]
    return {'direction': output, 'clipping': clipping}


def audit(agent, env, corpus, *, learner=0, cases=13, repeats=2, multipliers=(1, 100, 1000),
          weights=('initial', 'trained'), progress=None):
    if min(cases, repeats) < 1 or not 0 <= learner < agent.plasticity.state.learners:
        raise ValueError('positive cases/repeats and a valid learner required')
    if not multipliers or len(set(multipliers)) != len(multipliers) or any(not np.isfinite(m) or m <= 0 for m in multipliers):
        raise ValueError('positive unique finite update multipliers required')
    if not weights or set(weights) - {'initial', 'trained'}:
        raise ValueError('select initial and/or trained efficacy')
    before = plastic_hashes(agent.plasticity.state)
    env_before = env.snapshot_all()
    all_indices = corpus.indices('training')
    # Input/rank selection, never outcome-based. Prefer one hand per pair rank.
    by_rank = {}
    for index in all_indices:
        by_rank.setdefault(corpus.cases[index]['pair_rank'], index)
    ordered = list(by_rank.values()) + [i for i in all_indices if i not in by_rank.values()]
    selected = ordered[:cases]
    if len(selected) != cases:
        raise ValueError('not enough distinct training cases')
    state = agent.plasticity.state
    report = {'version': VERSION, 'case_indices': selected, 'learner': learner,
        'repeats': repeats, 'multipliers': [0, *multipliers], 'trials': [], 'clipping_trials': [],
        'learning_disabled_in_source': True, 'source_traces_discarded_in_probes': True,
        'stimulus_split': 'training only; no validation or final-test selection',
        'interventions': 'greedy selection and forced correct pair; actual native progress pulse; negative pulse is a sign control',
        'scaling': 'multiply a single actual efficacy delta, then clip to original bounds; not repeated online updates',
        'interpretation': 'local causal/sensitivity measurements, not training, pair-learning success or a biological valence claim',
        'exact_repeat_checks': 0, 'no_update_checks': 0, 'restoration_checks': 0}
    try:
        for index in selected:
            corpus.restore(env, 0, index)
            original_obs, original_masks = env.observe()
            original_masks['action_type_mask'][:] = False
            original_masks['action_type_mask'][:, 0] = True
            for repeat in range(repeats):
                rng = np.random.default_rng(derive_seed(VERSION, index, repeat, 'display'))
                order = np.r_[rng.permutation(8), 8, 9]
                obs, masks = permute_hand(original_obs, original_masks, order)
                transfer_order = np.r_[rng.permutation(8), 8, 9]
                changed_obs, changed_masks = permute_hand(obs, masks, transfer_order)
                source_seed = derive_seed(VERSION, corpus.cases[index]['seed'], repeat, 'source')
                other_seed = derive_seed(VERSION, corpus.cases[index]['seed'], repeat, 'held-noise')
                for label in weights:
                    efficacy = (state.initial_efficacy if label == 'initial' else state.efficacy)[learner:learner+1]
                    original = array(efficacy)
                    neural = _process(agent, obs, source_seed, efficacy)
                    _equal_neural(neural, _process(agent, obs, source_seed, efficacy))
                    report['exact_repeat_checks'] += 1
                    contexts = [('same_seed', obs, masks, source_seed, np.arange(10)),
                                ('held_noise', obs, masks, other_seed, np.arange(10)),
                                ('permuted_cards', changed_obs, changed_masks, other_seed, np.argsort(transfer_order))]
                    baselines = [array(_process(agent, o, seed, efficacy).output_activity) for _, o, _, seed, _ in contexts]
                    for (_, o, _, seed, _), baseline in zip(contexts, baselines):
                        duplicate = _process(agent, o, seed, efficacy_like(state, original))
                        if not np.array_equal(baseline, duplicate.output_activity):
                            raise ValueError('no-update copy changed output')
                        report['no_update_checks'] += 1
                    greedy = agent.motor.decode(neural.output_activity, masks, deterministic=True)
                    for action_name, cards in [('greedy', tuple(greedy['cards'][0, :2])), ('correct_pair', pair_slots(obs))]:
                        action = empty_action_batch(1); action['action_type'][:] = 0; action['n_cards'][:] = 2
                        action['cards'][0, :2] = cards
                        validate_strict_action_batch(action, masks)
                        native = {k: v.copy() for k, v in action.items()}; native['cards'][0, :2] = order[list(cards)]
                        corpus.restore(env, 0, index)
                        scored = env.step(native)
                        pulse = agent.reinforcement.map(scored.infos[0]['reward_components'], scored.infos[0])
                        if pulse.aversive or pulse.appetitive <= 0:
                            raise ValueError('opening pair-direction probe requires positive native progress only')
                        for sign in (1, -1):
                            delta, stats = one_update(agent, neural, action, masks, efficacy, pulse.appetitive, sign)
                            if sign > 0:
                                report['clipping_trials'].append(dict(case_index=index, repeat=repeat,
                                    weights=label, action=action_name, pulse=pulse.appetitive, **stats))
                            for multiplier in (0, *multipliers):
                                altered = np.clip(original + multiplier * delta,
                                    agent.plasticity.config.min_efficacy, agent.plasticity.config.max_efficacy)
                                if not np.isfinite(altered).all():
                                    raise ValueError('nonfinite intervention weights')
                                changes = altered - original
                                for context_index, (probe, probe_obs, probe_masks, seed, inverse) in enumerate(contexts):
                                    baseline = baselines[context_index]
                                    output = baseline if multiplier == 0 else _process(agent, probe_obs, seed, efficacy_like(state, altered)).output_activity
                                    target = inverse[list(cards)]
                                    initial_metrics = metrics(agent.motor, baseline, probe_masks, target)
                                    after_metrics = metrics(agent.motor, output, probe_masks, target)
                                    pair_target = pair_slots(probe_obs)
                                    row = {'case_index': index, 'repeat': repeat, 'weights': label, 'action': action_name,
                                        'target_cards': list(map(int, target)), 'source_cards': list(map(int, cards)),
                                        'probe': probe, 'sign': sign, 'multiplier': multiplier, 'native_pulse': pulse.appetitive,
                                        'max_edge_change': float(np.abs(changes).max()),
                                        'bound_fraction': float(np.mean((altered <= agent.plasticity.config.min_efficacy) | (altered >= agent.plasticity.config.max_efficacy))),
                                        'before': initial_metrics, 'after': after_metrics,
                                        'delta_mean_score_gap': after_metrics['mean_score_gap'] - initial_metrics['mean_score_gap'],
                                        'delta_topk_margin': after_metrics['topk_margin'] - initial_metrics['topk_margin'],
                                        'pair_selected_before': metrics(agent.motor, baseline, probe_masks, pair_target)['target_pair_selected'],
                                        'pair_selected_after': metrics(agent.motor, output, probe_masks, pair_target)['target_pair_selected']}
                                    report['trials'].append(row)
                    _equal_neural(neural, _process(agent, obs, source_seed, efficacy))
                    report['restoration_checks'] += 1
            if progress:
                progress({'completed_case': index, 'completed_cases': selected.index(index)+1, 'total_cases': cases})
    finally:
        env.restore_all(env_before)
        report['source_state_unchanged'] = plastic_hashes(agent.plasticity.state) == before
        if not report['source_state_unchanged']:
            raise ValueError('direction probe changed source plastic state')
    report['summary'] = summarize(report['trials'], report['clipping_trials'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--learner-index', type=int, default=0)
    parser.add_argument('--cases', type=int, default=13)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--multipliers', type=float, nargs='+', default=[1, 100, 1000])
    parser.add_argument('--weights', choices=['initial', 'trained'], nargs='+', default=['initial', 'trained'])
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    condition, corpus = load_condition(args.condition)
    if corpus is None:
        raise ValueError('use a declared pair-task condition')
    config = PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy)
    stack = build_condition_stack(config, condition, corpus)
    load_plastic_checkpoint(args.checkpoint, stack.trainer, expected_components=stack.components)
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False); env.reset([0])
    args.output_dir.mkdir(parents=True)
    hashes = {'condition': sha256_file(args.condition), 'checkpoint': sha256_file(args.checkpoint),
              'corpus': corpus.sha256, 'audit_script': sha256_file(Path(__file__))}
    started = time.perf_counter()
    write_json(args.output_dir / 'status.json', {'status': 'running', 'source_hashes': hashes})
    def progress(value):
        print(json.dumps(value), flush=True)
        write_json(args.output_dir / 'status.json', {'status': 'running', 'source_hashes': hashes, **value})
    try:
        report = audit(stack.agent, env, corpus, learner=args.learner_index, cases=args.cases,
            repeats=args.repeats, multipliers=args.multipliers, weights=args.weights, progress=progress)
        if sha256_file(args.checkpoint) != hashes['checkpoint'] or sha256_file(args.condition) != hashes['condition']:
            raise ValueError('source file changed during audit')
        report.update(status='completed', source_hashes=hashes, wall_seconds=time.perf_counter()-started)
        write_json(args.output_dir / 'report.json', report)
        write_json(args.output_dir / 'status.json', {'status': 'completed', 'source_hashes': hashes, 'wall_seconds': report['wall_seconds']})
    except BaseException as error:
        write_json(args.output_dir / 'status.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'source_hashes': hashes})
        raise
    print(json.dumps({'report': str(args.output_dir / 'report.json'), 'status': 'completed'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
