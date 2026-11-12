"""Train anatomical synapses locally from visible-state demonstrations.

Separate research version: no gradients or external learned policy. Evaluation
uses actual neural motor decoding with no teacher action substitution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import isolated_rule
from card_learning.audit import score
from card_learning.common import write_json
from card_learning.experiment import load_condition
from card_learning.pairs import PairCorpus, build_corpus
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.upstream_contract import validate_strict_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.seeds import derive_seed, SeedPlan
from local_teaching import LocalTeachingConfig, LocalTeachingPlasticity, edge_pool_indices, teaching_signals
from run_repaired_native_membership import CONDITION, QUALIFICATION, fresh_stack, verify_sources
from teacher_curriculum_data import (STAGES, collect_blind_demonstrations, decode_stage,
                                     evaluate_small_blind, opening_batch, shuffle_instruction)

VERSION = 'local-teacher-card-curriculum-v1'


def new_rule(agent, learners, config):
    single = isolated_rule(agent, agent.plasticity.state.initial_efficacy[:1])
    values = {k: np.repeat(v, learners, axis=0) for k, v in single.state.state_dict().items()}
    factory = type(single.state)
    state = (factory.from_state_dict(values, device=str(single.state.device)) if getattr(single.state, 'is_torch', False)
             else factory.from_state_dict(values))
    return LocalTeachingPlasticity(single.topology, state, config)


def prepare(folder, *, steps=1000, replicates=3, blind_games=32):
    if min(steps, replicates, blind_games) < 1:
        raise ValueError('positive curriculum budgets required')
    if folder.exists():
        raise FileExistsError(folder)
    qualification = json.loads((QUALIFICATION/'assessment.json').read_text())
    verify_sources(qualification['artifact_sha256'])
    condition, _ = load_condition(CONDITION)
    folder.mkdir(parents=True)
    build_corpus(folder/'pairs.json', train_per_rank=8, validation_per_rank=4,
                 training_offset=18500, validation_offset=460000, max_attempts=5000)
    build_corpus(folder/'fresh-pairs.json', train_per_rank=1, validation_per_rank=8,
                 training_offset=20500, validation_offset=480000, max_attempts=5000)
    pair_corpus = PairCorpus(folder/'pairs.json')
    count_targets = opening_batch(pair_corpus, pair_corpus.indices('training'), 'best_hand', key='count-prior')[2]['n_cards'].tolist()
    write_json(folder/'training-counts.json', {'best_hand': count_targets})
    training_seeds = list(SeedPlan().seeds('training', 32, offset=22500))
    demonstrations = collect_blind_demonstrations(training_seeds)
    obs, masks, targets, meta = demonstrations
    np.savez_compressed(folder/'blind-demonstrations.npz',
        **{'obs__'+k: v for k, v in obs.items()}, **{'mask__'+k: v for k, v in masks.items()},
        **{'target__'+k: v for k, v in targets.items()})
    write_json(folder/'demonstration-report.json', meta)
    paths = [QUALIFICATION/'config.toml', QUALIFICATION/'candidate-motor.json', QUALIFICATION/'assessment.json',
        CONDITION, folder/'pairs.json', folder/'fresh-pairs.json', folder/'blind-demonstrations.npz',
        folder/'demonstration-report.json', folder/'training-counts.json']
    paths += [Path(__file__).with_name(x) for x in ('run_local_teacher_curriculum.py', 'local_teaching.py',
        'teacher_curriculum_data.py', 'run_repaired_native_membership.py', 'run_card_membership.py')]
    sources = {str(path.resolve()): sha256_file(path) for path in paths}
    write_json(folder/'declaration.json', {'version': VERSION, 'source_hashes': sources,
        'config': str((QUALIFICATION/'config.toml').resolve()),
        'mapping_sha256': json.loads((QUALIFICATION/'status.json').read_text())['candidate_sha256'],
        'exploration': condition['exploration'], 'local_rule': asdict(LocalTeachingConfig()),
        'stages': list(STAGES), 'steps_per_stage': steps, 'replicates': replicates, 'warmup_frames': 64,
        'controls': ['teacher', 'random-instruction', 'frozen'],
        'teacher': 'visible-state card comparison / pair membership / pair / best immediate plain hand; output teaching only',
        'learning': 'local covariance eligibility times neuron-specific signed teaching pulse; target-minus-actual discrepancy; no derivatives/backprop/optimizer',
        'parameters': 'existing KC-to-MBON efficacies only; presynaptic means are local activity traces',
        'inference': 'fixed existing neural mapping; no teacher, fitted decoder or critic; PLAY-only early curriculum',
        'validation_blind_seeds': list(SeedPlan().seeds('validation', blind_games, offset=520000)),
        'fresh_blind_seeds': list(SeedPlan().seeds('validation', blind_games*2, offset=540000)),
        'promotion': 'advance only with fresh teacher-free card-play and native first-blind improvement against own initial/frozen controls',
        'promotion_thresholds': {'minimum_clear_gain_over_initial': .05, 'minimum_clear_gain_over_random_instruction': .05,
                                 'minimum_positive_learners': 2, 'conditional_case_interval_above_zero': True},
        'training_safety': {'maximum_bound_fraction': .1, 'maximum_mean_absolute_drift': .5},
        'future': 'DISCARD timing, full Ante-1 and shop curriculum remain gated on first-blind competence',
        'primary_model_changed': False, 'final_test_used': False})
    write_json(folder/'status.json', {'status': 'prepared'})


def _snapshot(rule):
    return {k: v.copy() for k, v in rule.state_dict().items()}


def _unchanged(rule, snapshot):
    current = rule.state_dict()
    for key in snapshot:
        np.testing.assert_array_equal(snapshot[key], current[key], err_msg='evaluation changed '+key)


def evaluate_openings(agent, rule, corpus, stage, ids, *, repeats=1):
    before = _snapshot(rule); rows = []
    sides = (0, 1) if stage in ('higher_card', 'pair_member') else (None,)
    for index in corpus.indices('validation'):
        for repeat in range(repeats):
            for side in sides:
                key = ('evaluation', stage, repeat, side)
                obs, masks, target, _ = opening_batch(corpus, [index]*len(ids), stage, key=key, correct_side=side)
                neural = agent.processor.process(obs, fly_seeds=[derive_seed(VERSION, key, index, r) for r in ids],
                                                 efficacy=rule.state.efficacy)
                actual = decode_stage(agent, neural.output_activity, masks, stage)
                validate_strict_action_batch(actual, masks)
                correct, chips, target_chips = [], [], []
                for lane in range(len(ids)):
                    a = actual['cards'][lane, :int(actual['n_cards'][lane])]
                    t = target['cards'][lane, :int(target['n_cards'][lane])]
                    correct.append(set(a.tolist()) == set(t.tolist()))
                    hand = [(int(c[:13].argmax())+2, int(c[13:17].argmax())) for c in obs['hand'][lane, :8]]
                    chips.append(score(hand, a)[0])
                    target_chips.append(score(hand, t)[0])
                rows.append({'case': int(index), 'side': side, 'repeat': repeat,
                    'correct': correct, 'chips': chips, 'target_chips': target_chips, 'cards': actual['cards'].tolist(), 'counts': actual['n_cards'].tolist()})
    _unchanged(rule, before)
    return {'stage': stage, 'cases': len(corpus.indices('validation')), 'rows': rows,
        'exact_accuracy_per_lane': np.asarray([r['correct'] for r in rows]).mean(axis=0).tolist(),
        'mean_chips_per_lane': np.asarray([r['chips'] for r in rows]).mean(axis=0).tolist(),
        'mean_teacher_chips_per_lane': np.asarray([r['target_chips'] for r in rows]).mean(axis=0).tolist(), 'all_state_fields_frozen': True}


def batch_for_stage(corpus, stage, indices, *, step, demos=None):
    if stage != 'small_blind':
        return opening_batch(corpus, indices, stage, key=('training', stage, step), correct_side=step % 2)[:3]
    obs, masks, targets = demos
    return ({k: v[indices].copy() for k, v in obs.items()}, {k: v[indices].copy() for k, v in masks.items()},
            {k: v[indices].copy() for k, v in targets.items()})


def save_checkpoint(folder, rule, cursor, declaration):
    directory = folder/'checkpoints'; directory.mkdir(exist_ok=True)
    path = directory/f"stage-{cursor['stage']:02}-step-{cursor['step']:06}.npz"
    values = rule.state_dict()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            for key, value in values.items():
                np.testing.assert_array_equal(saved[key], value, err_msg='checkpoint generation already differs')
    else:
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **values); temporary.replace(path)
    write_json(folder/'checkpoint-latest.json', {'version': VERSION, 'declaration_sha256': sha256_file(folder/'declaration.json'),
        'state_file': str(path.relative_to(folder)), 'state_sha256': sha256_file(path), 'cursor': cursor, 'local_rule': declaration['local_rule'],
        'scope': 'curriculum-only checkpoint; not interchangeable with legacy trainer checkpoints'})


def train(folder, *, heavy=False, resume=False):
    declaration = json.loads((folder/'declaration.json').read_text()); verify_sources(declaration['source_hashes'])
    config = PlasticExperimentConfig.load(Path(declaration['config'])); config.require_heavy_opt_in(heavy)
    stack = fresh_stack(config, declaration['exploration'], declaration['mapping_sha256']); agent = stack.agent
    source = plastic_hashes(agent.plasticity.state)
    replicates = declaration['replicates']; ids = np.repeat(np.arange(replicates), 3).tolist()
    roles = np.tile(np.arange(3), replicates); n = len(ids)
    rule = new_rule(agent, n, LocalTeachingConfig(**declaration['local_rule']))
    cursor = {'stage': 0, 'step': 0, 'warmup': 0}
    if resume:
        metadata = json.loads((folder/'checkpoint-latest.json').read_text())
        state_path = (folder/metadata['state_file']).resolve()
        if not state_path.is_relative_to(folder.resolve()):
            raise ValueError('curriculum checkpoint path leaves declared run')
        if metadata['declaration_sha256'] != sha256_file(folder/'declaration.json') or metadata['state_sha256'] != sha256_file(state_path):
            raise ValueError('curriculum resume identity differs')
        with np.load(state_path, allow_pickle=False) as values:
            factory = type(rule.state)
            rule.state = (factory.from_state_dict(values, device=str(rule.state.device)) if rule.torch else factory.from_state_dict(values))
            rule.load_traces(values)
        cursor = metadata['cursor']
    elif (folder/'checkpoint-latest.json').exists():
        raise FileExistsError('use --resume for an existing curriculum')
    corpus = PairCorpus(folder/'pairs.json'); fresh = PairCorpus(folder/'fresh-pairs.json')
    count_priors = json.loads((folder/'training-counts.json').read_text())
    with np.load(folder/'blind-demonstrations.npz', allow_pickle=False) as data:
        demos = tuple({k.split('__', 1)[1]: data[k].copy() for k in data.files if k.startswith(prefix+'__')}
                      for prefix in ('obs', 'mask', 'target'))
    edge_pool = edge_pool_indices(agent.motor.mapping, rule.topology)
    before_path = folder/'initial-evaluation.json'
    if resume and not before_path.exists():
        raise ValueError('curriculum initial evaluation missing; cannot reconstruct it from trained weights')
    def progress(row):
        print(json.dumps(row), flush=True); write_json(folder/'status.json', {'status': 'running', **row})
    started = time.perf_counter()
    if not before_path.exists():
        progress({'phase': 'initial-teacher-free-evaluation'})
        initial = {stage: evaluate_openings(agent, rule, fresh, stage, ids) for stage in STAGES[:-1]}
        frozen = _snapshot(rule)
        initial['small_blind'] = evaluate_small_blind(agent, rule, declaration['fresh_blind_seeds'], ids,
                                                    key='frozen-evaluation', progress=progress)
        _unchanged(rule, frozen); write_json(before_path, initial)
    for stage_index, stage in enumerate(declaration['stages']):
        if stage_index < cursor['stage']:
            continue
        candidates = np.arange(len(demos[0]['hand'])) if stage == 'small_blind' else np.asarray(corpus.indices('training'))
        orders = [np.random.default_rng(derive_seed(VERSION, 'training-order', stage, r)).permutation(candidates) for r in range(replicates)]
        targets_for_counts = demos[2]['n_cards'] if stage == 'small_blind' else np.asarray(count_priors['best_hand'])
        # Stage activity means are calibrated without labels or efficacy changes.
        warmup_start = cursor['warmup'] if stage_index == cursor['stage'] else 0
        if warmup_start == 0:
            rule.reset_activity_calibration()
        for warmup in range(warmup_start, declaration['warmup_frames']):
            indices = [int(orders[r][warmup % len(candidates)]) for r in ids]
            obs, masks, _ = batch_for_stage(corpus, stage, indices, step=warmup, demos=demos)
            neural = agent.processor.process(obs, fly_seeds=[derive_seed(VERSION, 'warmup', stage, r, warmup) for r in ids], efficacy=rule.state.efficacy)
            rule.observe(neural.edge_pre_activity, neural.edge_post_activity, calibrate=True)
        begin = cursor['step'] if stage_index == cursor['stage'] else 0
        progress({'phase': 'local-teaching', 'stage': stage, 'step': begin})
        stage_log = folder/(stage+'-training.jsonl')
        if begin:
            lines = stage_log.read_text().splitlines()
            kept = lines[:begin]
            if [json.loads(line)['step'] for line in kept] != list(range(begin)):
                raise ValueError('checkpoint training-log prefix differs')
            stage_log.write_text('\n'.join(kept)+'\n')
        with stage_log.open('a' if begin else 'w') as stream:
            for step in range(begin, declaration['steps_per_stage']):
                indices = [int(orders[r][step % len(candidates)]) for r in ids]
                obs, masks, target = batch_for_stage(corpus, stage, indices, step=step, demos=demos)
                neural = agent.processor.process(obs, fly_seeds=[derive_seed(VERSION, 'training', stage, r, step) for r in ids], efficacy=rule.state.efficacy)
                actual = decode_stage(agent, neural.output_activity, masks, stage)
                validate_strict_action_batch(actual, masks)
                controls = shuffle_instruction(target, stage, ids, key=step)
                if stage in ('best_hand', 'small_blind'):
                    for lane in np.flatnonzero(roles == 1):
                        rng = np.random.default_rng(derive_seed(VERSION, 'control-count', stage, ids[lane], step))
                        count = int(rng.choice(targets_for_counts)); controls['n_cards'][lane] = count; controls['cards'][lane] = -1
                        controls['cards'][lane, :count] = np.sort(rng.choice(8, count, replace=False))
                instruction = {k: v.copy() for k, v in target.items()}
                for key in instruction:
                    instruction[key][roles == 1] = controls[key][roles == 1]
                signals = teaching_signals(agent.motor.mapping, instruction, actual, teach_count=stage in ('best_hand', 'small_blind'))
                signals[roles == 2] = 0
                rule.observe(neural.edge_pre_activity, neural.edge_post_activity)
                events = rule.teach(signals, edge_pool)
                stream.write(json.dumps({'step': step, 'indices': indices, 'actual_cards': actual['cards'].tolist(),
                    'actual_counts': actual['n_cards'].tolist(), 'target_cards': target['cards'].tolist(),
                    'instruction_cards': instruction['cards'].tolist(), **events})+'\n')
                if (step+1) % 250 == 0 or step+1 == declaration['steps_per_stage']:
                    stream.flush()
                    state = rule.state.state_dict(); efficacy = state['efficacy']
                    safety = declaration['training_safety']
                    occupied = ((efficacy <= rule.config.min_efficacy) | (efficacy >= rule.config.max_efficacy)).mean(axis=1)
                    drift = np.abs(efficacy-state['initial_efficacy']).mean(axis=1)
                    if not np.isfinite(efficacy).all() or np.any(occupied > safety['maximum_bound_fraction']) or np.any(drift > safety['maximum_mean_absolute_drift']):
                        raise ValueError('local teaching plastic-state safety limit reached')
                    cursor = {'stage': stage_index, 'step': step+1, 'warmup': declaration['warmup_frames']}
                    save_checkpoint(folder, rule, cursor, declaration)
                    progress({'phase': 'local-teaching', 'stage': stage, 'step': step+1, 'last_update_l1': events['update_l1']})
        progress({'phase': 'teacher-free-stage-evaluation', 'stage': stage})
        stage_result = (evaluate_small_blind(agent, rule, declaration['validation_blind_seeds'], ids, key='stage-evaluation', progress=progress)
                        if stage == 'small_blind' else evaluate_openings(agent, rule, corpus, stage, ids))
        write_json(folder/(stage+'-evaluation.json'), stage_result)
        cursor = {'stage': stage_index+1, 'step': 0, 'warmup': 0}; save_checkpoint(folder, rule, cursor, declaration)
    progress({'phase': 'fresh-teacher-free-evaluation'})
    final = {stage: evaluate_openings(agent, rule, fresh, stage, ids) for stage in STAGES[:-1]}
    frozen = _snapshot(rule)
    final['small_blind'] = evaluate_small_blind(agent, rule, declaration['fresh_blind_seeds'], ids,
                                              key='frozen-evaluation', progress=progress)
    _unchanged(rule, frozen)
    np.savez_compressed(folder/'local-final-state.npz', **rule.state_dict())
    initial = json.loads(before_path.read_text())
    summary = {}
    for stage in STAGES:
        metric = 'clear_rate_per_lane' if stage == 'small_blind' else 'exact_accuracy_per_lane'
        a, b = np.asarray(initial[stage][metric]), np.asarray(final[stage][metric])
        np.testing.assert_array_equal(a[roles == 2], b[roles == 2])
        summary[stage] = {label: {'initial': a[roles == role].tolist(), 'trained': b[roles == role].tolist(),
            'change': (b-a)[roles == role].tolist(), 'mean_initial': float(a[roles == role].mean()),
            'mean_trained': float(b[roles == role].mean())} for role, label in enumerate(('teacher', 'random-instruction', 'frozen'))}
    values = rule.state.state_dict(); delta = values['efficacy']-values['initial_efficacy']
    if np.any(delta[roles == 2]) or np.any(delta[:, edge_pool < 0]):
        raise ValueError('frozen or non-PLAY weights changed')
    if plastic_hashes(agent.plasticity.state) != source:
        raise ValueError('source state changed')
    verify_sources(declaration['source_hashes'])
    gate = declaration['promotion_thresholds']; improvement = np.asarray(summary['small_blind']['teacher']['change'])
    advantage = (np.asarray(summary['small_blind']['teacher']['trained'])
                 -np.asarray(summary['small_blind']['random-instruction']['trained']))
    case_change = (np.asarray([r['cleared'] for r in final['small_blind']['rows']], dtype=float)[:, roles == 0]
                   -np.asarray([r['cleared'] for r in initial['small_blind']['rows']], dtype=float)[:, roles == 0]).mean(axis=1)
    rng = np.random.default_rng(derive_seed(VERSION, 'first-blind-case-bootstrap'))
    interval = np.quantile(case_change[rng.integers(0, len(case_change), size=(2000, len(case_change)))].mean(axis=1), [.025, .975]).tolist()
    write_json(folder/'report.json', {'version': VERSION, 'status': 'completed', 'source_hashes': declaration['source_hashes'],
        'initial': initial, 'final': final, 'summary': summary, 'roles': roles.tolist(), 'replicate_ids': ids,
        'source_state_unchanged': True, 'frozen_weights_exact': True, 'unrelated_weights_exact': True,
        'no_gradient_or_external_learned_policy': True, 'wall_seconds': time.perf_counter()-started,
        'first_blind_change_conditional_case_interval': interval,
        'first_blind_ready_for_next_curriculum': bool(improvement.mean() > gate['minimum_clear_gain_over_initial']
            and advantage.mean() > gate['minimum_clear_gain_over_random_instruction']
            and np.count_nonzero(improvement > 0) >= gate['minimum_positive_learners'] and interval[0] > 0),
        'limits': 'teacher-assisted local plasticity; PLAY-only first-blind curriculum; no claim of autonomous DISCARD/shop/Ante-1 competence'})
    write_json(folder/'status.json', {'status': 'completed'})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'train'), required=True); p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps-per-stage', type=int, default=1000); p.add_argument('--replicates', type=int, default=3)
    p.add_argument('--blind-games', type=int, default=32); p.add_argument('--heavy', action='store_true'); p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv); folder = args.output_dir.resolve()
    if args.stage == 'prepare':
        prepare(folder, steps=args.steps_per_stage, replicates=args.replicates, blind_games=args.blind_games)
    else:
        try:
            train(folder, heavy=args.heavy, resume=args.resume)
        except BaseException as error:
            write_json(folder/'status.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}); raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
