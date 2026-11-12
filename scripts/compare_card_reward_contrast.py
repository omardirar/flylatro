"""Exploratory centered native-progress feedback on disposable fixed-hand copies.

Feedback calibration is outcome-based and explicitly separate from reward-free
sensory/motor calibration. No card labels or unchosen-action scores are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from audit_card_credit_direction import array, isolated_rule
from compare_card_reference_scaling import evaluate
from probe_fixed_pair_learning import VERSION as SEED_VERSION
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


VERSION = 'centered-native-progress-capacity-v1'
LANES = ('native', 'centered', 'frozen')
CALIBRATION_ID_OFFSET = 10_000_000


@dataclass(frozen=True)
class CenteredFeedback:
    center: float
    scale: float
    max_pulse: float

    def __post_init__(self):
        if (not np.isfinite((self.center, self.scale, self.max_pulse)).all()
                or self.center < 0 or self.scale <= 0 or self.max_pulse <= 0):
            raise ValueError('finite nonnegative center and positive scale/cap required')

    def signed(self, native_pulse):
        pulse = np.asarray(native_pulse, dtype=np.float64)
        if not np.isfinite(pulse).all() or np.any(pulse < 0):
            raise ValueError('native progress pulse must be finite and nonnegative')
        return np.clip(self.scale * (pulse-self.center), -self.max_pulse, self.max_pulse)


def fit_feedback(native_pulses, *, max_pulse):
    """Freeze a scalar feedback transform; no state/action/card-label inputs."""
    values = np.asarray(native_pulses, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('at least two finite positive native progress pulses required')
    center = float(values.mean())
    absolute_deviation = float(np.mean(np.abs(values-center)))
    if absolute_deviation <= 1e-12:
        raise ValueError('feedback calibration has no measurable reward variation')
    rule = CenteredFeedback(center, center/absolute_deviation, max_pulse)
    signed = rule.signed(values)
    if np.any(np.abs(rule.scale*(values-rule.center)) >= max_pulse):
        raise ValueError('feedback cap prevents matching calibration pulse magnitude')
    if not np.any(signed > 0) or not np.any(signed < 0):
        raise ValueError('centered calibration requires both pulse signs')
    return rule, {'version': VERSION, 'rule': asdict(rule), 'trials': len(values),
        'native_pulses': values.tolist(), 'mean_native_pulse': center,
        'mean_native_absolute_deviation': absolute_deviation,
        'mean_centered_signed_pulse': float(signed.mean()),
        'mean_centered_absolute_pulse': float(np.abs(signed).mean()),
        'relative_absolute_pulse_matching_error': float(abs(np.abs(signed).mean()-center)/center),
        'matching_scope': 'frozen-policy calibration pulse magnitude only; no trajectory, eligibility or weight-change matching',
        'formula': 'center=mean(native); scale=mean(native)/mean(abs(native-center)); signed=clip(scale*(native-center), +/-max_pulse)',
        'labels_used': False, 'unchosen_actions_scored': False}


def local_rule(agent, learner):
    single = isolated_rule(agent, agent.plasticity.state.initial_efficacy[learner:learner+1])
    values = {key: np.repeat(value, 3, axis=0) for key, value in single.state.state_dict().items()}
    factory = type(single.state)
    state = (factory.from_state_dict(values, device=str(single.state.device))
             if getattr(single.state, 'is_torch', False) else factory.from_state_dict(values))
    return ThreeFactorPlasticity(agent.plasticity.topology, state, agent.plasticity.config)


def score_choices(agent, env, corpus, case_index, actions, order):
    native = {key: value.copy() for key, value in actions.items()}
    for row in range(3):
        corpus.restore(env, row, case_index)
        native['cards'][row, :2] = order[actions['cards'][row, :2]]
    result = env.step(native)
    if np.any(result.dones):
        raise ValueError('native scoring unexpectedly ended a run')
    pulses = [agent.reinforcement.map(info['reward_components'], info) for info in result.infos]
    if any(p.aversive != 0 or p.appetitive <= 0 or p.events != ('progress',) for p in pulses):
        raise ValueError('probe requires only positive native progress feedback')
    return np.asarray([p.appetitive for p in pulses], dtype=np.float64)


def calibrate_feedback(agent, env, corpus, case_index, obs, masks, order, efficacy,
                       *, learner=0, trials=128):
    if trials < 2:
        raise ValueError('at least two feedback calibration trials required')
    values, choices = [], []
    for trial in range(trials):
        seed = derive_seed(VERSION, case_index, 'feedback-calibration', trial)
        neural = agent.processor.process(obs, fly_seeds=[seed]*3, efficacy=efficacy)
        if not np.isfinite(neural.output_activity).all():
            raise ValueError('nonfinite calibration output')
        actions = agent.motor.decode(neural.output_activity, masks, deterministic=False,
            learner_ids=[learner]*3, decision_ids=[CALIBRATION_ID_OFFSET+trial]*3)
        if any(not np.array_equal(actions[key][0], actions[key][row])
               for key in actions for row in (1, 2)):
            raise ValueError('identical calibration lanes produced different actions')
        pulses = score_choices(agent, env, corpus, case_index, actions, order)
        if not np.all(pulses == pulses[0]):
            raise ValueError('identical calibration actions produced different native rewards')
        values.append(float(pulses[0])); choices.append(actions['cards'][0, :2].tolist())
    rule, report = fit_feedback(values, max_pulse=agent.reinforcement.config.max_total_pulse)
    report.update(selected_cards=choices,
        seed_recipe='version, training case, feedback-calibration, trial; independent of training/evaluation',
        source='native outcomes of actual frozen initial-policy choices on this training hand',
        reward_free=False, external_trainable_policy_parameters=0,
        additional_frozen_trials_per_lane=trials)
    return rule, report


def compare(agent, env, corpus, *, learner=0, case_index=None, steps=5000,
            evaluation_repeats=128, calibration_trials=128, progress=None):
    if (not 0 < steps < CALIBRATION_ID_OFFSET or evaluation_repeats < 1 or calibration_trials < 2
            or not 0 <= learner < agent.plasticity.state.learners):
        raise ValueError('positive budgets, at least two calibration trials and valid learner required')
    if env.num_envs != 3:
        raise ValueError('comparison requires native, centered and frozen lanes')
    if agent.plasticity.config.dopamine_decay != 0:
        raise ValueError('comparison requires decision-local dopamine')
    case_index = corpus.indices('training')[0] if case_index is None else case_index
    if case_index not in corpus.indices('training'):
        raise ValueError('reward-contrast comparison is training-corpus only')
    source = agent.plasticity.state
    before, env_before = plastic_hashes(source), env.snapshot_all()
    local = local_rule(agent, learner)
    credit = SelectedMotorCredit(agent.motor, local)
    initial = array(local.state.efficacy)
    rng = np.random.default_rng(derive_seed(SEED_VERSION, case_index, 'display'))
    order = np.r_[rng.permutation(8), 8, 9]
    correct, pulse_sum, signed_sum, absolute_sum, update_l1 = (np.zeros(3) for _ in range(5))
    positive_count, negative_count, zero_count = (np.zeros(3, dtype=int) for _ in range(3))
    block_correct, blocks, capped = np.zeros(3, dtype=int), [], 0
    try:
        for row in range(3):
            corpus.restore(env, row, case_index)
        raw_obs, raw_masks = env.observe()
        raw_masks['action_type_mask'][:] = False; raw_masks['action_type_mask'][:, 0] = True
        obs, masks = permute_hand(raw_obs, raw_masks, order)
        if progress:
            progress({'phase': 'feedback-calibration', 'trials_per_lane': calibration_trials})
        feedback, calibration = calibrate_feedback(agent, env, corpus, case_index,
            obs, masks, order, local.state.efficacy, learner=learner, trials=calibration_trials)
        if not np.array_equal(array(local.state.efficacy), initial):
            raise ValueError('feedback calibration changed weights')
        if progress:
            progress({'phase': 'training', 'through_step': 0, 'feedback_rule': asdict(feedback),
                      'calibration': calibration})
        initial_scores = evaluate(agent, obs, masks, local.state.efficacy, case_index, evaluation_repeats)
        if any(not (row[0] == row[1] == row[2]) for row in initial_scores['choices']):
            raise ValueError('identical initial lanes produced different choices')
        for step in range(steps):
            seed = derive_seed(SEED_VERSION, case_index, 'training', step)
            neural = agent.processor.process(obs, fly_seeds=[seed]*3, efficacy=local.state.efficacy)
            if not np.isfinite(neural.output_activity).all():
                raise ValueError('nonfinite training output')
            actions = agent.motor.decode(neural.output_activity, masks, deterministic=False,
                learner_ids=[learner]*3, decision_ids=[step]*3)
            pulses = score_choices(agent, env, corpus, case_index, actions, order)
            contrasted = float(feedback.signed(pulses[1]))
            capped += abs(feedback.scale*(pulses[1]-feedback.center)) >= feedback.max_pulse
            # Row two's observed native feedback is logged, but not applied.
            signed = np.asarray([pulses[0], contrasted, 0.], dtype=np.float32)
            gate = credit.mask(actions, masks)
            local.record_activity(neural.edge_pre_activity, neural.edge_post_activity, edge_credit=gate)
            events = local.apply_dopamine(np.maximum(signed, 0), np.maximum(-signed, 0))
            if any(e.lower_bound_hits or e.upper_bound_hits for e in events):
                raise ValueError('reward contrast reached an efficacy bound')
            local.state.reset_fast_traces(eligibility=True)
            pulse_sum += pulses; signed_sum += signed; absolute_sum += np.abs(signed)
            positive_count += signed > 0; negative_count += signed < 0; zero_count += signed == 0
            update_l1 += np.asarray([e.absolute_change for e in events])
            for row in range(3):
                hit = set(actions['cards'][row, :2].tolist()) == set(pair_slots(obs, row))
                correct[row] += hit; block_correct[row] += hit
            if (step+1) % 250 == 0 or step+1 == steps:
                n = 250 if (step+1) % 250 == 0 else (step+1) % 250
                block = {'phase': 'training', 'through_step': step+1, 'trials': n,
                    'pair_fraction': (block_correct/n).tolist(), 'native_pulse_sum': pulse_sum.tolist(),
                    'applied_signed_pulse_sum': signed_sum.tolist(),
                    'applied_absolute_pulse_sum': absolute_sum.tolist(),
                    'cumulative_actual_update_l1': update_l1.tolist()}
                blocks.append(block)
                if progress:
                    progress(block)
                block_correct.fill(0)
        final_scores = evaluate(agent, obs, masks, local.state.efficacy, case_index, evaluation_repeats)
        if any(a[2] != b[2] for a, b in zip(initial_scores['choices'], final_scores['choices'], strict=True)):
            raise ValueError('frozen individual evaluation choices changed')
        efficacy = array(local.state.efficacy)
        delta = efficacy-initial
        if np.any(delta[2]) or update_l1[2] != 0:
            raise ValueError('frozen efficacy changed')
        return {'version': VERSION, 'lane_order': list(LANES), 'learner': learner,
            'case_index': case_index, 'case_seed': corpus.cases[case_index]['seed'],
            'pair_rank': corpus.cases[case_index]['pair_rank'], 'display_order': order.tolist(),
            'steps_per_lane': steps, 'calibration': calibration,
            'unchanged_plasticity_config': asdict(local.config),
            'initial_evaluation': initial_scores, 'final_evaluation': final_scores,
            'training_pair_plays': correct.astype(int).tolist(), 'training_pair_fraction': (correct/steps).tolist(),
            'native_pulse_sum': pulse_sum.tolist(), 'applied_signed_pulse_sum': signed_sum.tolist(),
            'applied_absolute_pulse_sum': absolute_sum.tolist(),
            'applied_positive_pulse_counts': positive_count.tolist(),
            'applied_negative_pulse_counts': negative_count.tolist(),
            'applied_zero_pulse_counts': zero_count.tolist(), 'centered_pulse_cap_count': int(capped),
            'actual_update_l1': update_l1.tolist(),
            'mean_absolute_drift': np.abs(delta).mean(axis=1).tolist(),
            'mean_signed_drift': delta.mean(axis=1).tolist(),
            'max_absolute_drift': np.abs(delta).max(axis=1).tolist(),
            'increased_edge_fraction': (delta > 0).mean(axis=1).tolist(),
            'decreased_edge_fraction': (delta < 0).mean(axis=1).tolist(),
            'final_efficacy_min': efficacy.min(axis=1).tolist(),
            'final_efficacy_max': efficacy.max(axis=1).tolist(),
            'blocks': blocks, 'source_state_unchanged': True,
            'frozen_weights_and_individual_choices_unchanged': True,
            'feedback_calibration_weights_unchanged': True,
            'matching': 'same hand, display, initial weights, original references/LR, spike seeds, sampler and IDs; only centered lane feedback changes',
            'interpretation': 'one training hand and one initialization; fixed outcome-calibrated scalar reward transform, not a policy critic, biological DAN model or generalization result'}
    finally:
        env.restore_all(env_before)
        if plastic_hashes(source) != before:
            raise ValueError('reward comparison changed source plastic state')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('condition', 'checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--learner-index', type=int, default=0)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--evaluation-repeats', type=int, default=128)
    parser.add_argument('--calibration-trials', type=int, default=128)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    condition, corpus = load_condition(args.condition)
    if corpus is None:
        raise ValueError('a declared pair condition is required')
    config = PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(args.heavy)
    stack = build_condition_stack(config, condition, corpus)
    load_plastic_checkpoint(args.checkpoint, stack.trainer, expected_components=stack.components)
    env = BalatroSimAdapter(3, win_ante=1, allow_skip_blinds=False); env.reset([0]*3)
    paths = [args.condition, args.checkpoint, corpus.path, Path(__file__)]
    paths += [Path(__file__).with_name(name) for name in (
        'audit_card_credit_direction.py', 'compare_card_reference_scaling.py', 'probe_fixed_pair_learning.py')]
    hashes = {str(p.resolve()): sha256_file(p) for p in paths}
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    def progress(value):
        value = dict(value)
        calibration = value.pop('calibration', None)
        if calibration is not None:
            write_json(args.output_dir/'feedback-calibration.json', calibration)
        print(json.dumps(value), flush=True)
        write_json(args.output_dir/'status.json', {'status': 'running', 'source_hashes': hashes, **value})
    write_json(args.output_dir/'design.json', {'version': VERSION, 'source_hashes': hashes,
        'lane_order': list(LANES), 'steps_per_lane': args.steps,
        'calibration_trials_per_lane': args.calibration_trials, 'evaluation_repeats': args.evaluation_repeats,
        'learner': args.learner_index, 'case_index': args.case_index,
        'feedback_rule': 'center frozen native-pulse mean; match calibration mean absolute pulse via fixed scalar; cap at source max_total_pulse',
        'calibration_is_outcome_based': True, 'pair_labels_in_feedback': False,
        'unchanged_plasticity_config': asdict(stack.agent.plasticity.config),
        'unchanged_exploration': asdict(stack.agent.motor.exploration), 'seed_recipe': SEED_VERSION})
    try:
        report = compare(stack.agent, env, corpus, learner=args.learner_index, case_index=args.case_index,
            steps=args.steps, evaluation_repeats=args.evaluation_repeats,
            calibration_trials=args.calibration_trials, progress=progress)
        if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError('source file identity changed during comparison')
        report.update(source_hashes=hashes, pair_corpus_sha256=corpus.sha256,
                      status='completed', wall_seconds=time.perf_counter()-started)
        write_json(args.output_dir/'report.json', report)
        write_json(args.output_dir/'status.json', {'status': 'completed', 'wall_seconds': report['wall_seconds']})
    except BaseException as error:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'source_hashes': hashes,
                    'error': f'{type(error).__name__}: {error}'})
        raise
    print(json.dumps({'status': 'completed', 'report': str(args.output_dir/'report.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
