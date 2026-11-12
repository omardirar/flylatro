"""Fresh native membership learning on the qualified active-input motor mapping.

Keeps historical source/checkpoint identities intact; reuses the exact native
trial, credit, reward, saved-state audit and referee implementations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from assess_card_membership import replay, summarize
from assess_training_bottlenecks import verify_state
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from card_learning.motor import CardExploration, install_motor
from card_learning.pairs import PairCorpus, build_corpus
from confirm_card_membership import evaluate
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack, resolve_path
from run_card_membership import native_probe

VERSION = 'repaired-native-membership-v1'
QUALIFICATION = Path('artefacts/learnable-motor-20261028-v2')
LEGACY = Path('artefacts/card-membership-20261027-v2/native')
CONDITION = Path('runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json')
CHECKPOINT = Path('runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl')


def verify_sources(sources):
    for path, digest in sources.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('declared source changed: '+path)


def fresh_stack(config, exploration, expected_mapping):
    artifact = MotorMapping.load(resolve_path(config, config.fly.motor_mapping_path))
    if artifact.sha256 != expected_mapping:
        raise ValueError('fresh stack mapping differs from qualified declaration')
    stack = install_motor(build_plastic_stack(config), CardExploration(**exploration))
    agent = stack.agent
    if agent.motor.mapping.structure_sha256 != artifact.structure_sha256 or not agent.motor.mapping.routing.action_conditioned:
        raise ValueError('fresh stack mapping differs from qualified declaration')
    if agent.plasticity.config.dopamine_decay != 0:
        raise ValueError('decision-local dopamine required')
    state = agent.plasticity.state.state_dict()
    if np.any(state['efficacy'] != 1) or np.any(state['initial_efficacy'] != 1):
        raise ValueError('new mapping must start with fresh unit weights')
    if config.environment.allow_skip_blinds:
        raise ValueError('native comparison requires disabled blind skipping')
    return stack


def prepare(folder, steps=2000):
    if steps < 1:
        raise ValueError('positive training budget required')
    if folder.exists():
        raise FileExistsError(folder)
    qualified = json.loads((QUALIFICATION/'assessment.json').read_text())
    if qualified['status'] != 'completed' or not all(qualified['checks'].values()):
        raise ValueError('completed mapping qualification required')
    verify_sources(qualified['artifact_sha256'])
    old = json.loads((LEGACY/'report.json').read_text())
    old_assessment = json.loads((LEGACY/'assessment.json').read_text())
    if old['status'] != 'completed' or old['steps'] != steps or old['count'] != 2:
        raise ValueError('reuse only a matched completed legacy native study')
    if (old_assessment['source_report_sha256'] != sha256_file(LEGACY/'report.json')
            or old_assessment['final_state_sha256'] != sha256_file(LEGACY/'diagnostic-final-state.npz')):
        raise ValueError('legacy assessment/state identity differs')
    verify_sources(old['source_hashes'])
    verify_state({**old, 'study': 'followup'}, LEGACY/'diagnostic-final-state.npz')
    condition, _ = load_condition(CONDITION)
    folder.mkdir(parents=True)
    # New game roots are fixed before inspecting any new learning outcome.
    build_corpus(folder/'fresh-pairs.json', train_per_rank=1, validation_per_rank=8,
                 training_offset=17500, validation_offset=420000, max_attempts=5000)
    fresh = PairCorpus(folder/'fresh-pairs.json')
    old_corpus = PairCorpus(Path(old['corpus']))
    previous_confirmation = json.loads(Path('artefacts/card-membership-20261027-v2/fresh-confirmation/report.json').read_text())
    prior = PairCorpus(Path(previous_confirmation['corpus']))
    fresh_seeds = {c['seed'] for c in fresh.cases}
    if fresh_seeds & {c['seed'] for c in old_corpus.cases+prior.cases}:
        raise ValueError('fresh confirmation overlaps previous study')
    scripts = ['run_repaired_native_membership.py', 'run_card_membership.py', 'confirm_card_membership.py',
               'assess_card_membership.py', 'assess_training_bottlenecks.py', 'train_discard_play_curriculum.py']
    paths = [QUALIFICATION/'config.toml', QUALIFICATION/'candidate-motor.json', QUALIFICATION/'assessment.json',
             LEGACY/'report.json', LEGACY/'assessment.json', LEGACY/'diagnostic-final-state.npz',
             CONDITION, CHECKPOINT, Path(old['corpus']), folder/'fresh-pairs.json', Path(condition['config'])]
    paths += [Path(__file__).with_name(name) for name in scripts]
    sources = {str(path.resolve()): sha256_file(path) for path in paths}
    write_json(folder/'declaration.json', {'version': VERSION, 'source_hashes': sources,
        'config': str((QUALIFICATION/'config.toml').resolve()), 'mapping_sha256': json.loads((QUALIFICATION/'status.json').read_text())['candidate_sha256'],
        'exploration': condition['exploration'], 'training_corpus': old['corpus'],
        'fresh_corpus': str(fresh.path), 'steps': steps, 'replicates': 3, 'count': 2,
        'calibration_trials': 128, 'screen_repeats': 2, 'fresh_repeats': 2,
        'source_checkpoint_loaded_under_new_mapping': False,
        'design': 'neural native / reference native / training-only initial-update-matched reference / shared frozen; fixed count2; only DISCARD membership learns',
        'comparison': 'same training corpus, seeds, rewards and native trial implementation as completed legacy study; fresh states for new mapping',
        'fresh_validation': '104 new hands, disjoint previous screen and confirmation; both mappings, all feedback arms, own shared initial controls; greedy; no selection',
        'final_test_used': False,
        'readiness_rule': {'minimum_native_progress_gain': .005, 'minimum_positive_learners': 2,
            'conditional_case_interval_lower_bound_positive': True,
            'primary': 'new mapping neural-native feedback evaluated with actual frozen neural PLAY on fresh hands',
            'if_pass': 'advance to bounded free-action Ante-1 pilot; long training still requires native full-game validation against initial/frozen controls',
            'if_fail': 'do not scale this setup; diagnose neural PLAY or feedback/membership signal before longer training'}})
    write_json(folder/'status.json', {'status': 'prepared'})


def advance_gate(summary, rule):
    primary = summary['neural-native']['weak']
    positives = int(np.count_nonzero(np.asarray(primary['change_per_learner']) > 0))
    interval = primary['case_bootstrap_95_interval_conditional_on_these_learners']
    checks = {'meaningful_native_progress_gain': primary['mean_change'] >= rule['minimum_native_progress_gain'],
        'positive_in_multiple_learners': positives >= rule['minimum_positive_learners'],
        'conditional_case_interval_above_zero': interval[0] > 0}
    return {'advance_to_free_action_pilot': all(checks.values()), 'checks': checks,
        'native_progress_change': primary['mean_change'], 'positive_learners': positives,
        'conditional_case_interval': interval, 'long_training_ready': False,
        'meaning': 'this diagnostic can qualify a bounded free-action pilot; full-game evidence is required for long training'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'train', 'confirm-old', 'confirm-new', 'report'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--heavy', action='store_true'); args = p.parse_args(argv)
    folder = args.output_dir.resolve()
    if args.stage == 'prepare':
        prepare(folder); return 0
    declaration = json.loads((folder/'declaration.json').read_text())
    verify_sources(declaration['source_hashes'])
    def progress(row):
        write_json(folder/'status.json', {'status': 'running', 'stage': args.stage, **row})
        print(json.dumps({'stage': args.stage, **row}), flush=True)
    started = time.perf_counter()
    stage_dir = folder/('native' if args.stage == 'train' else args.stage)
    if args.stage != 'report':
        if stage_dir.exists():
            raise FileExistsError(stage_dir)
        stage_dir.mkdir()
    try:
        if args.stage == 'train':
            config = PlasticExperimentConfig.load(Path(declaration['config'])); config.require_heavy_opt_in(args.heavy)
            stack = fresh_stack(config, declaration['exploration'], declaration['mapping_sha256'])
            corpus = PairCorpus(Path(declaration['training_corpus']))
            progress({'phase': 'fresh-unit-weight-training'})
            with (stage_dir/'training-trials.jsonl').open('w') as stream:
                report, state = native_probe(stack.agent, corpus, steps=declaration['steps'],
                    replicates=declaration['replicates'], count=declaration['count'], repeats=declaration['screen_repeats'],
                    calibration_trials=declaration['calibration_trials'], progress=progress,
                    log=lambda row: stream.write(json.dumps(row, sort_keys=True)+'\n'))
            verify_sources(declaration['source_hashes'])
            np.savez_compressed(stage_dir/'diagnostic-final-state.npz', **state)
            report.update(status='completed', source_hashes=declaration['source_hashes'],
                corpus=str(corpus.path), corpus_sha256=corpus.sha256, wall_seconds=time.perf_counter()-started,
                source_checkpoint_loaded=False, fresh_unit_weights=True, mapping_sha256=stack.agent.motor.mapping.sha256)
            write_json(stage_dir/'report.json', report)
        elif args.stage.startswith('confirm-'):
            old = args.stage == 'confirm-old'
            source_dir = LEGACY if old else folder/'native'
            source_report = json.loads((source_dir/'report.json').read_text())
            verify_sources(source_report['source_hashes'])
            state_path = source_dir/'diagnostic-final-state.npz'
            verify_state({**source_report, 'study': 'followup'}, state_path)
            assessment = json.loads((source_dir/'assessment.json').read_text())
            if assessment['source_report_sha256'] != sha256_file(source_dir/'report.json') or assessment['final_state_sha256'] != sha256_file(state_path):
                raise ValueError('confirmation screen assessment differs')
            if old:
                condition, _ = load_condition(CONDITION)
                config = PlasticExperimentConfig.load(Path(condition['config'])); config.require_heavy_opt_in(args.heavy)
                stack = build_condition_stack(config, condition)
                load_plastic_checkpoint(CHECKPOINT, stack.trainer, expected_components=stack.components)
            else:
                config = PlasticExperimentConfig.load(Path(declaration['config'])); config.require_heavy_opt_in(args.heavy)
                stack = fresh_stack(config, declaration['exploration'], declaration['mapping_sha256'])
            corpus = PairCorpus(Path(declaration['fresh_corpus']))
            with np.load(state_path, allow_pickle=False) as data:
                state = {k: data[k].copy() for k in data.files}
            report = evaluate(stack.agent, corpus, state, replicates=declaration['replicates'],
                count=declaration['count'], repeats=declaration['fresh_repeats'], progress=progress)
            report['replay'] = replay(report)
            verify_sources(declaration['source_hashes'])
            report.update(source_hashes=declaration['source_hashes'], source_report_sha256=sha256_file(source_dir/'report.json'),
                final_state_sha256=sha256_file(state_path), mapping_sha256=stack.agent.motor.mapping.sha256,
                wall_seconds=time.perf_counter()-started)
            write_json(stage_dir/'report.json', report)
        else:
            old = json.loads((folder/'confirm-old/report.json').read_text())
            new = json.loads((folder/'confirm-new/report.json').read_text())
            if old['corpus_sha256'] != new['corpus_sha256']:
                raise ValueError('old/new confirmation cases differ')
            comparison = {arm: {policy: new['summary'][arm][policy]['mean_change']-old['summary'][arm][policy]['mean_change']
                for policy in ('weak', 'reference')} for arm in new['summary']}
            gate = advance_gate(new['summary'], declaration['readiness_rule'])
            paths = [folder/'declaration.json', folder/'native/report.json', folder/'native/assessment.json',
                folder/'native/diagnostic-final-state.npz', folder/'confirm-old/report.json', folder/'confirm-new/report.json']
            report = {'version': VERSION, 'status': 'completed', 'old': old['summary'], 'new': new['summary'],
                'difference_of_changes_new_minus_old': comparison, 'readiness': gate,
                'replay': {'new_screen': json.loads((folder/'native/assessment.json').read_text())['replay'],
                           'old_fresh': old['replay'], 'new_fresh': new['replay']},
                'source_hashes': declaration['source_hashes'],
                'artifact_sha256': {str(path.resolve()): sha256_file(path) for path in paths},
                'limits': 'three learner seeds on one mapping, constrained opening pair hands, fixed DISCARD timing/count; conditional case intervals; reference is diagnostic; no full-game competence claim'}
            write_json(folder/'report.json', report)
        write_json(folder/'status.json', {'status': 'completed' if args.stage == 'report' else 'stage-completed', 'stage': args.stage})
    except BaseException as error:
        write_json(folder/'status.json', {'status': 'failed', 'stage': args.stage, 'error': f'{type(error).__name__}: {error}'})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
