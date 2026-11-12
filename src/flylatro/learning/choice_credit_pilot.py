"""Prepare and run a matched, bounded pilot of choice-gated eligibility.

Run GPU jobs serially. Each arm contains independent learner rows, with matched
training seeds and 10,000 decisions per row by default. Frozen comparisons use
validation seeds only; this descriptive pilot is not a confirmatory protocol.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from flylatro.evaluation.compare import MATCHED_FIELDS
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.plasticity import GLOBAL_PLASTICITY_VERSION, CHOICE_GATED_PLASTICITY_VERSION, PlasticityConfig
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.protocol_materialize import render_toml
from flylatro.seeds import SeedPlan


def implementation_hash() -> str:
    digest = hashlib.sha256()
    source = Path(__file__).resolve().parents[1]
    for path in sorted(source.rglob('*')):
        if path.suffix in {'.py', '.cu', '.cpp'}:
            digest.update(str(path.relative_to(source)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def validate_pair(baseline: PlasticExperimentConfig, candidate: PlasticExperimentConfig) -> None:
    if baseline.plasticity.version != GLOBAL_PLASTICITY_VERSION or candidate.plasticity.version != CHOICE_GATED_PLASTICITY_VERSION:
        raise ValueError('pilot requires the global baseline and explicit choice-gated candidate')
    left, right = asdict(baseline), asdict(candidate)
    for document in (left, right):
        document.pop('name')
        document.pop('source_path')
        document['plasticity'].pop('version')
    if left != right:
        raise ValueError('pilot arms must differ only in rule version and experiment name')


def prepare(base: PlasticExperimentConfig, output: Path, *, steps: int, replicates: int,
            evaluation_episodes: int, evaluation_batch_size: int) -> dict:
    if min(steps, replicates, evaluation_episodes, evaluation_batch_size) < 1:
        raise ValueError('pilot sizes must be positive')
    if base.protocol.bound:
        raise ValueError('choice-credit pilot needs its own unbound exploratory configuration')
    if base.environment.allow_skip_blinds or base.reinforcement.run_failure_pulse != .01:
        raise ValueError('this pilot keeps the no-skip, failure-pulse-0.01 baseline')
    output.mkdir(parents=True, exist_ok=False)
    configs = output / 'configs'
    configs.mkdir()
    common = replace(base,
        environment=replace(base.environment, num_envs=replicates),
        fly=replace(base.fly, artifact_path=str(resolve_path(base, base.fly.artifact_path)),
            motor_mapping_path=str(resolve_path(base, base.fly.motor_mapping_path)) if base.fly.motor_mapping_path else ''),
        calibration=replace(base.calibration, corpus_path=str(resolve_path(base, base.calibration.corpus_path))
            if base.calibration.corpus_path else ''),
        training=replace(base.training, max_environment_decisions=steps * replicates,
            checkpoint_every_decisions=min(steps, 1000) * replicates,
            budget_basis='Predeclared choice-credit pilot; matched independent seeds, equal per-learner exposure'))
    arms = {}
    configurations = {}
    for label, version in [('baseline', GLOBAL_PLASTICITY_VERSION), ('choice', CHOICE_GATED_PLASTICITY_VERSION)]:
        config = replace(common, name=f'exploratory-choice-credit-{label}-v1',
            plasticity=replace(common.plasticity, version=version))
        config.validate()
        path = configs / (label + '.toml')
        document = asdict(config)
        document.pop('source_path')
        with path.open('x') as stream:
            stream.write(render_toml(document))
        loaded = PlasticExperimentConfig.load(path)
        configurations[label] = loaded
        arms[label] = {'config': str(path.resolve()), 'config_sha256': loaded.sha256,
            'config_file_sha256': sha256_file(path), 'rule_version': version}
    validate_pair(configurations['baseline'], configurations['choice'])
    declaration = {'version': 'choice-credit-pilot-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'implementation_sha256': implementation_hash(),
        'source_config': str(base.source_path.resolve()), 'source_config_sha256': base.sha256,
        'steps_per_learner': steps, 'replicates': replicates,
        'training_seeds': list(SeedPlan().seeds('training', replicates, offset=common.training.training_seed_offset)),
        'base_fly_seed': common.training.base_fly_seed,
        'learner_semantics': 'independent weights, eligibility, environments and RNG; no experience sharing',
        'evaluation_episodes': evaluation_episodes, 'evaluation_batch_size': evaluation_batch_size,
        'evaluation_seeds': list(SeedPlan().seeds('validation', evaluation_episodes)),
        'evaluation_win_ante': 1, 'arms': arms,
        'scale_up_rule': 'all budgets complete without safety stops; more blind clears in a majority of matched replicates and in total; total wins do not decrease',
        'interpretation': 'descriptive engineering pilot; three training replicates, not hundreds of independent training replicates',
        'final_test_seeds_used': False}
    with (output / 'declaration.json').open('x') as stream:
        json.dump(declaration, stream, indent=2)
    return declaration


def compare_replicate(baseline: dict, candidate: dict, index: int) -> dict:
    for report, version in [(baseline, GLOBAL_PLASTICITY_VERSION), (candidate, CHOICE_GATED_PLASTICITY_VERSION)]:
        if (report.get('learning_disabled') is not True or report.get('training_action_schedule_replayed') is not False
                or report.get('evaluation_weight_source') != 'trained-checkpoint'
                or report.get('action_policy') != 'independent-fixed-motor-choices-no-training-action-replay'):
            raise ValueError('pilot comparison requires independently acting frozen trained weights')
        if report['components']['plasticity_rule']['version'] != version:
            raise ValueError('pilot rule identity differs')
        if PlasticityConfig(**report['components']['plasticity_rule']).sha256 != report['components']['plasticity_rule_sha256']:
            raise ValueError('pilot plasticity identity hash differs from its parameters')
        if report.get('selected_learner_index') != index:
            raise ValueError('pilot learner row differs')
        if report['metrics']['evaluation/mean_skipped_blinds'] != 0:
            raise ValueError('pilot evaluation violated the no-skip rule')
    for key in ('seeds', 'seed_stream', 'win_ante', 'evaluation_batch_size', 'source_checkpoint_learners'):
        if baseline[key] != candidate[key]:
            raise ValueError(f'pilot evaluation {key} differs')
    if baseline['seed_stream'] != 'validation' or baseline['win_ante'] != 1:
        raise ValueError('pilot must retain validation seeds and Ante-1 target')
    for key in set(MATCHED_FIELDS) - {'plasticity_rule_sha256'}:
        if baseline['components'][key] != candidate['components'][key]:
            raise ValueError(f'pilot model field {key} differs')
    for key in ('action_constraints', 'motor_exploration', 'training_seeds', 'exposure_budget_decisions'):
        if baseline['components'].get(key) != candidate['components'].get(key):
            raise ValueError(f'pilot component {key} differs')
    if baseline['components'].get('action_constraints') != {'allow_skip_blinds': False}:
        raise ValueError('pilot requires explicit no-skip action identity')
    rules = [dict(report['components']['plasticity_rule']) for report in (baseline, candidate)]
    for rule in rules:
        rule.pop('version')
    if rules[0] != rules[1]:
        raise ValueError('pilot changed plasticity parameters other than credit assignment')
    if candidate['components'].get('credit_assignment', {}).get('version') != CHOICE_GATED_PLASTICITY_VERSION:
        raise ValueError('pilot candidate lacks credit-assignment provenance')
    seeds = baseline['seeds']
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('pilot validation seeds must be unique')
    for report in (baseline, candidate):
        if [row['seed'] for row in report['episodes']] != seeds:
            raise ValueError('pilot episode rows differ from declared seeds')
    deltas = []
    for left, right in zip(baseline['episodes'], candidate['episodes'], strict=True):
        if left['balatro_seed'] != right['balatro_seed']:
            raise ValueError('pilot simulator run seeds differ')
        delta = right['episode_return'] - left['episode_return']
        if not np.isfinite(delta):
            raise ValueError('pilot return must be finite')
        deltas.append(delta)
    def outcomes(report):
        return {'blinds_beaten': int(sum(row['blinds_beaten'] for row in report['episodes'])),
            'wins': int(sum(row['won'] for row in report['episodes'])),
            'mean_return': float(np.mean([row['episode_return'] for row in report['episodes']]))}
    return {'learner_index': index, 'episodes': len(seeds), 'baseline': outcomes(baseline),
        'choice': outcomes(candidate), 'mean_paired_return_delta': float(np.mean(deltas))}


def assess(output: Path, declaration: dict) -> dict:
    if implementation_hash() != declaration['implementation_sha256']:
        raise ValueError('pilot implementation changed during execution')
    for arm in declaration['arms'].values():
        if sha256_file(Path(arm['config'])) != arm['config_file_sha256']:
            raise ValueError('pilot config changed during execution')
    summaries = {arm: json.loads((output / arm / 'run-summary.json').read_text()) for arm in ('baseline', 'choice')}
    replicas = [compare_replicate(
        json.loads((output / f'baseline-eval-{i}/summary.json').read_text()),
        json.loads((output / f'choice-eval-{i}/summary.json').read_text()), i)
        for i in range(declaration['replicates'])]
    complete = all(summary['status'] == 'completed' and summary['safety_stop'] is None
        and summary['environment_decisions'] == declaration['steps_per_learner'] * declaration['replicates']
        for summary in summaries.values())
    totals = {arm: {key: sum(row[arm][key] for row in replicas) for key in ('blinds_beaten', 'wins')}
        for arm in summaries}
    improved = sum(row['choice']['blinds_beaten'] > row['baseline']['blinds_beaten'] for row in replicas)
    scale = (complete and improved > declaration['replicates'] / 2
        and totals['choice']['blinds_beaten'] > totals['baseline']['blinds_beaten']
        and totals['choice']['wins'] >= totals['baseline']['wins'])
    return {'version': 'choice-credit-pilot-assessment-v1', 'comparability': 'PASS',
        'all_training_budgets_complete_without_safety_stop': complete,
        'matched_training_replicates': declaration['replicates'], 'replicates': replicas,
        'held_out_totals': totals, 'replicates_with_more_blind_clears': improved,
        'predeclared_scale_up_signal': bool(scale),
        'interpretation': 'pilot signal only; no confirmatory significance or improved-learning claim from comparability',
        'declaration_sha256': sha256_file(output / 'declaration.json')}


def run(output: Path, declaration: dict) -> None:
    state = {'status': 'running', 'jobs': []}
    def save():
        path = output / 'suite.tmp'
        path.write_text(json.dumps(state, indent=2) + '\n')
        path.replace(output / 'suite.json')
    def execute(name, command, accepted=(0,)):
        row = {'name': name, 'command': list(map(str, command)), 'status': 'running'}
        state['jobs'].append(row)
        save()
        start = time.perf_counter()
        with (output / (name + '.stdout')).open('x') as stdout, (output / (name + '.stderr')).open('x') as stderr:
            process = subprocess.Popen(row['command'], stdout=stdout, stderr=stderr)
            row['pid'] = process.pid
            save()
            row['returncode'] = process.wait()
        row['wall_seconds'] = time.perf_counter() - start
        row['status'] = 'passed' if row['returncode'] == 0 else 'recorded_safety_stop' if row['returncode'] in accepted else 'failed'
        save()
        print(name, row['status'], flush=True)
        if row['returncode'] not in accepted:
            raise RuntimeError(name + ' failed; inspect captured output')
    def module(name, *args):
        return [sys.executable, '-m', name, *map(str, args)]
    try:
        width = declaration['replicates']
        execute('choice-resume-qualification', module('flylatro.analysis.resume_cli',
            '--config', declaration['arms']['choice']['config'], '--checkpoint', output / 'resume-probe.pkl',
            '--prefix-decisions', 8 * width, '--future-decisions', 8 * width,
            '--output', output / 'resume-report.json', '--heavy'))
        for arm in ('baseline', 'choice'):
            config = declaration['arms'][arm]['config']
            execute(arm + '-training', module('flylatro.learning.cli', '--config', config,
                '--run-dir', output / arm, '--heavy', '--dashboard', '--dashboard-refresh', '2', '--no-tensorboard'), (0, 2))
        for arm in ('baseline', 'choice'):
            config = declaration['arms'][arm]['config']
            summary = json.loads((output / arm / 'run-summary.json').read_text())
            for index in range(width):
                execute(f'{arm}-evaluation-{index}', module('flylatro.evaluation.plastic_cli',
                    '--config', config, '--checkpoint', summary['checkpoint'], '--learner-index', index,
                    '--episodes', declaration['evaluation_episodes'], '--batch-size', declaration['evaluation_batch_size'],
                    '--seed-stream', 'validation', '--win-ante', '1', '--max-vector-steps', '1000',
                    '--output-dir', output / f'{arm}-eval-{index}', '--heavy'))
        execute('checkpoint-log-audit', [sys.executable, 'scripts/verify_cli_artifacts.py',
            '--run-dir', str(output / 'baseline'), '--run-dir', str(output / 'choice'),
            '--output', str(output / 'checkpoint-audit.json')])
        report = assess(output, declaration)
        with (output / 'assessment.json').open('x') as stream:
            json.dump(report, stream, indent=2)
        state['status'] = 'completed'
    except BaseException as error:
        state['status'] = 'failed'
        state['error'] = str(error)
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/exploratory-event-no-skip-weak-failure-v1.toml'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps-per-learner', type=int, default=10000)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--evaluation-episodes', type=int, default=256)
    parser.add_argument('--evaluation-batch-size', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    base = PlasticExperimentConfig.load(args.config)
    if not args.prepare_only:
        base.require_heavy_opt_in(args.heavy)
    declaration = prepare(base, args.output_dir.resolve(), steps=args.steps_per_learner,
        replicates=args.replicates, evaluation_episodes=args.evaluation_episodes,
        evaluation_batch_size=args.evaluation_batch_size)
    if not args.prepare_only:
        run(args.output_dir.resolve(), declaration)
    print(json.dumps({'output_dir': str(args.output_dir), 'prepared_only': args.prepare_only}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
