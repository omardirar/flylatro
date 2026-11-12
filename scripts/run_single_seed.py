"""Declare, execute and resume a long single-game local-plasticity experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from card_learning.common import identity, write_json
from card_learning.experiment import load_condition
from card_learning.motor import CardExploration
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint, load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.protocol_materialize import render_toml
from flylatro.seeds import SeedPlan
from single_seed import VERSION, build_stack, evaluate, sources


def read(path):
    return json.loads(Path(path).read_text())


def prepare(config_path, output, *, seed=4000, steps=1_000_000, replicates=3,
            episodes=64, arms=('normalised', 'original')):
    if min(steps, replicates, episodes) < 1:
        raise ValueError('budgets, replicates and evaluation count must be positive')
    if seed not in range(10_000_000):
        raise ValueError('game seed must be in the training stream [0, 10000000)')
    if not arms or len(set(arms)) != len(arms) or set(arms) - {'normalised', 'original'}:
        raise ValueError('declare normalised and/or original arms')
    base = PlasticExperimentConfig.load(config_path)
    if (base.environment.backend != 'balatro_sim' or base.environment.allow_skip_blinds
            or base.protocol.bound or base.plasticity.version != 'three-factor-choice-gated-v1'
            or base.training.reinforcement_mode != 'outcome' or base.training.action_schedule_path
            or base.training.reinforcement_schedule_path or base.motor.deterministic_training):
        raise ValueError('requires native, no-skip, stochastic, unbound choice-gated outcome configuration')
    absolute = lambda p: str(resolve_path(base, p)) if p else ''
    common = replace(base, environment=replace(base.environment, num_envs=replicates),
        fly=replace(base.fly, artifact_path=absolute(base.fly.artifact_path),
            motor_mapping_path=absolute(base.fly.motor_mapping_path)),
        calibration=replace(base.calibration, corpus_path=absolute(base.calibration.corpus_path)),
        curriculum=replace(base.curriculum, enabled=True, ladder=(1,)),
        training=replace(base.training, max_environment_decisions=steps * replicates,
            checkpoint_every_decisions=min(10_000, steps) * replicates,
            plasticity_enabled=True, condition='plastic_real', matched_source_run_dir='',
            budget_basis='Fixed single-seed delayed-learning budget; no performance-based early stopping'))
    inputs = [common.fly.motor_mapping_path, common.calibration.corpus_path]
    if common.fly.backend == 'flywire':
        inputs.append(common.fly.artifact_path)
    inputs = {p: sha256_file(Path(p)) for p in inputs if p}
    stages = sorted({n for n in (10_000, 30_000, 100_000, 300_000, 1_000_000, steps) if n <= steps})
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'version': VERSION, 'implementation': identity(), 'research_sources': sources(),
        'source_config': str(config_path.resolve()), 'source_config_sha256': sha256_file(config_path),
        'game_seed': seed, 'steps_per_learner': steps, 'replicates': replicates,
        'total_training_decisions': steps * replicates * len(arms),
        'stages': [0, *stages], 'win_ante': 1,
        'evaluation_seeds': list(SeedPlan().seeds('validation', episodes, offset=910000)),
        'fresh_unit_weights': True, 'performance_early_stopping': False,
        'safety_gates_enabled': True, 'final_test_seeds_used': False,
        'evaluation': 'greedy frozen repeated-seed and fixed unseen seeds at each milestone',
        'scope': 'complete Ante-1 games (three blinds); no helper plays or forced card count',
        'arms': {}}
    for label in arms:
        config = replace(common, name=f'single-seed-{seed}-{label}-v1')
        config.validate()
        folder = output / 'conditions' / label
        folder.mkdir(parents=True)
        path = folder / 'config.toml'
        doc = asdict(config); doc.pop('source_path')
        path.write_text(render_toml(doc))
        condition = {'version': 'card-learning-condition-v1', 'label': label,
            'config': str(path), 'config_file_sha256': sha256_file(path),
            'implementation': plan['implementation'], 'input_files_sha256': inputs,
            'exploration': asdict(CardExploration(.05)), 'pair_corpus': None,
            'pair_corpus_sha256': None, 'task': 'repeated-native-ante-one',
            'single_game_seed': seed, 'single_seed_arm': label,
            'research_sources': plan['research_sources']}
        path = folder / 'condition.json'; write_json(path, condition)
        plan['arms'][label] = {'condition': str(path), 'sha256': sha256_file(path)}
    write_json(output / 'single-seed-plan.json', plan)
    (output / 'single-seed-plan.sha256').write_text(sha256_file(output / 'single-seed-plan.json') + '\n')
    return plan


def verify(output):
    if sha256_file(output / 'single-seed-plan.json') != (output / 'single-seed-plan.sha256').read_text().strip():
        raise ValueError('declared single-seed plan changed')
    plan = read(output / 'single-seed-plan.json')
    if plan['version'] != VERSION or plan['implementation'] != identity() or plan['research_sources'] != sources():
        raise ValueError('declared experiment implementation changed')
    for arm in plan['arms'].values():
        path = Path(arm['condition'])
        if sha256_file(path) != arm['sha256']:
            raise ValueError('declared condition changed')
        load_condition(path)
    return plan


def worker(output, label, command, *, checkpoint=None, destination=None, stop=None, heavy=False, dashboard=True):
    plan = verify(output)
    condition, _ = load_condition(Path(plan['arms'][label]['condition']))
    config = PlasticExperimentConfig.load(Path(condition['config']))
    config.require_heavy_opt_in(heavy)
    if command == 'train':
        from flylatro.learning import cli
        original_factory, original_overrides = cli.build_plastic_stack, cli._apply_overrides
        cli.build_plastic_stack = lambda cfg, **kw: build_stack(cfg, condition, **kw)
        cli._apply_overrides = lambda cfg, args: cfg
        try:
            return cli.main(['--config', condition['config'], '--run-dir', str(destination),
                '--stop-after-decisions', str(stop), '--no-tensorboard', '--dashboard-refresh', '2',
                '--dashboard' if dashboard else '--no-dashboard',
                *(['--heavy'] if heavy else []), *(['--resume', str(checkpoint)] if checkpoint else [])])
        finally:
            cli.build_plastic_stack, cli._apply_overrides = original_factory, original_overrides
    stack = build_stack(config, condition)
    if checkpoint:
        load_plastic_checkpoint(Path(checkpoint), stack.trainer, expected_components=stack.components)
    report = {'version': VERSION, 'arm': label,
        'steps_per_learner': stack.trainer.state.vector_steps,
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': sha256_file(Path(checkpoint)) if checkpoint else None,
        'repeated_seed': evaluate(stack, [plan['game_seed']]),
        'unseen_seeds': evaluate(stack, plan['evaluation_seeds'])}
    write_json(destination, report)
    return report


def next_attempt(folder):
    folder.mkdir(parents=True, exist_ok=True)
    index = 0
    while (folder / f'attempt-{index:03d}').exists():
        index += 1
    return folder / f'attempt-{index:03d}'


def recover_checkpoint(folder, previous):
    """Only fully hash-verified generations qualify for automatic recovery."""
    latest, count = previous, -1
    for attempt in sorted(folder.glob('attempt-*')):
        if (attempt / 'safety-stop.json').exists():
            raise RuntimeError(f'safety stop retained: {attempt}')
        summary = attempt / 'run-summary.json'
        if summary.exists() and read(summary).get('safety_stop') is not None:
            raise RuntimeError(f'safety stop retained: {attempt}')
        for path in attempt.glob('plastic-checkpoint-*.pkl'):
            if not path.with_suffix('.pkl.manifest.json').exists():
                continue
            manifest = load_plastic_manifest(path)
            if manifest['environment_decisions'] > count:
                latest, count = str(path), manifest['environment_decisions']
    return latest


def execute(output, *, heavy=False, dashboard=True):
    plan = verify(output)
    # No two launchers may mutate the same experiment, including recovery.
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = output / 'single-seed-status.json'
        status = read(status_path) if status_path.exists() else {'completed': [], 'checkpoints': {}}
        def call(label, command, **kw):
            args = [sys.executable, str(Path(__file__).resolve()), '--output-dir', str(output),
                '--worker', command, '--arm', label]
            for key, value in kw.items():
                if value is not None:
                    args += ['--' + key.replace('_', '-'), str(value)]
            if heavy: args.append('--heavy')
            if not dashboard: args.append('--no-dashboard')
            subprocess.run(args, check=True)
        try:
            for stage in plan['stages']:
                for label in plan['arms']:
                    key = f'{label}:{stage}'
                    if key in status['completed']:
                        receipt = read(output / 'receipts' / f'{label}-{stage:09d}.json')
                        if sha256_file(Path(receipt['evaluation'])) != receipt['sha256']:
                            raise ValueError('completed evaluation changed')
                        continue
                    status.pop('error', None)
                    status.update(status='running', phase='training', arm=label, stage_per_learner=stage)
                    write_json(status_path, status)
                    checkpoint = status['checkpoints'].get(label)
                    if stage:
                        folder = output / 'training' / label / f'stage-{stage:09d}'
                        checkpoint = recover_checkpoint(folder, checkpoint)
                        if checkpoint is None or load_plastic_manifest(Path(checkpoint))['environment_decisions'] < stage * plan['replicates']:
                            attempt = next_attempt(folder)
                            print(f'{label}: training to {stage:,} decisions per learner on game seed {plan["game_seed"]}', flush=True)
                            call(label, 'train', destination=attempt, checkpoint=checkpoint, stop=stage * plan['replicates'])
                            summary = read(attempt / 'run-summary.json')
                            if (summary['safety_stop'] is not None or summary['environment_decisions'] != stage * plan['replicates']):
                                raise RuntimeError('stage stopped before budget; checkpoint retained')
                            checkpoint = summary['checkpoint']
                        status['checkpoints'][label] = checkpoint
                    status['phase'] = 'frozen evaluation'; write_json(status_path, status)
                    destination = output / 'evaluations' / f'{label}-{stage:09d}.json'
                    # Incomplete evals have no atomically committed JSON result.
                    if not destination.exists():
                        call(label, 'evaluate', destination=destination, checkpoint=checkpoint)
                    report = read(destination)
                    if report['arm'] != label or report['steps_per_learner'] != stage:
                        raise ValueError('evaluation does not match stage')
                    if checkpoint and report['checkpoint_sha256'] != sha256_file(Path(checkpoint)):
                        raise ValueError('evaluation checkpoint changed')
                    write_json(output / 'receipts' / f'{label}-{stage:09d}.json',
                        {'evaluation': str(destination), 'sha256': sha256_file(destination)})
                    status['completed'].append(key)
                    print(f'{label} at {stage:,}: repeated-seed first-blind clear={report["repeated_seed"]["first_blind_clear_fraction"]:.1%}, unseen={report["unseen_seeds"]["first_blind_clear_fraction"]:.1%}', flush=True)
                    write_json(status_path, status)
            status.update(status='completed', phase='completed')
        except BaseException as error:
            status.update(status='stopped', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            write_json(status_path, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('artefacts/learnable-motor-20261028-v2/config.toml'))
    parser.add_argument('--seed', type=int, default=4000)
    parser.add_argument('--steps-per-learner', type=int, default=1_000_000)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--evaluation-seeds', type=int, default=64)
    parser.add_argument('--arms', nargs='+', choices=('normalised', 'original'), default=['normalised', 'original'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--heavy', action='store_true')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--worker', choices=('train', 'evaluate'), help=argparse.SUPPRESS)
    parser.add_argument('--arm', choices=('normalised', 'original'), help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--destination', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--stop', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.worker:
        return worker(output, args.arm, args.worker, checkpoint=args.checkpoint,
            destination=args.destination, stop=args.stop, heavy=args.heavy, dashboard=not args.no_dashboard)
    if not args.prepare_only and not args.heavy:
        parser.error('execution requires --heavy; use --prepare-only to declare without training')
    if not output.exists():
        prepare(args.config.resolve(), output, seed=args.seed, steps=args.steps_per_learner,
            replicates=args.replicates, episodes=args.evaluation_seeds, arms=tuple(args.arms))
    else:
        plan = verify(output)
        expected = (args.seed, args.steps_per_learner, args.replicates, args.evaluation_seeds, args.arms)
        declared = (plan['game_seed'], plan['steps_per_learner'], plan['replicates'],
            len(plan['evaluation_seeds']), list(plan['arms']))
        if expected[:4] != declared[:4] or set(expected[4]) != set(declared[4]):
            raise ValueError('requested experiment differs from declaration; use its original command or a new directory')
    if not args.prepare_only:
        execute(output, heavy=args.heavy, dashboard=not args.no_dashboard)
    print(f'Experiment: {output}', flush=True)
    return 0


if __name__ == '__main__':
    main()
