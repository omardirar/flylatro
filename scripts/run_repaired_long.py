"""Fixed-budget native local-plasticity run with staged frozen evaluations.

Uses the historical condition worker unchanged. Each training attempt has its
own logs, so resuming never appends a replayed tail to an earlier audit log.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

from card_learning.common import identity, write_json
from card_learning.experiment import load_condition, prepare as prepare_conditions
from flylatro.evaluation.compare import compare_frozen_evaluations
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml
from flylatro.seeds import SeedPlan


ARMS = ('subset-plastic', 'subset-frozen')
STAGES = (10_000, 30_000, 100_000, 300_000, 1_000_000)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(source, output, *, stages=STAGES, replicates=3, episodes=256,
            batch=4, evaluation_offset=850_000):
    stages = tuple(stages)
    if not stages or stages != tuple(sorted(set(stages))) or stages[0] < 1:
        raise ValueError('stages must be strictly increasing positive counts')
    seeds = SeedPlan().seeds('validation', episodes, offset=evaluation_offset)
    declaration = prepare_conditions(source, output, steps=stages[-1],
        replicates=replicates, episodes=episodes, batch=batch, probability=.05)
    # Set checkpoint cadence before sealing this experiment; historical source
    # configs and harness files are never edited.
    for arm in declaration['arms'].values():
        path = Path(arm['condition'])
        condition = read(path)
        config = PlasticExperimentConfig.load(Path(condition['config']))
        config = replace(config, training=replace(config.training,
            checkpoint_every_decisions=min(10_000, stages[0]) * replicates,
            budget_basis='Fixed delayed-learning budget; no performance-based early stopping'))
        document = asdict(config)
        document.pop('source_path')
        Path(condition['config']).write_text(render_toml(document))
        condition['config_file_sha256'] = sha256_file(Path(condition['config']))
        write_json(path, condition)
        arm['condition_sha256'] = sha256_file(path)
    declaration.update(evaluation_seed_offset=evaluation_offset, evaluation_seeds=list(seeds))
    write_json(output / 'declaration.json', declaration)
    plan = {'version': 'repaired-native-long-v1', 'implementation': identity(),
        'launcher_sha256': sha256_file(Path(__file__)),
        'declaration_sha256': sha256_file(output / 'declaration.json'),
        'selected_arms': list(ARMS), 'stages_per_learner': list(stages),
        'replicates': replicates, 'total_training_decisions': 2 * replicates * stages[-1],
        'fresh_unit_weights': True, 'performance_early_stopping': False,
        'safety_gates_enabled': True, 'final_test_seeds_used': False,
        'interpretation': 'Exploratory delayed-learning test; a flat finite run cannot rule out later learning'}
    write_json(output / 'long-plan.json', plan)
    return plan


def next_attempt(folder):
    folder.mkdir(parents=True, exist_ok=True)
    index = 0
    while (folder / f'attempt-{index:03d}').exists():
        index += 1
    return folder / f'attempt-{index:03d}'


def execute(output, *, dashboard=True):
    output = output.resolve()
    plan = read(output / 'long-plan.json')
    if (plan['implementation'] != identity()
            or plan['launcher_sha256'] != sha256_file(Path(__file__))
            or plan['declaration_sha256'] != sha256_file(output / 'declaration.json')):
        raise ValueError('declared implementation or plan inputs changed')
    declaration = read(output / 'declaration.json')
    for label in ARMS:
        arm = declaration['arms'][label]
        if sha256_file(Path(arm['condition'])) != arm['condition_sha256']:
            raise ValueError('declared condition changed')
        load_condition(Path(arm['condition']))
    launcher = Path(__file__).with_name('run_card_learning.py')
    status_path = output / 'long-status.json'
    status = read(status_path) if status_path.exists() else {'completed_stages': [], 'checkpoints': {}}

    def call(label, command, *args):
        subprocess.run([sys.executable, str(launcher), command, '--condition',
            declaration['arms'][label]['condition'], *map(str, args)], check=True)

    def evaluate(label, checkpoint, learner, folder, *, initial=False):
        receipt = folder / 'complete.json'
        if receipt.exists():
            saved = read(receipt)
            if sha256_file(Path(saved['summary'])) != saved['sha256']:
                raise ValueError('saved evaluation changed')
            return read(saved['summary'])
        destination = next_attempt(folder)
        call(label, 'evaluate', '--checkpoint', checkpoint, '--learner-index', learner,
            '--episodes', declaration['evaluation_episodes'],
            '--batch-size', declaration['evaluation_batch_size'],
            '--seed-stream', 'validation', '--seed-offset', declaration['evaluation_seed_offset'],
            '--win-ante', 1, '--max-vector-steps', 1000, '--output-dir', destination,
            '--heavy', *(['--initial-weights'] if initial else []))
        summary = destination / 'summary.json'
        write_json(receipt, {'summary': str(summary), 'sha256': sha256_file(summary)})
        return read(summary)

    try:
        for stage in plan['stages_per_learner']:
            if stage in status['completed_stages']:
                continue
            reports = {}
            for label in ARMS:
                status.update(status='running', stage_per_learner=stage, arm=label, phase='training')
                write_json(status_path, status)
                folder = output / 'training' / label / f'stage-{stage:07d}'
                receipt = folder / 'complete.json'
                if receipt.exists():
                    checkpoint = read(receipt)['checkpoint']
                    load_plastic_manifest(Path(checkpoint))
                else:
                    checkpoint = status['checkpoints'].get(label)
                    # Recover a clean interruption, or the newest fully written
                    # periodic checkpoint after an abrupt process termination.
                    for attempt in sorted(folder.glob('attempt-*')):
                        if (attempt / 'safety-stop.json').exists():
                            raise RuntimeError('safety stop retained; inspect before continuing')
                        summary_path = attempt / 'run-summary.json'
                        if summary_path.exists():
                            summary = read(summary_path)
                            if summary['safety_stop'] is not None:
                                raise RuntimeError('safety stop retained; inspect before continuing')
                            checkpoint = summary['checkpoint']
                        else:
                            candidates = sorted(attempt.glob('plastic-checkpoint-*.pkl'))
                            for candidate in candidates:
                                if candidate.with_suffix('.pkl.manifest.json').exists():
                                    load_plastic_manifest(candidate)
                                    checkpoint = str(candidate)
                    attempt = next_attempt(folder)
                    print(f'{label}: training to {stage:,} decisions per learner', flush=True)
                    call(label, 'train', '--run-dir', attempt,
                        '--stop-after-decisions', stage * plan['replicates'],
                        '--dashboard' if dashboard else '--no-dashboard', '--dashboard-refresh', 2,
                        '--no-tensorboard', '--heavy', *(['--resume', checkpoint] if checkpoint else []))
                    summary = read(attempt / 'run-summary.json')
                    if (summary['safety_stop'] is not None
                            or summary['status'] not in {'stage_complete', 'completed'}
                            or summary['environment_decisions'] != stage * plan['replicates']):
                        raise RuntimeError('stage interrupted; checkpoint retained; rerun the same command to resume')
                    checkpoint = summary['checkpoint']
                    write_json(receipt, {'checkpoint': checkpoint})
                status['checkpoints'][label] = checkpoint
                status.update(phase='evaluation')
                write_json(status_path, status)
                rows = []
                for learner in range(plan['replicates']):
                    initial = evaluate(label, checkpoint, learner,
                        output / 'evaluations' / label / f'initial-{learner}', initial=True)
                    trained = evaluate(label, checkpoint, learner,
                        output / 'evaluations' / label / f'{stage:07d}-{learner}')
                    rows.append({'learner': learner,
                        'blind_clears': sum(r['blinds_beaten'] for r in trained['episodes']),
                        'ante1_wins': sum(int(r['won']) for r in trained['episodes']),
                        'comparison_to_own_initial': compare_frozen_evaluations(trained, initial)})
                reports[label] = rows
            for learner in range(plan['replicates']):
                plastic = read(read(output / 'evaluations' / ARMS[0] / f'{stage:07d}-{learner}' / 'complete.json')['summary'])
                frozen = read(read(output / 'evaluations' / ARMS[1] / f'{stage:07d}-{learner}' / 'complete.json')['summary'])
                reports[ARMS[0]][learner]['comparison_to_frozen'] = compare_frozen_evaluations(plastic, frozen)
            write_json(output / f'learning-curve-{stage:07d}.json', {'steps_per_learner': stage, 'arms': reports})
            status['completed_stages'].append(stage)
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
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.heavy:
        parser.error('GPU execution requires --heavy')
    output = args.output_dir.resolve()
    if not output.exists():
        prepare(args.config.resolve(), output)
    elif not (output / 'long-plan.json').exists():
        raise ValueError('destination exists without a long-run declaration')
    if not args.prepare_only:
        execute(output)
    print(f'Experiment: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
