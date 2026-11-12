"""Declared factorial pilots, using the existing trainer and strict checkpoints."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

from flylatro.evaluation.compare import compare_frozen_evaluations
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack, resolve_path
from flylatro.learning.protocol_materialize import render_toml
from flylatro.seeds import SeedPlan

from .common import identity, write_json
from .motor import CardExploration, CardSubsetMotor, install_motor
from .pairs import PairCorpus, PairTaskEnvironment, TASK


def prepare(config_path: Path, output: Path, *, steps=20000, replicates=3,
            episodes=256, batch=4, probability=.05, pair_corpus=None):
    if min(steps, replicates, episodes, batch) < 1:
        raise ValueError('pilot counts must be positive')
    if not 0 < probability <= 1:
        raise ValueError('candidate probability must be in (0, 1]')
    base = PlasticExperimentConfig.load(config_path)
    if (base.protocol.bound or base.environment.allow_skip_blinds
            or base.plasticity.version != 'three-factor-choice-gated-v1'
            or base.training.action_schedule_path or base.training.reinforcement_mode != 'outcome'
            or base.motor.deterministic_training or not base.curriculum.enabled
            or base.curriculum.ladder != (1,)):
        raise ValueError('use an unbound no-skip, stochastic, choice-gated, fixed Ante-1 outcome experiment')
    corpus = PairCorpus(pair_corpus) if pair_corpus else None
    if corpus is not None and base.environment.backend != 'balatro_sim':
        raise ValueError('pair task requires the pinned native simulator')
    absolute = lambda p: str(resolve_path(base, p)) if p else ''
    common = replace(base, environment=replace(base.environment, num_envs=replicates),
        fly=replace(base.fly, artifact_path=absolute(base.fly.artifact_path),
                    motor_mapping_path=absolute(base.fly.motor_mapping_path)),
        calibration=replace(base.calibration, corpus_path=absolute(base.calibration.corpus_path)),
        training=replace(base.training, max_environment_decisions=steps*replicates,
            checkpoint_every_decisions=min(1000, steps)*replicates,
            budget_basis='Declared card-membership exploration factorial; equal independent learner exposure',
            matched_source_run_dir='', action_schedule_path='', reinforcement_schedule_path=''))
    output = output.resolve()
    input_paths = [common.fly.motor_mapping_path, common.calibration.corpus_path]
    if common.fly.backend == 'flywire':
        input_paths.append(common.fly.artifact_path)
    inputs = {str(Path(p)): sha256_file(Path(p)) for p in input_paths if p}
    output.mkdir(parents=True, exist_ok=False)
    declaration = {'version': 'card-learning-factorial-v1', 'implementation': identity(),
        'source_config': str(config_path.resolve()), 'source_config_sha256': sha256_file(config_path),
        'steps_per_learner': steps, 'replicates': replicates, 'evaluation_episodes': episodes,
        'evaluation_batch_size': batch, 'evaluation_seed_offset': 70000,
        'evaluation_seeds': list(SeedPlan().seeds('validation', episodes, offset=70000)),
        'training_seeds': list(SeedPlan().seeds('training', replicates, offset=common.training.training_seed_offset)),
        'design': '2 x 2: greedy top-k / explicit subset exploration, plastic / no-plasticity',
        'evaluation': 'frozen greedy full Ante-1 games; each condition versus its own initial weights',
        'pair_task': corpus is not None, 'final_test_seeds_used': False,
        'interpretation': 'descriptive pilot; validation roots are shared across learners, not independent training replicates',
        'input_files_sha256': inputs, 'arms': {}}
    for sampler, chance in [('topk', 0.0), ('subset', probability)]:
        for learning in [True, False]:
            label = f'{sampler}-' + ('plastic' if learning else 'frozen')
            config = replace(common, name=f'card-learning-{label}-v1', training=replace(common.training,
                condition='plastic_real' if learning else 'no_plasticity', plasticity_enabled=learning))
            config.validate()
            folder = output / 'conditions' / label
            folder.mkdir(parents=True)
            config_file = folder / 'config.toml'
            document = asdict(config); document.pop('source_path')
            config_file.write_text(render_toml(document))
            condition = {'version': 'card-learning-condition-v1', 'label': label,
                'config': str(config_file), 'config_file_sha256': sha256_file(config_file),
                'implementation': declaration['implementation'], 'exploration': asdict(CardExploration(chance)),
                'input_files_sha256': inputs,
                'pair_corpus': str(corpus.path) if corpus else None,
                'pair_corpus_sha256': corpus.sha256 if corpus else None,
                'task': TASK if corpus else 'normal-ante-one'}
            path = folder / 'condition.json'; write_json(path, condition)
            declaration['arms'][label] = {'condition': str(path), 'condition_sha256': sha256_file(path)}
    write_json(output / 'declaration.json', declaration)
    return declaration


def load_condition(path: Path):
    raw = path.read_bytes()
    condition = json.loads(raw)
    if condition.get('version') != 'card-learning-condition-v1':
        raise ValueError('unknown card learning condition')
    if condition['implementation'] != identity():
        raise ValueError('research or production implementation changed after declaration')
    if sha256_file(Path(condition['config'])) != condition['config_file_sha256']:
        raise ValueError('declared config changed')
    for filename, expected in condition['input_files_sha256'].items():
        if sha256_file(Path(filename)) != expected:
            raise ValueError(f'declared model input changed: {filename}')
    corpus = PairCorpus(Path(condition['pair_corpus'])) if condition['pair_corpus'] else None
    if corpus and corpus.sha256 != condition['pair_corpus_sha256']:
        raise ValueError('declared pair corpus changed')
    return condition, corpus


def build_condition_stack(config, condition, corpus=None, **kwargs):
    stack = build_plastic_stack(config, **kwargs)
    install_motor(stack, CardExploration(**condition['exploration']), pair_task=corpus is not None)
    if corpus is not None:
        stack.env = PairTaskEnvironment(stack.env, corpus)
        stack.trainer.env = stack.env
        stack.trainer.observations, stack.trainer.masks = stack.env.reset(stack.trainer.training_seeds)
    stack.components['card_learning_condition'] = {
        'version': condition['version'], 'label': condition['label'],
        'implementation': condition['implementation'], 'task': condition['task'],
        'pair_corpus_sha256': condition['pair_corpus_sha256']}
    return stack


def worker(condition_path, command, argv):
    """Use a process-local stack factory; no edits or global on-disk patching."""
    from flylatro.learning import cli as training
    from flylatro.evaluation import plastic_cli as evaluation
    condition, corpus = load_condition(Path(condition_path))
    forbidden = {'--config', '--condition', '--execution', '--sensory-mapping-seed', '--output-mode',
        '--action-schedule', '--reinforcement-schedule', '--dopamine-schedule',
        '--max-environment-decisions', '--budget-basis', '--checkpoint-every-decisions',
        '--curriculum-ladder', '--curriculum-evaluation-every-decisions', '--curriculum-evaluation-episodes',
        '--unlock-final-test'}
    if any(arg.split('=')[0] in forbidden for arg in argv):
        raise ValueError('declare a new condition instead of overriding experimental parameters')
    allowed = {'--help', '-h', '--heavy'} | ({'--run-dir', '--resume', '--stop-after-decisions',
        '--no-tensorboard', '--log-buffer-bytes', '--log-flush-seconds', '--dashboard', '--no-dashboard',
        '--dashboard-refresh', '--record-plasticity-events'} if command == 'train' else {
        '--checkpoint', '--output-dir', '--learner-index', '--initial-weights', '--episodes', '--batch-size',
        '--win-ante', '--progress', '--seed-stream', '--seed-offset', '--max-vector-steps', '--record-neural'})
    if any(arg.startswith('-') and arg.split('=')[0] not in allowed for arg in argv):
        raise ValueError('unsupported or abbreviated worker flag; experimental overrides are disabled')
    if '--seed-stream' in argv and argv[argv.index('--seed-stream') + 1] != 'validation':
        raise ValueError('research worker uses validation only')
    if any(arg.startswith('--seed-stream=') and arg != '--seed-stream=validation' for arg in argv):
        raise ValueError('research worker uses validation only')
    module = training if command == 'train' else evaluation
    parsed = module.parse_args(['--config', condition['config'], *argv])
    if command == 'evaluate' and (parsed.seed_stream != 'validation' or parsed.win_ante != 1):
        raise ValueError('card learning evaluation requires validation and --win-ante 1')
    original = module.build_plastic_stack
    original_evaluation_condition = evaluation._apply_evaluation_condition
    original_training_overrides = training._apply_overrides
    original_evaluator = evaluation.evaluate_plastic_fly
    module.build_plastic_stack = lambda config, **kw: build_condition_stack(config, condition, corpus, **kw)
    training._apply_overrides = lambda config, args: config
    if command == 'evaluate':
        # This factorial's frozen controls choose their own actions. Preserve
        # the declared training identity rather than the primary protocol's
        # matched-trajectory no-plasticity override/schedule convention.
        evaluation._apply_evaluation_condition = lambda config, args: config
        if corpus is not None:
            def transfer(env, agent, seeds, **kw):
                trained_motor = agent.motor
                agent.motor = CardSubsetMotor(trained_motor.base, trained_motor.exploration, pair_task=False)
                try:
                    return original_evaluator(env, agent, seeds, **kw)
                finally:
                    agent.motor = trained_motor
            evaluation.evaluate_plastic_fly = transfer
    try:
        result = module.main(['--config', condition['config'], *argv])
        if command == 'evaluate':
            path = parsed.output_dir / 'summary.json'
            report = json.loads(path.read_text())
            report['card_learning_evaluation'] = {
                'task': 'normal Ante-1 games; original free type/count heads',
                'training_task': condition['task'], 'subset_sampling_enabled': False,
                'forced_pair_action_and_count_enabled': False}
            write_json(path, report)
        return result
    finally:
        module.build_plastic_stack = original
        evaluation._apply_evaluation_condition = original_evaluation_condition
        training._apply_overrides = original_training_overrides
        evaluation.evaluate_plastic_fly = original_evaluator


def execute(output: Path):
    declaration = json.loads((output / 'declaration.json').read_text())
    if declaration['implementation'] != identity():
        raise ValueError('pilot implementation changed')
    launcher = Path(__file__).parents[1] / 'run_card_learning.py'
    status = {'status': 'running', 'completed_arms': []}
    result = {'version': 'card-learning-pilot-results-v1', 'arms': {},
              'learning_claim': 'descriptive only; no automatic promotion'}
    try:
        for label, arm in declaration['arms'].items():
            condition_path = Path(arm['condition'])
            if sha256_file(condition_path) != arm['condition_sha256']:
                raise ValueError('pilot condition changed')
            condition, corpus = load_condition(condition_path)
            status.update(arm=label, phase='training'); write_json(output / 'status.json', status)
            def call(command, *args):
                subprocess.run([sys.executable, str(launcher), command,
                    '--condition', str(condition_path), *map(str, args)], check=True)
            run_dir = output / label
            call('train', '--run-dir', run_dir, '--dashboard', '--no-tensorboard', '--heavy')
            summary = json.loads((run_dir / 'run-summary.json').read_text())
            if summary['status'] != 'completed' or summary['safety_stop'] is not None or summary['environment_decisions'] != declaration['steps_per_learner']*declaration['replicates']:
                raise RuntimeError('pilot did not complete its declared training budget')
            checkpoint = summary['checkpoint']
            status['phase'] = 'evaluation'; write_json(output / 'status.json', status)
            reports = []
            for learner in range(declaration['replicates']):
                paired = []
                # Own initial control for each learner. The sampler is disabled
                # by greedy evaluation in every factorial condition.
                for initial in (True, False):
                    destination = output / f'{label}-eval-{learner}-' / ('initial' if initial else 'trained')
                    call('evaluate', '--checkpoint', checkpoint, '--learner-index', learner,
                        '--episodes', declaration['evaluation_episodes'], '--batch-size', declaration['evaluation_batch_size'],
                        '--seed-stream', 'validation', '--seed-offset', declaration['evaluation_seed_offset'],
                        '--win-ante', 1, '--output-dir', destination, '--heavy',
                        *(['--initial-weights'] if initial else []))
                    paired.append(json.loads((destination / 'summary.json').read_text()))
                comparison = compare_frozen_evaluations(paired[1], paired[0])
                row = {'learner': learner, 'comparison_to_own_initial': comparison}
                if corpus is not None:
                    pair_report = output / f'{label}-pairs-{learner}.json'
                    call('pair-evaluate', '--checkpoint', checkpoint, '--learner-index', learner,
                         '--output', pair_report, '--heavy')
                    row['pair_selection'] = json.loads(pair_report.read_text())['summary']
                reports.append(row)
            result['arms'][label] = {'checkpoint': checkpoint, 'learners': reports}
            status['completed_arms'].append(label)
            write_json(output / 'results.json', result)
        status.update(status='completed', phase='completed')
    except BaseException as error:
        status.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', status)
