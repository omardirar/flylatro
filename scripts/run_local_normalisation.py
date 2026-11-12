"""Matched native two-card study: original / normalised / frozen, fresh learners."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from card_learning.common import identity, write_json
from card_learning.motor import CardExploration, install_motor
from card_learning.pairs import PairCorpus, build_corpus, pair_slots
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.interface.motor import MotorMapping
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack, resolve_path
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.seeds import derive_seed
from local_normalisation import NormalisedThreeFactorPlasticity, VERSION as RULE_VERSION
from run_card_membership import fixed_motor
from run_training_bottlenecks import head_edges
from train_discard_play_curriculum import forced, permute_rows, translate


VERSION = 'local-normalisation-fixed-count-study-v1'
ARMS = ('original', 'normalised', 'frozen')


def read(path):
    return json.loads(Path(path).read_text())


def verify(declaration):
    if declaration['implementation'] != identity():
        raise ValueError('historical harness/production implementation changed')
    for path, digest in declaration['input_sha256'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('declared input changed: ' + path)


def prepare(config_path, output, *, steps=20000, replicates=3, repeats=2,
            training_per_rank=8, validation_per_rank=8):
    if min(steps, replicates, repeats, training_per_rank, validation_per_rank) < 1:
        raise ValueError('positive counts required')
    config_path, output = config_path.resolve(), output.resolve()
    config = PlasticExperimentConfig.load(config_path)
    if (config.environment.allow_skip_blinds or config.plasticity.dopamine_decay != 0
            or config.plasticity.version != 'three-factor-choice-gated-v1'
            or config.reinforcement.run_failure_pulse != .01):
        raise ValueError('unchanged no-skip, choice-gated, decision-local native-feedback setup required')
    output.mkdir(parents=True, exist_ok=False)
    build_corpus(output/'pairs.json', train_per_rank=training_per_rank,
        validation_per_rank=validation_per_rank, training_offset=27000, validation_offset=940000, max_attempts=5000)
    build_corpus(output/'fresh-pairs.json', train_per_rank=1,
        validation_per_rank=validation_per_rank, training_offset=37000, validation_offset=960000, max_attempts=5000)
    paths = [config_path, output/'pairs.json', output/'fresh-pairs.json']
    paths += [Path(__file__).with_name(name) for name in ('local_normalisation.py',
        'run_local_normalisation.py', 'run_card_membership.py', 'run_training_bottlenecks.py',
        'train_discard_play_curriculum.py')]
    for name in (config.fly.artifact_path if config.fly.backend == 'flywire' else '',
                 config.fly.motor_mapping_path, config.calibration.corpus_path):
        if name: paths.append(resolve_path(config, name))
    sources = {str(p.resolve()): sha256_file(p) for p in paths}
    write_json(output/'declaration.json', {'version': VERSION, 'rule': RULE_VERSION,
        'implementation': identity(), 'input_sha256': sources, 'config': str(config_path),
        'steps': steps, 'replicates': replicates, 'repeats': repeats,
        'arms': list(ARMS), 'stages': sorted(set([min(5000, steps), steps])),
        'card_count': 2, 'exploration_probability': .05, 'fresh_unit_weights': True,
        'normalisation': 'fixed initial sum(anatomical weight * efficacy), per learner and MBON',
        'feedback': asdict(config.reinforcement), 'base_plasticity': asdict(config.plasticity),
        'training_safety': asdict(config.safety),
        'fixed_factors': 'same initial weights, neural/display/case/exploration seeds; native reinforcement; no teacher',
        'task_boundaries': 'one native opening PLAY; clear traces, no synthetic failure on artificial reset',
        'success_criteria': {'budget_relative_tolerance': 1e-6,
            'primary': 'fresh greedy pair selection improves by >=2 percentage points versus own initial AND original-rule arm',
            'minimum_positive_replicates': 2,
            'secondary': 'fresh native progress improves versus initial and original-rule arm'},
        'limits': 'fixed PLAY/two cards on opening one-pair hands; failure punishment unchanged but never exercised; no full-game claim',
        'final_test_seeds_used': False})
    write_json(output/'status.json', {'status': 'prepared'})


class Study:
    def __init__(self, stack, corpus, *, replicates=3):
        self.agent, self.corpus = stack.agent, corpus
        self.safety = stack.trainer.safety.config
        self.ids = np.repeat(np.arange(replicates), 3)
        self.roles = np.tile(np.arange(3), replicates)
        self.n = len(self.ids)
        source = self.agent.plasticity
        if np.any(state_numpy(source.state.efficacy) != 1):
            raise ValueError('study requires fresh unit weights')
        self.source_hash = plastic_hashes(source.state)
        values = {key: np.repeat(value[:1], self.n, axis=0)
            for key, value in source.state.state_dict().items()}
        factory = type(source.state)
        state = (factory.from_state_dict(values, device=str(source.state.device))
            if getattr(source.state, 'is_torch', False) else factory.from_state_dict(values))
        self.local = NormalisedThreeFactorPlasticity(source.topology, state, source.config,
            normalised_learners=self.roles == 1)
        self.motor = fixed_motor(self.agent, 0, 2)
        self.credit = SelectedMotorCredit(self.motor, self.local)
        self.allowed = head_edges(self.agent, ['card'])
        self.env = BalatroSimAdapter(self.n, win_ante=1, allow_skip_blinds=False)
        self.env.reset([0] * self.n)
        self.steps = 0
        self.update_l1 = np.zeros(self.n)

    def trial(self, indices, key, *, learn=False, initial=False, corpus=None):
        corpus = corpus or self.corpus
        for row, index in enumerate(indices): corpus.restore(self.env, row, index)
        original, legal = self.env.observe()
        orders = [np.r_[np.random.default_rng(derive_seed(VERSION, key, int(rep), index, 'display')).permutation(8), 8, 9]
            for rep, index in zip(self.ids, indices, strict=True)]
        obs, masks = permute_rows(original, forced(legal, 0), orders)
        seeds = [derive_seed(VERSION, key, int(rep), corpus.cases[index]['seed'], 'neural')
            for rep, index in zip(self.ids, indices, strict=True)]
        efficacy = self.local.state.initial_efficacy if initial else self.local.state.efficacy
        neural = self.agent.processor.process(obs, fly_seeds=seeds, efficacy=efficacy)
        if not np.isfinite(neural.output_activity).all():
            raise ValueError('nonfinite native model output')
        actions = self.motor.decode(neural.output_activity, masks, deterministic=not learn,
            learner_ids=self.ids, decision_ids=[derive_seed(VERSION, key, index, 'exploration') for index in indices])
        if np.any(actions['n_cards'] != 2) or np.any(actions['action_type'] != 0):
            raise ValueError('fixed PLAY/count changed')
        result = self.env.step(translate(actions, orders))
        pulses = [self.agent.reinforcement.map(info['reward_components'], info) for info in result.infos]
        if np.any(result.dones) or any(p.aversive or p.appetitive <= 0 for p in pulses):
            raise ValueError('opening two-card task emitted an unexpected terminal/pulse')
        update = np.zeros(self.n)
        if learn:
            self.local.record_activity(neural.edge_pre_activity, neural.edge_post_activity,
                edge_credit=self.credit.mask(actions, masks))
            positive = np.asarray([p.appetitive for p in pulses], dtype=np.float32)
            positive[self.roles == 2] = 0
            events = self.local.apply_dopamine(positive, 0)
            update = np.asarray([e.absolute_change for e in events])
            self.update_l1 += update
            self.local.state.reset_fast_traces(eligibility=True)
        # Pair labels only enter these diagnostics, after native choice/reward/update.
        return {'key': key, 'case_indices': list(map(int, indices)), 'orders': [o.tolist() for o in orders],
            'cards': actions['cards'][:, :2].tolist(),
            'pair_selected': [set(actions['cards'][i, :2]) == set(pair_slots(obs, i)) for i in range(self.n)],
            'native_progress': [info['reward_components']['blind_progress'] for info in result.infos],
            'native_rewards': result.rewards.tolist(), 'native_pulses': [p.appetitive for p in pulses],
            'update_l1': update.tolist()}

    def train_step(self):
        cases = self.corpus.indices('training')
        indices = [cases[derive_seed(VERSION, self.steps, int(rep), 'training-case') % len(cases)] for rep in self.ids]
        row = self.trial(indices, ('training', self.steps), learn=True)
        self.steps += 1
        return row

    def evaluate(self, split, repeats, *, corpus=None, initial=False):
        corpus = corpus or self.corpus
        before = plastic_hashes(self.local.state)
        rows = [self.trial([index] * self.n, ('evaluation', split, repeat),
            corpus=corpus, initial=initial) for index in corpus.indices(split) for repeat in range(repeats)]
        if plastic_hashes(self.local.state) != before:
            raise ValueError('frozen evaluation changed state')
        return rows

    def health(self):
        w, initial = state_numpy(self.local.state.efficacy), state_numpy(self.local.state.initial_efficacy)
        actual, budgets = state_numpy(self.local.incoming_strengths()), state_numpy(self.local.budgets)
        relative = np.max(abs(actual - budgets) / budgets, axis=1)
        bound_fraction = np.mean((w <= self.local.config.min_efficacy) | (w >= self.local.config.max_efficacy), axis=1)
        drift = np.mean(abs(w - initial), axis=1)
        if (not np.isfinite(w).all() or np.any(w < self.local.config.min_efficacy)
                or np.any(w > self.local.config.max_efficacy)):
            raise ValueError('invalid efficacy state')
        if self.safety.enabled and (np.any(bound_fraction > self.safety.maximum_bound_fraction)
                or np.any(drift > self.safety.maximum_mean_absolute_drift)):
            raise ValueError('study reached the configured saturation/drift safety limit')
        if np.any(relative[self.roles == 1] > 1e-6):
            raise ValueError('normalised postsynaptic budget drift')
        if np.any(w[self.roles == 2] != initial[self.roles == 2]) or np.any(w[:, ~self.allowed] != initial[:, ~self.allowed]):
            raise ValueError('frozen or unrelated efficacy changed')
        if plastic_hashes(self.agent.plasticity.state) != self.source_hash:
            raise ValueError('source state changed')
        return {'steps': self.steps, 'max_post_budget_relative_error_per_lane': relative.tolist(),
            'bound_fraction_per_lane': bound_fraction.tolist(),
            'mean_absolute_drift_per_lane': drift.tolist(),
            'update_l1': self.update_l1.tolist(), 'frozen_exact': True, 'unrelated_exact': True, 'source_unchanged': True}

    def save(self, output, log_bytes):
        path = output / f'checkpoint-{self.steps:07d}.npz'
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **self.local.state.state_dict(), steps=np.asarray(self.steps), update_l1=self.update_l1)
        temporary.replace(path)
        write_json(output/'checkpoint.json', {'path': str(path), 'sha256': sha256_file(path),
            'declaration_sha256': sha256_file(output/'declaration.json'),
            'rule_sha256': self.local.config.sha256, 'steps': self.steps, 'log_bytes': log_bytes,
            'health': self.health()})

    def restore(self, output):
        receipt = read(output/'checkpoint.json')
        if (receipt['rule_sha256'] != self.local.config.sha256
                or receipt['declaration_sha256'] != sha256_file(output/'declaration.json')
                or sha256_file(Path(receipt['path'])) != receipt['sha256']):
            raise ValueError('checkpoint rule/state differs')
        with np.load(receipt['path'], allow_pickle=False) as data:
            values = {key: data[key].copy() for key in self.local.state.state_dict()}
            if values['efficacy'].shape != self.local.state.efficacy.shape or np.any(values['initial_efficacy'] != 1):
                raise ValueError('checkpoint learner shape or initial budgets differ')
            self.steps = int(data['steps']); self.update_l1 = data['update_l1'].copy()
        factory = type(self.local.state)
        self.local.state = (factory.from_state_dict(values, device=str(self.local.state.device))
            if getattr(self.local.state, 'is_torch', False) else factory.from_state_dict(values))
        if self.steps != receipt['steps']:
            raise ValueError('checkpoint cursor differs')
        self.health()
        log = output/'training.jsonl'
        if not log.exists() or log.stat().st_size < receipt['log_bytes']:
            raise ValueError('checkpoint training log is missing/truncated')
        with log.open('r+b') as stream: stream.truncate(receipt['log_bytes'])


def execute(output, *, heavy=False):
    output = output.resolve()
    declaration = read(output/'declaration.json'); verify(declaration)
    if (output/'report.json').exists():
        return read(output/'report.json')
    config = PlasticExperimentConfig.load(Path(declaration['config'])); config.require_heavy_opt_in(heavy)
    stack = install_motor(build_plastic_stack(config), CardExploration(.05))
    mapping = MotorMapping.load(resolve_path(config, config.fly.motor_mapping_path)) if config.fly.motor_mapping_path else None
    if config.fly.backend == 'flywire' and (mapping is None or not mapping.routing.action_conditioned):
        raise ValueError('qualified separate PLAY/DISCARD mapping required')
    corpus, fresh = PairCorpus(output/'pairs.json'), PairCorpus(output/'fresh-pairs.json')
    if {c['seed'] for c in corpus.cases} & {c['seed'] for c in fresh.cases}:
        raise ValueError('fresh corpus overlaps training/screen cases')
    study = Study(stack, corpus, replicates=declaration['replicates'])
    started = time.perf_counter()
    (output/'training.jsonl').touch(exist_ok=True)
    if (output/'checkpoint.json').exists():
        study.restore(output)
    elif (output/'training.jsonl').stat().st_size:
        raise ValueError('uncheckpointed run exists; preserve it and start in a new directory')
    def evaluations(step):
        path = output/f'evaluation-{step:07d}.json'
        if not path.exists():
            rows = {split: study.evaluate(split, declaration['repeats']) for split in ('training', 'validation')}
            write_json(path, {'step': step, 'rows': rows, 'health': study.health()})
    if study.steps == 0: evaluations(0)
    elif study.steps in declaration['stages']: evaluations(study.steps)
    try:
        with (output/'training.jsonl').open('ab') as log:
            while study.steps < declaration['steps']:
                row = study.train_step()
                log.write((json.dumps(row, sort_keys=True) + '\n').encode())
                if study.steps % 250 == 0 or study.steps in declaration['stages']:
                    log.flush(); health = study.health()
                    write_json(output/'status.json', {'status': 'running', **health})
                    print(json.dumps({'steps': study.steps, 'budget_error': health['max_post_budget_relative_error_per_lane'],
                        'bound_fraction': health['bound_fraction_per_lane']}), flush=True)
                if study.steps % 1000 == 0 or study.steps in declaration['stages']:
                    log.flush(); study.save(output, log.tell())
                if study.steps in declaration['stages']: evaluations(study.steps)
        # All arms, all three replicas, predetermined fresh hands: no winner selection.
        fresh_rows = {'initial': study.evaluate('validation', declaration['repeats'], corpus=fresh, initial=True),
            'trained': study.evaluate('validation', declaration['repeats'], corpus=fresh)}
        write_json(output/'fresh-evaluation.json', fresh_rows)
        verify(declaration)
        report = {'version': VERSION, 'status': 'completed', 'steps': study.steps,
            'rule': asdict(study.local.config), 'replicate_ids': study.ids.tolist(), 'roles': study.roles.tolist(),
            'health': study.health(), 'wall_seconds_this_invocation': time.perf_counter() - started,
            'declaration_sha256': sha256_file(output/'declaration.json'), 'final_test_seeds_used': False,
            'summary': summarise(study, fresh_rows)}
        write_json(output/'report.json', report)
        write_json(output/'status.json', {'status': 'completed', 'steps': study.steps})
        return report
    except BaseException as error:
        write_json(output/'status.json', {'status': 'stopped', 'steps': study.steps, 'error': f'{type(error).__name__}: {error}'})
        raise


def summarise(study, rows):
    result = {}
    for role, arm in enumerate(ARMS):
        lanes = np.flatnonzero(study.roles == role)
        result[arm] = {}
        for metric in ('pair_selected', 'native_progress'):
            initial = np.asarray([r[metric] for r in rows['initial']], dtype=float)[:, lanes]
            trained = np.asarray([r[metric] for r in rows['trained']], dtype=float)[:, lanes]
            result[arm][metric] = {'initial': initial.mean(axis=0).tolist(), 'trained': trained.mean(axis=0).tolist(),
                'change': (trained - initial).mean(axis=0).tolist(), 'mean_change': float((trained - initial).mean())}
    pair = result['normalised']['pair_selected']
    original = result['original']['pair_selected']
    advantage = np.asarray(pair['trained']) - np.asarray(original['trained'])
    checks = {'fresh_pair_gain_at_least_two_points': pair['mean_change'] >= .02,
        'fresh_pair_advantage_at_least_two_points': float(advantage.mean()) >= .02,
        'gain_and_advantage_in_multiple_learners': int(np.count_nonzero((np.asarray(pair['change']) > 0) & (advantage > 0))) >= 2,
        'fresh_native_progress_gain': result['normalised']['native_progress']['mean_change'] > 0,
        'fresh_native_progress_advantage': float(np.mean(result['normalised']['native_progress']['trained'])) > float(np.mean(result['original']['native_progress']['trained']))}
    result['advance'] = {'checks': checks, 'fixed_count_learning_demonstrated': all(checks.values()),
        'full_game_learning_demonstrated': False, 'long_training_ready': False}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('artefacts/learnable-motor-20261028-v2/config.toml'))
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--training-per-rank', type=int, default=8)
    parser.add_argument('--validation-per-rank', type=int, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if not output.exists():
        prepare(args.config, output, steps=args.steps, replicates=args.replicates, repeats=args.repeats,
            training_per_rank=args.training_per_rank, validation_per_rank=args.validation_per_rank)
    if not args.prepare_only:
        report = execute(output, heavy=args.heavy)
        print(json.dumps(report['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
