"""Independently audit local teaching weights, controls and native inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from card_learning.common import write_json
from card_learning.experiment import load_condition
from card_learning.pairs import PairCorpus
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch, validate_strict_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.fly.prepared_cache import load_prepared_artifact
from flylatro.interface.motor import MotorMapping
from flylatro.learning.config import PlasticExperimentConfig, resolve_path, build_plastic_stack
from local_teaching import edge_pool_indices
from run_repaired_native_membership import CONDITION, verify_sources
from teacher_curriculum_data import STAGES, opening_batch
import train_discard_play_curriculum as delayed


def replay_blind(result):
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False); verified = 0
    for row in result['rows']:
        for lane in range(len(row['cleared'])):
            env.reset([row['seed']]); start = empty_action_batch(1)
            start['action_type'][:] = 2; start['n_cards'][:] = 0; env.step(start)
            total = 0.; plays = 0; cleared = False
            for step in row['trajectory']:
                if not step['active'][lane]:
                    continue
                action = empty_action_batch(1); action['action_type'][:] = 0
                count = step['counts'][lane]; action['n_cards'][:] = count
                action['cards'][0, :count] = step['cards'][lane][:count]
                native = env.step(action); verified += 1; plays += 1
                info = native.infos[0]; components = info['reward_components']
                cleared = bool(info.get('blind_cleared', False) or components.get('blind_clear', 0))
                np.testing.assert_allclose(components['blind_progress'], step['progress'][lane], atol=2e-6, rtol=0)
                if cleared != step['cleared'][lane] or bool(native.dones[0]) != step['dones'][lane]:
                    raise ValueError('native first-blind transition differs')
                total += components['blind_progress']
            np.testing.assert_allclose(total, row['progress'][lane], atol=2e-6, rtol=0)
            if plays != row['plays'][lane] or cleared != row['cleared'][lane]:
                raise ValueError('native first-blind result differs')
    np.testing.assert_allclose(np.asarray([r['cleared'] for r in result['rows']]).mean(axis=0), result['clear_rate_per_lane'], atol=0, rtol=0)
    return verified


def replay_openings(result, corpus):
    n = len(result['exact_accuracy_per_lane']); env = BalatroSimAdapter(n, win_ante=1, allow_skip_blinds=False)
    env.reset([0]*n); verified = 0; correctness = []
    for row in result['rows']:
        key = ('evaluation', result['stage'], row['repeat'], row['side'])
        obs, masks, target, orders = opening_batch(corpus, [row['case']]*n, result['stage'], key=key, correct_side=row['side'])
        actual = empty_action_batch(n); actual['action_type'][:] = 0
        actual['n_cards'][:] = row['counts']; actual['cards'][:] = row['cards']
        validate_strict_action_batch(actual, masks)
        correct = []
        for lane in range(n):
            a = actual['cards'][lane, :int(actual['n_cards'][lane])]
            t = target['cards'][lane, :int(target['n_cards'][lane])]
            correct.append(set(a.tolist()) == set(t.tolist()))
            corpus.restore(env, lane, row['case'])
        np.testing.assert_array_equal(correct, row['correct']); correctness.append(correct)
        outcome = env.step(delayed.translate(actual, orders)); verified += n
        np.testing.assert_allclose([info['reward_components']['blind_progress'] for info in outcome.infos],
            np.minimum(row['chips'], 300)/300, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(np.asarray(correctness).mean(axis=0), result['exact_accuracy_per_lane'])
    return verified


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args(argv); folder = args.run_dir
    declaration = json.loads((folder/'declaration.json').read_text()); report = json.loads((folder/'report.json').read_text())
    if report['status'] != 'completed':
        raise ValueError('completed local curriculum required')
    verify_sources(declaration['source_hashes']); load_condition(CONDITION)
    config = PlasticExperimentConfig.load(Path(declaration['config']))
    if config.fly.backend == 'synthetic':
        topology = build_plastic_stack(config).agent.plasticity.topology
    else:
        artifact = load_prepared_artifact(resolve_path(config, config.fly.artifact_path))
        topology = PlasticEdgeTopology.from_artifact(artifact, minimum_synapse_count=config.fly.minimum_synapse_count)
    mapping = MotorMapping.load(resolve_path(config, config.fly.motor_mapping_path))
    allowed = edge_pool_indices(mapping, topology) >= 0
    roles = np.asarray(report['roles']); rule = declaration['local_rule']
    with np.load(folder/'local-final-state.npz', allow_pickle=False) as state:
        efficacy, initial = state['efficacy'], state['initial_efficacy']
        if efficacy.shape != (len(roles), topology.edge_count) or not np.isfinite(efficacy).all() or np.any(initial != 1):
            raise ValueError('local final state shape, finiteness or initialization differs')
        if np.any(efficacy < rule['min_efficacy']) or np.any(efficacy > rule['max_efficacy']):
            raise ValueError('local efficacy outside declared bounds')
        delta = efficacy-initial
        if np.any(delta[roles == 2]) or np.any(delta[:, ~allowed]):
            raise ValueError('frozen or non-PLAY anatomical weights changed')
        for key in ('eligibility', 'dopamine'):
            if state[key].any():
                raise ValueError('local fast traces not clear')
        if np.any(state['decision_count'] != declaration['steps_per_stage']*len(declaration['stages'])):
            raise ValueError('local training exposure differs')
        if state['presynaptic_mean'].shape != efficacy.shape or not np.isfinite(state['presynaptic_mean']).all():
            raise ValueError('invalid local presynaptic trace')
        statistics = {'potentiated_edges_per_lane': (delta > 0).sum(axis=1).tolist(),
            'depressed_edges_per_lane': (delta < 0).sum(axis=1).tolist(),
            'mean_absolute_drift_per_lane': abs(delta).mean(axis=1).tolist(),
            'maximum_absolute_drift_per_lane': abs(delta).max(axis=1).tolist()}
        total_changed = np.zeros(len(roles), dtype=np.int64); total_bounds = total_changed.copy()
        for stage in declaration['stages']:
            lines = [json.loads(line) for line in (folder/(stage+'-training.jsonl')).open()]
            if [row['step'] for row in lines] != list(range(declaration['steps_per_stage'])):
                raise ValueError('local training log incomplete or duplicated')
            for row in lines:
                plus, minus, bounds = [np.asarray(row[k]) for k in ('potentiated', 'depressed', 'bound_hits')]
                if np.any((plus+minus)[roles == 2]) or not np.isfinite(row['update_l1']).all():
                    raise ValueError('invalid frozen update or local event')
                total_changed += plus+minus; total_bounds += bounds
        np.testing.assert_array_equal(total_changed, state['update_count'])
        np.testing.assert_array_equal(total_bounds, state['lower_bound_hits']+state['upper_bound_hits'])
        checkpoint = json.loads((folder/'checkpoint-latest.json').read_text())
        state_path = folder/checkpoint['state_file']
        if checkpoint['state_sha256'] != sha256_file(state_path) or checkpoint['declaration_sha256'] != sha256_file(folder/'declaration.json'):
            raise ValueError('latest local checkpoint identity differs')
        with np.load(state_path, allow_pickle=False) as saved:
            for key in state.files:
                np.testing.assert_array_equal(state[key], saved[key])
    corpus = PairCorpus(folder/'fresh-pairs.json'); train = PairCorpus(folder/'pairs.json')
    if {c['seed'] for c in corpus.cases} & {c['seed'] for c in train.cases}:
        raise ValueError('fresh curriculum cases overlap training/screen')
    actions = 0
    for phase in ('initial', 'final'):
        for stage in STAGES[:-1]:
            actions += replay_openings(report[phase][stage], corpus)
        actions += replay_blind(report[phase]['small_blind'])
    for stage in declaration['stages']:
        data = json.loads((folder/(stage+'-evaluation.json')).read_text())
        actions += replay_blind(data) if stage == 'small_blind' else replay_openings(data, train)
    paths = [folder/'declaration.json', folder/'report.json', folder/'local-final-state.npz', folder/'checkpoint-latest.json']
    paths += [folder/(stage+'-training.jsonl') for stage in declaration['stages']]
    result = {'status': 'completed', 'native_actions_replayed': actions, 'local_anatomical_weight_audit': statistics,
        'checks': {'frozen_weights_exact': True, 'only_existing_play_synapses_changed': True,
            'durable_checkpoint_matches_final_state': True, 'complete_training_exposure': True,
            'native_actions_and_scores_verified': True, 'legacy_resume_identity_preserved': True},
        'source_hashes': declaration['source_hashes'], 'assessor_sha256': sha256_file(Path(__file__)),
        'artifact_sha256': {str(path.resolve()): sha256_file(path) for path in paths},
        'claim': 'engineering local three-factor teacher experiment; no biological validation of the exact teaching rule or dopamine routing'}
    write_json(folder/'assessment.json', result)
    print(json.dumps({'native_actions_replayed': actions, 'weights': statistics}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
