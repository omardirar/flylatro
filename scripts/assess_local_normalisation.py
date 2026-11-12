"""Independent CPU replay and budget audit for the local-normalisation study."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from card_learning.common import write_json
from card_learning.pairs import PairCorpus, pair_slots
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.seeds import derive_seed
from run_local_normalisation import ARMS, VERSION, read, verify


def assess(output):
    output = output.resolve()
    declaration = read(output/'declaration.json'); verify(declaration)
    report = read(output/'report.json')
    checkpoint = read(output/'checkpoint.json')
    if (checkpoint['sha256'] != sha256_file(Path(checkpoint['path']))
            or checkpoint['declaration_sha256'] != sha256_file(output/'declaration.json')
            or report['declaration_sha256'] != sha256_file(output/'declaration.json')):
        raise ValueError('report/checkpoint declaration or state differs')
    roles, ids = np.asarray(report['roles']), np.asarray(report['replicate_ids'])
    if (not np.array_equal(roles, np.tile(np.arange(3), declaration['replicates']))
            or not np.array_equal(ids, np.repeat(np.arange(declaration['replicates']), 3))
            or checkpoint['steps'] != report['steps'] or report['steps'] != declaration['steps']):
        raise ValueError('declared actor layout or exposure differs')
    corpus, fresh = PairCorpus(output/'pairs.json'), PairCorpus(output/'fresh-pairs.json')
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False); env.reset([0])
    cache = {}
    counts = Counter()
    def replay_row(row, cases, *, training=False):
        for lane, index in enumerate(row['case_indices']):
            cards = row['cards'][lane]
            order = np.asarray(row['orders'][lane])
            expected_order = np.r_[np.random.default_rng(derive_seed(VERSION, row['key'], int(ids[lane]), index, 'display')).permutation(8), 8, 9]
            if (len(cards) != 2 or len(set(cards)) != 2 or min(cards) < 0 or max(cards) >= 8
                    or not np.array_equal(np.sort(order[:8]), np.arange(8))
                    or not np.array_equal(order[8:], [8, 9]) or not np.array_equal(order, expected_order)):
                raise ValueError('invalid fixed-count cards/permutation')
            native_cards = tuple(int(i) for i in order[cards])
            key = (cases.sha256, index, native_cards)
            if key not in cache:
                cases.restore(env, 0, index)
                obs, _ = env.observe()
                pair = set(pair_slots(obs, 0))
                action = empty_action_batch(1)
                action['action_type'][:] = 0; action['n_cards'][:] = 2
                action['cards'][0, :2] = native_cards
                result = env.step(action)
                if result.dones[0]: raise ValueError('unexpected native terminal')
                progress = result.infos[0]['reward_components']['blind_progress']
                cache[key] = (progress, float(result.rewards[0]), pair == set(native_cards))
            progress, reward, hit = cache[key]
            if (abs(progress - row['native_progress'][lane]) > 2e-6
                    or abs(reward - row['native_rewards'][lane]) > 2e-6
                    or bool(row['pair_selected'][lane]) != hit):
                raise ValueError('native replay/diagnostic pair mismatch')
            expected_pulse = min(progress, declaration['feedback']['max_progress_component']) * declaration['feedback']['progress_scale']
            if abs(expected_pulse - row['native_pulses'][lane]) > 2e-7:
                raise ValueError('unchanged native reinforcement differs')
            counts['training_records' if training else 'evaluation_records'] += 1

    cumulative = np.zeros(len(roles))
    with (output/'training.jsonl').open() as stream:
        for step, line in enumerate(stream):
            row = json.loads(line)
            if row['key'] != ['training', step]: raise ValueError('training cursor differs')
            choices = corpus.indices('training')
            expected = [choices[derive_seed(VERSION, step, int(rep), 'training-case') % len(choices)] for rep in ids]
            if row['case_indices'] != expected: raise ValueError('declared training schedule differs')
            if any(corpus.cases[i]['split'] != 'training' for i in row['case_indices']):
                raise ValueError('validation case used in training')
            replay_row(row, corpus, training=True)
            cumulative += row['update_l1']
    if counts['training_records'] != declaration['steps'] * len(roles):
        raise ValueError('training budget differs')
    initial = read(output/'evaluation-0000000.json')
    evaluation_paths = sorted(output.glob('evaluation-*.json'))
    for path in evaluation_paths:
        document = read(path)
        for split, rows in document['rows'].items():
            expected = [index for index in corpus.indices(split) for _ in range(declaration['repeats'])]
            if len(rows) != len(expected) or [r['case_indices'][0] for r in rows] != expected:
                raise ValueError('evaluation case coverage differs')
            for index, row in enumerate(rows):
                if any(corpus.cases[i]['split'] != split for i in row['case_indices']):
                    raise ValueError('evaluation split differs')
                replay_row(row, corpus)
                before = initial['rows'][split][index]
                frozen = roles == 2
                if np.any(np.asarray(row['cards'])[frozen] != np.asarray(before['cards'])[frozen]):
                    raise ValueError('frozen individual choices changed')
    for rows in initial['rows'].values():
        for row in rows:
            for rep in np.unique(ids):
                choices = np.asarray(row['cards'])[ids == rep]
                if np.any(choices != choices[:1]): raise ValueError('initial matched choices differ')
    fresh_rows = read(output/'fresh-evaluation.json')
    for rows in fresh_rows.values():
        expected = [index for index in fresh.indices('validation') for _ in range(declaration['repeats'])]
        if len(rows) != len(expected) or [r['case_indices'][0] for r in rows] != expected:
            raise ValueError('fresh evaluation case coverage differs')
        for row in rows:
            if any(fresh.cases[i]['split'] != 'validation' for i in row['case_indices']):
                raise ValueError('fresh validation split differs')
            replay_row(row, fresh)
    for a, b in zip(fresh_rows['initial'], fresh_rows['trained'], strict=True):
        if np.any(np.asarray(a['cards'])[roles == 2] != np.asarray(b['cards'])[roles == 2]):
            raise ValueError('fresh frozen individual choices changed')
    for role, arm in enumerate(ARMS):
        for metric in ('pair_selected', 'native_progress'):
            left = np.asarray([r[metric] for r in fresh_rows['initial']], dtype=float)[:, roles == role]
            right = np.asarray([r[metric] for r in fresh_rows['trained']], dtype=float)[:, roles == role]
            for name, actual in [('initial', left.mean(axis=0)), ('trained', right.mean(axis=0)),
                                 ('change', (right - left).mean(axis=0))]:
                np.testing.assert_allclose(report['summary'][arm][metric][name], actual, atol=1e-12, rtol=0)
    with np.load(checkpoint['path'], allow_pickle=False) as z:
        efficacy, original = z['efficacy'], z['initial_efficacy']
        np.testing.assert_allclose(cumulative, z['update_l1'], rtol=1e-9, atol=1e-9)
        if (np.any(efficacy[roles == 2] != original[roles == 2])
                or np.any(z['eligibility']) or np.any(z['dopamine'])):
            raise ValueError('frozen weights changed or fast traces remain at boundary')
    config = PlasticExperimentConfig.load(Path(declaration['config']))
    if config.fly.backend == 'flywire':
        # Read the pinned anatomy directly, independently of the rule's grouped
        # reduction/projection implementation.
        with np.load(resolve_path(config, config.fly.artifact_path), allow_pickle=False) as z:
            edge_ids = z['kc_mbon_edge_indices']
            anatomy = z['signed_synapse_counts'][edge_ids]
            roots = z['root_ids'][z['post_indices'][edge_ids]]
        keep = anatomy >= config.fly.minimum_synapse_count
        anatomy, roots = anatomy[keep].astype(np.float64), roots[keep]
        if efficacy.shape[1] != len(roots): raise ValueError('anatomical edge alignment differs')
        errors = np.zeros(len(roles))
        for root in np.unique(roots):
            edges = roots == root
            budget = (original[:, edges] * anatomy[edges]).sum(axis=1)
            strength = (efficacy[:, edges] * anatomy[edges]).sum(axis=1)
            errors = np.maximum(errors, abs(strength - budget) / budget)
        if np.any(errors[roles == 1] > 1e-6): raise ValueError('weighted postsynaptic budget differs')
        mapping = MotorMapping.load(resolve_path(config, config.fly.motor_mapping_path))
        pools = mapping.routing.head_routes['card']
        play_roots = [mapping.output_root_ids[i] for p in pools if p >= 0 for i in mapping.pool_indices[p]]
        if np.any(efficacy[:, ~np.isin(roots, play_roots)] != original[:, ~np.isin(roots, play_roots)]):
            raise ValueError('unrelated anatomical efficacy changed')
    else:
        errors = np.asarray(report['health']['max_post_budget_relative_error_per_lane'])
    # Average repeat observations per case before resampling shared game cases.
    conditional = {}
    for metric in ('pair_selected', 'native_progress'):
        changes = []
        for a, b in zip(fresh_rows['initial'], fresh_rows['trained'], strict=True):
            left = np.asarray(a[metric], dtype=float); right = np.asarray(b[metric], dtype=float)
            changes.append((b['case_indices'][0], float(np.mean(right[roles == 1] - left[roles == 1])),
                float(np.mean(right[roles == 1] - right[roles == 0]))))
        cases = sorted(set(row[0] for row in changes))
        matrix = np.asarray([[np.mean([r[i] for r in changes if r[0] == case]) for i in (1, 2)] for case in cases])
        rng = np.random.default_rng(20261031)
        samples = matrix[rng.integers(0, len(cases), size=(10000, len(cases)))].mean(axis=1)
        conditional[metric] = {'gain_vs_initial_95ci': np.quantile(samples[:, 0], [.025, .975]).tolist(),
            'advantage_vs_original_95ci': np.quantile(samples[:, 1], [.025, .975]).tolist()}
    result = {'status': 'completed', 'source_report_sha256': sha256_file(output/'report.json'),
        'checkpoint_sha256': checkpoint['sha256'], 'assessor_sha256': sha256_file(Path(__file__)),
        'recorded_actions_verified_against_native': sum(counts.values()), 'record_counts': dict(counts),
        'unique_native_actions_replayed': len(cache),
        'replay_semantics': 'all records checked; identical snapshot/action results cached after one independent native replay',
        'normalised_budget_error': errors[roles == 1].tolist(),
        'frozen_weights_and_individual_choices_exact': True, 'initial_matched_choices_exact': True,
        'native_rewards_and_pulses_verified': True, 'conditional_case_intervals': conditional,
        'interval_scope': 'shared fresh hands resampled after averaging repeats; conditional on these three learners',
        'summary': report['summary'], 'final_test_used': False}
    write_json(output/'assessment.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    report = assess(args.output_dir)
    print(json.dumps({key: report[key] for key in ('status', 'record_counts', 'unique_native_actions_replayed', 'normalised_budget_error')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
