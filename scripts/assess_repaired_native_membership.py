"""Audit all training transitions, pulse assignment, states and fresh comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from assess_card_membership import replay
from assess_training_bottlenecks import verify_state
from card_learning.common import write_json
from card_learning.audit import score
from card_learning.pairs import PairCorpus
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.fly.prepared_cache import load_prepared_artifact
from flylatro.interface.motor import MotorMapping
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.reinforcement import ReinforcementMapper
from run_repaired_native_membership import advance_gate, verify_sources
from run_card_membership import VERSION
from flylatro.seeds import derive_seed


def audit_training_rows(report, rows, reinforcement_config=None):
    roles = np.asarray(report['roles']); n = len(roles)
    update = np.zeros(n); pulse = np.zeros(n); rewarded_zero = np.zeros(n, dtype=int)
    rewarded = np.zeros(n, dtype=int)
    mapper = ReinforcementMapper(reinforcement_config) if reinforcement_config is not None else None
    for step, row in enumerate(rows):
        if row['step'] != step or list(row['key']) != ['training', step]:
            raise ValueError('training rows incomplete or out of order')
        if row['discard_counts'] != [report['count']]*n:
            raise ValueError('fixed discard count changed')
        weak = np.asarray(row['weak_pulse'], dtype=np.float32)
        reference = np.asarray(row['reference_pulse'], dtype=np.float32)
        if mapper is not None:
            for followup, observed in (('weak', weak), ('reference', reference)):
                expected_native = np.asarray([mapper.map({'blind_progress': progress, 'blind_clear': cleared}).appetitive
                    for progress, cleared in zip(row[followup+'_progress'], row[followup+'_clears'], strict=True)], dtype=np.float32)
                np.testing.assert_array_equal(observed, expected_native)
        expected = np.where(roles == 0, weak, reference)
        expected[roles == 2] *= report['calibration']['gain']; expected[roles == 3] = 0
        actual = np.asarray(row['applied_pulse']); delta = np.asarray(row['update_l1'])
        mass = np.asarray(row['eligibility_mass'])
        if any(x.shape != (n,) for x in (actual, delta, mass)) or not np.isfinite(np.r_[actual, delta, mass]).all():
            raise ValueError('invalid training activity or updates')
        np.testing.assert_array_equal(actual, expected)
        if np.any(delta < 0) or np.any(mass < 0) or np.any(delta[roles == 3]):
            raise ValueError('invalid updates or frozen actor changed')
        update += delta; pulse += actual
        rewarded += actual > 0; rewarded_zero += (actual > 0) & (mass == 0)
    if len(rows) != report['steps']:
        raise ValueError('training rows incomplete or out of order')
    np.testing.assert_allclose(update, report['actual_update_l1'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(pulse, report['training_applied_pulse_sum'], atol=1e-9, rtol=0)
    weak_pulses = np.asarray([row['weak_pulse'] for row in rows])
    reference_pulses = np.asarray([row['reference_pulse'] for row in rows])
    correlations = [float(np.corrcoef(weak_pulses[:, lane], reference_pulses[:, lane])[0, 1])
        if weak_pulses[:, lane].std() > 0 and reference_pulses[:, lane].std() > 0 else None for lane in range(n)]
    return {'training_rows': len(rows), 'rewarded_trials_per_lane': rewarded.tolist(),
        'rewarded_zero_eligibility_trials_per_lane': rewarded_zero.tolist(),
        'pulse_assignment_verified': True, 'native_reinforcement_mapping_verified': mapper is not None,
        'mean_neural_pulse': float(weak_pulses.mean()), 'mean_reference_pulse': float(reference_pulses.mean()),
        'neural_reference_pulse_correlation_per_lane': correlations,
        'updates_and_pulses_sum_to_report': True}


def audit_target_weights(config, mapping_path, state_path):
    """Independently align saved efficacy columns with anatomical post roots."""
    artifact = load_prepared_artifact(resolve_path(config, config.fly.artifact_path))
    topology = PlasticEdgeTopology.from_artifact(artifact, minimum_synapse_count=config.fly.minimum_synapse_count)
    mapping = MotorMapping.load(mapping_path)
    with np.load(state_path, allow_pickle=False) as state:
        delta = state['efficacy']-state['initial_efficacy']
        if delta.shape[1] != topology.edge_count:
            raise ValueError('saved state plastic topology differs')
        # This study applies positive native reinforcement only.
        if np.any(delta < 0):
            raise ValueError('positive-only native study depressed efficacy')
        allowed = set()
        roles = []
        for option, pool in enumerate(mapping.routing.head_routes['discard_card']):
            roots = mapping.output_root_ids[list(mapping.pool_indices[pool])]
            edges = np.isin(topology.post_root_ids, roots)
            allowed.update(np.flatnonzero(edges).tolist())
            modes = []
            for changes in delta[:, edges]:
                changed = changes[changes > 0]
                modes.append(float(np.unique(changed, return_counts=True)[1].max()/len(changed)) if len(changed) else None)
            roles.append({'slot': option, 'root_ids': roots.tolist(), 'plastic_edges': int(edges.sum()),
                'changed_edges_per_lane': np.count_nonzero(delta[:, edges], axis=1).tolist(),
                'modal_change_fraction_among_changed_per_lane': modes,
                'final_update_l1_per_lane': delta[:, edges].sum(axis=1, dtype=np.float64).tolist()})
        unrelated = np.ones(topology.edge_count, dtype=bool); unrelated[list(allowed)] = False
        if np.any(delta[:, unrelated]):
            raise ValueError('unrelated non-DISCARD-card weights changed')
    return roles


def describe_followup_choices(report):
    """Post-hoc one-step PLAY descriptions; never training input or selection."""
    corpus = PairCorpus(Path(report['corpus']))
    if corpus.sha256 != report['corpus_sha256']:
        raise ValueError('decision-audit corpus changed')
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False); env.reset([0])
    summaries = {name: {'trials': 0, 'weak_play_count_histogram': {}, 'missed_immediate_clear': 0,
        'weak_high_card_despite_stronger_available_hand': 0, 'weak_progress_sum': 0., 'reference_progress_sum': 0.}
        for name in ('neural-native', 'reference-native', 'reference-matched', 'frozen')}
    for row in report['after']['greedy']:
        for lane, (replicate, index) in enumerate(zip(report['replicate_ids'], row['case_indices'], strict=True)):
            rng = np.random.default_rng(derive_seed(VERSION, 'native', row['key'], replicate, index, 'display'))
            order = np.r_[rng.permutation(8), 8, 9]; next_order = np.r_[rng.permutation(8), 8, 9]
            corpus.restore(env, 0, index)
            action = empty_action_batch(1); action['action_type'][:] = 1; action['n_cards'][:] = report['count']
            action['cards'][0, :report['count']] = order[row['discard_cards'][lane][:report['count']]]
            visible = env.step(action).observations
            hand = [(int(c[:13].argmax())+2, int(c[13:17].argmax())) for c in visible['hand'][0, :8]]
            weak = next_order[[c for c in row['weak_play_cards'][lane] if c >= 0]]
            reference = next_order[[c for c in row['reference_play_cards'][lane] if c >= 0]]
            weak_score, weak_kind = score(hand, weak); ref_score, ref_kind = score(hand, reference)
            # Plain opening cards: independently check the score against native
            # progress already verified by the separate full referee replay.
            np.testing.assert_allclose(min(weak_score, 300)/300, row['weak_progress'][lane], atol=2e-6, rtol=0)
            np.testing.assert_allclose(min(ref_score, 300)/300, row['reference_progress'][lane], atol=2e-6, rtol=0)
            name = ('neural-native', 'reference-native', 'reference-matched', 'frozen')[report['roles'][lane]]
            out = summaries[name]; out['trials'] += 1
            histogram = out['weak_play_count_histogram']; key = str(len(weak)); histogram[key] = histogram.get(key, 0)+1
            out['missed_immediate_clear'] += int(ref_score >= 300 and weak_score < 300)
            out['weak_high_card_despite_stronger_available_hand'] += int(weak_kind == 'high_card' and ref_kind != 'high_card')
            out['weak_progress_sum'] += row['weak_progress'][lane]
            out['reference_progress_sum'] += row['reference_progress'][lane]
    for out in summaries.values():
        out['missed_immediate_clear_fraction'] = out['missed_immediate_clear']/out['trials']
        out['weak_high_card_despite_stronger_available_hand_fraction'] = out['weak_high_card_despite_stronger_available_hand']/out['trials']
        out['mean_weak_progress'] = out.pop('weak_progress_sum')/out['trials']
        out['mean_reference_progress'] = out.pop('reference_progress_sum')/out['trials']
    return {'scope': 'descriptive immediate-score comparisons on plain opening hands; not optimal long-term action labels', 'actors': summaries}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args(argv); folder = args.run_dir
    declaration = json.loads((folder/'declaration.json').read_text())
    result = json.loads((folder/'report.json').read_text())
    if result['status'] != 'completed':
        raise ValueError('completed comparison required')
    verify_sources(result['source_hashes']); verify_sources(result['artifact_sha256'])
    report = json.loads((folder/'native/report.json').read_text())
    verify_state({**report, 'study': 'followup'}, folder/'native/diagnostic-final-state.npz')
    with (folder/'native/training-trials.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    config = PlasticExperimentConfig.load(Path(declaration['config']))
    training = audit_training_rows(report, rows, config.reinforcement)
    native = replay({**report, 'calibration_rows': rows, 'before': {}, 'after': {}})
    weights = audit_target_weights(config, resolve_path(config, config.fly.motor_mapping_path),
                                  folder/'native/diagnostic-final-state.npz')
    fresh = json.loads((folder/'confirm-new/report.json').read_text())
    if advance_gate(fresh['summary'], declaration['readiness_rule']) != result['readiness']:
        raise ValueError('readiness evaluation differs')
    gate = next(row for row in weights if row['slot'] == 3)
    if not all(gate['changed_edges_per_lane'][lane] > 0 for lane in range(12) if lane % 4 != 3):
        raise ValueError('repaired output did not receive native learning in all plastic actors')
    artifacts = [folder/'report.json', folder/'native/training-trials.jsonl', folder/'native/diagnostic-final-state.npz']
    audit = {'status': 'completed', 'training': training, 'training_native_replay': native,
        'fresh_neural_play_decisions': describe_followup_choices(fresh),
        'anatomical_card_weight_audit': weights, 'readiness_recomputed': result['readiness'],
        'source_hashes_unchanged': True, 'assessor_sha256': sha256_file(Path(__file__)),
        'artifact_sha256': {str(path.resolve()): sha256_file(path) for path in artifacts}}
    write_json(folder/'assessment.json', audit)
    print(json.dumps({'training_native_actions': native['native_actions'], 'repaired_output': gate,
                      'readiness': result['readiness']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
