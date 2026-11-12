"""Independently check saved old/new mapping qualification and learner states."""
import argparse
import json
from pathlib import Path

import numpy as np

from assess_training_bottlenecks import verify_state
from card_learning.common import write_json
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from learnable_motor import learning_support
from qualify_learnable_motor import align


def replay_interface(checks, corpus):
    """Replay saved interface choices through an independent native referee."""
    from flylatro.env.balatro_sim import BalatroSimAdapter
    from flylatro.env.upstream_contract import empty_action_batch
    from flylatro.seeds import derive_seed
    from qualify_action_conditioned_motor import VERSION
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    env.reset([0]); verified = 0
    trials = list(checks['actual_neural_trials'])
    trials += [{'case_index': r['case_index'], 'counts': [2, 5],
                'cards': [r['play_cards'], r['discard_cards']]}
               for r in checks['artificial_readout_opposing_choice_checks']]
    for trial in trials:
        index = trial['case_index']
        order = np.r_[np.random.default_rng(derive_seed(VERSION, index, 'display')).permutation(8), 8, 9]
        for kind in (0, 1):
            corpus.restore(env, 0, index)
            action = empty_action_batch(1); action['action_type'][:] = kind
            count = trial['counts'][kind]; action['n_cards'][:] = count
            action['cards'][0, :count] = order[trial['cards'][kind]]
            env.step(action); verified += 1
    return verified


def summarize(report):
    before = np.asarray([r['rewarded_probability'] for r in report['before']]).mean(axis=0)
    after = np.asarray([r['rewarded_probability'] for r in report['after']]).mean(axis=0)
    plastic = np.asarray(report['plastic'])
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('nonfinite choice probabilities')
    if not np.array_equal(before[~plastic], after[~plastic]):
        raise ValueError('frozen choice probabilities changed')
    return {'plastic_before': before[plastic].tolist(), 'plastic_after': after[plastic].tolist(),
        'plastic_change': (after-before)[plastic].tolist(),
        'mean_before': float(before[plastic].mean()), 'mean_after': float(after[plastic].mean()),
        'frozen_choices_exact': True, 'update_l1': np.asarray(report['actual_update_l1'])[plastic].tolist(),
        'target_slots': np.asarray(report['target_slots'])[plastic].tolist()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args(argv); folder = args.run_dir
    declaration = json.loads((folder/'declaration.json').read_text())
    validation = json.loads((folder/'validation/report.json').read_text())
    if validation['status'] != 'completed':
        raise ValueError('completed validation required')
    for path, digest in declaration['source_hashes'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('source changed: '+path)
    mapping = MotorMapping.load(folder/'candidate-motor.json')
    if mapping.sha256 != validation['mapping_sha256']:
        raise ValueError('validated candidate changed')
    if 'motor_mapping_sha256' not in validation['legacy_checkpoint_rejection']:
        raise ValueError('legacy checkpoint rejection does not identify changed motor routing')
    fit = json.loads((folder/'calibration-report.json').read_text())
    if fit['qualification']['gates']['status'] != 'PASS' or fit['after']['status'] != 'PASS':
        raise ValueError('calibration gates failed')
    results = {}
    for name in ('old', 'new'):
        report = json.loads((folder/f'validation/{name}-binary.json').read_text())
        verify_state(report, folder/f'validation/{name}-state.npz')
        if not report['source_state_unchanged'] or not report['unrelated_weights_exact']:
            raise ValueError('source or unrelated weights changed')
        results[name] = summarize(report)
    # Replicates 0 and 2 use other slot pairs. A two-pool mapping repair must
    # leave their binary trajectories bit-identical, including fast state.
    unchanged_lanes = [0, 1, 2, 3, 8, 9, 10, 11]
    with np.load(folder/'validation/old-state.npz', allow_pickle=False) as old_state, \
            np.load(folder/'validation/new-state.npz', allow_pickle=False) as new_state:
        for key in old_state.files:
            np.testing.assert_array_equal(old_state[key][unchanged_lanes], new_state[key][unchanged_lanes])
    # Check actual input-gate reproduction on independently seeded game states
    # and supplemental inventory display fixtures, without reading outcomes.
    from flylatro.analysis.corpus import CalibrationCorpus
    corpus = CalibrationCorpus.load(folder/'validation/reward-free-corpus.npz')
    masks = {k: v.copy() for k, v in corpus.masks.items()}; masks['action_type_mask'][:, 3] = False
    with np.load(folder/'validation/context-display-fixtures.npz', allow_pickle=False) as fixtures:
        masks = {k: np.concatenate([v, fixtures['mask_'+k]]) for k, v in masks.items()}
    with np.load(folder/'validation/reward-free-activity.npz', allow_pickle=False) as activity:
        coactivity = align(activity, mapping.output_root_ids, 'coactivity')
        support, _ = learning_support(mapping, coactivity,
            masks=masks, order=np.repeat(np.arange(len(masks['action_type_mask'])), 2))
        fresh_count = len(corpus.observations['hand'])
        scopes = {}
        for name, start, stop in (('fresh_games', 0, fresh_count),
                                  ('inventory_display_fixtures', fresh_count, len(masks['action_type_mask']))):
            check, _ = learning_support(mapping, coactivity[2*start:2*stop],
                masks={k: v[start:stop] for k, v in masks.items()}, order=np.repeat(np.arange(stop-start), 2))
            if check['status'] != 'PASS':
                raise ValueError('active-input gate failed separately on '+name)
            scopes[name] = check
    if support['status'] != 'PASS':
        raise ValueError('saved heldout activity fails input gate')
    # This lane was the zero-update target in the original experiment.
    lane = 3
    repaired = {'slot': results['new']['target_slots'][lane],
        'old_update_l1': results['old']['update_l1'][lane], 'new_update_l1': results['new']['update_l1'][lane],
        'old_probability_change': results['old']['plastic_change'][lane],
        'new_probability_change': results['new']['plastic_change'][lane]}
    if repaired['slot'] != 3 or repaired['old_update_l1'] != 0 or repaired['new_update_l1'] <= 0:
        raise ValueError('previously inactive target did not demonstrate restored updates')
    from card_learning.pairs import PairCorpus
    pair_paths = [Path(path) for path in declaration['source_hashes'] if Path(path).name == 'pairs.json']
    if len(pair_paths) != 1:
        raise ValueError('expected one declared binary/interface corpus')
    native_actions = replay_interface(validation['native_interface'], PairCorpus(pair_paths[0]))
    artifacts = [folder/'declaration.json', folder/'config.toml', folder/'candidate-motor.json', folder/'calibration-report.json',
        folder/'validation/report.json', folder/'validation/learning-support.json',
        folder/'validation/reward-free-corpus.npz', folder/'validation/context-display-fixtures.npz',
        folder/'validation/reward-free-activity.npz']
    artifacts += [folder/f'validation/{name}-{kind}.{ext}' for name in ('old', 'new')
                  for kind, ext in (('binary', 'json'), ('state', 'npz'))]
    result = {'status': 'completed', 'summary': results, 'repaired_target': repaired,
        'native_interface_actions_replayed': native_actions,
        'reward_free_validation_scopes': scopes,
        'checks': {'saved_states_verified': True, 'heldout_input_gate_reproduced': True,
                   'original_quality_gates_pass': True, 'source_files_unchanged': True,
                   'unaffected_binary_replicate_states_exact': True},
        'artifact_sha256': {str(path.resolve()): sha256_file(path) for path in artifacts},
        'assessor_sha256': sha256_file(Path(__file__)),
        'limits': 'three learner seeds on one mapping, counterbalanced fixed targets; diagnostic reward; no native game-performance inference'}
    write_json(folder/'assessment.json', result)
    write_json(folder/'validation/status.json', {'status': 'completed', 'phase': 'independently-assessed',
                                               'assessment': str((folder/'assessment.json').resolve())})
    print(json.dumps(result['repaired_target'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
