"""Generate and validate a fresh, reward-free active-input-qualified mapping."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from audit_motor_learning_opportunities import audit
from card_learning.common import write_json
from card_learning.experiment import build_condition_stack, load_condition
from card_learning.motor import CardExploration, install_motor
from card_learning.pairs import PairCorpus
from flylatro.analysis.corpus import CalibrationCorpus, build_calibration_corpus, INVENTORY_NAVIGATION_RULE
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.protocol_materialize import render_toml
from learnable_motor import VERSION, learning_support, repair_mapping
from qualify_action_conditioned_motor import actual_checks
from run_card_membership import binary_probe
import train_discard_play_curriculum as delayed


def align(data, roots, key):
    positions = {int(root): i for i, root in enumerate(data['post_root_ids'])}
    return data[key][:, :, [positions[int(r)] for r in roots]].reshape(-1, len(roots))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--validate', action='store_true', help='validate a previously generated candidate; never refit from outcomes')
    p.add_argument('--heavy', action='store_true')
    p.add_argument('--steps', type=int, default=1500)
    args = p.parse_args(argv)
    folder = args.output_dir.resolve()
    source_dir = Path('artefacts/play-discard-motor-20261027-v4')
    cache = Path('artefacts/card-membership-20261027-v2/motor-opportunity-audit')
    activity_path = Path('artefacts/separated-motor-20261024-pilot-v2/calibration-activity.npz')
    condition_path = Path('runs/play-discard-learning-pilot-20261027-v1/conditions/subset-plastic/condition.json')
    checkpoint = Path('runs/play-discard-learning-pilot-20261027-v1/subset-plastic/plastic-checkpoint-final.pkl')
    pairs_path = Path('artefacts/card-membership-20261027-smoke-v1/binary/pairs.json')
    condition, _ = load_condition(condition_path)
    config = PlasticExperimentConfig.load(Path(condition['config']))
    source = MotorMapping.load(source_dir/'action-conditioned-motor.json')
    corpus_path = Path(config.calibration.corpus_path)
    corpus = CalibrationCorpus.load(corpus_path)
    masks = {k: v.copy() for k, v in corpus.masks.items()}
    masks['action_type_mask'][:, 3] = False
    if not args.validate:
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir(parents=True)
        original_report = json.loads((source_dir/'declaration.json').read_text())
        audit_report = json.loads((cache/'report.json').read_text())
        for report in (original_report, audit_report):
            for path, digest in report['source_hashes'].items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError('source provenance changed: '+path)
        if audit_report['corpus_sha256'] != corpus.sha256 or not audit_report['source_state_unchanged']:
            raise ValueError('coactivity corpus provenance differs')
        sources = {str(x.resolve()): sha256_file(x) for x in (
            source_dir/'action-conditioned-motor.json', source_dir/'config.toml', corpus_path,
            activity_path, cache/'activity.npz', cache/'report.json', condition_path, checkpoint, pairs_path,
            Path(__file__), Path(__file__).with_name('learnable_motor.py'))}
        write_json(folder/'declaration.json', {'version': VERSION, 'source_hashes': sources,
            'selection': 'repair failed width-one pools only; rank unused candidates by reward-free neural response; original quality gates unchanged',
            'minimum_coactive_fraction': .5, 'disabled_action_types': [3],
            'validation': 'fresh unit-weight learners; independent reward-free environment seeds; fixed old/new counterbalanced binary probe',
            'binary_steps': args.steps, 'binary_replicates': 3, 'binary_evaluation_repeats': 64,
            'reward_pulse': .05, 'no_candidate_selection_from_validation': True,
            'heldout_environment_seeds': list(range(281000, 281008)),
            'context_validation': 'fresh game states plus deterministic new display permutations of the existing inventory corpus; fresh neural seeds; report these scopes separately',
            'legacy_checkpoints': 'preserved; never resumed under the new mapping'})
        with np.load(activity_path, allow_pickle=False) as data, np.load(cache/'activity.npz', allow_pickle=False) as co:
            if not np.array_equal(data['output_root_ids'], source.output_root_ids):
                raise ValueError('canonical output universe differs')
            mapping, report = repair_mapping(source, data['output_hz'], align(co, source.output_root_ids, 'coactivity'),
                masks=masks, activity_order=data['corpus_indices'], coactivity_order=np.repeat(np.arange(len(corpus.observations['hand'])), co['coactivity'].shape[1]),
                metadata={'source_hashes': sources, 'initial_unit_weights': True, 'reward_free': True})
        mapping_path = mapping.save(folder/'candidate-motor.json')
        write_json(folder/'calibration-report.json', report)
        base = PlasticExperimentConfig.load(source_dir/'config.toml')
        fresh = replace(base, name='active-plastic-input-motor-v1',
            fly=replace(base.fly, motor_mapping_path=str(mapping_path)),
            calibration=replace(base.calibration, motor_calibration_report=str(folder/'calibration-report.json')))
        doc = asdict(fresh); doc.pop('source_path')
        (folder/'config.toml').write_text(render_toml(doc))
        write_json(folder/'status.json', {'status': 'candidate', 'validation_required': True,
            'candidate_sha256': mapping.sha256, 'replacements': report['replacements']})
        print(json.dumps({'status': 'candidate', 'replacements': report['replacements']}, indent=2))
        return 0

    declaration = json.loads((folder/'declaration.json').read_text())
    for path, digest in declaration['source_hashes'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('source changed: '+path)
    fresh = PlasticExperimentConfig.load(folder/'config.toml'); fresh.require_heavy_opt_in(args.heavy)
    candidate = MotorMapping.load(Path(fresh.fly.motor_mapping_path))
    status = json.loads((folder/'status.json').read_text())
    if status['candidate_sha256'] != candidate.sha256:
        raise ValueError('candidate changed after allocation')
    if (folder/'validation').exists():
        raise FileExistsError(folder/'validation')
    (folder/'validation').mkdir()
    def progress(row):
        print(json.dumps(row), flush=True)
        write_json(folder/'validation/status.json', {'status': 'running', **row})
    try:
        # Strictly load the source first. The comparison uses fresh local unit
        # states, not trained efficacy. New mapping is built in a separate stack.
        old = build_condition_stack(config, condition)
        load_plastic_checkpoint(checkpoint, old.trainer, expected_components=old.components)
        pairs = PairCorpus(pairs_path)
        progress({'phase': 'old-binary'})
        old_report, old_state = binary_probe(old.agent, pairs, steps=declaration['binary_steps'], replicates=3, repeats=64, progress=progress)
        write_json(folder/'validation/old-binary.json', old_report)
        np.savez_compressed(folder/'validation/old-state.npz', **old_state)
        del old
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        new = build_plastic_stack(fresh)
        install_motor(new, CardExploration(**condition['exploration']))
        # The checkpoint loader must reject the new mapping identity.
        try:
            load_plastic_checkpoint(checkpoint, new.trainer, expected_components=new.components)
        except ValueError as error:
            rejection = str(error)
        else:
            raise ValueError('legacy checkpoint was accepted under a changed mapping')
        progress({'phase': 'heldout-reward-free-contexts'})
        heldout = build_calibration_corpus(BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False),
            environment_seeds=declaration['heldout_environment_seeds'], states_per_seed=24,
            navigation_seed=281099, navigation_rule=INVENTORY_NAVIGATION_RULE)
        heldout.save(folder/'validation/reward-free-corpus.npz')
        heldout_masks = {k: v.copy() for k, v in heldout.masks.items()}
        heldout_masks['action_type_mask'][:, 3] = False
        # Random navigation at this budget does not reach inventory reliably.
        # Cover those heads with declared display perturbations of the original
        # corpus, reported separately from independently seeded game states.
        orders = [np.r_[np.random.default_rng(281500+i).permutation(int(n)), np.arange(int(n), 10)]
                  for i, n in enumerate(corpus.observations['hand_len'])]
        extra_obs, extra_masks = delayed.permute_rows(corpus.observations, masks, orders)
        combined_obs = {k: np.concatenate([v, extra_obs[k]]) for k, v in heldout.observations.items()}
        combined_masks = {k: np.concatenate([v, extra_masks[k]]) for k, v in heldout_masks.items()}
        np.savez_compressed(folder/'validation/context-display-fixtures.npz',
            **{'observation_'+k: v for k, v in extra_obs.items()}, **{'mask_'+k: v for k, v in extra_masks.items()})
        _, activity = audit(new.agent, combined_obs, combined_masks, repeats=2)
        np.savez_compressed(folder/'validation/reward-free-activity.npz', **activity)
        heldout_check, _ = learning_support(candidate, align(activity, candidate.output_root_ids, 'coactivity'),
            masks=combined_masks, order=np.repeat(np.arange(len(combined_obs['hand'])), 2))
        heldout_check['fresh_game_cases'] = len(heldout.observations['hand'])
        heldout_check['calibration_display_permutation_cases'] = len(extra_obs['hand'])
        write_json(folder/'validation/learning-support.json', heldout_check)
        if heldout_check['status'] != 'PASS':
            raise ValueError('candidate failed heldout active-input gate; no promotion')
        progress({'phase': 'new-binary'})
        new_report, new_state = binary_probe(new.agent, pairs, steps=declaration['binary_steps'], replicates=3, repeats=64, progress=progress)
        write_json(folder/'validation/new-binary.json', new_report)
        np.savez_compressed(folder/'validation/new-state.npz', **new_state)
        del new
        gc.collect(); torch.cuda.empty_cache()
        progress({'phase': 'native-interface-checks'})
        interface = actual_checks(fresh, pairs, cases=3, repeats=2)
        if any(sha256_file(Path(path)) != digest for path, digest in declaration['source_hashes'].items()):
            raise ValueError('source changed during validation')
        report = {'status': 'completed', 'mapping_sha256': candidate.sha256,
            'heldout_support': heldout_check['status'], 'legacy_checkpoint_rejection': rejection,
            'native_interface': interface, 'binary_steps': declaration['binary_steps'],
            'source_files_unchanged': True,
            'scope': 'learnability and interface qualification; no claim of improved native game performance'}
        write_json(folder/'validation/report.json', report)
        write_json(folder/'status.json', {'status': 'completed', 'candidate_sha256': candidate.sha256})
        print(json.dumps(report, indent=2))
    except BaseException as error:
        write_json(folder/'validation/status.json', {'status': 'failed', 'error': str(error)})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
