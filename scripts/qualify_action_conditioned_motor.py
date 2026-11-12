"""Calibrate a fresh PLAY/DISCARD mapping and verify actual decoding and credit.

The cached neural responses are hash-verified reward-free unit-weight activity,
not trained weights. Old checkpoints are never loaded into the new interface.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from card_learning.common import write_json
from card_learning.pairs import PairCorpus, pair_slots
from flylatro.analysis.card_selection import permute_hand, plastic_hashes
from flylatro.analysis.corpus import CalibrationCorpus
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import UpstreamActionType as A, empty_action_batch, validate_strict_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.interface.action_conditioned_motor import calibrate_action_conditioned
from flylatro.interface.motor import FixedMotorInterface, MotorMapping
from flylatro.learning.choice_credit_pilot import implementation_hash
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack, resolve_path
from flylatro.learning.protocol_materialize import render_toml
from flylatro.seeds import derive_seed


VERSION = 'play-discard-motor-qualification-v1'


def actual_checks(config, corpus, *, cases=13, repeats=4):
    """Fresh real flies: same neural activity/hand, forced action type only.

    Separately, a labelled artificial output stimulus verifies that the decoder
    can play the pair and discard non-pair cards; this is not learned behavior.
    """
    stack = build_plastic_stack(config)
    agent, env = stack.agent, BalatroSimAdapter(2, win_ante=1, allow_skip_blinds=False)
    env.reset([0, 0])
    before = plastic_hashes(agent.plasticity.state)
    rows, forced, update_checks, distinct, distinct_top2 = [], [], 0, 0, 0
    by_rank = {}
    for index in corpus.indices('training'):
        by_rank.setdefault(corpus.cases[index]['pair_rank'], index)
    indices = list(by_rank.values())[:cases]
    if len(indices) != cases:
        raise ValueError('not enough distinct training ranks')
    mapping = agent.motor.mapping
    for index in indices:
        for row in range(2):
            corpus.restore(env, row, index)
        raw_obs, raw_masks = env.observe()
        order = np.r_[np.random.default_rng(derive_seed(VERSION, index, 'display')).permutation(8), 8, 9]
        obs, masks = permute_hand(raw_obs, raw_masks, order)
        masks['action_type_mask'][:] = False
        masks['action_type_mask'][0, int(A.PLAY_HAND)] = True
        masks['action_type_mask'][1, int(A.DISCARD)] = True
        # Labelled decoder fixture: inject known readout activity AFTER neural
        # processing. No labels enter calibration or learned weights.
        pair = pair_slots(obs)
        nonpair = [slot for slot in range(8) if slot not in pair][:5]
        normalization = mapping.normalization
        fixture = np.zeros((2, len(mapping.output_root_ids)), dtype=np.float64)
        for pool, members in enumerate(mapping.pool_indices):
            if mapping.routing.pool_groups[pool] in ('card_count', 'card_slot', 'discard_card_count', 'discard_card_slot'):
                fixture[:, list(members)] = normalization.baseline[pool]
        for head, choices in (('card', pair), ('card_count', [2]),
                              ('discard_card', nonpair), ('discard_card_count', [5])):
            for choice in choices:
                pool = mapping.routing.head_routes[head][choice]
                fixture[:, list(mapping.pool_indices[pool])] = normalization.baseline[pool]+10*normalization.scale[pool]
        selected = agent.motor.decode(fixture, masks, deterministic=True)
        if (set(selected['cards'][0, :2]) != set(pair) or selected['n_cards'].tolist() != [2, 5]
                or set(selected['cards'][1, :5]) & set(pair)):
            raise ValueError('decoder cannot express opposing PLAY/DISCARD preferences')
        forced.append({'case_index': index, 'pair_slots': list(pair),
                       'play_cards': selected['cards'][0, :2].tolist(),
                       'discard_cards': selected['cards'][1, :5].tolist(),
                       'labelled_artificial_readout_fixture': True})
        submitted = {key: value.copy() for key, value in selected.items()}
        for row in range(2):
            corpus.restore(env, row, index)
            count = int(selected['n_cards'][row])
            submitted['cards'][row, :count] = order[selected['cards'][row, :count]]
        env.step(submitted)
        for repeat in range(repeats):
            seed = derive_seed(VERSION, index, repeat, 'neural')
            neural = agent.processor.process(obs, fly_seeds=[seed, seed], efficacy=agent.plasticity.state.efficacy)
            if not np.array_equal(neural.output_activity[0], neural.output_activity[1]):
                raise ValueError('identical stimuli/seeds did not produce identical neural responses')
            actions = agent.motor.decode(neural.output_activity, masks, deterministic=True)
            validate_strict_action_batch(actions, masks)
            scores = agent.motor.head_scores(neural.output_activity)
            legal_slots = np.flatnonzero(masks['card_select_mask'][0])
            top2 = [legal_slots[np.argsort(-scores[head][0, legal_slots], kind='stable')[:2]].tolist()
                    for head in ('card', 'discard_card')]
            distinct_top2 += set(top2[0]) != set(top2[1])
            gate = agent.credit.mask(actions, masks)
            selected_edges = np.asarray(state_numpy(gate)).copy()
            # Diagnostic equal positive pulses check immediate choice credit;
            # they are not native discard reward or an online training run.
            agent.plasticity.record_activity(neural.edge_pre_activity, neural.edge_post_activity, edge_credit=gate)
            agent.plasticity.apply_dopamine(.01, 0)
            delta = agent.plasticity.state.state_dict()['efficacy']-1
            if np.any(delta[~selected_edges] != 0):
                raise ValueError('unselected edge received a fresh update')
            for row, inactive in ((0, ('discard_card', 'discard_card_count')), (1, ('card', 'card_count'))):
                pools = [pool for head in inactive for pool in mapping.routing.head_routes[head]]
                roots = {int(mapping.output_root_ids[i]) for pool in pools for i in mapping.pool_indices[pool]}
                inactive_edges = np.isin(agent.plasticity.topology.post_root_ids, list(roots))
                if selected_edges[row, inactive_edges].any() or np.any(delta[row, inactive_edges]):
                    raise ValueError('PLAY/DISCARD update leaked into inactive action heads')
                if not np.any(delta[row] > 0):
                    raise ValueError('credit test did not exercise a real plastic update')
            update_checks += 2
            # Restore exact initial state after the diagnostic intervention.
            values = agent.plasticity.state.state_dict()
            for key in values:
                if key == 'efficacy':
                    values[key][...] = values['initial_efficacy']
                elif key != 'initial_efficacy':
                    values[key].fill(0)
            factory = type(agent.plasticity.state)
            state = factory.from_state_dict(values, device=str(agent.plasticity.state.device))
            agent.plasticity.state = state
            # Credit's reusable mask remains valid on the same device/shape.
            cards = [actions['cards'][row, :int(actions['n_cards'][row])].tolist() for row in range(2)]
            distinct += set(cards[0]) != set(cards[1])
            native = {key: value.copy() for key, value in actions.items()}
            for row in range(2):
                corpus.restore(env, row, index)
                count = int(actions['n_cards'][row])
                native['cards'][row, :count] = order[actions['cards'][row, :count]]
            env.step(native)  # Strict native referee checks both action paths.
            rows.append({'case_index': index, 'repeat': repeat, 'cards': cards,
                         'same_count_top2': top2,
                         'counts': actions['n_cards'].tolist(), 'inactive_action_update_zero': True})
    if plastic_hashes(agent.plasticity.state) != before:
        raise ValueError('qualification did not restore exact initial plastic state')
    return {'fresh_unit_weights': True, 'source_checkpoint_loaded': False,
            'cases': cases, 'repeats': repeats, 'actual_neural_trials': rows,
            'actual_different_card_sets': int(distinct), 'actual_trials': len(rows),
            'actual_different_top2_rankings': int(distinct_top2),
            'inactive_action_zero_update_checks': update_checks,
            'artificial_readout_opposing_choice_checks': forced,
            'plastic_state_restored_exactly': True,
            'interpretation': 'interface expressivity, native legality and immediate credit isolation; no learned pair/discard strategy claim'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pilot', type=Path, default=Path('artefacts/separated-motor-20261024-pilot-v2'))
    parser.add_argument('--pair-corpus', type=Path, default=Path('artefacts/card-learning-20261024-v1/pairs.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=32)
    parser.add_argument('--pool-width', type=int, choices=(1, 2), default=1)
    parser.add_argument('--cases', type=int, default=13)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--heavy', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    base = PlasticExperimentConfig.load(args.source_pilot/'configs/separated.toml')
    base.require_heavy_opt_in(args.heavy)
    source = MotorMapping.load(args.source_pilot/'separated-motor.json')
    activity_path = args.source_pilot/'calibration-activity.npz'
    old_report = json.loads((args.source_pilot/'calibration-report.json').read_text())
    if sha256_file(activity_path) != old_report['activity']['activity_sha256']:
        raise ValueError('cached reward-free activity hash differs')
    corpus = CalibrationCorpus.load(resolve_path(base, base.calibration.corpus_path))
    if corpus.sha256 != old_report['activity']['corpus_sha256'] or not old_report['activity']['initial_unit_weights']:
        raise ValueError('cached activity corpus or unit-weight provenance differs')
    with np.load(activity_path, allow_pickle=False) as data:
        values, indices, roots = data['output_hz'].copy(), data['corpus_indices'].copy(), data['output_root_ids'].copy()
    if not np.array_equal(roots, source.output_root_ids):
        raise ValueError('cached neural output universe differs')
    masks = {k: v.copy() for k, v in corpus.masks.items()}
    masks['action_type_mask'][:, int(A.SKIP_BLIND)] = False
    args.output_dir.mkdir(parents=True)
    started, implementation = time.perf_counter(), implementation_hash()
    sources = {str(p.resolve()): sha256_file(p) for p in (base.source_path, activity_path,
        args.source_pilot/'separated-motor.json', args.source_pilot/'calibration-report.json',
        resolve_path(base, base.calibration.corpus_path), args.pair_corpus, Path(__file__))}
    declaration = {'version': VERSION, 'implementation_sha256': implementation, 'source_hashes': sources,
        'allocation': 'reward-free cached unit-weight rates and legality; exclusive core; declared width; original response/diversity/coverage gates',
        'pool_width': args.pool_width, 'exclusive_core_neurons': 45*args.pool_width,
        'attempts': args.attempts, 'test_cases': args.cases, 'neural_repeats': args.repeats,
        'legacy_checkpoints': 'unchanged; this new mapping uses fresh unit weights',
        'tests': 'artificial opposing readout, real identical-neural-input decoding, native legality, isolated positive diagnostic credit'}
    write_json(args.output_dir/'declaration.json', declaration)
    def progress(row):
        print(json.dumps(row), flush=True)
        write_json(args.output_dir/'status.json', {'status': 'running', **row})
    try:
        mapping, report = calibrate_action_conditioned(roots, values, masks=masks, order=indices,
            metadata={'cached_activity_sha256': sha256_file(activity_path), 'corpus_sha256': corpus.sha256,
                      'initial_unit_weights': True, 'reward_free': True}, attempts=args.attempts,
            candidate_set_sha256=source.candidate_set_sha256, pool_width=args.pool_width, progress=progress)
        mapping_path = mapping.save(args.output_dir/'action-conditioned-motor.json')
        write_json(args.output_dir/'calibration-report.json', report)
        write_json(args.output_dir/'preflight-thresholds.json', {'minimum_motor_pool_width': args.pool_width})
        config = replace(base, name='play-discard-separated-v1',
            environment=replace(base.environment, num_envs=2),
            fly=replace(base.fly, artifact_path=str(resolve_path(base, base.fly.artifact_path)),
                        motor_mapping_path=str(mapping_path.resolve()), motor_pool_width=args.pool_width),
            calibration=replace(base.calibration, corpus_path=str(resolve_path(base, base.calibration.corpus_path))))
        document = asdict(config); document.pop('source_path')
        config_path = args.output_dir/'config.toml'; config_path.write_text(render_toml(document))
        progress({'phase': 'native-interface-checks'})
        checks = actual_checks(PlasticExperimentConfig.load(config_path), PairCorpus(args.pair_corpus),
                               cases=args.cases, repeats=args.repeats)
        if implementation_hash() != implementation or any(sha256_file(Path(p)) != h for p, h in sources.items()):
            raise ValueError('source identity changed during qualification')
        write_json(args.output_dir/'report.json', {**declaration, 'status': 'completed',
            'checks': checks, 'mapping_sha256': mapping.sha256, 'config': str(config_path.resolve()),
            'calibration_gates': report['qualification']['gates'], 'wall_seconds': time.perf_counter()-started})
        write_json(args.output_dir/'status.json', {'status': 'completed'})
    except BaseException as error:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
            'report': getattr(error, 'report', None)})
        raise
    print(json.dumps({'status': 'completed', 'output': str(args.output_dir)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
