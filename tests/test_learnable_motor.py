"""Active outputs need active plastic inputs in their actual choice contexts."""
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from learnable_motor import learning_support, repair_mapping
from flylatro.env.upstream_contract import MASK_SPEC, UpstreamActionType as A
from flylatro.interface.action_conditioned_motor import structural_mapping, calibrate_action_conditioned, separation_summary
from flylatro.interface.motor import MotorMapping
from flylatro.interface.motor_calibration import MotorCalibrationError, MotorCalibrationThresholds
from test_discard_studies import native_corpus


def legal(n=12):
    masks = {k: np.zeros((n, *shape), dtype=dtype) for k, (shape, dtype) in MASK_SPEC.items()}
    masks['action_type_mask'][:, [int(A.PLAY_HAND), int(A.DISCARD)]] = True
    masks['card_select_mask'][:, :8] = True
    return masks


def test_firing_output_with_zero_kc_coactivity_fails_but_disabled_skip_does_not():
    mapping = structural_mapping(np.arange(96), pool_width=1)
    bad = mapping.pool_indices[mapping.routing.head_routes['discard_card'][3]][0]
    skip = mapping.pool_indices[mapping.routing.head_routes['action_type'][int(A.SKIP_BLIND)]][0]
    coactivity = np.ones((12, 96)); coactivity[:, [bad, skip]] = 0
    report, _ = learning_support(mapping, coactivity, masks=legal())
    failures = [r for r in report['roles'] if r['status'] == 'FAIL']
    assert len(failures) == 1 and failures[0]['head'] == 'discard_card' and failures[0]['option'] == 3
    assert next(r for r in report['roles'] if r['head'] == 'action_type' and r['option'] == 3)['status'] == 'UNEXERCISED'
    coactivity[:, bad] = 1
    assert learning_support(mapping, coactivity, masks=legal())[0]['status'] == 'PASS'


def test_support_is_context_specific_and_ignores_forced_heads_and_padding():
    mapping = structural_mapping(np.arange(96), pool_width=1)
    masks = legal()
    masks['action_type_mask'][:] = False
    masks['action_type_mask'][:6, int(A.PLAY_HAND)] = True
    masks['action_type_mask'][6:, int(A.DISCARD)] = True
    post = mapping.pool_indices[mapping.routing.head_routes['discard_card'][3]][0]
    coactivity = np.ones((12, 96)); coactivity[6:, post] = 0
    r, eligible = learning_support(mapping, coactivity, masks=masks)
    assert r['status'] == 'FAIL' and not eligible['discard_card_slot'][post]
    # Activity while PLAY is read cannot qualify the DISCARD output.
    row = next(x for x in r['roles'] if x['head'] == 'discard_card' and x['option'] == 3)
    assert row['competing_samples'] == 6 and row['coactive_fractions'] == [0]
    assert all(x['status'] == 'UNEXERCISED' for x in r['roles'] if x['head'] == 'action_type')
    assert next(x for x in r['roles'] if x['head'] == 'discard_card' and x['option'] == 9)['status'] == 'UNEXERCISED'


def test_shared_target_requires_coactivity_in_each_exercised_interpretation():
    mapping = structural_mapping(np.arange(96), pool_width=1)
    masks = legal()
    masks['action_type_mask'][:] = False
    masks['action_type_mask'][:6, int(A.BUY_SHOP)] = True
    masks['action_type_mask'][6:, int(A.SELL_JOKER)] = True
    masks['shop_target_mask'][:, :2] = True
    masks['joker_target_mask'][:, :2] = True
    post = mapping.pool_indices[mapping.routing.head_routes['shop'][0]][0]
    mass = np.ones((12, 96)); mass[6:, post] = 0
    report, eligible = learning_support(mapping, mass, masks=masks)
    assert report['status'] == 'FAIL' and not eligible['context_slot'][post]
    assert next(x for x in report['roles'] if x['head'] == 'shop' and x['option'] == 0)['status'] == 'PASS'
    assert next(x for x in report['roles'] if x['head'] == 'joker' and x['option'] == 0)['status'] == 'FAIL'


@pytest.fixture
def calibrated():
    values = np.maximum(40+np.random.default_rng(293).normal(0, 10, (128, 96)), 0)
    masks = {k: np.ones((128, *shape), dtype=dtype) for k, (shape, dtype) in MASK_SPEC.items()}
    mapping, _ = calibrate_action_conditioned(np.arange(96), values, masks=masks,
        attempts=2, annealing_steps=100, pool_width=1)
    return mapping, values, masks


def test_repair_changes_only_failed_pool_and_keeps_old_gates_identity_and_architecture(calibrated, tmp_path):
    source, values, masks = calibrated
    bad_pool = source.routing.head_routes['discard_card'][3]
    mass = np.ones_like(values); mass[:, source.pool_indices[bad_pool][0]] = 0
    repaired, report = repair_mapping(source, values, mass, masks=masks)
    assert len(report['replacements']) == 1 and report['replacements'][0]['pool'] == bad_pool
    assert [i for i, (a, b) in enumerate(zip(source.pool_indices, repaired.pool_indices)) if a != b] == [bad_pool]
    assert report['qualification']['gates']['status'] == 'PASS' and report['after']['status'] == 'PASS'
    assert report['thresholds'] == asdict(MotorCalibrationThresholds(minimum_pool_width=1))
    assert all(x == 0 for x in separation_summary(repaired)['core_overlap_neuron_counts'].values())
    assert np.array_equal(repaired.output_root_ids, source.output_root_ids)
    assert not report['reward_used'] and not report['outcome_information_used']
    assert repaired.sha256 != source.sha256
    assert MotorMapping.load(repaired.save(tmp_path/'new.json')).sha256 == repaired.sha256


def test_no_supported_replacement_fails_closed(calibrated):
    source, values, masks = calibrated
    with pytest.raises(MotorCalibrationError, match='no repair passed'):
        repair_mapping(source, values, np.zeros_like(values), masks=masks)
    with pytest.raises(ValueError, match='nonnegative'):
        learning_support(source, np.full_like(values, -1), masks=masks)
    with pytest.raises(ValueError, match='samples and mask order differ'):
        learning_support(source, np.ones((1, 96)), masks=masks)
    with pytest.raises(ValueError, match='fraction'):
        learning_support(source, np.ones_like(values), masks=masks, minimum_fraction=0)


def test_active_inputs_cannot_bypass_original_response_gates(calibrated):
    source, values, masks = calibrated
    with pytest.raises(MotorCalibrationError, match='no repair passed'):
        repair_mapping(source, np.zeros_like(values), np.ones_like(values), masks=masks)


def test_binary_assessment_refuses_changed_frozen_controls():
    from assess_learnable_motor import summarize
    report = {'before': [{'rewarded_probability': [.025, .025]}],
        'after': [{'rewarded_probability': [.975, .025]}], 'plastic': [True, False],
        'actual_update_l1': [1., 0.], 'target_slots': [3, 3]}
    summary = summarize(report)
    assert summary['plastic_change'] == pytest.approx([.95]) and summary['frozen_choices_exact']
    report['after'][0]['rewarded_probability'][1] = .975
    with pytest.raises(ValueError, match='frozen choice probabilities changed'):
        summarize(report)


def test_coactivity_mask_order_and_threshold_boundary_are_explicit():
    mapping = structural_mapping(np.arange(96), pool_width=1)
    mass = np.ones((24, 96))
    post = mapping.pool_indices[mapping.routing.head_routes['discard_card'][3]][0]
    mass[::2, post] = 0
    report, _ = learning_support(mapping, mass, masks=legal(), order=np.repeat(np.arange(12), 2))
    assert report['status'] == 'PASS'
    mass[1, post] = 0
    assert learning_support(mapping, mass, masks=legal(), order=np.repeat(np.arange(12), 2))[0]['status'] == 'FAIL'


def test_illegal_option_samples_do_not_dilute_its_learning_support():
    mapping = structural_mapping(np.arange(96), pool_width=1)
    masks = legal(); masks['card_select_mask'][:8, 3] = False
    post = mapping.pool_indices[mapping.routing.head_routes['discard_card'][3]][0]
    mass = np.ones((12, 96)); mass[:8, post] = 0
    report, _ = learning_support(mapping, mass, masks=masks)
    row = next(r for r in report['roles'] if r['head'] == 'discard_card' and r['option'] == 3)
    assert row['competing_samples'] == 4 and row['status'] == 'PASS'


def test_cached_coactivity_is_reordered_by_root_identity():
    from qualify_learnable_motor import align
    data = {'post_root_ids': np.array([300, 100, 200]),
            'coactivity': np.array([[[3., 1., 2.], [30., 10., 20.]]])}
    np.testing.assert_array_equal(align(data, [100, 200, 300], 'coactivity'), [[1., 2., 3.], [10., 20., 30.]])
    with pytest.raises(KeyError):
        align(data, [400], 'coactivity')


def test_saved_interface_actions_are_checked_by_native_referee(native_corpus):
    from assess_learnable_motor import replay_interface
    check = {'actual_neural_trials': [{'case_index': native_corpus.indices('training')[0],
        'counts': [2, 2], 'cards': [[0, 1], [2, 3]]}],
        'artificial_readout_opposing_choice_checks': []}
    assert replay_interface(check, native_corpus) == 2
    check['actual_neural_trials'][0]['cards'][1] = [2, 2]
    with pytest.raises(ValueError):
        replay_interface(check, native_corpus)
