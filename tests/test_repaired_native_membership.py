"""Readiness needs actual neural PLAY gains and exact source/control handling."""
from dataclasses import replace
from pathlib import Path
import sys
from copy import deepcopy

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_repaired_native_membership import advance_gate, fresh_stack, verify_sources
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.action_conditioned_motor import structural_mapping
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from test_discard_studies import conditioned_toy, native_corpus


RULE = {'minimum_native_progress_gain': .005, 'minimum_positive_learners': 2}


def result(change=.01, learners=(.01, .01, .01), interval=(.003, .02)):
    return {'neural-native': {'weak': {'mean_change': change, 'change_per_learner': list(learners),
        'case_bootstrap_95_interval_conditional_on_these_learners': list(interval)}},
        'reference-native': {'reference': {'mean_change': 1.}}}


def test_reference_gain_alone_does_not_qualify_native_scale_up():
    assert not advance_gate(result(change=-.001, learners=(-.001,)*3, interval=(-.003, .001)), RULE)['advance_to_free_action_pilot']


@pytest.mark.parametrize('summary', [result(change=.001), result(learners=(.05, -.01, -.01)),
                                     result(interval=(-.001, .03))])
def test_readiness_requires_mean_multiple_learners_and_case_signal(summary):
    assert not advance_gate(summary, RULE)['advance_to_free_action_pilot']


def test_native_signal_only_qualifies_bounded_free_action_pilot():
    gate = advance_gate(result(), RULE)
    assert gate['advance_to_free_action_pilot']
    assert not gate['long_training_ready']


def test_declared_source_changes_are_rejected(tmp_path):
    path = tmp_path/'source'; path.write_text('first')
    sources = {str(path): sha256_file(path)}
    verify_sources(sources)
    path.write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        verify_sources(sources)


def test_fresh_stack_requires_qualified_mapping_and_no_skip(tmp_path):
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, environment=replace(config.environment, allow_skip_blinds=False),
        plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    roots = build_plastic_stack(config).agent.processor.output_root_ids
    mapping = structural_mapping(roots, pool_width=1)
    path = mapping.save(tmp_path/'mapping.json')
    config = replace(config, fly=replace(config.fly, motor_mapping_path=str(path), motor_pool_width=1))
    exploration = {'probability': .05}
    stack = fresh_stack(config, exploration, mapping.sha256)
    assert np.all(stack.agent.plasticity.state.state_dict()['efficacy'] == 1)
    with pytest.raises(ValueError, match='mapping differs'):
        fresh_stack(config, exploration, 'wrong')
    with pytest.raises(ValueError, match='disabled blind skipping'):
        fresh_stack(replace(config, environment=replace(config.environment, allow_skip_blinds=True)), exploration, mapping.sha256)


def test_native_training_audit_checks_treatment_pulses_and_complete_sequence(native_corpus, tmp_path):
    from assess_repaired_native_membership import audit_training_rows, describe_followup_choices
    from run_card_membership import native_probe
    from confirm_card_membership import evaluate
    rows = []
    agent = conditioned_toy(tmp_path).agent
    report, state = native_probe(agent, native_corpus,
        steps=3, replicates=1, repeats=1, calibration_trials=3, log=rows.append)
    result = audit_training_rows(report, rows, agent.reinforcement.config)
    assert result['training_rows'] == 3 and result['pulse_assignment_verified'] and result['native_reinforcement_mapping_verified']
    altered = deepcopy(rows); altered[0]['applied_pulse'][2] += .01
    with pytest.raises(AssertionError):
        audit_training_rows(report, altered)
    with pytest.raises(ValueError, match='incomplete or out of order'):
        audit_training_rows(report, rows[:-1])
    altered = deepcopy(rows); altered[0]['update_l1'][3] = .01
    with pytest.raises(ValueError, match='frozen actor changed'):
        audit_training_rows(report, altered)
    confirmation = evaluate(agent, native_corpus, state, replicates=1, repeats=1)
    descriptions = describe_followup_choices(confirmation)
    assert all(row['trials'] == len(native_corpus.indices('validation')) for row in descriptions['actors'].values())
    confirmation['after']['greedy'][0]['weak_progress'][0] += .1
    with pytest.raises(AssertionError):
        describe_followup_choices(confirmation)
