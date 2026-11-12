"""Recovery screening must retain comparability despite deliberate rule changes."""
import importlib
from pathlib import Path

import pytest

from flylatro.evaluation.compare import MATCHED_FIELDS


@pytest.fixture
def compare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('run_autonomous_recovery').paired_screen


def report():
    return {'version': 'frozen-plastic-evaluation-v2', 'learning_disabled': True,
        'training_action_schedule_replayed': False,
        'action_policy': 'independent-fixed-motor-choices-no-training-action-replay',
        'seed_stream': 'validation', 'win_ante': 1, 'seeds': [20,21],
        'components': {key: 'same' for key in MATCHED_FIELDS},
        'episodes': [{'seed': seed, 'balatro_seed': str(seed), 'blinds_beaten': 0,
            'won': False, 'decisions': 8, 'episode_return': .1} for seed in (20,21)]}


def test_recovery_allows_declared_rule_changes_and_pairs_actual_outcomes(compare):
    initial, trained = report(), report()
    trained['components']['plasticity_rule_sha256'] = 'new-rule'
    trained['components']['reinforcement_mapping_sha256'] = 'new-pulses'
    trained['episodes'][0].update(blinds_beaten=1, won=True)
    result = compare(trained, initial)
    assert result['mean_blinds_gain'] == .5
    assert result['trained_only_wins'] == 1
    assert result['episodes'] == 2


@pytest.mark.parametrize('mismatch', ['motor', 'seeds', 'episode-order', 'simulator-seed', 'replay', 'nonfinite'])
def test_recovery_rejects_incomparable_or_invalid_measurements(compare, mismatch):
    initial, trained = report(), report()
    if mismatch == 'motor': trained['components']['motor_mapping_sha256'] = 'different'
    if mismatch == 'seeds': trained['seeds'] = [22,23]
    if mismatch == 'episode-order': trained['episodes'].reverse()
    if mismatch == 'simulator-seed': trained['episodes'][0]['balatro_seed'] = 'different'
    if mismatch == 'replay': trained['training_action_schedule_replayed'] = True
    if mismatch == 'nonfinite': trained['episodes'][0]['episode_return'] = float('nan')
    with pytest.raises(ValueError): compare(trained, initial)
