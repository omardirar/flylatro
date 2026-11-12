"""Offline diagnostics must not accept mismatched simulator traces."""
import importlib
import json
from pathlib import Path

import pytest

from flylatro.env.balatro_sim import BalatroSimAdapter, composite_actions_to_batch
from flylatro.env.types import ActionType, CompositeAction
from flylatro.evaluation.state_hash import hash_observation_row, terminal_hash


@pytest.fixture
def trace(tmp_path):
    pytest.importorskip('balatro_sim')
    env = BalatroSimAdapter(1, strict=True, win_ante=1)
    observations, _ = env.reset([20])
    rows = []
    for decision in range(5):
        action = CompositeAction(ActionType.SELECT_BLIND) if decision == 0 else CompositeAction(ActionType.PLAY_HAND, (0,))
        before = hash_observation_row(observations, 0)
        step = env.step(composite_actions_to_batch([action]))
        rows.append({'seed':20,'decision_id':decision,'action':action.to_payload(),
            'done':bool(step.dones[0]),'state_hash_before':before,
            'state_hash_after':terminal_hash(step.infos[0]) if step.dones[0] else hash_observation_row(step.observations, 0)})
        observations = step.observations
    assert rows[-1]['done']
    (tmp_path/'summary.json').write_text(json.dumps({'seeds':[20], 'win_ante':1}))
    (tmp_path/'transitions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return tmp_path


def audit_function(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('audit_frozen_replays').audit


def test_cpu_audit_checks_complete_real_trace_without_changing_evidence(trace, monkeypatch):
    before = (trace/'transitions.jsonl').read_bytes()
    result = audit_function(monkeypatch)(trace)
    assert result['transitions_checked'] == 5
    assert result['visible_play_card_counts'] == {1:4}
    assert result['plays_with_selected_visible_rank_pair'] == 0
    assert (trace/'transitions.jsonl').read_bytes() == before


def test_cpu_audit_refuses_wrong_state_hash(trace, monkeypatch):
    rows = [json.loads(line) for line in (trace/'transitions.jsonl').read_text().splitlines()]
    rows[2]['state_hash_after'] = 'wrong'
    (trace/'transitions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='Post-action replay mismatch'):
        audit_function(monkeypatch)(trace)
