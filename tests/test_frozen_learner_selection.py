"""Multi-fly evaluation selects a verified row, never pooled learned weights."""
import hashlib
import json
import pickle

import numpy as np
import pytest

from flylatro.evaluation.plastic_cli import _select_frozen_learner, main as evaluate_main
from flylatro.fly.mushroom_body.state import PlasticEdgeState, TorchPlasticEdgeState, state_numpy
from flylatro.learning.cli import main as train_main
from helpers import write_config


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_selected_row_is_an_independent_copy_of_all_plastic_state(backend):
    if backend == 'torch': pytest.importorskip('torch')
    state = PlasticEdgeState.initialize(3, learners=3)
    state.efficacy[:] = [[1,1,1], [.8,.9,1], [1.1,1.2,1.3]]
    state.eligibility[:] = [[.1,.1,.1], [.2,.2,.2], [.3,.3,.3]]
    state.decision_count[:] = [10,20,30]
    source = TorchPlasticEdgeState.from_state_dict(state.state_dict(), device='cpu') if backend == 'torch' else state
    before = source.state_dict()
    selected = _select_frozen_learner(source, 1)
    assert selected.learners == 1
    for key, array in before.items():
        np.testing.assert_array_equal(selected.state_dict()[key], array[1:2])
    if backend == 'torch': selected.efficacy.fill_(.7)
    else: selected.efficacy[:] = .7
    for key, array in before.items(): np.testing.assert_array_equal(source.state_dict()[key], array)
    with pytest.raises(ValueError, match='outside'): _select_frozen_learner(source, 3)


def test_multi_fly_cli_requires_selection_and_reports_selected_weight_identity(tmp_path):
    config = write_config(tmp_path, overrides={'environment': {'num_envs':2}})
    run = tmp_path/'run'
    assert train_main(['--config',str(config),'--run-dir',str(run),'--no-dashboard']) == 0
    checkpoint = run/'plastic-checkpoint-final.pkl'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with checkpoint.open('rb') as stream: payload = pickle.load(stream)
    expected = hashlib.sha256(payload['trainer']['plastic_state']['efficacy'][1:2].astype('<f4').tobytes()).hexdigest()
    common = ['--config',str(config),'--checkpoint',str(checkpoint),'--episodes','3','--batch-size','2','--win-ante','1']
    with pytest.raises(ValueError, match='explicit --learner-index'):
        evaluate_main([*common,'--output-dir',str(tmp_path/'missing-selection')])
    output = tmp_path/'selected'
    assert evaluate_main([*common,'--learner-index','1','--output-dir',str(output)]) == 0
    report = json.loads((output/'summary.json').read_text())
    assert report['source_checkpoint_learners'] == 2
    assert report['selected_learner_index'] == 1
    assert report['plastic_weight_sha256'] == expected
    assert report['source_checkpoint_weight_sha256'] == report['checkpoint_manifest']['plastic_weight_sha256']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    assert not list((output/'replay-candidates').iterdir())
