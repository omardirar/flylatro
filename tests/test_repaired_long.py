"""Exercise the actual condition worker, stage resumes and frozen comparisons."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml
from flylatro.learning.reinforcement import exploratory_condition
from run_repaired_long import execute, prepare, read


def test_staged_long_run_initial_controls_and_resume(tmp_path, monkeypatch):
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config,
        environment=replace(config.environment, allow_skip_blinds=False),
        plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'),
        reinforcement=exploratory_condition('exploratory-weak-failure-v1'),
        curriculum=replace(config.curriculum, enabled=True, ladder=(1,)))
    document = asdict(config)
    document.pop('source_path')
    source = tmp_path / 'source.toml'
    source.write_text(render_toml(document))
    output = tmp_path / 'experiment'
    plan = prepare(source, output, stages=(3, 6), replicates=1, episodes=2, batch=2)
    assert plan['total_training_decisions'] == 12
    import run_repaired_long
    original_run = run_repaired_long.subprocess.run
    def interrupt_evaluation(command, **kwargs):
        if command[2] == 'evaluate':
            raise RuntimeError('simulated evaluation interruption')
        return original_run(command, **kwargs)
    monkeypatch.setattr(run_repaired_long.subprocess, 'run', interrupt_evaluation)
    with pytest.raises(RuntimeError, match='simulated'):
        execute(output, dashboard=False)
    monkeypatch.setattr(run_repaired_long.subprocess, 'run', original_run)
    execute(output, dashboard=False)
    assert len(list((output / 'training' / 'subset-plastic' / 'stage-0000003').glob('attempt-*'))) == 1
    assert read(output / 'long-status.json')['completed_stages'] == [3, 6]
    for label in ('subset-plastic', 'subset-frozen'):
        receipt = read(output / 'evaluations' / label / 'initial-0' / 'complete.json')
        initial = read(receipt['summary'])
        assert initial['evaluation_weight_source'] == 'initial-efficacy-from-verified-checkpoint'
    report = read(output / 'learning-curve-0000006.json')
    assert report['arms']['subset-plastic'][0]['comparison_to_frozen']['gates']['status'] == 'PASS'
    frozen = report['arms']['subset-frozen'][0]['comparison_to_own_initial']
    assert frozen['mean_return_delta'] == 0
    assert frozen['trained_plastic_weight_sha256'] == frozen['control_plastic_weight_sha256']
    import pickle
    checkpoint = read(output / 'long-status.json')['checkpoints']['subset-frozen']
    with open(checkpoint, 'rb') as stream:
        state = pickle.load(stream)['trainer']['plastic_state']
    np.testing.assert_array_equal(state['efficacy'], state['initial_efficacy'])
    # A completed launch is idempotent and neither retrains nor appends logs.
    files = {p: p.stat().st_mtime_ns for p in output.rglob('training-actions.jsonl')}
    execute(output, dashboard=False)
    assert files == {p: p.stat().st_mtime_ns for p in files}
    declaration = output / 'declaration.json'
    declaration.write_text(json.dumps({}))
    with pytest.raises(ValueError, match='changed'):
        execute(output, dashboard=False)
