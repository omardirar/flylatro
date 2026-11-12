"""Repeated native episodes, terminal accounting, isolation and staged recovery."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import sys
import copy

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.evaluation.state_hash import hash_observation_row
from flylatro.learning.checkpoints import load_plastic_checkpoint, save_plastic_checkpoint
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml
from flylatro.learning.reinforcement import exploratory_condition
from run_single_seed import prepare, read, worker, execute, recover_checkpoint
from single_seed import RepeatedSeedEnvironment, build_stack, evaluate
from strategy_puzzles import action


def assert_state_equal(left, right):
    import torch
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_state_equal(a, b)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def declaration(tmp_path, *, steps=18, replicates=2, arms=('normalised', 'original')):
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config,
        environment=replace(config.environment, backend='balatro_sim', allow_skip_blinds=False),
        plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'),
        reinforcement=exploratory_condition('exploratory-weak-failure-v1'))
    document = asdict(config); document.pop('source_path')
    source = tmp_path / 'source.toml'; source.write_text(render_toml(document))
    output = tmp_path / 'experiment'
    plan = prepare(source, output, seed=4000, steps=steps, replicates=replicates,
        episodes=2, arms=arms)
    return output, plan


def stack_for(plan, arm='normalised'):
    condition = read(plan['arms'][arm]['condition'])
    config = PlasticExperimentConfig.load(Path(condition['config']))
    return build_stack(config, condition), config


def test_native_terminal_rows_repeat_exact_opening_without_resetting_other_rows():
    raw = BalatroSimAdapter(2, win_ante=1, allow_skip_blinds=False)
    repeated = RepeatedSeedEnvironment(raw, 4000)
    obs, masks = repeated.reset([42, 43])
    opening = [hash_observation_row(obs, i) for i in range(2)]
    assert opening[0] == opening[1]
    # Row 0 plays one card; row 1 deliberately discards before playing so their
    # terminal boundaries differ. The unaffected row must exactly match native.
    control = BalatroSimAdapter(2, win_ante=1, allow_skip_blinds=False)
    control.reset([4000, 4000])
    seen = False
    for turn in range(7):
        acts = action(0, [0], batch=2)
        if turn == 0:
            acts = action(2, batch=2)
        elif turn == 1:
            acts['action_type'][1] = 1
        expected = control.step(acts)
        actual = repeated.step(acts)
        np.testing.assert_array_equal(actual.rewards, expected.rewards)
        np.testing.assert_array_equal(actual.dones, expected.dones)
        for i in range(2):
            assert actual.infos[i].get('episode') == expected.infos[i].get('episode')
            assert actual.infos[i]['reward_components'] == expected.infos[i]['reward_components']
        if actual.dones[0]:
            assert not actual.dones[1]
            assert hash_observation_row(actual.observations, 0) == opening[0]
            assert hash_observation_row(actual.observations, 1) == hash_observation_row(expected.observations, 1)
            assert actual.infos[0]['repeated_game_seed'] == 4000
            seen = True
            break
    assert seen


def test_native_repeat_restores_same_hidden_deck_across_episodes():
    repeated = RepeatedSeedEnvironment(BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False), 4000)
    repeated.reset([999])
    trajectories = []
    for _ in range(3):
        hashes = []
        repeated.step(action(2))
        for _ in range(4):
            result = repeated.step(action(0, [0]))
            hashes.append(hash_observation_row(result.observations, 0))
        assert result.dones[0]
        trajectories.append(hashes)
    assert trajectories[0] == trajectories[1] == trajectories[2]


def test_checkpoint_resume_and_frozen_evaluation_preserve_next_decisions(tmp_path):
    output, plan = declaration(tmp_path)
    first, config = stack_for(plan)
    assert len(set(first.trainer.training_seeds)) == 2
    for _ in range(7): first.trainer.step()
    checkpoint = tmp_path / 'checkpoint.pkl'
    save_plastic_checkpoint(checkpoint, first.trainer, experiment_config=config.to_dict(), component_metadata=first.components)
    restored, _ = stack_for(plan)
    load_plastic_checkpoint(checkpoint, restored.trainer, expected_components=restored.components)
    before = copy.deepcopy(restored.trainer.state_dict())
    native_before = [json.loads(restored.env.base._env.state_json(i)) for i in range(2)]
    report = evaluate(restored, [4000, *plan['evaluation_seeds']])
    assert {row['seed'] for row in report['episodes']} == {4000, *plan['evaluation_seeds']}
    assert {row['learner'] for row in report['episodes']} == {0, 1}
    assert len(report['episodes']) == 6
    after = restored.trainer.state_dict()
    # Native snapshots serialize hash maps; byte order can change on restore.
    # Compare the canonical native state plus exact future transitions instead.
    before.pop('environment_snapshots'); after.pop('environment_snapshots')
    assert_state_equal(before, after)
    assert native_before == [json.loads(restored.env.base._env.state_json(i)) for i in range(2)]
    for _ in range(8):
        first.trainer.step(); restored.trainer.step()
        for key in first.trainer.last_executed_actions:
            np.testing.assert_array_equal(first.trainer.last_executed_actions[key], restored.trainer.last_executed_actions[key])
        np.testing.assert_array_equal(first.agent.plasticity.state.efficacy, restored.agent.plasticity.state.efficacy)
        for i in range(2):
            assert json.loads(first.env.base._env.state_json(i)) == json.loads(restored.env.base._env.state_json(i))
        np.testing.assert_array_equal(first.trainer.last_step.rewards, restored.trainer.last_step.rewards)
    other, _ = stack_for(plan, 'original')
    with pytest.raises(ValueError, match='plasticity rule differs'):
        load_plastic_checkpoint(checkpoint, other.trainer, expected_components=other.components)


def test_launcher_recovers_evaluation_interruption_and_is_idempotent(tmp_path, monkeypatch):
    output, plan = declaration(tmp_path, steps=10, replicates=1)
    import run_single_seed
    # Use the real CLI worker in-process to keep this synthetic-native regression
    # fast; production invokes identical entrypoints in isolated subprocesses.
    calls = []
    original = run_single_seed.subprocess.run
    interrupted = False
    def invoke(args, **kwargs):
        nonlocal interrupted
        if '--worker' not in args:
            return original(args, **kwargs)
        command = args[args.index('--worker') + 1]
        calls.append(command)
        if command == 'evaluate' and '--checkpoint' in args and not interrupted:
            interrupted = True
            raise RuntimeError('simulated evaluation interruption')
        return run_single_seed.main(args[2:])
    monkeypatch.setattr(run_single_seed.subprocess, 'run', invoke)
    with pytest.raises(RuntimeError, match='simulated'):
        execute(output, heavy=True, dashboard=False)
    trained = len(list(output.glob('training/*/stage-*/attempt-*')))
    execute(output, heavy=True, dashboard=False)
    assert read(output / 'single-seed-status.json')['status'] == 'completed'
    assert len(list(output.glob('training/*/stage-*/attempt-*'))) == trained + 1
    count = len(calls)
    execute(output, heavy=True, dashboard=False)
    assert len(calls) == count
    final = read(output / 'evaluations' / 'normalised-000000010.json')
    assert final['steps_per_learner'] == 10
    assert final['repeated_seed']['seeds'] == [4000]
    assert final['unseen_seeds']['seeds'] == plan['evaluation_seeds']
    # Completed receipts and safety stops must never be silently bypassed.
    final_path = output / 'evaluations' / 'normalised-000000010.json'
    final_path.write_text('{}')
    with pytest.raises(ValueError, match='evaluation changed'):
        execute(output, heavy=True, dashboard=False)
    folder = tmp_path / 'stopped'; attempt = folder / 'attempt-000'
    attempt.mkdir(parents=True); (attempt / 'safety-stop.json').write_text('{}')
    with pytest.raises(RuntimeError, match='safety stop'):
        recover_checkpoint(folder, None)


def test_source_and_requested_budget_changes_are_rejected(tmp_path):
    output, plan = declaration(tmp_path)
    import run_single_seed
    with pytest.raises(ValueError, match='differs from declaration'):
        run_single_seed.main(['--output-dir', str(output), '--prepare-only'])
    condition = Path(plan['arms']['normalised']['condition'])
    original = condition.read_text()
    condition.write_text(original + ' ')
    with pytest.raises(ValueError, match='condition changed'):
        run_single_seed.verify(output)
    condition.write_text(original)
    path = output / 'single-seed-plan.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='plan changed'):
        run_single_seed.verify(output)
