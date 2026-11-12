"""Local-only teacher plasticity must learn associations without a decoder fit."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from local_teaching import LocalTeachingConfig, LocalTeachingPlasticity, edge_pool_indices, teaching_signals
from teacher_curriculum_data import opening_batch, decode_stage
from flylatro.fly.mushroom_body.state import PlasticEdgeState, TorchPlasticEdgeState, state_numpy
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.interface.action_conditioned_motor import structural_mapping
from flylatro.env.upstream_contract import empty_action_batch
from test_discard_studies import native_corpus, conditioned_toy


def rule(torch_state=False):
    topology = PlasticEdgeTopology.synthetic(np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 2]), np.ones(5, dtype=np.float32))
    state = PlasticEdgeState.initialize(5, learners=2)
    if torch_state:
        state = TorchPlasticEdgeState.from_state_dict(state.state_dict(), device='cpu')
    return LocalTeachingPlasticity(topology, state)


@pytest.mark.parametrize('torch_state', [False, True])
def test_local_rule_learns_contextual_association_and_preserves_frozen_edges(torch_state):
    model = rule(torch_state); routing = np.array([0, 0, 1, 1, -1])
    activities = [np.tile([40, 0, 40, 0, 20], (2, 1)), np.tile([0, 40, 0, 40, 20], (2, 1))]
    for x in activities:
        model.observe(x, np.full((2, 5), 20), calibrate=True)
    for step in range(20):
        target = step % 2; pre = activities[target]
        efficacy = state_numpy(model.state.efficacy)
        drives = np.stack([(efficacy[:, :2]*pre[:, :2]).sum(axis=1), (efficacy[:, 2:4]*pre[:, 2:4]).sum(axis=1)], axis=1)
        actual = drives.argmax(axis=1)
        signals = np.zeros((2, 2), dtype=np.float32); signals[0, target] += 1; signals[0, actual[0]] -= 1
        post = np.repeat(np.maximum(drives, 1), 2, axis=1); post = np.c_[post, np.full(2, 20)]
        model.observe(pre, post); model.teach(signals, routing)
    efficacy = state_numpy(model.state.efficacy)
    for target, pre in enumerate(activities):
        drives = [(efficacy[0, :2]*pre[0, :2]).sum(), (efficacy[0, 2:4]*pre[0, 2:4]).sum()]
        assert np.argmax(drives) == target
    assert np.any(efficacy[0] > 1) and np.any(efficacy[0] < 1)
    np.testing.assert_array_equal(efficacy[1], 1)
    assert efficacy[0, 4] == 1 and not state_numpy(model.state.eligibility).any()


def test_synaptic_changes_do_not_depend_on_unrelated_synapse_inputs():
    a, b = rule(), rule()
    for model in (a, b):
        model.observe(np.full((2, 5), 20), np.full((2, 5), 20), calibrate=True)
    pre = np.full((2, 5), 40); changed = pre.copy(); changed[:, 4] = 10000
    for model, x in ((a, pre), (b, changed)):
        model.observe(x, np.full((2, 5), 40)); model.teach(np.ones((2, 2)), np.array([0, 0, 1, 1, -1]))
    np.testing.assert_array_equal(a.state.efficacy, b.state.efficacy)


def test_local_trace_checkpoint_restores_exact_future_updates():
    model = rule(); model.observe(np.full((2, 5), 20), np.full((2, 5), 20), calibrate=True)
    values = model.state_dict(); restored = LocalTeachingPlasticity(model.topology, PlasticEdgeState.from_state_dict(values))
    restored.load_traces(values)
    for item in (model, restored):
        item.observe(np.full((2, 5), 40), np.full((2, 5), 30)); item.teach(np.array([[1, -1], [0, 0]]), np.array([0, 0, 1, 1, -1]))
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value, restored.state_dict()[key])
    with pytest.raises(ValueError, match='calibration'):
        rule().observe(np.ones((2, 5)), np.ones((2, 5)))
    with pytest.raises(ValueError, match='nonnegative'):
        rule().observe(np.full((2, 5), -1), np.ones((2, 5)))


def test_teacher_correct_side_follows_visible_cards_and_never_changes_inference(native_corpus, tmp_path, monkeypatch):
    import teacher_curriculum_data as data
    agent = conditioned_toy(tmp_path).agent
    index = native_corpus.indices('training')[0]
    for stage in ('higher_card', 'pair_member'):
        for side in (0, 1):
            obs, masks, target, _ = opening_batch(native_corpus, [index, index], stage, key='test', correct_side=side)
            assert target['cards'][:, 0].tolist() == [side, side]
            if stage == 'higher_card':
                ranks = obs['hand'][0, :2, :13].argmax(axis=1)+2
                values = np.where(ranks == 14, 11, np.minimum(ranks, 10))
                assert values[side] > values[1-side]
            else:
                ranks = obs['hand'][0, :8, :13].argmax(axis=1)
                assert np.count_nonzero(ranks == ranks[side]) == 2
                assert np.count_nonzero(ranks == ranks[1-side]) == 1
            monkeypatch.setattr(data, 'visible_teacher', lambda *a: (_ for _ in ()).throw(AssertionError('teacher leaked into inference')))
            neural = agent.processor.process(obs, fly_seeds=[7, 7], efficacy=np.repeat(agent.plasticity.state.efficacy, 2, axis=0))
            actual = decode_stage(agent, neural.output_activity, masks, stage)
            assert actual['n_cards'].tolist() == [1, 1]


def test_only_named_anatomical_play_pools_receive_teaching():
    mapping = structural_mapping(np.arange(96)+1000, pool_width=1)
    topology = PlasticEdgeTopology.synthetic(np.arange(96), np.arange(96), np.ones(96), post_root_ids=mapping.output_root_ids)
    routed = edge_pool_indices(mapping, topology)
    teacher = empty_action_batch(1); actual = empty_action_batch(1)
    for action in (teacher, actual):
        action['action_type'][:] = 0; action['n_cards'][:] = 1
    teacher['cards'][0, 0] = 0; actual['cards'][0, 0] = 1
    signal = teaching_signals(mapping, teacher, actual)
    assert signal.sum() == 0 and np.count_nonzero(signal) == 2
    allowed = {pool for head in ('card', 'card_count') for pool in mapping.routing.head_routes[head] if pool >= 0}
    assert set(routed[routed >= 0]) == allowed
    assert all(signal[0, p] == 0 for p in mapping.routing.head_routes['discard_card'])


def test_rate_instruction_is_neuron_local_and_teaches_margin_even_if_choice_correct():
    from local_rate_teaching import rate_teaching_signals, LocalRateTeachingConfig
    from flylatro.interface.motor import MotorNormalization
    from test_action_conditioned_motor import masks as legal_masks
    mapping = structural_mapping(np.arange(96)+1000, pool_width=1)
    mapping = replace(mapping, normalization=replace(mapping.normalization, baseline=np.full(mapping.routing.pool_count, 40.),
                                                     scale=np.full(mapping.routing.pool_count, 10.)))
    action = empty_action_batch(1); action['action_type'][:] = 0; action['n_cards'][:] = 1; action['cards'][0, 0] = 0
    masks = legal_masks(1); masks['card_select_mask'][:] = False; masks['card_select_mask'][:, :2] = True
    rates = np.full((1, 96), 40.)
    signal = rate_teaching_signals(mapping, action, action, activity=rates, masks=masks)
    pools = mapping.routing.head_routes['card']
    assert signal[0, pools[0]] == 1 and signal[0, pools[1]] == -1
    assert np.count_nonzero(signal) == 2
    rates[:, mapping.pool_indices[pools[2]]] = 199
    np.testing.assert_array_equal(signal, rate_teaching_signals(mapping, action, action, activity=rates, masks=masks))
    assert LocalRateTeachingConfig().version == 'local-rate-target-teaching-v1'


@pytest.mark.parametrize('variant', ['choice', 'rate'])
def test_complete_local_curriculum_and_resume_preserve_inference_and_checkpoints(native_corpus, tmp_path, variant):
    import json
    import shutil
    from card_learning.common import write_json
    from flylatro.fly.flywire_artifact import sha256_file
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
    from flylatro.learning.protocol_materialize import render_toml
    if variant == 'choice':
        from run_local_teacher_curriculum import train
        local_config = LocalTeachingConfig()
    else:
        from run_local_rate_teacher import train
        from local_rate_teaching import LocalRateTeachingConfig
        local_config = LocalRateTeachingConfig()
    from teacher_curriculum_data import collect_blind_demonstrations, STAGES
    from card_learning.pairs import build_corpus
    folder = tmp_path/'curriculum'; folder.mkdir()
    config = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(config, environment=replace(config.environment, allow_skip_blinds=False),
        plasticity=replace(config.plasticity, version='three-factor-choice-gated-v1'))
    source = build_plastic_stack(config)
    mapping = structural_mapping(source.agent.processor.output_root_ids, pool_width=1)
    mapping_path = mapping.save(folder/'motor.json')
    config = replace(config, fly=replace(config.fly, motor_mapping_path=str(mapping_path), motor_pool_width=1))
    document = asdict(config); document.pop('source_path')
    config_path = folder/'config.toml'; config_path.write_text(render_toml(document))
    shutil.copy(native_corpus.path, folder/'pairs.json')
    build_corpus(folder/'fresh-pairs.json', train_per_rank=1, validation_per_rank=1,
                 training_offset=35500, validation_offset=960000, max_attempts=5000)
    obs, masks, target, _ = collect_blind_demonstrations([701])
    np.savez_compressed(folder/'blind-demonstrations.npz', **{'obs__'+k: v for k, v in obs.items()},
        **{'mask__'+k: v for k, v in masks.items()}, **{'target__'+k: v for k, v in target.items()})
    write_json(folder/'training-counts.json', {'best_hand': [2, 5]})
    stages = list(STAGES) if variant == 'choice' else ['higher_card', 'best_hand', 'small_blind']
    write_json(folder/'declaration.json', {'config': str(config_path), 'source_hashes': {str(config_path): sha256_file(config_path)},
        'mapping_sha256': mapping.sha256, 'exploration': {'probability': .05}, 'replicates': 1,
        'local_rule': asdict(local_config), 'stages': stages, 'steps_per_stage': 2, 'warmup_frames': 2,
        'training_safety': {'maximum_bound_fraction': .1, 'maximum_mean_absolute_drift': .5},
        'validation_blind_seeds': [702], 'fresh_blind_seeds': [703, 704],
        'promotion_thresholds': {'minimum_clear_gain_over_initial': .05, 'minimum_clear_gain_over_random_instruction': .05,
                                 'minimum_positive_learners': 2}})
    train(folder)
    first = json.loads((folder/'report.json').read_text())
    assert first['frozen_weights_exact'] and first['unrelated_weights_exact'] and first['no_gradient_or_external_learned_policy']
    metadata = json.loads((folder/'checkpoint-latest.json').read_text())
    assert metadata['cursor']['stage'] == len(stages)
    train(folder, resume=True)
    assert json.loads((folder/'report.json').read_text())['summary'] == first['summary']
    # Rewind to a durable stage-boundary checkpoint, including a torn log tail.
    # Re-execution must reproduce every later checkpoint and frozen evaluation.
    early = folder/'checkpoints/stage-00-step-000002.npz'
    metadata.update(state_file=str(early.relative_to(folder)), state_sha256=sha256_file(early),
                    cursor={'stage': 0, 'step': 2, 'warmup': 2})
    write_json(folder/'checkpoint-latest.json', metadata)
    with (folder/'higher_card-training.jsonl').open('a') as stream:
        stream.write('{"incomplete')
    train(folder, resume=True)
    assert json.loads((folder/'report.json').read_text())['summary'] == first['summary']
    assert len((folder/'higher_card-training.jsonl').read_text().splitlines()) == 2
    metadata = json.loads((folder/'checkpoint-latest.json').read_text())
    with np.load(folder/metadata['state_file']) as values:
        assert np.any(values['efficacy'][0] != 1)
        np.testing.assert_array_equal(values['efficacy'][2], 1)
    from assess_local_teacher_curriculum import replay_blind, replay_openings
    assert replay_blind(first['final']['small_blind']) > 0
    from card_learning.pairs import PairCorpus
    assert replay_openings(first['final']['pair'], PairCorpus(folder/'fresh-pairs.json')) > 0
    from assess_local_teacher_curriculum import main as assess
    assert assess(['--run-dir', str(folder)]) == 0
    first['final']['small_blind']['rows'][0]['progress'][0] += .1
    with pytest.raises(AssertionError):
        replay_blind(first['final']['small_blind'])
