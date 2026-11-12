from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.env.upstream_contract import MASK_SPEC, empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.interface.motor import FixedMotorInterface, MotorMapping
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.interface.motor_calibration import MotorCalibrationError
from flylatro.interface.separated_motor import _match_slots, calibrate_separated, separation_summary
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.separated_pilot import prepare, validate_pair, validate_evaluation


def source_mapping():
    mapping = MotorMapping.contiguous_pools(np.arange(96)+1000, mode="mbon_direct", pool_width=2)
    pools = []
    for count in mapping.routing.group_pool_counts:
        pools.extend(tuple(range(2*i, 2*i+2)) for i in range(count))
    return replace(mapping, pool_indices=tuple(pools))


def test_capacity_matching_reassigns_earlier_choices_and_refuses_impossible_allocation():
    assert _match_slots(["a", "b"], {"a": [1, 2], "b": [1]}) == {"a": 2, "b": 1}
    with pytest.raises(MotorCalibrationError, match="insufficient"):
        _match_slots(["a", "b"], {"a": [1], "b": [1]})


def test_checkpoint_rejects_another_fixed_motor_layout_even_for_frozen_loading(tmp_path):
    from flylatro.learning.config import build_plastic_stack
    from flylatro.learning.checkpoints import save_plastic_checkpoint, load_plastic_checkpoint
    base = PlasticExperimentConfig.load(Path("configs/plastic-smoke.toml"))
    base = replace(base, plasticity=replace(base.plasticity, version="three-factor-choice-gated-v1"))
    stack = build_plastic_stack(base)
    checkpoint = tmp_path/"source.pkl"
    save_plastic_checkpoint(checkpoint, stack.trainer, experiment_config=base.to_dict(), component_metadata=stack.components)
    mapping = stack.agent.motor.mapping
    pools = list(mapping.pool_indices); pools[0], pools[1] = pools[1], pools[0]
    path = tmp_path/"different-motor.json"
    replace(mapping, pool_indices=tuple(pools)).save(path)
    candidate = build_plastic_stack(replace(base, fly=replace(base.fly, motor_mapping_path=str(path))))
    with pytest.raises(ValueError, match="motor"):
        load_plastic_checkpoint(checkpoint, candidate.trainer, expected_components=candidate.components,
                                require_matching_exposure_budget=False)


def test_reward_free_separation_passes_unchanged_gates_and_eliminates_card_alias(tmp_path):
    from types import SimpleNamespace
    source = source_mapping()
    rng = np.random.default_rng(900)
    values = np.maximum(40+rng.normal(0, 10, (128, 96)), 0)
    masks = {key: np.ones((128, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}
    baseline, candidate, report = calibrate_separated(source, values,
        contexts=motor_context_windows(masks), metadata={"reward_free_fixture": True}, attempts=2)
    assert report["baseline"]["gates"]["status"] == report["separated"]["gates"]["status"] == "PASS"
    assert baseline.pool_indices == source.pool_indices
    assert not report["reward_used"] and not report["outcome_information_used"]
    assert not separation_summary(baseline)["card_isolated_and_core_disjoint"]
    assert separation_summary(candidate)["card_isolated_and_core_disjoint"]
    assert separation_summary(candidate)["unique_core_neurons"] == 58
    assert candidate.routing == source.routing
    np.testing.assert_array_equal(candidate.output_root_ids, source.output_root_ids)
    assert candidate.exploration_epsilon == source.exploration_epsilon
    path = tmp_path/"candidate.json"; candidate.save(path)
    assert MotorMapping.load(path).structure_sha256 == candidate.structure_sha256
    # Legality, rather than a relaxed quality threshold, excludes a disabled
    # option from responsiveness requirements. Its quiet pool stays disjoint.
    constrained = {key: value.copy() for key, value in masks.items()}
    constrained["action_type_mask"][:, 3] = False
    quieter = values.copy(); quieter[:, 94:] = 0
    _, constrained_candidate, constrained_report = calibrate_separated(source, quieter,
        contexts=motor_context_windows(constrained), metadata={"allow_skip_blinds": False}, attempts=2)
    skip_pool = constrained_candidate.routing.head_routes["action_type"][3]
    assert set(constrained_candidate.pool_indices[skip_pool]) == {94, 95}
    assert constrained_report["separated"]["gates"]["status"] == "PASS"
    assert constrained_report["separated"]["calibration"]["quality"]["by_context"]["action_type"]["active_option_count"] == 12
    topology = PlasticEdgeTopology.synthetic(np.arange(96), np.arange(96), np.ones(96, dtype=np.float32), post_root_ids=source.output_root_ids)
    action = empty_action_batch(1); action["action_type"][:] = 0; action["n_cards"][:] = 1; action["cards"][0, 0] = 0
    legal = {key: value[:1].copy() for key, value in masks.items()}
    legal["action_type_mask"][:] = False; legal["action_type_mask"][:, :2] = True
    for mapping, expected in ((baseline, False), (candidate, True)):
        credit = SelectedMotorCredit(FixedMotorInterface(mapping), SimpleNamespace(topology=topology, is_torch=False))
        gate = credit.mask(action, legal)
        assert gate.shape == (1, 96)
        assert (credit.metrics["credit/card_alias_edge_fraction_mean"] == 0) == expected
    with pytest.raises(MotorCalibrationError, match="nonnegative"):
        calibrate_separated(source, values*np.nan, contexts=motor_context_windows(masks), metadata={})
    with pytest.raises(MotorCalibrationError, match="qualification"):
        calibrate_separated(source, np.ones_like(values), contexts=motor_context_windows(masks), metadata={})


def test_bounded_pair_search_is_deterministic_and_preserves_unique_neurons():
    from types import SimpleNamespace
    from flylatro.interface.separated_motor import _anneal_pairs
    core = ("action_type", "card_count", "card_slot")
    ids = {g: [2*i, 2*i+1] for i, g in enumerate(core)}
    source = SimpleNamespace(routing=SimpleNamespace(group_pool_ids=lambda g: ids[g]))
    latent = np.random.default_rng(10).normal(size=(256, 3))
    values = np.repeat(latent, 4, axis=1)
    evidence = {g: {"values": values, "eligible": np.ones(12, dtype=bool)} for g in core}
    pools = [tuple(range(2*i, 2*i+2)) for i in range(6)]
    result = _anneal_pairs(list(pools), core, source, evidence, seed=42)
    assert result == _anneal_pairs(list(pools), core, source, evidence, seed=42)
    assert len(set(i for pool in result for i in pool)) == 12
    assert result != pools


def test_pilot_declaration_and_pair_only_allow_mapping_difference(tmp_path):
    from helpers import write_config
    artifact = tmp_path/"artifact.npz"; artifact.write_bytes(b"fixture")
    motor = tmp_path/"motor.json"; motor.write_text("fixture")
    corpus = tmp_path/"corpus.npz"; corpus.write_bytes(b"fixture")
    path = write_config(tmp_path, corpus_path=corpus, overrides={
        "environment": {"backend": "balatro_sim", "num_envs": 3, "allow_skip_blinds": False},
        "fly": {"backend": "flywire", "duration_ms": 100, "artifact_path": str(artifact), "motor_mapping_path": str(motor)},
        "reinforcement": {"run_failure_pulse": .01}, "plasticity": {"version": "three-factor-choice-gated-v1"}})
    base = PlasticExperimentConfig.load(path)
    candidate = replace(base, name="separated", fly=replace(base.fly, motor_mapping_path="candidate.json"))
    validate_pair(base, candidate)
    with pytest.raises(ValueError, match="only"):
        validate_pair(base, replace(candidate, plasticity=replace(candidate.plasticity, learning_rate=.01)))
    with pytest.raises(ValueError, match="retains"):
        validate_pair(base, replace(candidate, environment=replace(candidate.environment, allow_skip_blinds=True)))
    d = prepare(base, tmp_path/"pilot", steps=10, replicates=3, episodes=2, batch=4)
    assert d["training_seeds"] == [4000, 4001, 4002]
    assert d["evaluation_seeds"] == [20030000, 20030001]
    assert not d["final_test_seeds_used"]
    assert d["calibration_repeats"] == 4


def test_assessment_requires_learning_beyond_better_initial_policy(monkeypatch, tmp_path):
    from flylatro.learning import separated_pilot as pilot
    from flylatro.evaluation.compare import MATCHED_FIELDS
    d = {"replicates": 3, "steps_per_learner": 10, "evaluation_seeds": [20030000, 20030001],
         "evaluation_batch_size": 4, "training_seeds": [4000, 4001, 4002], "interpretation": "fixture"}
    monkeypatch.setattr(pilot, "check_sources", lambda declaration: None)
    arms = {}
    for name in ("baseline", "separated"):
        f = tmp_path/f"{name}.json"; f.write_text("fixture")
        arms[name] = {"config": str(f), "config_sha256": sha256_file(f), "motor": str(f),
                      "motor_file_sha256": sha256_file(f), "motor_mapping_sha256": name}
        run = tmp_path/name; run.mkdir()
        (run/"run-summary.json").write_text(json.dumps({"status": "completed", "safety_stop": None, "environment_decisions": 30}))
        (run/"metrics.jsonl").write_text(json.dumps({"metrics": {"credit/card_alias_edge_fraction_mean": .1 if name == "baseline" else 0}})+"\n")
    def evaluation(name, index, blinds, initial=False):
        components = {key: "same" for key in MATCHED_FIELDS}
        components.update(motor_mapping_sha256=name, motor_routing_sha256="same", motor_exploration={"epsilon": .05},
            action_constraints={"allow_skip_blinds": False}, credit_assignment={"version": "three-factor-choice-gated-v1"},
            training_seeds=d["training_seeds"], exposure_budget_decisions=30)
        return {"version": "frozen-plastic-evaluation-v2", "learning_disabled": True,
            "training_action_schedule_replayed": False, "evaluation_weight_source": "initial-efficacy-from-verified-checkpoint" if initial else "trained-checkpoint",
            "action_policy": "independent-fixed-motor-choices-no-training-action-replay", "selected_learner_index": index,
            "seeds": d["evaluation_seeds"], "seed_stream": "validation", "win_ante": 1, "evaluation_batch_size": 4,
            "source_checkpoint_learners": 3, "components": components, "plastic_weight_sha256": "weights",
            "metrics": {"evaluation/mean_skipped_blinds": 0},
            "episodes": [{"seed": seed, "balatro_seed": str(seed), "episode_return": .1, "ante": 1,
                           "decisions": 10, "score": 0, "blinds_beaten": blinds if i == 0 else 0, "won": False}
                          for i, seed in enumerate(d["evaluation_seeds"])]}
    def write(name, index, blinds, initial=False):
        directory = tmp_path/(f"{name}-initial" if initial else f"{name}-eval-{index}")
        directory.mkdir(exist_ok=True)
        (directory/"summary.json").write_text(json.dumps(evaluation(name, index, blinds, initial)))
    for name in arms:
        count = int(name == "separated")
        write(name, 0, count, initial=True)
        for i in range(3): write(name, i, count)
    report = pilot.assess(tmp_path, d, {"arms": arms})
    assert not report["predeclared_scale_up_signal"]
    for i in range(3): write("separated", i, 2)
    assert pilot.assess(tmp_path, d, {"arms": arms})["predeclared_scale_up_signal"]
    bad = evaluation("separated", 0, 2); bad["components"]["training_seeds"] = [1, 2, 3]
    with pytest.raises(ValueError, match="exposure"):
        validate_evaluation(bad, arms["separated"], 0, d)
