from dataclasses import replace

import numpy as np
import pytest

from flylatro.analysis.card_decoding import (
    RidgeReadout, check_partitions, labels_for, primary_assessment, probe, scores, select_states,
)
from test_motor_normalization_study import corpus


def test_selection_randomizes_complete_cards_without_mutating_source_and_groups_actual_runs():
    source = corpus("probe", 20)
    before = source.observations["hand"].copy()
    records, observations = select_states(source, 10, seed=200)
    assert len({row["run_seed"] for row in records}) == 10
    assert any(row["permutation"] != list(range(8)) for row in records)
    for row, obs in zip(records, observations):
        np.testing.assert_array_equal(obs["hand"][0, :8], before[row["state_index"], row["permutation"]])
    np.testing.assert_array_equal(source.observations["hand"], before)
    labels, direct = labels_for(observations)
    assert labels["rank"].shape == (10, 8)
    assert direct.shape == (10, 136)
    assert len(select_states(source, 5, seed=201, excluded_runs=[row["run_seed"] for row in records])[0]) == 5
    with pytest.raises(ValueError, match="found"):
        select_states(replace(source, source_run_seed=("same-run",) * 20), 2, seed=1)
    with pytest.raises(ValueError, match="provenance"):
        select_states(replace(source, source_run_seed=()), 2, seed=1)


def test_partition_audit_checks_actual_runs_original_states_and_root_streams():
    partitions = {part: [{"run_seed": part, "root_seed": i, "state_hash": part}]
                  for i, part in enumerate(("train", "validation", "test"))}
    check_partitions(partitions)
    for field in ("run_seed", "root_seed", "state_hash"):
        changed = {key: [dict(row) for row in rows] for key, rows in partitions.items()}
        changed["test"][0][field] = changed["train"][0][field]
        with pytest.raises(ValueError, match="overlap"):
            check_partitions(changed)


def test_linear_positive_control_recovers_rank_and_suit_and_null_destroys_identity():
    rng = np.random.default_rng(123)
    labels, features = {}, {}
    for part, count in (("train", 200), ("validation", 60), ("test", 60)):
        labels[part] = {"rank": rng.integers(0, 13, (count, 8)), "suit": rng.integers(0, 4, (count, 8))}
        features[part] = np.concatenate([np.eye(13)[labels[part]["rank"]].reshape(count, -1),
                                         np.eye(4)[labels[part]["suit"]].reshape(count, -1)], axis=1)
    metrics, arrays = probe(*(features[part] for part in labels), labels, permutations=19, seed=200)
    for task in ("rank", "suit"):
        assert metrics[task]["accuracy"] > .98
        assert metrics[task]["null_mean"] < metrics[task]["uniform_chance"] + .04
        assert metrics[task]["permutation_p_uncorrected"] == .05
        assert metrics[task]["excess_over_slot_majority_95_run_bootstrap"][0] > .5
        assert arrays[task + "_prediction"].shape == (60, 8)


def test_preprocessing_uses_training_only_and_constant_activity_is_intercept_only():
    train = np.ones((40, 3))
    model = RidgeReadout(train, np.full((10, 3), 100.), np.full((12, 3), -100.))
    assert model.features == 0
    y = np.tile(np.arange(8) % 4, (40, 1))
    prediction, _ = model.predict(y, y[:10], 4)
    np.testing.assert_array_equal(prediction, np.tile(np.arange(8) % 4, (12, 1)))
    with pytest.raises(ValueError, match="finite"):
        RidgeReadout(train * np.nan, train, train)


def test_metric_balances_classes_and_holm_controls_six_primary_tests():
    metric = scores(np.array([[0, 0], [0, 1]]), np.zeros((2, 2), int), 2)
    assert metric["accuracy"] == .75
    assert metric["balanced_accuracy"] == .5
    def record():
        return {"metrics": {task: {"balanced_accuracy": .8,
                "slot_majority": {"balanced_accuracy": .3}} for task in ("rank", "suit")},
                "arrays": {task + "_null": np.full(199, .3) for task in ("rank", "suit")}}
    reports = {"sensory": record(), **{f"trained-{index}-{stage}": record()
               for index in range(3) for stage in ("kc", "mbon")}}
    assessment = primary_assessment(reports)
    assert len(assessment["primary_tests"]) == 6
    assert all(row["above_controls"] and row["holm_p"] == .03 for row in assessment["primary_tests"])
    reports["sensory"]["metrics"]["rank"]["slot_majority"]["balanced_accuracy"] = .9
    assert not primary_assessment(reports)["primary_tests"][0]["above_controls"]


def test_cli_keeps_probes_outside_agent_and_preserves_frozen_state(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from flylatro.analysis import card_decoding_cli as cli
    from flylatro.fly.flywire_artifact import sha256_file
    from flylatro.fly.mushroom_body.state import PlasticEdgeState
    from helpers import write_config
    from test_card_information import fly

    agent = fly()
    agent.plasticity.state = PlasticEdgeState.initialize(3, learners=3)
    agent.plasticity.state.efficacy[1:] = 1.1
    motor_path = tmp_path / "motor.json"
    agent.motor.mapping.save(motor_path)
    artifact = tmp_path / "artifact.npz"
    artifact.write_bytes(b"fixture")
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"fixture checkpoint")
    config = write_config(tmp_path, overrides={
        "environment": {"backend": "balatro_sim", "num_envs": 3, "allow_skip_blinds": False},
        "reinforcement": {"run_failure_pulse": .01}, "plasticity": {"version": "three-factor-choice-gated-v1"},
        "fly": {"backend": "flywire", "duration_ms": 100, "artifact_path": str(artifact),
                "motor_mapping_path": str(motor_path)}})
    def collect(*args, **kwargs):
        seeds = kwargs["environment_seeds"]
        return corpus(str(seeds[0]), 12, seeds=seeds)
    monkeypatch.setattr(cli, "build_calibration_corpus", collect)
    monkeypatch.setattr(cli, "build_plastic_environment", lambda config: object())
    monkeypatch.setattr(cli, "build_plastic_stack", lambda config: SimpleNamespace(
        agent=agent, trainer=object(), components={"fixture": True}))
    loaded = []
    def load(*args, **kwargs):
        assert kwargs["expected_components"] == {"fixture": True}
        loaded.append(True)
    monkeypatch.setattr(cli, "load_plastic_checkpoint", load)
    original = agent.processor.process
    def process(*args, **kwargs):
        assert loaded and kwargs["efficacy"].shape == (1, 3)
        response = original(*args, **kwargs)
        response.kc_activity = np.maximum(response.kc_activity, 0)
        response.mbon_activity = np.maximum(response.mbon_activity, 0)
        return response
    agent.processor.process = process
    output = tmp_path / "study"
    assert cli.main(["--config", str(config), "--checkpoint", str(checkpoint), "--output-dir", str(output),
                     "--train-states", "8", "--validation-states", "4", "--test-states", "4",
                     "--repeats", "2", "--permutations", "3", "--heavy"]) == 0
    report = json.loads((output / "report.json").read_text())
    declaration = json.loads((output / "declaration.json").read_text())
    assert report["status"] == "completed" and report["sources_unchanged"]
    assert report["diagnostic_only"] and report["learning_disabled"] and report["plastic_state_unchanged"]
    assert report["plastic_state_hashes_before"] == report["plastic_state_hashes_after"]
    assert report["exact_repeat_checks"] == 16 * 4
    check_partitions(declaration["records"])
    assert not declaration["final_test_seeds_used"]
    assert len(report["probes"]) == 10
    for group in (report["responses"], report["probes"]):
        for row in group.values():
            assert sha256_file(output / row["file"]) == row["sha256"]
    assert not list(output.glob("*checkpoint*"))
    # Identical sampling seeds are intentional across weights, never partitions.
    with np.load(output / "initial-0-responses.npz") as data:
        assert not set(data["train_neural_seeds"].ravel()) & set(data["test_neural_seeds"].ravel())
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        cli.check_sources(declaration)
