from __future__ import annotations

import json

import pytest

from flylatro.evaluation.compare import MATCHED_FIELDS, compare_frozen_evaluations
from flylatro.evaluation.plastic import PlasticEpisodeEvaluation, PlasticEvaluationResult
from flylatro.evaluation.plastic_cli import main as evaluate_main, parse_args
from flylatro.learning.cli import main as train_main
from helpers import write_config


def summary():
    return {
        "version": "frozen-plastic-evaluation-v2", "learning_disabled": True,
        "training_action_schedule_replayed": False,
        "action_policy": "independent-fixed-motor-choices-no-training-action-replay",
        "win_ante": 1, "seed_stream": "validation", "seeds": [20_000_000, 20_000_001],
        "components": {key: "same-model" for key in MATCHED_FIELDS}, "plastic_weight_sha256": "weights",
        "episodes": [
            {"seed": seed, "balatro_seed": str(seed), "won": False, "ante": 1,
             "decisions": 10, "episode_return": -1.0, "score": None}
            for seed in (20_000_000, 20_000_001)
        ],
    }


def test_paired_comparison_reports_outcome_deltas_not_a_learning_gate():
    trained, control = summary(), summary()
    trained["episodes"][0].update(won=True, episode_return=1.0)
    result = compare_frozen_evaluations(trained, control)
    assert result["win_rate_delta"] == 0.5
    assert result["mean_return_delta"] == 1.0
    assert result["trained_only_wins"] == 1
    assert result["scored_episode_pairs"] == 0
    assert result["gates"]["status"] == "PASS"
    assert "not learning improvement" in result["interpretation"]


def test_unavailable_scores_are_not_reported_as_zero():
    episode = PlasticEpisodeEvaluation(seed=1, won=False, ante=1, decisions=10, episode_return=-1.0, score=None)
    result = PlasticEvaluationResult(episodes=(episode,), transitions=(), frozen_weight_sha256="weights")
    assert result.metrics()["evaluation/mean_score"] is None
    assert result.metrics()["evaluation/scored_episodes"] == 0


def test_initial_weight_evaluation_is_labelled_and_preserves_source_checkpoint(tmp_path):
    import hashlib
    import pickle
    path = write_config(tmp_path)
    run = tmp_path / "run"
    assert train_main(["--config", str(path), "--run-dir", str(run), "--no-tensorboard"]) == 0
    checkpoint = run / "plastic-checkpoint-final.pkl"
    source_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with checkpoint.open("rb") as stream:
        payload = pickle.load(stream)
    initial = payload['trainer']['plastic_state']['initial_efficacy']
    expected = hashlib.sha256(initial.astype('<f4').tobytes()).hexdigest()
    output = tmp_path / "initial"
    assert evaluate_main(["--config", str(path), "--checkpoint", str(checkpoint),
        "--initial-weights", "--episodes", "3", "--batch-size", "2",
        "--win-ante", "1", "--output-dir", str(output)]) == 0
    report = json.loads((output / 'summary.json').read_text())
    assert report['evaluation_weight_source'] == 'initial-efficacy-from-verified-checkpoint'
    assert report['source_checkpoint_weight_sha256'] == report['checkpoint_manifest']['plastic_weight_sha256']
    assert report['plastic_weight_sha256'] == expected
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_digest
    assert report['learning_disabled'] and not report['training_action_schedule_replayed']
    assert not list((output / 'replay-candidates').iterdir())


def test_milestone_metrics_keep_skips_separate_from_actual_clears():
    episode = PlasticEpisodeEvaluation(seed=1, won=False, ante=1, decisions=10,
        episode_return=0.2, score=None, blinds_beaten=1, skipped_blinds=2, highest_played_round=2)
    metrics = PlasticEvaluationResult((episode,), (), 'weights').metrics()
    assert metrics['evaluation/blinds_beaten'] == 1
    assert metrics['evaluation/mean_skipped_blinds'] == 2
    assert metrics['evaluation/mean_highest_played_round'] == 2


@pytest.mark.parametrize("field,value", [
    ("learning_disabled", False), ("training_action_schedule_replayed", True),
    ("action_policy", "replay"), ("win_ante", 8), ("seeds", [2, 3]),
    ("seed_stream", "final_test"), ("version", "old"),
])
def test_paired_comparison_refuses_incomparable_evaluations(field, value):
    control = summary()
    control[field] = value
    with pytest.raises(ValueError):
        compare_frozen_evaluations(summary(), control)


def test_paired_comparison_rejects_different_model_or_episode_rows():
    control = summary()
    control["components"]["motor_mapping_sha256"] = "different"
    with pytest.raises(ValueError, match="motor_mapping"):
        compare_frozen_evaluations(summary(), control)
    control = summary()
    control["episodes"].pop()
    with pytest.raises(ValueError, match="episode rows"):
        compare_frozen_evaluations(summary(), control)


def test_paired_comparison_keeps_action_restrictions_matched():
    trained, control = summary(), summary()
    trained['components']['action_constraints'] = {'allow_skip_blinds': False}
    with pytest.raises(ValueError, match='action constraints'):
        compare_frozen_evaluations(trained, control)
    control['components']['action_constraints'] = {'allow_skip_blinds': False}
    assert compare_frozen_evaluations(trained, control)['gates']['status'] == 'PASS'


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('allow_skip', [True, False])
def test_frozen_real_environment_preserves_skip_rule_in_every_lane(width, allow_skip):
    pytest.importorskip('balatro_sim')
    from dataclasses import replace
    from pathlib import Path
    from flylatro.learning.config import PlasticExperimentConfig
    from flylatro.evaluation.plastic_cli import _frozen_environment
    from flylatro.env.upstream_contract import UpstreamActionType

    original = PlasticExperimentConfig.load(Path('configs/plastic-smoke.toml'))
    config = replace(original, environment=replace(original.environment,
        backend='balatro_sim', allow_skip_blinds=allow_skip))
    env = _frozen_environment(config, width, 1, .25)
    _, masks = env.reset(tuple(range(11, 11 + width)))
    assert env.allow_skip_blinds == allow_skip
    assert env._beta == .25
    assert masks['action_type_mask'][:, int(UpstreamActionType.SELECT_BLIND)].all()
    assert (masks['action_type_mask'][:, int(UpstreamActionType.SKIP_BLIND)] == allow_skip).all()


def test_ante_target_and_explicit_free_action_evaluation_metadata(tmp_path):
    path = write_config(tmp_path)
    run = tmp_path / "run"
    assert train_main(["--config", str(path), "--run-dir", str(run), "--no-tensorboard"]) == 0
    output = tmp_path / "evaluation"
    assert evaluate_main(["--config", str(path), "--checkpoint", str(run / "plastic-checkpoint-final.pkl"),
                          "--win-ante", "1", "--episodes", "2", "--output-dir", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["win_ante"] == 1
    assert report["learning_disabled"] is True
    assert report["training_action_schedule_replayed"] is False
    assert report["seeds"] == [20_000_000, 20_000_001]
    assert [row["seed"] for row in report["episodes"]] == report["seeds"]
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 2


def test_frozen_seed_batches_preserve_checkpoint_identity_and_independent_trajectories(tmp_path):
    path = write_config(tmp_path)
    run = tmp_path / "run"
    assert train_main(["--config", str(path), "--run-dir", str(run), "--no-tensorboard"]) == 0
    reports = []
    trajectories = []
    for width in (1, 2):
        output = tmp_path / f"batch-{width}"
        assert evaluate_main(["--config", str(path), "--checkpoint", str(run / "plastic-checkpoint-final.pkl"),
            "--win-ante", "1", "--episodes", "5", "--batch-size", str(width), "--output-dir", str(output)]) == 0
        reports.append(json.loads((output / "summary.json").read_text()))
        transitions = [json.loads(line) for line in (output / "transitions.jsonl").read_text().splitlines()]
        trajectories.append(sorted(transitions, key=lambda row: (row["seed"], row["decision_id"])))
    assert reports[0]["plastic_weight_sha256"] == reports[1]["plastic_weight_sha256"]
    assert reports[0]["episodes"] == reports[1]["episodes"]
    assert trajectories[0] == trajectories[1]
    assert reports[1]["evaluation_batch_size"] == 2


@pytest.mark.parametrize("ante", [0, 9])
def test_invalid_evaluation_ante_is_refused(ante):
    with pytest.raises(SystemExit):
        parse_args(["--config", "unused", "--checkpoint", "unused", "--output-dir", "unused", "--win-ante", str(ante)])
