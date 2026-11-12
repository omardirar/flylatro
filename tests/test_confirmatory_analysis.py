import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.evaluation.confirmatory import paired_seed_block_analysis
from flylatro.evaluation import confirmatory
from flylatro.evaluation.compare import MATCHED_FIELDS
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import SeedPlan


def test_exact_seed_block_permutation_agrees_with_enumeration():
    effects = np.array([[1, 1, -1, 0], [1, 0, -1, 0], [0, 1, 1, 0]])
    indices = np.array([0, 1, 2, 0])
    totals = effects[indices, np.arange(4)]
    nonzero = np.abs(totals[totals != 0])
    null = [sum(a * b for a, b in zip(signs, nonzero)) for signs in itertools.product((-1, 1), repeat=len(nonzero))]
    report = paired_seed_block_analysis(effects, primary_fly_indices=indices)
    assert report["one_sided_exact_sign_permutation_p"] == np.mean(np.array(null) >= totals.sum())
    assert report["seed_blocks"] == 4  # Shared seeds are not 12 independent rows.


def test_zero_discordance_and_single_lucky_win_do_not_establish_learning():
    effects = np.zeros((3, 128), dtype=int)
    report = paired_seed_block_analysis(effects)
    assert report["one_sided_exact_sign_permutation_p"] == 1
    assert report["primary_paired_95_ci"][0] < 0 < report["primary_paired_95_ci"][1]
    assert not report["statistical_improvement_demonstrated"]
    effects[0, 0] = 1
    report = paired_seed_block_analysis(effects, primary_fly_indices=np.zeros(128, dtype=int))
    assert report["one_sided_exact_sign_permutation_p"] == 0.5
    assert not report["statistical_improvement_demonstrated"]


def test_consistent_paired_improvement_passes_and_negative_effect_fails():
    report = paired_seed_block_analysis(np.ones((3, 128)))
    assert report["statistical_improvement_demonstrated"]
    assert report["paired_seed_bootstrap_95_ci"] == [1.0, 1.0]
    assert not paired_seed_block_analysis(-np.ones((3, 128)))["statistical_improvement_demonstrated"]


@pytest.mark.parametrize("effects", [np.zeros((2, 128)), np.full((3, 128), 0.1), np.full((3, 128), np.nan)])
def test_analysis_refuses_invalid_observations(effects):
    with pytest.raises(ValueError):
        paired_seed_block_analysis(effects)


def test_confirmatory_cli_binds_actual_files_and_prospective_assignment(tmp_path):
    seeds = list(SeedPlan().seeds("final_test", 128))
    assignment = np.random.default_rng(910011).integers(0, 3, size=128).tolist()
    pairing = tmp_path / "pairing.json"
    pairing.write_text(json.dumps({"primary_fly_indices": assignment}))
    lock = {"analysis_module_sha256": sha256_file(Path(confirmatory.__file__)),
        "pairing_predeclaration": str(pairing), "pairing_predeclaration_sha256": sha256_file(pairing),
        "model_identity": {key: "synthetic-development-fixture" for key in MATCHED_FIELDS},
        "evaluation_batch_size": 1,
        "evaluation_implementation_sha256": {str(Path(confirmatory.__file__)): sha256_file(Path(confirmatory.__file__))},
        "primary_fly_indices": assignment, "plasticity_rule_sha256": "synthetic-development-fixture", "pairs": []}
    commands = []
    for index in range(3):
        pair = {}
        for role in ("trained", "control"):
            prefix = tmp_path / f"{role}-{index}"
            checkpoint = prefix.with_suffix(".pkl"); checkpoint.write_bytes(b"local-fixture")
            config = prefix.with_suffix(".toml"); config.write_text("synthetic fixture")
            config_sha = sha256_file(config)
            pair.update({role + "_checkpoint": str(checkpoint), role + "_checkpoint_sha256": sha256_file(checkpoint),
                role + "_weight_sha256": f"synthetic-weights-{index}-{role}", role + "_config": str(config),
                role + "_config_file_sha256": config_sha, role + "_config_sha256": config_sha})
            summary = {"version": "frozen-plastic-evaluation-v2", "learning_disabled": True,
                "training_action_schedule_replayed": False, "action_policy": "independent-fixed-motor-choices-no-training-action-replay",
                "win_ante": 1, "seed_stream": "final_test", "seeds": seeds,
                "config_sha256": config_sha, "plastic_weight_sha256": pair[role + "_weight_sha256"],
                "checkpoint_manifest": {"checkpoint_sha256": sha256_file(checkpoint)},
                "components": {key: "synthetic-development-fixture" for key in MATCHED_FIELDS},
                "episodes": [{"seed": seed, "balatro_seed": str(seed), "won": False,
                    "ante": 1, "decisions": 3, "episode_return": 0.0, "score": None} for seed in seeds]}
            summary_path = prefix.with_suffix(".json"); summary_path.write_text(json.dumps(summary))
            commands += ["--" + role, str(summary_path)]
        lock["pairs"].append(pair)
    lock_path = tmp_path / "lock.json"; lock_path.write_text(json.dumps(lock))
    output = tmp_path / "analysis.json"
    assert confirmatory.main([*commands, "--lock", str(lock_path), "--output", str(output)]) == 0
    assert not json.loads(output.read_text())["statistical_improvement_demonstrated"]
    Path(lock["pairs"][0]["trained_config"]).write_text("changed after lock")
    with pytest.raises(ValueError, match="configuration file differs"):
        confirmatory.main([*commands, "--lock", str(lock_path), "--output", str(tmp_path / "tampered.json")])
