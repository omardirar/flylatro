from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.corpus import (
    CALIBRATION_CORPUS_VERSION,
    CalibrationCorpus,
    FORBIDDEN_CORPUS_FIELDS,
    build_calibration_corpus,
    scripted_navigation_actions,
)
from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.env.upstream_contract import MASK_SPEC, OBS_SPEC
from helpers import build_mock_corpus


def test_corpus_generation_is_deterministic_and_hashable(tmp_path: Path) -> None:
    first = build_mock_corpus(tmp_path / "a.npz")
    second = build_mock_corpus(tmp_path / "b.npz")
    assert first.sha256 == second.sha256
    assert first.version == CALIBRATION_CORPUS_VERSION
    changed = build_mock_corpus(tmp_path / "c.npz", navigation_seed=999)
    assert changed.sha256 != first.sha256
    other_seeds = build_mock_corpus(tmp_path / "d.npz", seeds=(7, 8, 9))
    assert other_seeds.sha256 != first.sha256


def test_corpus_round_trip_verifies_its_own_hash(tmp_path: Path) -> None:
    path = tmp_path / "corpus.npz"
    corpus = build_mock_corpus(path)
    loaded = CalibrationCorpus.load(path)
    assert loaded.sha256 == corpus.sha256
    assert len(loaded) == len(corpus)
    for key in OBS_SPEC:
        np.testing.assert_array_equal(loaded.observations[key], corpus.observations[key])
    for key in MASK_SPEC:
        np.testing.assert_array_equal(loaded.masks[key], corpus.masks[key])

    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        CalibrationCorpus.load(path)


def test_corpus_stores_states_and_masks_only(tmp_path: Path) -> None:
    corpus = build_mock_corpus(tmp_path / "corpus.npz")
    manifest = corpus.to_manifest()
    assert manifest["contains_reward_or_strategy_labels"] is False
    assert set(manifest["excluded_fields"]) == set(FORBIDDEN_CORPUS_FIELDS)
    encoded = json.dumps(manifest).lower()
    for forbidden in ("best_action", "expected_value", "hand_strength"):
        # The names appear only in the explicit exclusion list, never as data.
        assert encoded.count(forbidden) == 1
    assert set(corpus.observations) == set(OBS_SPEC)
    assert set(corpus.masks) == set(MASK_SPEC)
    assert not hasattr(corpus, "rewards")
    assert not hasattr(corpus, "actions")


def test_corpus_records_provenance_and_coverage(tmp_path: Path) -> None:
    corpus = build_mock_corpus(tmp_path / "corpus.npz", seeds=(5, 6), states_per_seed=3)
    manifest = corpus.to_manifest()
    assert manifest["environment_seeds"] == [5, 6]
    assert manifest["simulator_version"] == "mock-array-balatro-v1"
    assert len(manifest["state_hashes"]) == len(corpus) == 6
    assert "scripted" in manifest["generation_method"]
    coverage = manifest["coverage"]
    assert coverage["states"] == 6
    assert coverage["phases_observed"]["PLAYING"] == 6
    # The mock cannot reach shop or pack phases; that is reported, not hidden.
    assert "SHOP" in coverage["phases_missing"]


def test_scripted_navigation_is_legal_and_never_stored() -> None:
    env = MockArrayBalatroEnv(1, blind_target=18.0)
    _, masks = env.reset((3,))
    rng = np.random.default_rng(0)
    actions = scripted_navigation_actions(masks, rng)
    assert masks["action_type_mask"][0, int(actions["action_type"][0])]
    count = int(actions["n_cards"][0])
    assert count >= 1
    assert masks["card_select_mask"][0, actions["cards"][0, :count]].all()
    corpus = build_calibration_corpus(
        env, environment_seeds=(3,), states_per_seed=2, navigation_seed=0
    )
    assert "actions" not in corpus.to_manifest()


def test_corpus_rows_are_usable_as_observation_batches(tmp_path: Path) -> None:
    corpus = build_mock_corpus(tmp_path / "corpus.npz")
    observations, masks = corpus.row(1)
    assert observations["global"].shape[0] == 1
    assert masks["action_type_mask"].shape[0] == 1
    batch_obs, _ = corpus.batch([0, 2])
    assert batch_obs["global"].shape[0] == 2
