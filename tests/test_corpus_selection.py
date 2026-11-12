from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.corpus import (
    INVENTORY_NAVIGATION_RULE,
    CalibrationCorpus,
    build_calibration_corpus,
    scripted_navigation_actions,
)
from flylatro.analysis.corpus_selection import select_calibration_corpus
from flylatro.analysis.corpus_selection_cli import main
from helpers import ScriptedPhaseEnv, build_mock_corpus
from flylatro.env.upstream_contract import UpstreamActionType


def test_selection_keeps_rare_late_phases_and_exact_provenance(tmp_path: Path) -> None:
    env = ScriptedPhaseEnv()
    # A long playing prefix followed by just enough examples of each rare phase.
    env.PHASES = (1,) * 100 + (0, 2, 3, 4) * 4
    source = build_calibration_corpus(
        env, environment_seeds=(5, 6), states_per_seed=116,
        navigation_seed=31, store_snapshots=True,
    )
    selected, report = select_calibration_corpus(source, states=40)
    repeated, repeated_report = select_calibration_corpus(source, states=40)
    assert report["status"] == "PASS"
    assert selected.sha256 == repeated.sha256
    assert report == repeated_report
    assert len(selected) == 40
    assert len(set(selected.state_hashes)) == 40
    assert source.sha256 in selected.generation_method
    indices = report["source_row_indices"]
    assert any(index >= 100 for index in indices)
    assert indices == sorted(indices)
    for key in source.observations:
        np.testing.assert_array_equal(selected.observations[key], source.observations[key][indices])
    for key in source.masks:
        np.testing.assert_array_equal(selected.masks[key], source.masks[key][indices])
    for field in ("source_seed", "source_decision", "source_run_seed", "source_episode", "source_collection_index", "snapshots"):
        assert getattr(selected, field) == tuple(getattr(source, field)[index] for index in indices)
    selected.save(tmp_path / "selected.npz")
    assert CalibrationCorpus.load(tmp_path / "selected.npz").sha256 == selected.sha256


def test_missing_contexts_and_small_budgets_remain_failures(tmp_path: Path) -> None:
    source = build_mock_corpus(tmp_path / "mock.npz")
    selected, report = select_calibration_corpus(source, states=2)
    assert len(selected) == 2
    assert report["status"] == "FAIL"
    assert not report["source_coverage"]["satisfied"]
    assert any("SHOP" in reason for reason in report["coverage"]["deficiency_summary"])
    assert any("shop_target" in reason for reason in report["coverage"]["deficiency_summary"])


def test_duplicate_states_do_not_pad_the_selected_corpus(tmp_path: Path) -> None:
    source = build_mock_corpus(tmp_path / "mock.npz")
    repeated = replace(source, state_hashes=(source.state_hashes[0],) * len(source))
    selected, report = select_calibration_corpus(repeated, states=10)
    assert len(selected) == 1
    assert report["selected_states"] == 1
    assert report["status"] == "FAIL"


def test_cli_reports_failure_and_preserves_existing_files(tmp_path: Path) -> None:
    source_path = tmp_path / "mock.npz"
    source = build_mock_corpus(source_path)
    output = tmp_path / "selected.npz"
    args = ["--source", str(source_path), "--output", str(output), "--states", "4"]
    assert main(args) == 2
    selected_hash = CalibrationCorpus.load(output).sha256
    with pytest.raises(FileExistsError, match="already exists"):
        main(args)
    assert CalibrationCorpus.load(output).sha256 == selected_hash
    assert CalibrationCorpus.load(source_path).sha256 == source.sha256


@pytest.mark.parametrize("states,per_stratum", [(0, 8), (5, 0), (-1, 8)])
def test_selection_rejects_invalid_counts(tmp_path: Path, states: int, per_stratum: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        select_calibration_corpus(build_mock_corpus(tmp_path / "mock.npz"), states=states, states_per_stratum=per_stratum)


def test_inventory_navigation_retains_lone_items_but_allows_competition() -> None:
    env = ScriptedPhaseEnv(jokers=1, consumables=1)
    _, masks = env.reset((7,))
    rng = np.random.default_rng(31)
    inventory_actions = {
        int(UpstreamActionType.SELL_JOKER),
        int(UpstreamActionType.SELL_CONSUMABLE),
        int(UpstreamActionType.USE_CONSUMABLE),
    }
    for _ in range(40):
        actions = scripted_navigation_actions(masks, rng, navigation_rule=INVENTORY_NAVIGATION_RULE)
        assert int(actions["action_type"][0]) not in inventory_actions
        env._validate(actions, masks)
    # If inventory is the only legal action, retaining it must not deadlock.
    masks["action_type_mask"][:] = False
    masks["action_type_mask"][:, int(UpstreamActionType.SELL_JOKER)] = True
    actions = scripted_navigation_actions(masks, rng, navigation_rule=INVENTORY_NAVIGATION_RULE)
    assert int(actions["action_type"][0]) == int(UpstreamActionType.SELL_JOKER)
    env._validate(actions, masks)
    env = ScriptedPhaseEnv(jokers=2, consumables=2)
    _, masks = env.reset((7,))
    observed = {
        int(scripted_navigation_actions(masks, rng, navigation_rule=INVENTORY_NAVIGATION_RULE)["action_type"][0])
        for _ in range(100)
    }
    assert inventory_actions <= observed


def test_navigation_rule_is_deterministic_and_bound_to_corpus_identity() -> None:
    kwargs = dict(environment_seeds=(5,), states_per_seed=8, navigation_seed=31)
    baseline = build_calibration_corpus(ScriptedPhaseEnv(), **kwargs)
    first = build_calibration_corpus(ScriptedPhaseEnv(), **kwargs, navigation_rule=INVENTORY_NAVIGATION_RULE)
    second = build_calibration_corpus(ScriptedPhaseEnv(), **kwargs, navigation_rule=INVENTORY_NAVIGATION_RULE)
    assert first.sha256 == second.sha256
    assert first.sha256 != baseline.sha256
    assert first.navigation_rule == INVENTORY_NAVIGATION_RULE
    with pytest.raises(ValueError, match="unknown calibration navigation"):
        build_calibration_corpus(ScriptedPhaseEnv(), **kwargs, navigation_rule="unknown")
