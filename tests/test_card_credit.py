from dataclasses import replace

import numpy as np
import pytest

from flylatro.analysis.card_credit import PathwayLayout, feature_matrices, root_columns, weight_summary
from flylatro.analysis.card_credit_cli import assess
from flylatro.env.upstream_contract import MASK_SPEC
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.interface.motor import FixedMotorInterface, MotorMapping, MOTOR_POOL_COUNT


def fixture(shared=True, runs=4):
    roots = np.arange(MOTOR_POOL_COUNT) + 1000
    mapping = MotorMapping.contiguous_pools(roots, mode="mbon_direct", pool_width=1)
    if shared:
        pools = list(mapping.pool_indices)
        pools[mapping.routing.head_routes["action_type"][0]] = pools[mapping.routing.head_routes["card"][2]]
        pools[mapping.routing.head_routes["card_count"][1]] = pools[mapping.routing.head_routes["card"][1]]
        mapping = replace(mapping, pool_indices=tuple(pools))
    motor = FixedMotorInterface(mapping)
    topology = PlasticEdgeTopology.synthetic(np.tile(np.arange(5), len(roots)),
        np.repeat(np.arange(len(roots)), 5), np.ones(5 * len(roots), dtype=np.float32),
        pre_root_ids=np.tile(np.arange(5) + 100, len(roots)), post_root_ids=np.repeat(roots, 5))
    layout = PathwayLayout(topology, motor, np.arange(5) + 100, roots)
    kc = np.random.default_rng(4).uniform(0, 40, (runs, 2, 5)).astype(np.float32)
    mbon = np.zeros((runs, 2, len(roots)), dtype=np.float32)
    mbon[:, :, motor.mapping.pool_indices[mapping.routing.head_routes["card"][0]]] = 60
    mbon[:, :, motor.mapping.pool_indices[mapping.routing.head_routes["card"][1]]] = 2
    mbon[:, :, motor.mapping.pool_indices[mapping.routing.head_routes["card"][2]]] = 4
    mbon[:, :, motor.mapping.pool_indices[mapping.routing.head_routes["card_count"][1]]] = 2
    mbon[:, :, motor.mapping.pool_indices[mapping.routing.head_routes["action_type"][0]]] = 4
    masks = {key: np.zeros((runs, *shape), dtype=dtype) for key, (shape, dtype) in MASK_SPEC.items()}
    masks["action_type_mask"][:, :2] = True
    masks["card_select_mask"][:, :8] = True
    config = PlasticityConfig(version="three-factor-choice-gated-v1", kc_reference_hz=20, mbon_reference_hz=20)
    return layout, kc, mbon, masks, config


def test_shared_action_and_count_neurons_admit_credit_to_unchosen_cards():
    layout, kc, mbon, masks, config = fixture()
    values, diagnostic, by_run, actions = feature_matrices(layout, kc, mbon,
        np.ones(layout.topology.edge_count), masks, config)
    assert np.all(actions["cards"][:, 0] == 0)
    assert np.all(actions["n_cards"] == 1)
    assert diagnostic["slot_selected_fraction"] == [1.] + [0.] * 9
    assert diagnostic["slot_edge_admitted_fraction"] == [1., 1., 1.] + [0.] * 7
    assert diagnostic["aliased_fraction_of_admitted_card_edges"] == pytest.approx(2 / 3)
    assert diagnostic["trials_with_credit_to_unchosen_card_output_fraction"] == 1
    assert diagnostic["aliased_fraction_of_nonzero_card_increments"] == pytest.approx(2 / 3)
    # Selected output saturates at 1, while aliased count/action outputs contribute .1 and .2.
    assert diagnostic["aliased_fraction_of_card_increment_mass"] == pytest.approx(.3 / 1.3)
    assert values["gated_card_edges"].shape == (4, 50)
    np.testing.assert_array_equal(by_run["aliased_edge_count"], np.full(4, 20))
    separate = fixture(shared=False)
    layout, kc, mbon, masks, config = separate
    _, diagnostic, _, _ = feature_matrices(layout, kc, mbon, np.ones(layout.topology.edge_count), masks, config)
    assert diagnostic["aliased_fraction_of_admitted_card_edges"] == 0
    assert diagnostic["aliased_fraction_of_card_increment_mass"] == 0


def test_cached_eligibility_matches_real_rule_per_trial_before_repeat_averaging():
    layout, kc, mbon, masks, config = fixture()
    efficacy = np.ones(layout.topology.edge_count, dtype=np.float32)
    features, _, _, actions = feature_matrices(layout, kc, mbon, efficacy, masks, config)
    repeated_masks = {key: np.repeat(value, 2, axis=0) for key, value in masks.items()}
    gate = layout.credit.mask(actions, repeated_masks)
    state = PlasticEdgeState.initialize(layout.topology.edge_count)
    rule = ThreeFactorPlasticity(layout.topology, state, config)
    expected = []
    for trial in range(8):
        state.reset_fast_traces(eligibility=True)
        rule.record_activity(kc.reshape(8, -1)[trial:trial + 1, layout.pre],
                             mbon.reshape(8, -1)[trial:trial + 1, layout.post],
                             edge_credit=gate[trial:trial + 1])
        expected.append(state.eligibility[0, layout.card_edges].copy())
    np.testing.assert_array_equal(features["gated_card_edges"], np.asarray(expected).reshape(4, 2, -1).mean(axis=1))
    # Saturation is nonlinear: clipping after averaging would change this signal.
    assert not np.array_equal(np.clip(kc / 20, 0, 1).mean(axis=1), np.clip(kc.mean(axis=1) / 20, 0, 1))


def test_projection_uses_anatomical_efficacy_and_validates_root_alignment():
    layout, kc, mbon, masks, config = fixture()
    projection = layout.projection(np.ones(layout.topology.edge_count))
    np.testing.assert_allclose(projection.sum(axis=0), 1)
    np.testing.assert_allclose(projection, .2)
    weights = np.ones(layout.topology.edge_count)
    weights[0] = 2
    changed = layout.projection(weights)
    assert changed[0, 0] == pytest.approx(1 / 3)
    assert changed[1, 0] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="absent"):
        root_columns([1, 2], [3])
    with pytest.raises(ValueError, match="unique"):
        root_columns([1, 1], [1])
    report = weight_summary(layout, np.ones(len(weights)), weights)
    assert report["all"]["changed_fraction"] == 1 / len(weights)
    assert report["card"]["changed_fraction"] == 0
    with pytest.raises(ValueError, match="nonnegative"):
        feature_matrices(layout, kc * np.nan, mbon, weights, masks, config)


def test_followup_corrects_all_sixteen_tests_and_requires_above_baseline():
    from flylatro.analysis.card_credit import STAGES
    reports = {}
    for stage in STAGES:
        for learner in range(3):
            reports[f"trained-{learner}-{stage}"] = {
                "metrics": {task: {"balanced_accuracy": .8, "slot_majority": {"balanced_accuracy": .3}}
                            for task in ("rank", "suit")},
                "arrays": {task + "_null": np.full(499, .3) for task in ("rank", "suit")}}
    tests = assess(reports)["exploratory_tests"]
    assert len(tests) == 16
    assert all(row["holm_p"] == .032 and row["above_controls"] for row in tests)
