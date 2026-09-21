"""Context-aware motor evidence, and action construction against the real contract.

Two different defects are covered here:

* a motor pool that varies strongly in states that never read it and is flat in
  the states that do must not produce a passing motor gate, either in
  reward-free calibration or in the post-motor representation report;
* the fixed motor interface must build actions the pinned simulator's strict
  referee accepts for **every** supported action type, not only for the two the
  development mock exercises.
"""

from __future__ import annotations

import numpy as np
import pytest

from flylatro.analysis.representation import (
    RepresentationThresholds,
    representation_diagnostics,
)
from flylatro.env.upstream_contract import (
    MASK_SPEC,
    MAX_CARD_PICKS,
    UpstreamActionType,
    empty_action_batch,
    validate_strict_action_batch,
    validate_strict_action_row,
)
from flylatro.interface.motor import (
    MOTOR_POOL_COUNT,
    FixedMotorInterface,
    MotorMapping,
)
from flylatro.interface.motor_calibration import (
    MotorCalibrationError,
    MotorCalibrationThresholds,
    calibrate_reward_free_motor,
)
from flylatro.interface.motor_contexts import (
    MOTOR_CONTEXT_NAMES,
    motor_context_counts,
    motor_context_windows,
)
from helpers import ScriptedPhaseEnv

POOL_WIDTH = 2
CANDIDATES = MOTOR_POOL_COUNT * POOL_WIDTH

#: Row blocks of the synthetic mask fixture, ten states each.
PLAYING = slice(0, 10)
SHOP = slice(10, 20)
PACK = slice(20, 30)
INVENTORY = slice(30, 40)
SAMPLES = 40


def _fixture_masks() -> dict[str, np.ndarray]:
    """Legality masks that exercise every motor interpretation context."""

    masks = {
        key: np.zeros((SAMPLES, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    types = masks["action_type_mask"]
    types[PLAYING, int(UpstreamActionType.PLAY_HAND)] = True
    types[PLAYING, int(UpstreamActionType.DISCARD)] = True
    masks["card_select_mask"][PLAYING, :6] = True
    types[SHOP, int(UpstreamActionType.BUY_SHOP)] = True
    types[SHOP, int(UpstreamActionType.REROLL)] = True
    types[SHOP, int(UpstreamActionType.LEAVE_SHOP)] = True
    masks["shop_target_mask"][SHOP, :4] = True
    types[PACK, int(UpstreamActionType.PICK_PACK)] = True
    types[PACK, int(UpstreamActionType.SKIP_PACK)] = True
    masks["pack_target_mask"][PACK, :3] = True
    types[INVENTORY, int(UpstreamActionType.SELECT_BLIND)] = True
    types[INVENTORY, int(UpstreamActionType.SKIP_BLIND)] = True
    types[INVENTORY, int(UpstreamActionType.SELL_JOKER)] = True
    types[INVENTORY, int(UpstreamActionType.SELL_CONSUMABLE)] = True
    masks["joker_target_mask"][INVENTORY, :3] = True
    masks["consumable_target_mask"][INVENTORY, :2] = True
    return masks


def _activity(*, varying: tuple[slice, ...], seed: int = 3) -> np.ndarray:
    """Candidate activity that only varies inside the requested row blocks."""

    rng = np.random.default_rng(seed)
    offsets = 10.0 + rng.uniform(0.0, 5.0, size=CANDIDATES)
    values = np.tile(offsets[None, :], (SAMPLES, 1))
    for block in varying:
        rows = np.arange(SAMPLES)[block]
        ramp = np.linspace(0.0, 30.0, len(rows))
        values[rows] += ramp[:, None] * (1.0 + rng.uniform(0.0, 1.0, size=CANDIDATES))
        values[rows] += rng.uniform(0.0, 4.0, size=(len(rows), CANDIDATES))
    return values


# --------------------------------------------------------------------------
# context derivation
# --------------------------------------------------------------------------


def test_contexts_are_derived_from_legality_masks_only() -> None:
    windows = motor_context_windows(_fixture_masks())
    assert set(windows) == set(MOTOR_CONTEXT_NAMES)
    assert windows["action_type"].relevant_states == SAMPLES
    assert windows["card_slot"].relevant_states == 10
    assert windows["card_count"].relevant_states == 10
    assert windows["shop_target"].relevant_states == 10
    assert windows["pack_target"].relevant_states == 10
    assert windows["joker_target"].relevant_states == 10
    assert windows["consumable_target"].relevant_states == 10
    assert list(windows["shop_target"].active_options) == [0, 1, 2, 3]
    assert list(windows["pack_target"].active_options) == [0, 1, 2]
    # Count zero is only legal for a consumable that targets no hand card.
    assert 0 not in windows["card_count"].active_options
    assert windows["card_slot"].competing_states == 10


def test_context_counts_summarize_the_same_windows() -> None:
    counts = motor_context_counts(_fixture_masks())
    assert counts["shop_target"]["group"] == "context_slot"
    assert counts["shop_target"]["states_with_competing_options"] == 10
    assert counts["consumable_target"]["active_option_count"] == 2


def test_contexts_can_be_aligned_with_repeated_samples() -> None:
    masks = _fixture_masks()
    order = list(range(SAMPLES)) * 3
    windows = motor_context_windows(masks, order=order)
    assert windows["shop_target"].relevant.shape == (SAMPLES * 3,)
    assert windows["shop_target"].relevant_states == 30


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def test_calibration_refuses_a_group_with_no_variation_in_its_own_contexts() -> None:
    """Global diversity cannot stand in for a group's own contexts."""

    roots = np.arange(5_000, 5_000 + CANDIDATES, dtype=np.int64)
    # Thirty PLAYING states give every candidate ample *global* diversity, so
    # the whole-corpus eligibility precondition passes; the ten states that
    # interpret the shared contextual slot pools are flat.
    masks = {
        key: np.zeros((SAMPLES, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    wide_playing = slice(0, 30)
    masks["action_type_mask"][wide_playing, int(UpstreamActionType.PLAY_HAND)] = True
    masks["action_type_mask"][wide_playing, int(UpstreamActionType.DISCARD)] = True
    masks["card_select_mask"][wide_playing, :6] = True
    masks["action_type_mask"][30:40, int(UpstreamActionType.BUY_SHOP)] = True
    masks["action_type_mask"][30:40, int(UpstreamActionType.LEAVE_SHOP)] = True
    masks["shop_target_mask"][30:40, :4] = True
    values = _activity(varying=(wide_playing,))
    with pytest.raises(MotorCalibrationError, match="group context_slot exhausted"):
        calibrate_reward_free_motor(
            roots,
            values,
            mode="mbon_direct",
            pool_width=POOL_WIDTH,
            contexts=motor_context_windows(masks),
        )


def test_variation_outside_a_context_is_not_evidence_for_it() -> None:
    roots = np.arange(5_000, 5_000 + CANDIDATES, dtype=np.int64)
    masks = _fixture_masks()
    # Strong variation in PLAYING, SHOP and INVENTORY states; the PACK states
    # that interpret the very same pools as pack slots are flat.
    values = _activity(varying=(PLAYING, SHOP, INVENTORY))
    result = calibrate_reward_free_motor(
        roots,
        values,
        mode="mbon_direct",
        pool_width=POOL_WIDTH,
        contexts=motor_context_windows(masks),
        thresholds=MotorCalibrationThresholds(
            minimum_effective_signal_fraction=0.0,
            minimum_normalized_option_range=0.25,
        ),
    )
    quality = result.report["calibration"]["quality"]
    # The whole-corpus group statistic looks healthy...
    assert quality["by_group"]["context_slot"]["normalized_option_range"]["min"] > 0.25
    # ...while the context that actually reads those pools carries nothing.
    pack = quality["by_context"]["pack_target"]
    assert pack["relevant_states"] == 10
    assert pack["normalized_option_range"]["min"] == pytest.approx(0.0)
    assert quality["by_context"]["shop_target"]["normalized_option_range"]["min"] > 0.25
    assert result.status == "FAIL"
    assert "option_dynamic_range" in result.report["gates"]["failed"]


def test_a_context_without_enough_states_is_not_treated_as_evidence() -> None:
    roots = np.arange(5_000, 5_000 + CANDIDATES, dtype=np.int64)
    masks = _fixture_masks()
    # Only two shop states: too few to demonstrate anything.
    masks["action_type_mask"][SHOP, int(UpstreamActionType.BUY_SHOP)] = False
    masks["action_type_mask"][10:12, int(UpstreamActionType.BUY_SHOP)] = True
    result = calibrate_reward_free_motor(
        roots,
        _activity(varying=(PLAYING, SHOP, PACK, INVENTORY)),
        mode="mbon_direct",
        pool_width=POOL_WIDTH,
        contexts=motor_context_windows(masks),
        thresholds=MotorCalibrationThresholds(
            minimum_effective_signal_fraction=0.0,
            minimum_normalized_option_range=0.0,
            minimum_context_states=4,
        ),
    )
    shop = result.report["calibration"]["quality"]["by_context"]["shop_target"]
    assert shop["relevant_states"] == 2
    assert shop["insufficient_evidence"] is True
    assert "2 relevant state(s)" in shop["reason"]
    assert result.status == "FAIL"
    assert "context_evidence_sufficient" in result.report["gates"]["failed"]
    assert "shop_target" in result.report["calibration"]["quality"][
        "contexts_with_insufficient_evidence"
    ]


def test_group_selection_evidence_records_the_states_it_used() -> None:
    roots = np.arange(5_000, 5_000 + CANDIDATES, dtype=np.int64)
    result = calibrate_reward_free_motor(
        roots,
        _activity(varying=(PLAYING, SHOP, PACK, INVENTORY)),
        mode="mbon_direct",
        pool_width=POOL_WIDTH,
        contexts=motor_context_windows(_fixture_masks()),
        thresholds=MotorCalibrationThresholds(
            minimum_effective_signal_fraction=0.0,
            minimum_normalized_option_range=0.0,
        ),
    )
    evidence = result.report["calibration"]["selection_evidence_by_group"]
    assert evidence["card_slot"]["relevant_states"] == 10
    assert evidence["card_slot"]["context_restricted"] is True
    assert evidence["context_slot"]["relevant_states"] == 30
    assert evidence["action_type"]["relevant_states"] == SAMPLES
    assert result.report["calibration"]["context_aware"] is True


def test_calibration_without_contexts_declares_missing_context_evidence() -> None:
    roots = np.arange(5_000, 5_000 + CANDIDATES, dtype=np.int64)
    result = calibrate_reward_free_motor(
        roots,
        _activity(varying=(PLAYING, SHOP, PACK, INVENTORY)),
        mode="mbon_direct",
        pool_width=POOL_WIDTH,
        thresholds=MotorCalibrationThresholds(
            minimum_effective_signal_fraction=0.0,
            minimum_normalized_option_range=0.0,
        ),
    )
    assert result.report["calibration"]["context_aware"] is False
    assert result.status == "FAIL"
    assert "context_evidence_sufficient" in result.report["gates"]["failed"]


# --------------------------------------------------------------------------
# post-motor representation
# --------------------------------------------------------------------------


def _interface() -> FixedMotorInterface:
    roots = np.arange(9_000, 9_000 + CANDIDATES, dtype=np.int64)
    return FixedMotorInterface(
        MotorMapping.contiguous_pools(roots, mode="mbon_direct", pool_width=POOL_WIDTH)
    )


def _post_report(values: np.ndarray, masks: dict[str, np.ndarray], **thresholds):
    return representation_diagnostics(
        values,
        values,
        values,
        motor_activity=values,
        motor_activity_population="mbon",
        motor_interface=_interface(),
        motor_contexts=motor_context_windows(masks),
        stage="post",
        thresholds=RepresentationThresholds(**thresholds),
    )


def test_post_motor_diagnostics_judge_each_head_in_its_own_context() -> None:
    masks = _fixture_masks()
    report = _post_report(
        _activity(varying=(PLAYING, SHOP, INVENTORY)),
        masks,
        minimum_action_coverage_fraction=0.0,
        maximum_competing_pool_correlation=1.0,
        minimum_separation_ratio=0.0,
    )
    interface = report["motor_interface"]
    assert interface["context_aware"] is True
    # Whole-corpus figures stay descriptive and look healthy.
    assert interface["by_group"]["context_slot"]["normalized_option_range"]["min"] > 1.0
    # The pack context that reads those same pools is flat, so the readiness
    # figure — and therefore the gate — is not.
    assert interface["by_context"]["pack_target"]["normalized_option_range"][
        "min"
    ] == pytest.approx(0.0)
    assert interface["minimum_normalized_option_range"] == pytest.approx(0.0)
    assert report["gates"]["status"] == "FAIL"
    assert "motor_option_dynamic_range" in report["gates"]["failed"]


def test_post_motor_diagnostics_pass_when_every_context_carries_signal() -> None:
    masks = _fixture_masks()
    report = _post_report(
        _activity(varying=(PLAYING, SHOP, PACK, INVENTORY)),
        masks,
        minimum_action_coverage_fraction=0.0,
        maximum_competing_pool_correlation=1.0,
        minimum_separation_ratio=0.0,
    )
    interface = report["motor_interface"]
    assert not interface["contexts_with_insufficient_evidence"]
    assert interface["minimum_normalized_option_range"] > 0.25
    for name in ("shop_target", "pack_target", "joker_target", "consumable_target"):
        entry = interface["by_context"][name]
        assert entry["insufficient_evidence"] is False
        assert entry["states_with_competing_legal_options"] == 10
        assert entry["distinct_argmax_options"] >= 1
        assert 0.0 <= entry["argmax_coverage_fraction"] <= 1.0
    assert report["gates"]["checks"]["motor_option_dynamic_range"] is True
    assert report["gates"]["checks"]["motor_context_evidence_sufficient"] is True


def test_post_motor_diagnostics_flag_a_context_with_too_few_states() -> None:
    masks = _fixture_masks()
    masks["action_type_mask"][PACK, int(UpstreamActionType.PICK_PACK)] = False
    masks["action_type_mask"][20:22, int(UpstreamActionType.PICK_PACK)] = True
    report = _post_report(
        _activity(varying=(PLAYING, SHOP, PACK, INVENTORY)),
        masks,
        minimum_action_coverage_fraction=0.0,
        maximum_competing_pool_correlation=1.0,
        minimum_separation_ratio=0.0,
        minimum_normalized_option_range=0.0,
    )
    interface = report["motor_interface"]
    assert interface["by_context"]["pack_target"]["eligible_states"] == 2
    assert interface["by_context"]["pack_target"]["insufficient_evidence"] is True
    assert "pack_target" in interface["contexts_with_insufficient_evidence"]
    assert report["gates"]["checks"]["motor_context_evidence_sufficient"] is False


def test_post_motor_without_contexts_makes_no_context_claim() -> None:
    values = _activity(varying=(PLAYING,))
    report = representation_diagnostics(
        values,
        values,
        values,
        motor_activity=values,
        motor_activity_population="mbon",
        motor_interface=_interface(),
        stage="post",
        thresholds=RepresentationThresholds(minimum_separation_ratio=0.0),
    )
    assert report["motor_interface"]["context_aware"] is False
    assert report["motor_interface"]["contexts_with_insufficient_evidence"] == [
        "<no legality context supplied>"
    ]
    assert report["gates"]["checks"]["motor_context_evidence_sufficient"] is False


# --------------------------------------------------------------------------
# strict action construction
# --------------------------------------------------------------------------


def _single_type_masks(action_type: int) -> dict[str, np.ndarray]:
    masks = {
        key: np.zeros((1, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    masks["action_type_mask"][0, action_type] = True
    if action_type in (
        int(UpstreamActionType.PLAY_HAND),
        int(UpstreamActionType.DISCARD),
        int(UpstreamActionType.USE_CONSUMABLE),
    ):
        masks["card_select_mask"][0, :7] = True
    if action_type == int(UpstreamActionType.SELL_JOKER):
        masks["joker_target_mask"][0, :3] = True
    if action_type in (
        int(UpstreamActionType.USE_CONSUMABLE),
        int(UpstreamActionType.SELL_CONSUMABLE),
    ):
        masks["consumable_target_mask"][0, :2] = True
    if action_type == int(UpstreamActionType.BUY_SHOP):
        masks["shop_target_mask"][0, :4] = True
    if action_type == int(UpstreamActionType.PICK_PACK):
        masks["pack_target_mask"][0, :3] = True
    return masks


@pytest.mark.parametrize(
    "action_type", list(MotorMapping.contiguous_pools(
        np.arange(9_000, 9_000 + CANDIDATES, dtype=np.int64),
        mode="mbon_direct",
        pool_width=POOL_WIDTH,
    ).routing.supported_action_types)
)
@pytest.mark.parametrize("deterministic", [True, False])
def test_every_supported_action_type_satisfies_the_strict_referee(
    action_type: int, deterministic: bool
) -> None:
    interface = _interface()
    masks = _single_type_masks(action_type)
    rng = np.random.default_rng(action_type)
    for trial in range(6):
        activity = rng.uniform(0.0, 60.0, size=(1, CANDIDATES))
        actions = interface.decode(
            activity,
            masks,
            deterministic=deterministic,
            learner_ids=np.asarray([0]),
            decision_ids=np.asarray([trial]),
        )
        assert int(actions["action_type"][0]) == action_type
        validate_strict_action_batch(actions, masks)


def test_actions_that_consume_no_cards_carry_n_cards_zero_not_minus_one() -> None:
    """The regression that would only have surfaced on the real simulator."""

    interface = _interface()
    activity = np.random.default_rng(1).uniform(0.0, 40.0, size=(1, CANDIDATES))
    for action_type in (
        int(UpstreamActionType.SELECT_BLIND),
        int(UpstreamActionType.CASH_OUT),
        int(UpstreamActionType.BUY_SHOP),
        int(UpstreamActionType.SELL_JOKER),
        int(UpstreamActionType.SELL_CONSUMABLE),
        int(UpstreamActionType.PICK_PACK),
        int(UpstreamActionType.SKIP_PACK),
        int(UpstreamActionType.LEAVE_SHOP),
        int(UpstreamActionType.REROLL),
    ):
        masks = _single_type_masks(action_type)
        actions = interface.decode(activity, masks)
        assert int(actions["n_cards"][0]) == 0
        assert list(actions["cards"][0]) == [-1] * MAX_CARD_PICKS
        validate_strict_action_batch(actions, masks)


def test_the_strict_referee_rejects_the_pre_fix_padding() -> None:
    """The mirror is load-bearing: -1 padding must be an error, not a pass."""

    masks = _single_type_masks(int(UpstreamActionType.SELECT_BLIND))
    actions = empty_action_batch(1)
    actions["action_type"][0] = int(UpstreamActionType.SELECT_BLIND)
    with pytest.raises(ValueError, match="card params must be empty"):
        validate_strict_action_row(actions, masks, 0)


def test_the_strict_referee_rejects_a_pointer_for_the_wrong_type() -> None:
    masks = _single_type_masks(int(UpstreamActionType.SELECT_BLIND))
    masks["joker_target_mask"][0, :3] = True
    actions = empty_action_batch(1)
    actions["n_cards"][0] = 0
    actions["action_type"][0] = int(UpstreamActionType.SELECT_BLIND)
    actions["joker_target"][0] = 1
    with pytest.raises(ValueError, match="joker_target=1 must be -1"):
        validate_strict_action_row(actions, masks, 0)


def test_use_consumable_may_target_no_hand_card() -> None:
    interface = _interface()
    masks = _single_type_masks(int(UpstreamActionType.USE_CONSUMABLE))
    masks["card_select_mask"][0, :] = False
    activity = np.random.default_rng(2).uniform(0.0, 40.0, size=(1, CANDIDATES))
    actions = interface.decode(activity, masks)
    assert int(actions["n_cards"][0]) == 0
    assert int(actions["consumable_target"][0]) in (0, 1)
    validate_strict_action_batch(actions, masks)


def test_decoded_actions_survive_a_full_scripted_phase_episode() -> None:
    """Every phase, every decision, judged by the strict referee."""

    interface = _interface()
    env = ScriptedPhaseEnv(1)
    _, masks = env.reset((4_242,))
    rng = np.random.default_rng(9)
    for step in range(24):
        activity = rng.uniform(0.0, 50.0, size=(1, CANDIDATES))
        actions = interface.decode(
            activity,
            masks,
            deterministic=False,
            learner_ids=np.asarray([0]),
            decision_ids=np.asarray([step]),
        )
        validate_strict_action_batch(actions, masks)
        masks = env.step(actions).masks
