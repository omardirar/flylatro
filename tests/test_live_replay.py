from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from flylatro.env.types import ActionType, CompositeAction
from flylatro.replay.live import (
    BalatrobotClient,
    CrossvalMirror,
    LiveReplayDiverged,
    _verify_live_signature,
    _verify_crossval_state,
    live_state_hash,
    live_state_signature,
)


class CapturingClient(BalatrobotClient):
    def __init__(self):
        self.calls = []

    def call(self, method, params=None):
        self.calls.append((method, params))
        return {"state": "SHOP"}


def test_live_client_resolves_combined_simulator_shop_slot() -> None:
    client = CapturingClient()
    state = {
        "shop": {"cards": [{}, {}]},
        "vouchers": {"cards": [{}]},
        "packs": {"cards": [{}, {}]},
    }

    client.apply(CompositeAction(ActionType.BUY, shop_target=3), state)

    assert client.calls == [("buy", {"pack": 0})]


def test_live_client_uses_resolved_sim_pack_slot_mapping() -> None:
    client = CapturingClient()

    client.apply_direct(
        {"kind": "buy_pack", "slot": 2},
        {"state": "SHOP"},
        {"packs": {"cards": [{"slot": 0}, {"slot": 2}]}},
    )

    assert client.calls == [("buy", {"pack": 1})]


def test_live_state_hash_is_order_independent() -> None:
    assert live_state_hash({"state": "SHOP", "ante": 2}) == live_state_hash(
        {"ante": 2, "state": "SHOP"}
    )


def test_live_signature_normalises_shared_state_and_reports_first_difference() -> None:
    state = {
        "state": "SELECTING_HAND",
        "ante_num": 2,
        "round_num": 4,
        "money": 7,
        "round": {"hands_left": 3, "discards_left": 2, "chips": 12},
    }
    signature = live_state_signature(state)
    _verify_live_signature({"state_signature_before": signature}, state, 3)

    bad = dict(signature, money=8)
    with pytest.raises(LiveReplayDiverged, match="decision 3"):
        _verify_live_signature({"state_signature_before": bad}, state, 3)


def test_crossval_comparison_normalises_sim_and_live_card_shapes() -> None:
    sim = {
        "state": "SELECTING_HAND", "ante_num": 1, "round_num": 1,
        "money": 4, "won": False,
        "round": {"hands_left": 3, "hands_played": 1, "discards_left": 2,
                  "discards_used": 0, "reroll_cost": 5, "chips": 10},
        "blinds": {stage: {"name": stage, "score": 100, "status": "CURRENT"}
                   for stage in ("small", "big", "boss")},
        "hand": {"cards": [{"key": "H_A", "suit": "H", "rank": "A",
                               "edition": None, "enhancement": None,
                               "seal": None, "debuff": False, "hidden": False}]},
        "cards": {"cards": []}, "jokers": {"cards": []},
        "consumables": {"cards": []},
    }
    live = {
        **sim,
        "hand": {"cards": [{"key": "H_A", "set": "DEFAULT",
                               "value": {"suit": "H", "rank": "A"},
                               "modifier": {"edition": None, "enhancement": None,
                                            "seal": None},
                               "state": {"debuff": False, "hidden": False},
                               "cost": {"sell": 1}}]},
    }
    _verify_crossval_state(sim, live, 0, "before")

    live["money"] = 5
    with pytest.raises(LiveReplayDiverged, match=r"\$\.money"):
        _verify_crossval_state(sim, live, 0, "before")


def test_crossval_mirror_resolves_and_applies_complete_action() -> None:
    class FakeRun:
        def __init__(self, seed):
            self.seed = seed
            self.applied = None

        def state_json(self):
            return json.dumps({"seed": self.seed})

        def action_json(self, action_type, cards, joker, consumable, shop, pack):
            assert (action_type, cards, joker, consumable, shop, pack) == (
                0, [1, 3], -1, -1, -1, -1
            )
            return json.dumps({"kind": "play", "cards": cards})

        def apply_json(self, payload):
            self.applied = json.loads(payload)

    mirror = CrossvalMirror(
        "LIVESEED", SimpleNamespace(CrossvalRun=FakeRun)
    )

    direct = mirror.apply(
        CompositeAction(ActionType.PLAY_HAND, cards=(1, 3))
    )

    assert direct == {"kind": "play", "cards": [1, 3]}
    assert mirror.run.applied == direct
