"""Live Balatro replay boundary for the balatrobot local JSON-RPC mod."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping, Protocol
from urllib.request import Request, urlopen

from flylatro.env.types import ActionType, CompositeAction
from flylatro.env.balatro_sim import composite_actions_to_batch
from flylatro.replay.bundle import ReplayBundle
from flylatro.replay.verify import composite_action_from_payload


SETTLED_STATES = {
    "MENU", "BLIND_SELECT", "SELECTING_HAND", "ROUND_EVAL", "SHOP",
    "SMODS_BOOSTER_OPENED", "GAME_OVER",
}

class LiveBalatroClient(Protocol):
    def start(self, *, deck: str, stake: str, seed: str) -> Mapping[str, Any]: ...
    def state(self) -> Mapping[str, Any]: ...
    def apply(self, action: CompositeAction, state: Mapping[str, Any]) -> Mapping[str, Any]: ...


class BalatrobotClient:
    """Small standard-library client compatible with balatrobot v1.5 JSON-RPC."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:12346",
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._request_id = 0

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self._request_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0", "method": method, "id": self._request_id,
        }
        if params is not None:
            payload["params"] = dict(params)
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read())
        if "error" in result:
            error = result["error"]
            raise RuntimeError(
                f"balatrobot {method} failed: {error.get('code')} {error.get('message')}"
            )
        return result["result"]

    def start(self, *, deck: str, stake: str, seed: str) -> Mapping[str, Any]:
        state = self.state()
        if state.get("state") != "MENU":
            self.call("menu")
        return self.call("start", {"deck": deck, "stake": stake, "seed": seed})

    def state(self) -> Mapping[str, Any]:
        return self.call("gamestate")

    def stable_state(
        self, *, timeout_seconds: float = 60.0, interval_seconds: float = 0.35
    ) -> Mapping[str, Any]:
        """Wait for three identical actionable/terminal API observations."""

        deadline = time.monotonic() + timeout_seconds
        previous: Mapping[str, Any] | None = None
        identical = 0
        while True:
            current = self.state()
            if current.get("state") in SETTLED_STATES and current == previous:
                identical += 1
                if identical >= 2:
                    return current
            else:
                previous = current
                identical = 0
            if time.monotonic() >= deadline:
                raise TimeoutError("live Balatro did not reach a stable actionable state")
            time.sleep(interval_seconds)

    def apply(
        self, action: CompositeAction, state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        action_type = action.action_type
        if action_type is ActionType.PLAY_HAND:
            return self.call("play", {"cards": list(action.cards)})
        if action_type is ActionType.DISCARD:
            return self.call("discard", {"cards": list(action.cards)})
        if action_type is ActionType.SELECT_BLIND:
            return self.call("select")
        if action_type is ActionType.SKIP_BLIND:
            return self.call("skip")
        if action_type is ActionType.CASH_OUT:
            return self.call("cash_out")
        if action_type is ActionType.REROLL:
            return self.call("reroll")
        if action_type is ActionType.END_SHOP:
            return self.call("next_round")
        if action_type is ActionType.BUY:
            method, target = _resolve_shop_target(action, state)
            return self.call("buy", {method: target})
        if action_type is ActionType.USE_CONSUMABLE:
            return self.call(
                "use",
                {
                    "consumable": _required(action.consumable_target, "consumable"),
                    "cards": list(action.cards),
                },
            )
        if action_type is ActionType.SELL_JOKER:
            return self.call(
                "sell", {"joker": _required(action.joker_target, "joker")}
            )
        if action_type is ActionType.SELL_CONSUMABLE:
            return self.call(
                "sell",
                {"consumable": _required(action.consumable_target, "consumable")},
            )
        if action_type is ActionType.PICK_PACK:
            return self.call(
                "pack",
                {
                    "card": _required(action.pack_target, "pack"),
                    "targets": list(action.cards),
                },
            )
        if action_type is ActionType.SKIP_PACK:
            return self.call("pack", {"skip": True})
        raise ValueError(f"unsupported live action {action_type.value}")

    def apply_direct(
        self,
        action: Mapping[str, Any],
        state: Mapping[str, Any],
        expected_sim_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Apply CrossvalRun's resolved action dialect without reinterpretation."""

        kind = action["kind"]
        if kind == "select_blind":
            return self.call("select")
        if kind == "skip_blind":
            return self.call("skip")
        if kind == "play":
            return self.call("play", {"cards": list(action["cards"])})
        if kind == "discard":
            return self.call("discard", {"cards": list(action["cards"])})
        if kind == "cash_out":
            return self.call("cash_out")
        if kind == "leave_shop":
            return self.call("next_round")
        if kind == "reroll":
            return self.call("reroll")
        if kind == "buy_card":
            return self.call("buy", {"card": int(action["slot"])})
        if kind == "redeem_voucher":
            return self.call("buy", {"voucher": int(action["slot"])})
        if kind == "buy_pack":
            sim_packs = _area_cards(expected_sim_state.get("packs"))
            game_index = next(
                index for index, pack in enumerate(sim_packs)
                if int(pack.get("slot", -1)) == int(action["slot"])
            )
            return self.call("buy", {"pack": game_index})
        if kind == "use_consumable":
            return self.call(
                "use",
                {"consumable": int(action["slot"]),
                 "cards": list(action.get("targets", ()))},
            )
        if kind == "sell_joker":
            return self.call("sell", {"joker": int(action["slot"])})
        if kind == "sell_consumable":
            return self.call("sell", {"consumable": int(action["slot"])})
        if kind == "pick_pack":
            return self.call(
                "pack",
                {"card": int(action["slot"]),
                 "targets": list(action.get("targets", ()))},
            )
        if kind == "skip_pack":
            return self.call("pack", {"skip": True})
        raise ValueError(f"unsupported resolved live action kind {kind!r}")


@dataclass(frozen=True, slots=True)
class LiveReplayReport:
    decisions_applied: int
    final_state: Mapping[str, Any]


class LiveReplayDiverged(RuntimeError):
    pass


class CrossvalMirror:
    """Pinned simulator mirror used for rich live-state verification."""

    def __init__(self, balatro_seed: str, sim_module: Any | None = None) -> None:
        if sim_module is None:
            try:
                import balatro_sim as sim_module
            except ImportError as error:
                raise ImportError(
                    "live replay needs Flylatro's pinned balatro_sim build"
                ) from error
        if not hasattr(sim_module, "CrossvalRun"):
            raise RuntimeError("pinned balatro_sim build does not expose CrossvalRun")
        self.run = sim_module.CrossvalRun(str(balatro_seed))

    def state(self) -> Mapping[str, Any]:
        return json.loads(self.run.state_json())

    def apply(self, action: CompositeAction) -> Mapping[str, Any] | None:
        row = composite_actions_to_batch((action,))
        count = int(row["n_cards"][0])
        direct_json = self.run.action_json(
            int(row["action_type"][0]),
            [int(value) for value in row["cards"][0, :count]],
            int(row["joker_target"][0]),
            int(row["consumable_target"][0]),
            int(row["shop_target"][0]),
            int(row["pack_target"][0]),
        )
        direct = json.loads(direct_json)
        if direct is None:
            return None
        self.run.apply_json(direct_json)
        return direct


def play_live_replay(
    bundle: ReplayBundle,
    client: LiveBalatroClient,
    *,
    deck: str = "RED",
    stake: str = "WHITE",
    settle_seconds: float = 0.0,
    cash_out_delay_seconds: float = 8.0,
    verify_full_state: bool = True,
    mirror: CrossvalMirror | None = None,
) -> LiveReplayReport:
    """Apply a certified bundle and stop at the first available live mismatch.

    Encoded simulator hashes are verified separately by ``verify_replay``.
    Live hashes are checked when the recording contains
    ``live_state_hash_before``/``after`` fields.
    """

    state = client.start(
        deck=deck, stake=stake, seed=str(bundle.identity.balatro_seed)
    )
    if verify_full_state and mirror is None:
        mirror = CrossvalMirror(str(bundle.identity.balatro_seed))
    stable = getattr(client, "stable_state", None)
    if stable is not None:
        state = stable()
    for decision in bundle.decisions:
        decision_id = int(decision["decision_id"])
        _verify_live_hash(decision, "before", state, decision_id)
        _verify_live_signature(decision, state, decision_id)
        expected_sim_state = None
        if mirror is not None:
            expected_sim_state = mirror.state()
            _verify_crossval_state(expected_sim_state, state, decision_id, "before")
        action = composite_action_from_payload(decision["action"])
        simulator_noop = False
        direct = None
        if mirror is not None:
            direct = mirror.apply(action)
            simulator_noop = direct is None
        # balatrobot exposes ROUND_EVAL before its dollar animation has staged
        # the payout. An early cash_out can therefore apply the previous round's
        # value. The conservative delay is live-presentation time only.
        if (
            not simulator_noop
            and action.action_type is ActionType.CASH_OUT
            and cash_out_delay_seconds > 0
        ):
            time.sleep(cash_out_delay_seconds)
        if not simulator_noop:
            apply_direct = getattr(client, "apply_direct", None)
            if direct is not None and apply_direct is not None:
                state = apply_direct(direct, state, expected_sim_state)
            else:
                state = client.apply(action, state)
        if stable is not None:
            state = stable()
        elif settle_seconds > 0:
            time.sleep(settle_seconds)
            state = client.state()
        _verify_live_hash(decision, "after", state, decision_id)
    if mirror is not None:
        _verify_crossval_state(
            mirror.state(), state, len(bundle.decisions) - 1, "final"
        )
    return LiveReplayReport(len(bundle.decisions), state)


def live_state_hash(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def live_state_signature(state: Mapping[str, Any]) -> dict[str, Any]:
    round_info = state.get("round", {})
    if not isinstance(round_info, Mapping):
        round_info = {}
    return {
        "state": state.get("state"),
        "ante_num": state.get("ante_num"),
        "round_num": state.get("round_num"),
        "money": state.get("money"),
        "hands_left": round_info.get("hands_left"),
        "discards_left": round_info.get("discards_left"),
        "chips": float(round_info.get("chips", 0.0) or 0.0),
    }


def _verify_live_hash(
    decision: Mapping[str, Any], stage: str, state: Mapping[str, Any], decision_id: int
) -> None:
    key = f"live_state_hash_{stage}"
    expected = decision.get(key)
    if expected is None:
        return
    actual = live_state_hash(state)
    if actual != expected:
        raise LiveReplayDiverged(
            f"live replay diverged at decision {decision_id} ({stage}): "
            f"expected {expected}, got {actual}"
        )


def _verify_live_signature(
    decision: Mapping[str, Any], state: Mapping[str, Any], decision_id: int
) -> None:
    expected = decision.get("state_signature_before")
    if expected is None:
        return
    actual = live_state_signature(state)
    differences = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if differences:
        raise LiveReplayDiverged(
            f"live replay diverged at decision {decision_id} (state signature): "
            + json.dumps(differences, sort_keys=True)
        )


def normalise_replay_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical common subset of CrossvalRun and balatrobot state JSON."""

    round_info = _mapping(state.get("round"))
    result: dict[str, Any] = {
        "state": state.get("state"),
        "ante": state.get("ante_num"),
        "round": state.get("round_num"),
        "money": state.get("money"),
        "won": bool(state.get("won", False)),
        "round_info": {
            key: round_info.get(key)
            for key in (
                "hands_left", "hands_played", "discards_left", "discards_used",
                "reroll_cost", "chips",
            )
        },
    }
    blinds = _mapping(state.get("blinds"))
    result["blinds"] = {
        stage: {
            key: _mapping(blinds.get(stage)).get(key)
            for key in ("name", "score", "status")
        }
        for stage in ("small", "big", "boss")
    }
    for key in ("hand", "cards", "jokers", "consumables"):
        result[key] = [_card_signature(card) for card in _area_cards(state.get(key))]
    if state.get("state") in {"SHOP", "SMODS_BOOSTER_OPENED"}:
        for key in ("shop", "vouchers", "packs"):
            result[key] = [
                _card_signature(card) for card in _area_cards(state.get(key))
            ]
    if state.get("state") == "SMODS_BOOSTER_OPENED":
        result["pack"] = [
            _card_signature(card) for card in _area_cards(state.get("pack"))
        ]
    return result


def _verify_crossval_state(
    expected_state: Mapping[str, Any], actual_state: Mapping[str, Any],
    decision_id: int, stage: str,
) -> None:
    expected = normalise_replay_state(expected_state)
    actual = normalise_replay_state(actual_state)
    difference = _first_difference(expected, actual)
    if difference is not None:
        path, expected_value, actual_value = difference
        raise LiveReplayDiverged(
            f"live replay diverged at decision {decision_id} ({stage}) field "
            f"{path}: expected {expected_value!r}, got {actual_value!r}"
        )


def _first_difference(
    expected: Any, actual: Any, path: str = "$"
) -> tuple[str, Any, Any] | None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return f"{path}.{key}", expected.get(key), actual.get(key)
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}.length", len(expected), len(actual)
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return None if float(expected) == float(actual) else (path, expected, actual)
    return None if expected == actual else (path, expected, actual)


def _area_cards(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    area = _mapping(value)
    cards = area.get("cards", [])
    return [item for item in cards if isinstance(item, Mapping)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _card_signature(card: Mapping[str, Any]) -> dict[str, Any]:
    modifier = _mapping(card.get("modifier"))
    value = _mapping(card.get("value"))
    state = _mapping(card.get("state"))
    key = card.get("key")
    # The game's forced first-shop Buffoon pack and the simulator may choose
    # equivalent `_1`/`_2` keys through different RNG timing.
    if isinstance(key, str) and key.startswith("p_buffoon_normal_"):
        key = "p_buffoon_normal_x"
    return {
        "key": key,
        "suit": value.get("suit", card.get("suit")),
        "rank": value.get("rank", card.get("rank")),
        "edition": modifier.get("edition", card.get("edition")),
        "enhancement": modifier.get("enhancement", card.get("enhancement")),
        "seal": modifier.get("seal", card.get("seal")),
        "debuff": bool(state.get("debuff", card.get("debuff", False))),
        "hidden": bool(state.get("hidden", card.get("hidden", False))),
    }


def _resolve_shop_target(
    action: CompositeAction, state: Mapping[str, Any]
) -> tuple[str, int]:
    slot = _required(action.shop_target, "shop")
    # Simulator contract order: cards, vouchers, unopened packs; first six.
    groups = (("card", "shop"), ("voucher", "vouchers"), ("pack", "packs"))
    offset = slot
    for parameter, key in groups:
        area = state.get(key, {})
        cards = area.get("cards", []) if isinstance(area, Mapping) else []
        if offset < len(cards):
            return parameter, offset
        offset -= len(cards)
    raise ValueError(f"shop target {slot} cannot be resolved from live state")


def _required(value: int | None, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} target is required")
    return value
