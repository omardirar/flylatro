"""Deterministic simulator replay with fail-fast divergence reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flylatro.env.balatro_sim import ArrayBalatroEnv, composite_actions_to_batch
from flylatro.env.types import ActionType, CompositeAction
from flylatro.evaluation.state_hash import hash_observation_row, terminal_hash
from flylatro.replay.bundle import ReplayBundle


@dataclass(frozen=True, slots=True)
class ReplayDivergence:
    decision_id: int
    stage: str
    expected_hash: str
    actual_hash: str


class ReplayDivergedError(RuntimeError):
    def __init__(self, divergence: ReplayDivergence) -> None:
        self.divergence = divergence
        super().__init__(
            f"replay diverged at decision {divergence.decision_id} "
            f"({divergence.stage}): expected {divergence.expected_hash}, "
            f"got {divergence.actual_hash}"
        )


def verify_replay(
    bundle: ReplayBundle,
    env: ArrayBalatroEnv,
    *,
    stop_on_divergence: bool = True,
) -> tuple[ReplayDivergence, ...]:
    if env.num_envs != 1:
        raise ValueError("standalone replay verification requires one environment")
    if env.simulator_version != bundle.identity.simulator_version:
        raise ValueError(
            "replay simulator version does not match the environment: "
            f"{bundle.identity.simulator_version} != {env.simulator_version}"
        )
    reset_seed = bundle.identity.simulator_seed
    if reset_seed is None:
        reset_seed = int(bundle.identity.balatro_seed)
    observations, _ = env.reset((reset_seed,))
    if hasattr(env, "run_seed"):
        actual_seed = str(env.run_seed(0))
        if actual_seed != str(bundle.identity.balatro_seed):
            raise ValueError(
                f"replay Balatro seed mismatch: expected "
                f"{bundle.identity.balatro_seed}, got {actual_seed}"
            )
    divergences: list[ReplayDivergence] = []
    for decision in bundle.decisions:
        decision_id = int(decision["decision_id"])
        _check(
            divergences, decision_id, "before",
            str(decision["state_hash_before"]),
            hash_observation_row(observations, 0), stop_on_divergence,
        )
        step = env.step(
            composite_actions_to_batch(
                (composite_action_from_payload(decision["action"]),)
            )
        )
        actual_after = (
            terminal_hash(step.infos[0])
            if bool(step.dones[0])
            else hash_observation_row(step.observations, 0)
        )
        _check(
            divergences, decision_id, "after",
            str(decision["state_hash_after"]), actual_after, stop_on_divergence,
        )
        observations = step.observations
        if bool(step.dones[0]) != bool(decision["done"]):
            _check(
                divergences, decision_id, "done", str(bool(decision["done"])),
                str(bool(step.dones[0])), stop_on_divergence,
            )
        if step.dones[0] and decision_id != len(bundle.decisions) - 1:
            raise ValueError("replay contains decisions after terminal state")
    return tuple(divergences)


def composite_action_from_payload(payload: dict[str, Any]) -> CompositeAction:
    return CompositeAction(
        action_type=ActionType(str(payload["type"])),
        cards=tuple(int(value) for value in payload.get("cards", ())),
        target=_optional_int(payload.get("target")),
        joker_target=_optional_int(payload.get("joker_target")),
        consumable_target=_optional_int(payload.get("consumable_target")),
        shop_target=_optional_int(payload.get("shop_target")),
        pack_target=_optional_int(payload.get("pack_target")),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _check(
    divergences: list[ReplayDivergence], decision_id: int, stage: str,
    expected: str, actual: str, stop: bool,
) -> None:
    if expected == actual:
        return
    divergence = ReplayDivergence(decision_id, stage, expected, actual)
    divergences.append(divergence)
    if stop:
        raise ReplayDivergedError(divergence)
