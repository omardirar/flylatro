"""Is `three-factor-global-v1` action-specific, or diffuse?

A single global reinforcement scalar multiplies the eligibility of every
plastic edge.  These diagnostics quantify how much of the eligibility and of
the resulting weight change sits on the motor pool that actually produced the
chosen action, on the pools that lost the comparison, and on MBONs that carry
no motor role at all.

This is measurement only.  V1 deliberately does not bias plasticity toward the
chosen output; doing so to improve results would change the learning rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.interface.motor import MotorMapping


SPECIFICITY_REPORT_VERSION = "chosen-action-plasticity-specificity-v1"

#: Which routing group each structured choice belongs to.
CHOICE_HEADS: tuple[str, ...] = (
    "action_type",
    "card_count",
    "card",
    "joker",
    "consumable",
    "shop",
    "pack",
)


@dataclass(slots=True)
class ChosenActionSpecificity:
    """Accumulate per-decision eligibility and weight-change attribution."""

    topology: PlasticEdgeTopology
    mapping: MotorMapping
    edge_pool: NDArray[np.int64] = field(init=False)
    pool_group: tuple[str, ...] = field(init=False)
    decisions: int = field(init=False, default=0)
    totals: dict[str, float] = field(init=False, default_factory=dict)
    per_decision: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        position = {
            int(root): index
            for index, root in enumerate(self.mapping.output_root_ids.tolist())
        }
        pool_of_output = np.full(len(self.mapping.output_root_ids), -1, dtype=np.int64)
        for pool_id, pool in enumerate(self.mapping.pool_indices):
            for index in pool:
                pool_of_output[index] = pool_id
        edge_pool = np.full(self.topology.edge_count, -1, dtype=np.int64)
        for edge, root in enumerate(self.topology.post_root_ids.tolist()):
            output = position.get(int(root), -1)
            if output >= 0:
                edge_pool[edge] = pool_of_output[output]
        self.edge_pool = edge_pool
        self.pool_group = self.mapping.routing.pool_groups
        self.totals = {
            "chosen_eligibility": 0.0,
            "competing_eligibility": 0.0,
            "unconsulted_motor_eligibility": 0.0,
            "non_motor_eligibility": 0.0,
            "total_eligibility": 0.0,
            "chosen_update": 0.0,
            "competing_update": 0.0,
            "unconsulted_motor_update": 0.0,
            "non_motor_update": 0.0,
            "total_update": 0.0,
        }

    @property
    def motor_edge_fraction(self) -> float:
        return float(np.mean(self.edge_pool >= 0))

    def record(
        self,
        *,
        eligibility: NDArray[np.floating],
        efficacy_delta: NDArray[np.floating],
        choices: Mapping[str, Any],
        keep_detail: bool = False,
    ) -> dict[str, float]:
        elig = np.abs(np.asarray(eligibility, dtype=np.float64))
        delta = np.abs(np.asarray(efficacy_delta, dtype=np.float64))
        if elig.shape != (self.topology.edge_count,) or delta.shape != elig.shape:
            raise ValueError("eligibility and delta must be one value per plastic edge")
        chosen_pools, consulted_groups = self._chosen_pools(choices)
        pool_count = self.mapping.routing.pool_count
        motor = self.edge_pool >= 0
        pool_elig = np.bincount(
            self.edge_pool[motor], weights=elig[motor], minlength=pool_count
        )
        pool_delta = np.bincount(
            self.edge_pool[motor], weights=delta[motor], minlength=pool_count
        )
        competing = [
            pool
            for pool in range(pool_count)
            if self.pool_group[pool] in consulted_groups and pool not in chosen_pools
        ]
        # A head the chosen action type never consulted (a shop slot on a PLAY
        # decision) is neither the chosen pool nor a losing alternative, so it
        # gets its own column and the four columns sum to the total.
        unconsulted = [
            pool
            for pool in range(pool_count)
            if self.pool_group[pool] not in consulted_groups
        ]
        values = {
            "chosen_eligibility": float(sum(pool_elig[pool] for pool in chosen_pools)),
            "competing_eligibility": float(sum(pool_elig[pool] for pool in competing)),
            "unconsulted_motor_eligibility": float(
                sum(pool_elig[pool] for pool in unconsulted)
            ),
            "non_motor_eligibility": float(elig[~motor].sum()),
            "total_eligibility": float(elig.sum()),
            "chosen_update": float(sum(pool_delta[pool] for pool in chosen_pools)),
            "competing_update": float(sum(pool_delta[pool] for pool in competing)),
            "unconsulted_motor_update": float(
                sum(pool_delta[pool] for pool in unconsulted)
            ),
            "non_motor_update": float(delta[~motor].sum()),
            "total_update": float(delta.sum()),
        }
        for key, value in values.items():
            self.totals[key] += value
        self.decisions += 1
        if keep_detail:
            self.per_decision.append(
                {
                    **values,
                    "chosen": {
                        key: (list(value) if isinstance(value, tuple) else value)
                        for key, value in choices.items()
                    },
                    "chosen_pools": sorted(chosen_pools),
                    "competing_pool_count": len(competing),
                }
            )
        return values

    def _chosen_pools(self, choices: Mapping[str, Any]) -> tuple[set[int], set[str]]:
        routes = self.mapping.routing.head_routes
        pools: set[int] = set()
        groups: set[str] = set()

        def add(head: str, option: int) -> None:
            route = routes[head]
            if 0 <= option < len(route) and route[option] >= 0:
                pool = route[option]
                pools.add(int(pool))
                groups.add(self.pool_group[pool])

        for head in CHOICE_HEADS:
            if head not in choices:
                continue
            value = choices[head]
            if head == "card":
                continue
            add(head, int(value))
        for card in choices.get("cards", ()):  # simultaneously ranked slots
            add("card", int(card))
        if "cards" in choices:
            groups.add("card_slot")
        return pools, groups

    def report(self) -> dict[str, Any]:
        totals = self.totals
        elig_total = max(totals["total_eligibility"], 1e-12)
        update_total = max(totals["total_update"], 1e-12)
        return {
            "version": SPECIFICITY_REPORT_VERSION,
            "diagnostic_only_learning_rule_unchanged": True,
            "plasticity_rule": "three-factor-global-v1",
            "decisions": self.decisions,
            "plastic_edges": int(self.topology.edge_count),
            "motor_attributable_edge_fraction": self.motor_edge_fraction,
            "motor_attribution_note": (
                "in whole_brain mode no plastic KC->MBON edge terminates on a "
                "motor output, so a zero attributable fraction is expected and "
                "the non-motor column carries every update"
                if self.motor_edge_fraction == 0.0
                else "plastic edges terminate on the measured motor pools"
            ),
            "totals": dict(totals),
            "chosen_motor_eligibility_fraction": totals["chosen_eligibility"] / elig_total,
            "competing_motor_eligibility_fraction": totals["competing_eligibility"] / elig_total,
            "unconsulted_motor_eligibility_fraction": totals[
                "unconsulted_motor_eligibility"
            ] / elig_total,
            "non_motor_eligibility_fraction": totals["non_motor_eligibility"] / elig_total,
            "chosen_motor_update_fraction": totals["chosen_update"] / update_total,
            "competing_motor_update_fraction": totals["competing_update"] / update_total,
            "unconsulted_motor_update_fraction": totals[
                "unconsulted_motor_update"
            ] / update_total,
            "non_motor_update_fraction": totals["non_motor_update"] / update_total,
            "chosen_over_competing_update_ratio": (
                totals["chosen_update"] / max(totals["competing_update"], 1e-12)
            ),
            "per_decision": self.per_decision,
            "interpretation": (
                "a chosen-pool update fraction close to the motor-attributable "
                "edge fraction indicates diffuse, non-action-specific plasticity; "
                "this is evidence about whether three-factor-global-v1 suffices, "
                "not a reason to bias the rule in V1"
            ),
        }


def summarize(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Aggregate raw per-decision attribution dictionaries."""

    keys = sorted({key for record in records for key in record})
    return {key: float(sum(float(record.get(key, 0.0)) for record in records)) for key in keys}
