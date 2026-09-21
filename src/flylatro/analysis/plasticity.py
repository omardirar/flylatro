"""Inspect what changed inside the sparse KC->MBON learning state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from flylatro.fly.mushroom_body.plasticity import (
    PlasticityConfig,
    ThreeFactorPlasticity,
)
from flylatro.fly.mushroom_body.state import PlasticEdgeState, state_numpy
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


@dataclass(frozen=True, slots=True)
class PlasticitySafetyThresholds:
    minimum_modified_fraction: float = 1e-6
    maximum_modified_fraction: float = 0.80
    maximum_bound_fraction: float = 0.10
    maximum_mean_absolute_change: float = 0.50
    maximum_eligibility: float = 4.0


def synaptic_change_report(
    topology: PlasticEdgeTopology,
    state: PlasticEdgeState,
    *,
    learner: int = 0,
) -> dict[str, Any]:
    if not 0 <= learner < state.learners:
        raise IndexError("learner index is outside plastic state")
    initial = state_numpy(state.initial_efficacy)[learner].astype(np.float64)
    current = state_numpy(state.efficacy)[learner].astype(np.float64)
    delta = current - initial
    relative = np.divide(
        delta,
        initial,
        out=np.zeros_like(delta),
        where=initial != 0,
    )

    def groups(values: np.ndarray) -> list[dict[str, Any]]:
        result = []
        for key in sorted(set(str(value) for value in values)):
            selected = values.astype(str) == key
            result.append(
                {
                    "group": key,
                    "synapses": int(selected.sum()),
                    "modified": int(np.count_nonzero(delta[selected])),
                    "mean_change": float(delta[selected].mean()),
                    "absolute_change": float(np.abs(delta[selected]).sum()),
                    "positive": int(np.count_nonzero(delta[selected] > 0)),
                    "negative": int(np.count_nonzero(delta[selected] < 0)),
                }
            )
        return result

    return {
        "learner": learner,
        "topology_sha256": topology.sha256,
        "weight_sha256": state.weight_sha256,
        "synapses": topology.edge_count,
        "modified_synapses": int(np.count_nonzero(delta)),
        "positive_changes": int(np.count_nonzero(delta > 0)),
        "negative_changes": int(np.count_nonzero(delta < 0)),
        "absolute_change": float(np.abs(delta).sum()),
        "mean_change": float(delta.mean()),
        "mean_absolute_relative_change": float(np.abs(relative).mean()),
        "change_quantiles": {
            str(quantile): float(np.quantile(delta, quantile))
            for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "by_mbon_root_id": groups(topology.post_root_ids),
        "by_mbon_type": groups(topology.mbon_types),
        "by_kc_type": groups(topology.kc_types),
        "by_compartment": groups(topology.compartments),
    }


def plasticity_diagnostics(
    state: PlasticEdgeState,
    config: PlasticityConfig,
) -> dict[str, Any]:
    efficacy = state_numpy(state.efficacy).astype(np.float64)
    delta = efficacy - state_numpy(state.initial_efficacy)
    eligibility = state_numpy(state.eligibility).astype(np.float64)
    finite = bool(np.isfinite(efficacy).all() and np.isfinite(eligibility).all())
    lower_fraction = float(np.mean(efficacy <= config.min_efficacy))
    upper_fraction = float(np.mean(efficacy >= config.max_efficacy))
    return {
        "finite": finite,
        "modified_fraction": float(np.mean(delta != 0)),
        "absolute_change": float(np.abs(delta).sum()),
        "mean_efficacy": float(efficacy.mean()),
        "median_efficacy": float(np.median(efficacy)),
        "efficacy_std": float(efficacy.std()),
        "absolute_drift": float(np.abs(delta).sum()),
        "mean_absolute_drift": float(np.abs(delta).mean()),
        "mean_absolute_relative_drift": float(
            np.mean(
                np.divide(
                    np.abs(delta),
                    np.maximum(np.abs(state_numpy(state.initial_efficacy)), 1e-12),
                )
            )
        ),
        "mean_cumulative_efficacy_drift": float(np.abs(delta).mean()),
        "max_cumulative_efficacy_drift": float(np.abs(delta).max()),
        "changed_synapse_fraction": float(np.mean(delta != 0)),
        "positive_changes": int(np.count_nonzero(delta > 0)),
        "negative_changes": int(np.count_nonzero(delta < 0)),
        "eligibility_mean_absolute": float(np.abs(eligibility).mean()),
        "eligibility_max_absolute": float(np.abs(eligibility).max()),
        "lower_bound_fraction": lower_fraction,
        "upper_bound_fraction": upper_fraction,
        "all_weights_saturated": bool(lower_fraction + upper_fraction >= 0.999),
        "no_synaptic_change": bool(not np.any(delta)),
        "update_counts": state_numpy(state.update_count).tolist(),
        "decision_counts": state_numpy(state.decision_count).tolist(),
        "episode_counts": state_numpy(state.episode_count).tolist(),
    }


def plasticity_calibration_report(
    state: PlasticEdgeState,
    config: PlasticityConfig,
    *,
    thresholds: PlasticitySafetyThresholds | None = None,
    update_magnitudes: Sequence[float] = (),
    eligible_reinforcement_events: int = 0,
) -> dict[str, Any]:
    limits = thresholds or PlasticitySafetyThresholds(
        maximum_eligibility=config.max_eligibility
    )
    diagnostics = plasticity_diagnostics(state, config)
    delta = state_numpy(state.efficacy).astype(np.float64) - state_numpy(state.initial_efficacy)
    bound_fraction = diagnostics["lower_bound_fraction"] + diagnostics["upper_bound_fraction"]
    mean_absolute_change = float(np.mean(np.abs(delta)))
    measured_updates = np.asarray(update_magnitudes, dtype=np.float64)
    mean_update = float(measured_updates.mean()) if measured_updates.size else 0.0
    max_update = float(measured_updates.max()) if measured_updates.size else 0.0
    safety_flags = {
        "rapid_bound_saturation": bound_fraction > limits.maximum_bound_fraction,
        "non_finite_state": not diagnostics["finite"],
        "global_uncontrolled_drift": (
            diagnostics["modified_fraction"] > limits.maximum_modified_fraction
            or mean_absolute_change > limits.maximum_mean_absolute_change
        ),
        "zero_learning_despite_eligible_reinforcement": (
            eligible_reinforcement_events > 0 and measured_updates.size == 0
        ),
    }
    checks = {
        "finite": diagnostics["finite"],
        "synapses_change": diagnostics["modified_fraction"] >= limits.minimum_modified_fraction,
        "not_runaway_global_change": diagnostics["modified_fraction"] <= limits.maximum_modified_fraction,
        "bounds_not_crowded": bound_fraction <= limits.maximum_bound_fraction,
        "mean_change_bounded": mean_absolute_change <= limits.maximum_mean_absolute_change,
        "eligibility_bounded": diagnostics["eligibility_max_absolute"] <= limits.maximum_eligibility + 1e-7,
        "nonzero_learning_when_eligible": not safety_flags[
            "zero_learning_despite_eligible_reinforcement"
        ],
    }
    return {
        "unit_contract": {
            "kc_activity": "spikes_per_second_hz",
            "mbon_activity": "spikes_per_second_hz",
            "efficacy": "dimensionless_multiplier",
            "eligibility": "bounded_normalized_coincidence_trace",
        },
        "thresholds": asdict(limits),
        "diagnostics": diagnostics,
        "mean_absolute_efficacy_change": mean_absolute_change,
        "mean_update_magnitude": mean_update,
        "max_update_magnitude": max_update,
        "measured_update_count": int(measured_updates.size),
        "eligible_reinforcement_events": int(eligible_reinforcement_events),
        "safety_flags": safety_flags,
        "gates": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "failed": [name for name, passed in checks.items() if not passed],
        },
    }


def run_controlled_plasticity_sequence(
    rule: ThreeFactorPlasticity,
    *,
    decisions: int,
) -> tuple[float, ...]:
    """Exercise the rule with deterministic, reward-independent Hz inputs."""

    if decisions < 2:
        raise ValueError("controlled calibration needs at least two decisions")
    edge_axis = np.arange(rule.state.edge_count, dtype=np.float32)
    edge_scale = 0.5 + 0.5 * (edge_axis % 7) / 6.0
    normalized_levels = (0.0, 0.05, 0.25, 1.0)
    reinforcement = ((0.0, 0.0), (0.20, 0.0), (0.0, 0.15), (0.10, 0.0))
    updates: list[float] = []
    for decision in range(decisions):
        level = normalized_levels[decision % len(normalized_levels)]
        pre = np.broadcast_to(
            edge_scale * level * rule.config.kc_reference_hz,
            (rule.state.learners, rule.state.edge_count),
        ).copy()
        post = np.broadcast_to(
            edge_scale[::-1] * level * rule.config.mbon_reference_hz,
            (rule.state.learners, rule.state.edge_count),
        ).copy()
        rule.record_activity(pre, post)
        appetitive, aversive = reinforcement[decision % len(reinforcement)]
        events = rule.apply_dopamine(
            appetitive,
            aversive,
            include_sparse_changes=True,
        )
        updates.extend(abs(value) for event in events for value in event.efficacy_changes)
    return tuple(updates)
