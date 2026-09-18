"""Inspect what changed inside the sparse KC->MBON learning state."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from flylatro.fly.mushroom_body.plasticity import PlasticityConfig
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


def synaptic_change_report(
    topology: PlasticEdgeTopology,
    state: PlasticEdgeState,
    *,
    learner: int = 0,
) -> dict[str, Any]:
    if not 0 <= learner < state.learners:
        raise IndexError("learner index is outside plastic state")
    initial = state.initial_efficacy[learner].astype(np.float64)
    current = state.efficacy[learner].astype(np.float64)
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
    efficacy = state.efficacy.astype(np.float64)
    delta = efficacy - state.initial_efficacy
    eligibility = state.eligibility.astype(np.float64)
    finite = bool(np.isfinite(efficacy).all() and np.isfinite(eligibility).all())
    lower_fraction = float(np.mean(efficacy <= config.min_efficacy))
    upper_fraction = float(np.mean(efficacy >= config.max_efficacy))
    return {
        "finite": finite,
        "modified_fraction": float(np.mean(delta != 0)),
        "absolute_change": float(np.abs(delta).sum()),
        "mean_efficacy": float(efficacy.mean()),
        "efficacy_std": float(efficacy.std()),
        "eligibility_mean_absolute": float(np.abs(eligibility).mean()),
        "eligibility_max_absolute": float(np.abs(eligibility).max()),
        "lower_bound_fraction": lower_fraction,
        "upper_bound_fraction": upper_fraction,
        "all_weights_saturated": bool(lower_fraction + upper_fraction >= 0.999),
        "no_synaptic_change": bool(not np.any(delta)),
        "update_counts": state.update_count.tolist(),
        "decision_counts": state.decision_count.tolist(),
        "episode_counts": state.episode_count.tolist(),
    }
