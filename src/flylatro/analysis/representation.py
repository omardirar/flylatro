"""Representation diagnostics with explicit spike-rate units and gates.

Two stages exist deliberately:

``pre``
    neural representation only, before any motor artifact exists.  It measures
    KC/MBON/descending distributions, same-state repeatability, different-state
    separation and observable-category probes.  It makes **no** claim about the
    final motor interface.

``post``
    re-run with the persisted final motor mapping.  Only this stage may satisfy
    motor-related readiness gates, because only it measures the normalized pool
    scores the decoder actually compares.

The motor population measured is always exactly the population the motor
interface consumes: MBON output activity in ``mbon_direct`` and descending
output activity in ``whole_brain``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from flylatro.interface.motor_contexts import (
    MOTOR_CONTEXT_VERSION,
    MotorContextWindow,
)


REPRESENTATION_REPORT_VERSION = "flylatro-representation-v3"
MOTOR_POPULATION_BY_MODE: dict[str, str] = {
    "mbon_direct": "mbon",
    "whole_brain": "descending",
}


@dataclass(frozen=True, slots=True)
class RepresentationThresholds:
    silence_hz: float = 0.0
    high_rate_hz: float = 200.0
    maximum_silent_fraction: float = 0.95
    maximum_high_rate_fraction: float = 0.25
    minimum_separation_ratio: float = 1.10
    minimum_motor_dynamic_range_hz: float = 1.0
    minimum_action_coverage_fraction: float = 0.50
    minimum_normalized_option_range: float = 0.25
    maximum_competing_pool_correlation: float = 0.99
    #: A head is only judged where it is actually read.  These mirror the motor
    #: calibration gates; 0 deliberately opts out (development doubles).
    minimum_context_states: int = 4
    minimum_competing_context_states: int = 2


def representation_diagnostics(
    kc_activity: NDArray[np.floating],
    mbon_activity: NDArray[np.floating],
    descending_activity: NDArray[np.floating],
    *,
    motor_activity: NDArray[np.floating],
    motor_activity_population: str,
    state_labels: NDArray[np.integer] | None = None,
    observable_categories: Mapping[str, NDArray[np.integer]] | None = None,
    motor_interface: Any | None = None,
    motor_contexts: Mapping[str, MotorContextWindow] | None = None,
    stage: str = "pre",
    thresholds: RepresentationThresholds | None = None,
) -> dict[str, Any]:
    if stage not in {"pre", "post"}:
        raise ValueError("representation stage must be pre or post")
    if motor_activity_population not in {"mbon", "descending"}:
        raise ValueError("motor_activity_population must be mbon or descending")
    if stage == "post" and motor_interface is None:
        raise ValueError(
            "post-motor representation requires the persisted final motor mapping"
        )
    limits = thresholds or RepresentationThresholds()
    kc = _matrix(kc_activity, "KC")
    mbon = _matrix(mbon_activity, "MBON")
    descending = _matrix(descending_activity, "descending")
    motor = _matrix(motor_activity, "motor")
    if not (len(kc) == len(mbon) == len(descending) == len(motor)):
        raise ValueError("all activity matrices must have the same samples")
    reference = mbon if motor_activity_population == "mbon" else descending
    result: dict[str, Any] = {
        "version": REPRESENTATION_REPORT_VERSION,
        "stage": stage,
        "activity_unit": "spikes_per_second_hz",
        "samples": kc.shape[0],
        "thresholds": asdict(limits),
        "motor_activity_population": motor_activity_population,
        "motor_activity_neurons": int(motor.shape[1]),
        "motor_activity_matches_named_population": bool(
            motor.shape == reference.shape and np.allclose(motor, reference)
        ),
        "populations": {
            "kc": _population(kc, limits),
            "mbon": _population(mbon, limits),
            "descending": _population(descending, limits),
            "motor_output": _population(motor, limits),
        },
        "state_geometry": _state_geometry(mbon, state_labels),
        "motor_population": _motor_population(motor, limits),
        "observable_category_probes": {},
    }
    for name, labels in (observable_categories or {}).items():
        result["observable_category_probes"][name] = _centroid_probe(
            mbon, np.asarray(labels)
        )
    geometry = result["state_geometry"]
    mbon_stats = result["populations"]["mbon"]
    checks = {
        "kc_not_silent": result["populations"]["kc"]["silent_neuron_fraction"]
        <= limits.maximum_silent_fraction,
        "mbon_not_silent": mbon_stats["silent_neuron_fraction"] <= limits.maximum_silent_fraction,
        "mbon_not_high_rate": mbon_stats["high_rate_sample_fraction"]
        <= limits.maximum_high_rate_fraction,
        "state_separation": geometry["separation_ratio"] is None
        or geometry["separation_ratio"] >= limits.minimum_separation_ratio,
        "motor_population_not_constant": result["motor_population"]["mean_dynamic_range_hz"]
        >= limits.minimum_motor_dynamic_range_hz,
    }
    if stage == "post":
        interface = _motor_interface_diagnostics(
            motor_interface, motor, limits, motor_contexts
        )
        result["motor_interface"] = interface
        checks.update(
            {
                # Every motor readiness number below is measured in the states
                # where the head is actually read, never across irrelevant
                # states that cannot exercise it.
                "motor_context_evidence_sufficient": not interface[
                    "contexts_with_insufficient_evidence"
                ],
                "motor_option_dynamic_range": interface["minimum_normalized_option_range"]
                >= limits.minimum_normalized_option_range,
                "action_option_coverage": interface["action_option_coverage_fraction"]
                >= limits.minimum_action_coverage_fraction,
                "competing_pools_distinguishable": interface[
                    "maximum_competing_pool_correlation"
                ]
                <= limits.maximum_competing_pool_correlation,
            }
        )
        result["final_motor_interface_evaluated"] = True
    else:
        result["motor_interface"] = {
            "evaluated": False,
            "reason": (
                "pre-motor stage: the final reward-free motor artifact does not "
                "exist yet and no motor readiness claim is made"
            ),
        }
        result["final_motor_interface_evaluated"] = False
    result["gates"] = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "motor_claim": (
            "final motor interface evaluated"
            if stage == "post"
            else "no motor interface claim"
        ),
    }
    result.update(
        kc_active_fraction=float(np.mean(kc > limits.silence_hz)),
        kc_mean_activity=float(kc.mean()),
        mbon_mean_activity=float(mbon.mean()),
        mbon_silent_fraction=mbon_stats["silent_neuron_fraction"],
        mbon_saturated_fraction=mbon_stats["high_rate_sample_fraction"],
        descending_mean_activity=float(descending.mean()),
        descending_silent_fraction=result["populations"]["descending"]["silent_neuron_fraction"],
        same_state_variability=geometry["within_state_mean_distance_hz"],
        different_state_separability=geometry["between_state_mean_distance_hz"],
    )
    return result


def _population(values: NDArray[np.float64], limits: RepresentationThresholds) -> dict[str, float | int]:
    return {
        "neurons": int(values.shape[1]),
        "mean_hz": float(values.mean()),
        "median_hz": float(np.median(values)),
        "p95_hz": float(np.quantile(values, 0.95)),
        "max_hz": float(values.max()),
        "active_sample_fraction": float(np.mean(values > limits.silence_hz)),
        "silent_neuron_fraction": float(np.mean(np.all(values <= limits.silence_hz, axis=0))),
        "high_rate_sample_fraction": float(np.mean(values >= limits.high_rate_hz)),
        "mean_neuron_variance_hz2": float(np.mean(values.var(axis=0))),
        "mean_neuron_dynamic_range_hz": float(np.mean(np.ptp(values, axis=0))),
    }


def _state_geometry(values: NDArray[np.float64], labels: NDArray[np.integer] | None) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "within_state_variance_hz2": None,
        "between_state_variance_hz2": None,
        "within_state_mean_distance_hz": None,
        "between_state_mean_distance_hz": None,
        "separation_ratio": None,
    }
    if labels is None:
        return result
    labels = np.asarray(labels)
    if labels.shape != (len(values),):
        raise ValueError("state_labels must identify every sample")
    unique = np.unique(labels)
    centroids = np.stack([values[labels == label].mean(axis=0) for label in unique])
    within_var = [values[labels == label].var(axis=0).mean() for label in unique]
    same: list[float] = []
    different: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            target = same if labels[left] == labels[right] else different
            target.append(float(np.linalg.norm(values[left] - values[right])))
    same_mean = float(np.mean(same)) if same else None
    different_mean = float(np.mean(different)) if different else None
    ratio = different_mean / max(same_mean, 1e-12) if different_mean is not None and same_mean is not None else None
    result.update(
        within_state_variance_hz2=float(np.mean(within_var)),
        between_state_variance_hz2=float(centroids.var(axis=0).mean()) if len(unique) > 1 else 0.0,
        within_state_mean_distance_hz=same_mean,
        between_state_mean_distance_hz=different_mean,
        separation_ratio=ratio,
    )
    return result


def _motor_population(
    values: NDArray[np.float64], limits: RepresentationThresholds
) -> dict[str, Any]:
    """Raw statistics of the population the motor interface actually reads."""

    dynamic = np.ptp(values, axis=0)
    return {
        "neurons": int(values.shape[1]),
        "mean_dynamic_range_hz": float(dynamic.mean()),
        "minimum_dynamic_range_hz": float(dynamic.min()),
        "maximum_dynamic_range_hz": float(dynamic.max()),
        "constant_neuron_fraction": float(np.mean(dynamic <= 0)),
        "silent_neuron_fraction": float(np.mean(np.all(values <= limits.silence_hz, axis=0))),
        "responsive_neuron_fraction": float(
            np.mean(dynamic >= limits.minimum_motor_dynamic_range_hz)
        ),
    }


def _motor_interface_diagnostics(
    interface: Any,
    motor: NDArray[np.float64],
    limits: RepresentationThresholds,
    contexts: Mapping[str, MotorContextWindow] | None,
) -> dict[str, Any]:
    """Post-motor stage: measure the decoder's own normalized comparisons.

    Whole-corpus ``by_group``/``by_head`` numbers are retained as descriptive
    context, but every readiness figure is computed per motor interpretation
    context: the states where the head is actually read, restricted to the
    options that are actually legal there.  A pool that swings widely outside
    its context and is flat inside it can no longer produce a passing gate.
    """

    mapping = interface.mapping
    pool_activity = interface.pool_activity(motor)
    scores = interface.head_scores(motor)
    normalization = mapping.normalization
    normalized_pools = (
        normalization.apply(pool_activity)
        if normalization is not None
        else pool_activity
    )
    ranges = np.ptp(normalized_pools, axis=0)
    by_group: dict[str, Any] = {}
    for group in mapping.routing.group_names:
        pool_ids = list(mapping.routing.group_pool_ids(group))
        block = normalized_pools[:, pool_ids]
        correlation = _correlation(block)
        off = correlation[~np.eye(len(pool_ids), dtype=bool)]
        eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
        by_group[group] = {
            "pools": len(pool_ids),
            "measured_over": "every corpus state (descriptive, not a gate)",
            "normalized_option_range": {
                "min": float(ranges[pool_ids].min()),
                "median": float(np.median(ranges[pool_ids])),
                "max": float(ranges[pool_ids].max()),
            },
            "raw_pool_hz": {
                "min": float(pool_activity[:, pool_ids].min()),
                "median": float(np.median(pool_activity[:, pool_ids])),
                "max": float(pool_activity[:, pool_ids].max()),
            },
            "competing_pool_correlation_max_absolute": (
                float(np.abs(off).max()) if off.size else 0.0
            ),
            "effective_distinct_signals": float(
                eigenvalues.sum() ** 2 / max(float((eigenvalues**2).sum()), 1e-12)
            ),
        }
    by_head: dict[str, Any] = {}
    for head, options in mapping.represented_pools().items():
        option_indices = sorted(options)
        head_scores = scores[head][:, option_indices]
        head_ranges = np.ptp(head_scores, axis=0)
        selected = np.asarray(option_indices)[np.argmax(head_scores, axis=1)]
        counts = np.bincount(
            np.searchsorted(np.asarray(option_indices), selected),
            minlength=len(option_indices),
        ).astype(np.float64)
        probabilities = counts / max(counts.sum(), 1)
        observed = probabilities[probabilities > 0]
        by_head[head] = {
            "represented_options": len(option_indices),
            "contract_options": len(scores[head][0]),
            "measured_over": "every corpus state (descriptive, not a gate)",
            "option_normalized_range": [float(value) for value in head_ranges],
            "minimum_option_normalized_range": float(head_ranges.min()),
            "distinct_unmasked_argmax_options": int(np.count_nonzero(counts)),
            "unmasked_argmax_entropy": float(-np.sum(observed * np.log(observed))),
            "score_distribution": {
                "mean": float(head_scores.mean()),
                "std": float(head_scores.std()),
                "min": float(head_scores.min()),
                "max": float(head_scores.max()),
            },
        }
    by_context, summary = _context_diagnostics(
        mapping, scores, normalized_pools, pool_activity, limits, contexts
    )
    return {
        "evaluated": True,
        "context_version": MOTOR_CONTEXT_VERSION,
        "context_aware": contexts is not None,
        "motor_mapping_sha256": mapping.structure_sha256,
        "motor_artifact_sha256": mapping.sha256,
        "motor_routing_sha256": mapping.routing.sha256,
        "motor_normalization_sha256": (
            None if mapping.normalization is None else mapping.normalization.sha256
        ),
        "motor_candidate_set_sha256": mapping.candidate_set_sha256,
        "selection_method": mapping.selection_method,
        "pool_count": mapping.routing.pool_count,
        "pool_width": mapping.pool_width,
        "by_group": by_group,
        "by_head": by_head,
        "by_context": by_context,
        "readiness_rule": (
            "motor readiness is decided from the contexts where each head is "
            "interpreted and from the options legal there, never from variation "
            "in states that never read the head"
        ),
        **summary,
        "reserved_action_slots": list(mapping.routing.reserved_action_types),
    }


def _context_diagnostics(
    mapping: Any,
    scores: Mapping[str, NDArray[np.float64]],
    normalized_pools: NDArray[np.float64],
    pool_activity: NDArray[np.float64],
    limits: RepresentationThresholds,
    contexts: Mapping[str, MotorContextWindow] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-context decoder behaviour plus the aggregated readiness figures."""

    if contexts is None:
        # Without legality masks the decoder cannot be judged in context. That
        # is reported as missing evidence rather than silently falling back to
        # the whole-corpus numbers.
        insufficient = (
            [] if limits.minimum_context_states <= 0 else ["<no legality context supplied>"]
        )
        ranges = np.ptp(normalized_pools, axis=0)
        return {}, {
            "contexts_with_insufficient_evidence": insufficient,
            "minimum_normalized_option_range": float(ranges.min()),
            "maximum_competing_pool_correlation": max(
                (
                    float(
                        np.abs(
                            _correlation(
                                normalized_pools[
                                    :, list(mapping.routing.group_pool_ids(group))
                                ]
                            )[
                                ~np.eye(
                                    len(mapping.routing.group_pool_ids(group)),
                                    dtype=bool,
                                )
                            ]
                        ).max()
                    )
                    for group in mapping.routing.group_names
                    if len(mapping.routing.group_pool_ids(group)) > 1
                ),
                default=0.0,
            ),
            "action_option_coverage_fraction": float(
                np.mean(ranges >= limits.minimum_normalized_option_range)
            ),
            "argmax_coverage_fraction": 0.0,
        }
    by_context: dict[str, Any] = {}
    insufficient: list[str] = []
    worst_range = np.inf
    worst_correlation = 0.0
    covered = 0
    total = 0
    argmax_covered = 0
    argmax_total = 0
    for name, window in contexts.items():
        entry = _one_context(
            name, window, mapping, scores, normalized_pools, pool_activity, limits
        )
        by_context[name] = entry
        if entry["insufficient_evidence"]:
            if limits.minimum_context_states > 0:
                insufficient.append(name)
            continue
        worst_range = min(worst_range, float(entry["normalized_option_range"]["min"]))
        worst_correlation = max(
            worst_correlation, float(entry["competing_pool_correlation_max_absolute"])
        )
        covered += int(entry["options_above_minimum_range"])
        total += int(entry["active_option_count"])
        argmax_covered += int(entry["distinct_argmax_options"])
        argmax_total += int(entry["active_option_count"])
    if worst_range is np.inf:
        worst_range = 0.0
    return by_context, {
        "contexts_with_insufficient_evidence": insufficient,
        "minimum_normalized_option_range": worst_range,
        "maximum_competing_pool_correlation": worst_correlation,
        "action_option_coverage_fraction": covered / max(total, 1),
        "argmax_coverage_fraction": argmax_covered / max(argmax_total, 1),
        "evaluated_contexts": [
            name for name, entry in by_context.items() if not entry["insufficient_evidence"]
        ],
    }


def _one_context(
    name: str,
    window: MotorContextWindow,
    mapping: Any,
    scores: Mapping[str, NDArray[np.float64]],
    normalized_pools: NDArray[np.float64],
    pool_activity: NDArray[np.float64],
    limits: RepresentationThresholds,
) -> dict[str, Any]:
    head = window.spec.head
    route = np.asarray(mapping.routing.head_routes[head], dtype=np.int64)
    rows = window.relevant_indices
    active = np.asarray(
        [int(option) for option in window.active_options if route[int(option)] >= 0],
        dtype=np.int64,
    )
    competing_rows = np.flatnonzero(
        window.relevant & (window.option_legal.sum(axis=1) > 1)
    ).astype(np.int64)
    insufficient = (
        rows.size < max(limits.minimum_context_states, 2)
        or competing_rows.size < max(limits.minimum_competing_context_states, 1)
        or active.size < 2
    )
    base: dict[str, Any] = {
        "group": window.spec.group,
        "head": head,
        "description": window.spec.description,
        "eligible_states": int(rows.size),
        "states_with_competing_legal_options": int(competing_rows.size),
        "active_option_count": int(active.size),
        "contract_option_count": int(route.size),
        "insufficient_evidence": bool(insufficient),
    }
    if insufficient:
        return {
            **base,
            "reason": (
                f"{int(rows.size)} eligible state(s), {int(competing_rows.size)} with "
                f"more than one legal option and {int(active.size)} active option(s): "
                "this context cannot demonstrate that its pools are usable"
            ),
            "normalized_option_range": {"min": 0.0, "median": 0.0, "max": 0.0},
            "raw_pool_hz": {"min": 0.0, "median": 0.0, "max": 0.0},
            "options_above_minimum_range": 0,
            "distinct_argmax_options": 0,
            "argmax_coverage_fraction": 0.0,
            "competing_pool_correlation_max_absolute": 0.0,
            "effective_distinct_signals": 0.0,
            "effective_signal_fraction": 0.0,
        }
    pool_ids = [int(route[int(option)]) for option in active]
    block = normalized_pools[np.ix_(rows, pool_ids)]
    option_ranges = np.ptp(block, axis=0)
    correlation = _correlation(block)
    off = correlation[~np.eye(len(pool_ids), dtype=bool)]
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    head_scores = scores[head]
    argmax_options: list[int] = []
    for row in competing_rows:
        legal = window.option_legal[row].copy()
        legal[route < 0] = False
        candidates = np.flatnonzero(legal)
        if not candidates.size:
            continue
        argmax_options.append(int(candidates[int(np.argmax(head_scores[row, candidates]))]))
    distinct_argmax = len(set(argmax_options))
    return {
        **base,
        "normalized_option_range": {
            "min": float(option_ranges.min()),
            "median": float(np.median(option_ranges)),
            "max": float(option_ranges.max()),
        },
        "raw_pool_hz": {
            "min": float(pool_activity[np.ix_(rows, pool_ids)].min()),
            "median": float(np.median(pool_activity[np.ix_(rows, pool_ids)])),
            "max": float(pool_activity[np.ix_(rows, pool_ids)].max()),
        },
        "options_above_minimum_range": int(
            np.count_nonzero(option_ranges >= limits.minimum_normalized_option_range)
        ),
        "distinct_argmax_options": distinct_argmax,
        "argmax_coverage_fraction": distinct_argmax / max(int(active.size), 1),
        "competing_pool_correlation_max_absolute": (
            float(np.abs(off).max()) if off.size else 0.0
        ),
        "effective_distinct_signals": float(
            eigenvalues.sum() ** 2 / max(float((eigenvalues**2).sum()), 1e-12)
        ),
        "effective_signal_fraction": float(
            eigenvalues.sum() ** 2
            / max(float((eigenvalues**2).sum()), 1e-12)
            / max(len(pool_ids), 1)
        ),
    }


def _correlation(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centred = values - values.mean(axis=0)
    std = values.std(axis=0)
    standardized = centred / np.where(std > 0, std, 1.0)
    matrix = standardized.T @ standardized / max(values.shape[0] - 1, 1)
    constant = std <= 0
    if constant.any():
        matrix[constant, :] = 0.0
        matrix[:, constant] = 0.0
    np.fill_diagonal(matrix, 1.0)
    return np.clip(matrix, -1.0, 1.0)


def _centroid_probe(values: NDArray[np.float64], labels: NDArray[np.integer]) -> dict[str, object]:
    """Leave-one-out nearest-centroid probe without singleton-class leakage.

    A class with a single sample cannot supply a held-out centroid: using the
    held-out sample itself leaks the answer and inflates accuracy.  Such classes
    and their samples are excluded and reported explicitly.
    """

    if labels.shape != (len(values),):
        raise ValueError("observable category labels must identify every sample")
    unique, counts = np.unique(labels, return_counts=True)
    support = {int(label): int(count) for label, count in zip(unique, counts, strict=True)}
    usable_classes = unique[counts >= 2]
    excluded_classes = unique[counts < 2]
    usable_mask = np.isin(labels, usable_classes)
    base: dict[str, object] = {
        "diagnostic_only_not_a_policy_input": True,
        "classes": int(len(unique)),
        "usable_classes": int(len(usable_classes)),
        "excluded_singleton_classes": [int(value) for value in excluded_classes],
        "excluded_singleton_samples": int(np.count_nonzero(~usable_mask)),
        "sample_support": support,
        "evaluated_samples": int(np.count_nonzero(usable_mask)),
    }
    if len(usable_classes) < 2:
        return {
            **base,
            "nearest_centroid_leave_one_out_accuracy": None,
            "majority_class_reference_accuracy": None,
            "uniform_chance_accuracy": None,
            "reason": "probe requires at least two categories with support >= 2",
        }
    rows = np.flatnonzero(usable_mask)
    predictions = []
    for row in rows:
        distances = []
        for label in usable_classes:
            mask = labels == label
            mask[row] = False
            if not mask.any():
                distances.append(np.inf)
                continue
            distances.append(float(np.linalg.norm(values[row] - values[mask].mean(axis=0))))
        predictions.append(usable_classes[int(np.argmin(distances))])
    usable_counts = counts[counts >= 2]
    return {
        **base,
        "nearest_centroid_leave_one_out_accuracy": float(
            np.mean(np.asarray(predictions) == labels[rows])
        ),
        "majority_class_reference_accuracy": float(
            usable_counts.max() / usable_counts.sum()
        ),
        "uniform_chance_accuracy": float(1.0 / len(usable_classes)),
    }


def _matrix(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[1] or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} activity must be a finite non-negative sample-by-neuron Hz matrix")
    return array
