"""Representation diagnostics with explicit spike-rate units and gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RepresentationThresholds:
    silence_hz: float = 0.0
    high_rate_hz: float = 200.0
    maximum_silent_fraction: float = 0.95
    maximum_high_rate_fraction: float = 0.25
    minimum_separation_ratio: float = 1.10
    minimum_motor_dynamic_range_hz: float = 1.0
    minimum_action_coverage_fraction: float = 0.50


def representation_diagnostics(
    kc_activity: NDArray[np.floating],
    mbon_activity: NDArray[np.floating],
    descending_activity: NDArray[np.floating],
    *,
    state_labels: NDArray[np.integer] | None = None,
    observable_categories: Mapping[str, NDArray[np.integer]] | None = None,
    motor_pools: Mapping[str, tuple[tuple[int, ...], ...]] | None = None,
    thresholds: RepresentationThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or RepresentationThresholds()
    kc = _matrix(kc_activity, "KC")
    mbon = _matrix(mbon_activity, "MBON")
    descending = _matrix(descending_activity, "descending")
    if not (len(kc) == len(mbon) == len(descending)):
        raise ValueError("all activity matrices must have the same samples")
    result: dict[str, Any] = {
        "activity_unit": "spikes_per_second_hz",
        "samples": kc.shape[0],
        "thresholds": asdict(limits),
        "populations": {
            "kc": _population(kc, limits),
            "mbon": _population(mbon, limits),
            "descending": _population(descending, limits),
        },
        "state_geometry": _state_geometry(mbon, state_labels),
        "motor": _motor_diagnostics(descending, motor_pools, limits),
        "observable_category_probes": {},
    }
    for name, labels in (observable_categories or {}).items():
        result["observable_category_probes"][name] = _centroid_probe(mbon, np.asarray(labels))
    geometry = result["state_geometry"]
    mbon_stats = result["populations"]["mbon"]
    motor = result["motor"]
    checks = {
        "mbon_not_silent": mbon_stats["silent_neuron_fraction"] <= limits.maximum_silent_fraction,
        "mbon_not_high_rate": mbon_stats["high_rate_sample_fraction"] <= limits.maximum_high_rate_fraction,
        "state_separation": geometry["separation_ratio"] is None or geometry["separation_ratio"] >= limits.minimum_separation_ratio,
        "motor_dynamic_range": motor["mean_dynamic_range_hz"] >= limits.minimum_motor_dynamic_range_hz,
        "action_coverage": motor["action_coverage_fraction"] >= limits.minimum_action_coverage_fraction,
    }
    result["gates"] = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
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
    result = {
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


def _motor_diagnostics(values: NDArray[np.float64], pools: Mapping[str, tuple[tuple[int, ...], ...]] | None, limits: RepresentationThresholds) -> dict[str, Any]:
    dynamic = np.ptp(values, axis=0)
    coverage = float(np.mean(dynamic >= limits.minimum_motor_dynamic_range_hz))
    by_head: dict[str, object] = {}
    if pools:
        covered = 0
        total = 0
        for name, options in pools.items():
            ranges = [float(np.ptp(values[:, pool].mean(axis=1))) for pool in options]
            by_head[name] = {"option_dynamic_range_hz": ranges}
            covered += sum(value >= limits.minimum_motor_dynamic_range_hz for value in ranges)
            total += len(ranges)
        coverage = covered / max(total, 1)
    return {
        "mean_dynamic_range_hz": float(dynamic.mean()),
        "minimum_dynamic_range_hz": float(dynamic.min()),
        "maximum_dynamic_range_hz": float(dynamic.max()),
        "action_coverage_fraction": coverage,
        "by_head": by_head,
    }


def _centroid_probe(values: NDArray[np.float64], labels: NDArray[np.integer]) -> dict[str, object]:
    if labels.shape != (len(values),):
        raise ValueError("observable category labels must identify every sample")
    unique = np.unique(labels)
    if len(unique) < 2:
        return {
            "diagnostic_only_not_a_policy_input": True,
            "classes": int(len(unique)),
            "nearest_centroid_leave_one_out_accuracy": None,
            "reason": "probe requires at least two observed categories",
        }
    predictions = []
    for row in range(len(values)):
        distances = []
        for label in unique:
            mask = labels == label
            mask[row] = False
            centroid = values[mask].mean(axis=0) if mask.any() else values[row]
            distances.append(np.linalg.norm(values[row] - centroid))
        predictions.append(unique[int(np.argmin(distances))])
    return {
        "diagnostic_only_not_a_policy_input": True,
        "classes": int(len(unique)),
        "nearest_centroid_leave_one_out_accuracy": float(np.mean(np.asarray(predictions) == labels)),
    }


def _matrix(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[1] or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} activity must be a finite non-negative sample-by-neuron Hz matrix")
    return array
