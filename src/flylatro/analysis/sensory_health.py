"""State-conditioned sensory health on real encoded Balatro observations.

The scientific question is not how many dormant feature channels share an ALPN.
It is whether, in the states the fly actually sees, collisions saturate the
input population or destroy the observable-state information the experiment
depends on.  Those are different quantities:

* **structural collision load** counts every feature channel assigned to an
  ALPN across the whole pinned contract, active or not;
* **simultaneous collision load** counts only the channels that are non-zero in
  one observed state and therefore actually sum into one ALPN's drive.

Only the second can attenuate or destroy state information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import ObsDict
from flylatro.fly.plastic_features import observation_features
from flylatro.interface.sensory import FixedPlasticSensoryEncoder, SensoryMapping


SENSORY_HEALTH_VERSION = "state-conditioned-sensory-health-v1"


@dataclass(frozen=True, slots=True)
class SensoryHealthThresholds:
    saturation_tolerance_hz: float = 1e-3
    maximum_saturated_alpn_fraction: float = 0.25
    maximum_collision_saturated_fraction: float = 0.10
    maximum_identical_vector_fraction: float = 0.0
    minimum_active_alpn_fraction: float = 0.01
    minimum_state_separation_hz: float = 1.0


def sensory_health_report(
    mapping: SensoryMapping,
    observations: ObsDict,
    *,
    state_hashes: tuple[str, ...] | None = None,
    thresholds: SensoryHealthThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or SensoryHealthThresholds()
    encoder = FixedPlasticSensoryEncoder(mapping)
    rates = np.asarray(encoder.encode(observations).rates_hz, dtype=np.float64)
    repeat = np.asarray(encoder.encode(observations).rates_hz, dtype=np.float64)
    deterministic = bool(np.array_equal(rates, repeat))
    alpn = rates[:, mapping.available_alpn_indices]
    samples, population = alpn.shape
    features = observation_features(observations)
    contributors, single_max = _simultaneous_contributors(mapping, features)
    saturation = mapping.max_rate_hz - limits.saturation_tolerance_hz
    saturated = alpn >= saturation
    active = alpn > 0
    collision_saturated = saturated & (single_max < 1.0 - 1e-6) & (contributors > 1)
    active_rates = alpn[active]
    contributor_counts = contributors[active]
    separation = _state_separation(alpn, state_hashes)
    checks = {
        "encoder_deterministic": deterministic,
        "alpn_drive_present": float(active.mean()) >= limits.minimum_active_alpn_fraction,
        "saturation_bounded": float(saturated.mean()) <= limits.maximum_saturated_alpn_fraction,
        "collision_saturation_bounded": (
            float(collision_saturated.mean()) <= limits.maximum_collision_saturated_fraction
        ),
        "distinct_states_remain_distinguishable": (
            separation["distinct_state_pairs_with_identical_vector_fraction"]
            <= limits.maximum_identical_vector_fraction
        ),
        "state_separation_present": (
            separation["different_state_mean_distance_hz"] is None
            or separation["different_state_mean_distance_hz"]
            >= limits.minimum_state_separation_hz
        ),
    }
    return {
        "version": SENSORY_HEALTH_VERSION,
        "activity_unit": "input_drive_hz",
        "states": int(samples),
        "alpn_population": int(population),
        "max_rate_hz": mapping.max_rate_hz,
        "thresholds": asdict(limits),
        "state_conditioned": {
            "fraction_alpns_silent": float(np.mean(~active.any(axis=0))),
            "fraction_alpns_ever_active": float(np.mean(active.any(axis=0))),
            "fraction_state_alpn_active": float(active.mean()),
            "fraction_state_alpn_at_max_rate": float(saturated.mean()),
            "fraction_alpns_ever_at_max_rate": float(np.mean(saturated.any(axis=0))),
            "mean_alpn_rate_hz": float(alpn.mean()),
            "median_active_alpn_rate_hz": (
                float(np.median(active_rates)) if active_rates.size else 0.0
            ),
            "p95_active_alpn_rate_hz": (
                float(np.quantile(active_rates, 0.95)) if active_rates.size else 0.0
            ),
        },
        "simultaneous_collision_load": {
            "metric_kind": "state_conditioned_simultaneous_active_contributors",
            "active_contributors_per_active_alpn": _quantiles(contributor_counts),
            "fraction_active_alpns_with_multiple_features": (
                float(np.mean(contributor_counts > 1)) if contributor_counts.size else 0.0
            ),
            "fraction_state_alpn_saturated_by_collision": float(collision_saturated.mean()),
            "fraction_of_saturated_entries_caused_by_collision": (
                float(collision_saturated.sum() / saturated.sum())
                if saturated.any()
                else 0.0
            ),
            "mean_active_feature_channels_per_state": float(np.mean(features > 0)) * features.shape[1],
        },
        "state_separation": separation,
        "determinism": {
            "same_input_produces_identical_vector": deterministic,
            "repeat_max_absolute_difference_hz": float(np.abs(rates - repeat).max()),
        },
        "structural_assignment_metrics": mapping.collision_audit(),
        "gates": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "failed": [name for name, passed in checks.items() if not passed],
        },
    }


def _simultaneous_contributors(
    mapping: SensoryMapping, features: NDArray[np.floating]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Per state and ALPN: active contributing channels and their largest value."""

    values = np.asarray(features, dtype=np.float64)
    samples = values.shape[0]
    position = np.full(mapping.neuron_count, -1, dtype=np.int64)
    position[mapping.available_alpn_indices] = np.arange(
        len(mapping.available_alpn_indices), dtype=np.int64
    )
    routed = position[mapping.population_indices]
    counts = np.zeros((samples, len(mapping.available_alpn_indices)), dtype=np.int64)
    largest = np.zeros_like(counts, dtype=np.float64)
    for row in range(samples):
        active = np.flatnonzero(values[row] > 0)
        if not active.size:
            continue
        targets = routed[active].ravel()
        magnitudes = np.repeat(values[row, active], routed.shape[1])
        valid = targets >= 0
        np.add.at(counts[row], targets[valid], 1)
        np.maximum.at(largest[row], targets[valid], magnitudes[valid])
    return counts, largest


def _state_separation(
    alpn: NDArray[np.float64], state_hashes: tuple[str, ...] | None
) -> dict[str, Any]:
    samples = alpn.shape[0]
    same: list[float] = []
    different: list[float] = []
    identical_vector_pairs = 0
    different_pairs = 0
    labels = (
        list(state_hashes)
        if state_hashes is not None and len(state_hashes) == samples
        else [row.tobytes() for row in alpn]
    )
    if state_hashes is None or len(state_hashes) != samples:
        labels = [str(index) for index in range(samples)]
    for left in range(samples):
        for right in range(left + 1, samples):
            distance = float(np.linalg.norm(alpn[left] - alpn[right]))
            if labels[left] == labels[right]:
                same.append(distance)
            else:
                different.append(distance)
                different_pairs += 1
                if distance == 0.0:
                    identical_vector_pairs += 1
    return {
        "compared_pairs": samples * (samples - 1) // 2,
        "identical_state_pairs": len(same),
        "distinct_state_pairs": different_pairs,
        "same_state_vectors_identical": bool(all(value == 0.0 for value in same)),
        "same_state_max_distance_hz": float(max(same)) if same else 0.0,
        "different_state_mean_distance_hz": (
            float(np.mean(different)) if different else None
        ),
        "different_state_median_distance_hz": (
            float(np.median(different)) if different else None
        ),
        "different_state_min_distance_hz": float(min(different)) if different else None,
        "distinct_state_pairs_with_identical_vector_fraction": (
            identical_vector_pairs / different_pairs if different_pairs else 0.0
        ),
    }


def _quantiles(values: NDArray[np.number]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {name: 0.0 for name in ("median", "p90", "p95", "p99", "max")}
    return {
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def merge_identity(report: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(report), "evidence_identity": dict(identity)}
