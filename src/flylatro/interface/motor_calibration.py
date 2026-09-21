"""Reward-free selection of fixed motor pools and their fixed normalization.

Nothing in this module observes reward, win rate, game score, a correct action
or any strategy label.  Selection uses only structural and statistical
properties of neural activity recorded over a frozen, reward-free calibration
state corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.interface.motor import (
    CONTEXTUAL_ROUTING,
    MotorMapping,
    MotorNormalization,
    MotorRouting,
)


MOTOR_CALIBRATION_VERSION = "reward-free-neural-motor-calibration-v2"
MOTOR_SELECTION_METHOD = "reward-free-robust-scale-with-within-group-decorrelation"


class MotorCalibrationError(ValueError):
    """Raised when reward-free calibration cannot build a usable interface."""

    def __init__(self, message: str, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report or {})


@dataclass(frozen=True, slots=True)
class MotorCalibrationThresholds:
    silence_hz: float = 0.0
    high_rate_hz: float = 200.0
    minimum_candidate_robust_scale_hz: float = 0.5
    minimum_scale_hz: float = 0.5
    maximum_within_group_correlation: float = 0.95
    minimum_effective_signal_fraction: float = 0.50
    minimum_normalized_option_range: float = 0.25
    maximum_pool_silent_fraction: float = 0.90
    minimum_pool_width: int = 2


@dataclass(frozen=True, slots=True)
class MotorCalibrationResult:
    mapping: MotorMapping
    report: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.report["gates"]["status"])


def calibrate_reward_free_motor(
    candidate_root_ids: NDArray[np.int64],
    activity_hz: NDArray[np.floating],
    *,
    mode: str,
    pool_width: int = 2,
    routing: MotorRouting = CONTEXTUAL_ROUTING,
    thresholds: MotorCalibrationThresholds | None = None,
    exploration_epsilon: float = 0.0,
    exploration_temperature: float = 1.0,
    exploration_seed: int = 0,
    candidate_set_sha256: str | None = None,
    candidate_details: Sequence[Mapping[str, Any]] | None = None,
    calibration_metadata: Mapping[str, Any] | None = None,
) -> MotorCalibrationResult:
    limits = thresholds or MotorCalibrationThresholds()
    roots = np.asarray(candidate_root_ids, dtype=np.int64)
    values = np.asarray(activity_hz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(roots):
        raise MotorCalibrationError("calibration activity must be sample-by-candidate")
    if pool_width < limits.minimum_pool_width:
        raise MotorCalibrationError(
            f"calibrated motor pools need at least {limits.minimum_pool_width} neurons"
        )
    if values.shape[0] < 4 or not np.isfinite(values).all():
        raise MotorCalibrationError("insufficient finite reward-free calibration states")

    statistics = _candidate_statistics(values, limits)
    eligible = statistics["eligible"]
    required = routing.pool_count * pool_width
    if int(eligible.sum()) < required:
        raise MotorCalibrationError(
            f"reward-free calibration found {int(eligible.sum())} usable motor "
            f"candidates; the contextual routing needs {required} "
            f"({routing.pool_count} pools x width {pool_width})",
            {"candidate_statistics": _summarize_candidates(statistics, roots)},
        )

    order = _ranked_candidates(statistics, roots)
    standardized = _standardize(values, statistics["std"])
    assignment = _assign_pools(
        order=order,
        eligible=eligible,
        standardized=standardized,
        routing=routing,
        pool_width=pool_width,
        maximum_correlation=limits.maximum_within_group_correlation,
    )
    pool_indices = assignment["pool_indices"]
    pool_activity = np.stack(
        [values[:, list(pool)].mean(axis=1) for pool in pool_indices], axis=1
    )
    normalization = MotorNormalization.from_pool_activity(
        pool_activity, minimum_scale_hz=limits.minimum_scale_hz
    )
    normalized = normalization.apply(pool_activity)
    quality = _quality_report(
        pool_activity=pool_activity,
        normalized=normalized,
        routing=routing,
        limits=limits,
    )
    selected_positions = np.asarray(
        [index for pool in pool_indices for index in pool], dtype=np.int64
    )
    raw_stats: dict[str, Any] = {
        "calibration_version": MOTOR_CALIBRATION_VERSION,
        "reward_used": False,
        "outcome_information_used": False,
        "sample_count": int(values.shape[0]),
        "candidate_count": int(len(roots)),
        "eligible_candidate_count": int(eligible.sum()),
        "selected_count": int(len(selected_positions)),
        "pool_count": routing.pool_count,
        "pool_width": int(pool_width),
        "required_outputs": required,
        "thresholds": asdict(limits),
        "selected_root_ids": roots[selected_positions].tolist(),
        "pool_root_ids": [
            [int(roots[index]) for index in pool] for pool in pool_indices
        ],
        "relaxed_correlation_assignments": assignment["relaxed"],
        "candidate_statistics": _summarize_candidates(statistics, roots),
        "selected_candidate_details": (
            [dict(candidate_details[int(position)]) for position in selected_positions]
            if candidate_details is not None
            else None
        ),
        "pool_reuse": routing.reuse_summary(),
        "quality": quality,
        "state_sample_set": dict(calibration_metadata or {}),
    }
    calibration_hash = hashlib.sha256(
        json.dumps(raw_stats, sort_keys=True, separators=(",", ":"), default=_encode).encode()
    ).hexdigest()
    mapping = MotorMapping(
        version=MOTOR_CALIBRATION_VERSION,
        mode=mode,
        output_root_ids=roots.copy(),
        pool_indices=pool_indices,
        routing=routing,
        normalization=normalization,
        exploration_epsilon=exploration_epsilon,
        exploration_temperature=exploration_temperature,
        exploration_seed=exploration_seed,
        selection_method=MOTOR_SELECTION_METHOD,
        candidate_set_sha256=candidate_set_sha256,
        calibration_sha256=calibration_hash,
        calibration_stats=raw_stats,
    )
    checks = quality["checks"]
    report = {
        "version": MOTOR_CALIBRATION_VERSION,
        "mode": mode,
        "motor_mapping_sha256": mapping.structure_sha256,
        "motor_artifact_sha256": mapping.sha256,
        "routing_sha256": routing.sha256,
        "normalization_sha256": normalization.sha256,
        "candidate_set_sha256": candidate_set_sha256,
        "calibration_sha256": calibration_hash,
        "calibration": raw_stats,
        "normalization": normalization.to_manifest(),
        "gates": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "failed": [name for name, passed in checks.items() if not passed],
        },
    }
    return MotorCalibrationResult(mapping=mapping, report=report)


def _candidate_statistics(
    values: NDArray[np.float64], limits: MotorCalibrationThresholds
) -> dict[str, NDArray[Any]]:
    quantiles = np.quantile(values, (0.25, 0.5, 0.75), axis=0)
    iqr = quantiles[2] - quantiles[0]
    robust = iqr / 1.349
    std = values.std(axis=0)
    dynamic = np.ptp(values, axis=0)
    active = np.mean(values > limits.silence_hz, axis=0)
    high = np.mean(values >= limits.high_rate_hz, axis=0)
    eligible = (
        (active > 0)
        & (high < 1.0)
        & (robust >= limits.minimum_candidate_robust_scale_hz)
        & np.isfinite(robust)
    )
    return {
        "median": quantiles[1],
        "iqr": iqr,
        "robust_scale": robust,
        "std": std,
        "dynamic_range": dynamic,
        "active_fraction": active,
        "high_rate_fraction": high,
        "mean": values.mean(axis=0),
        "eligible": eligible,
    }


def _summarize_candidates(
    statistics: Mapping[str, NDArray[Any]], roots: NDArray[np.int64]
) -> dict[str, Any]:
    eligible = statistics["eligible"]
    def quantiles(values: NDArray[np.float64]) -> dict[str, float]:
        if not values.size:
            return {}
        return {
            name: float(np.quantile(values, quantile))
            for name, quantile in (
                ("min", 0.0), ("p05", 0.05), ("median", 0.5),
                ("p95", 0.95), ("max", 1.0),
            )
        }
    return {
        "candidates": int(len(roots)),
        "eligible": int(eligible.sum()),
        "eligible_fraction": float(eligible.mean()),
        "always_silent": int(np.count_nonzero(statistics["active_fraction"] <= 0)),
        "always_high_rate": int(np.count_nonzero(statistics["high_rate_fraction"] >= 1.0)),
        "below_minimum_robust_scale": int(
            np.count_nonzero(~eligible & (statistics["active_fraction"] > 0))
        ),
        "baseline_firing_hz": quantiles(statistics["median"]),
        "mean_firing_hz": quantiles(statistics["mean"]),
        "dynamic_range_hz": quantiles(statistics["dynamic_range"]),
        "variance_hz2": quantiles(statistics["std"] ** 2),
        "robust_scale_hz": quantiles(statistics["robust_scale"]),
    }


def _ranked_candidates(
    statistics: Mapping[str, NDArray[Any]], roots: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Deterministic reward-free ranking: robust scale, range, then root ID."""

    return np.lexsort(
        (roots, -statistics["dynamic_range"], -statistics["robust_scale"])
    ).astype(np.int64)


def _standardize(
    values: NDArray[np.float64], std: NDArray[np.float64]
) -> NDArray[np.float64]:
    safe = np.where(std > 0, std, 1.0)
    return (values - values.mean(axis=0)) / safe


def _assign_pools(
    *,
    order: NDArray[np.int64],
    eligible: NDArray[np.bool_],
    standardized: NDArray[np.float64],
    routing: MotorRouting,
    pool_width: int,
    maximum_correlation: float,
) -> dict[str, Any]:
    samples = standardized.shape[0]
    ranked = [int(index) for index in order if bool(eligible[index])]
    used: set[int] = set()
    pools: list[tuple[int, ...] | None] = [None] * routing.pool_count
    relaxed: list[dict[str, Any]] = []
    for group, count in zip(routing.group_names, routing.group_pool_counts, strict=True):
        pool_ids = routing.group_pool_ids(group)
        need = count * pool_width
        chosen: list[int] = []
        for slot in range(need):
            available = [index for index in ranked if index not in used]
            if not available:
                raise MotorCalibrationError(
                    f"group {group} exhausted usable reward-free motor candidates"
                )
            picked = None
            if chosen:
                reference = standardized[:, chosen]
                for candidate in available:
                    correlation = np.abs(
                        reference.T @ standardized[:, candidate] / max(samples - 1, 1)
                    )
                    if correlation.max(initial=0.0) <= maximum_correlation:
                        picked = candidate
                        break
            else:
                picked = available[0]
            if picked is None:
                picked = available[0]
                relaxed.append(
                    {"group": group, "slot": slot, "candidate": int(picked)}
                )
            chosen.append(picked)
            used.add(picked)
        # Interleave so that no single option monopolises the most responsive
        # neurons: pool j receives ranks j, j+count, j+2*count, ...
        for position, pool_id in enumerate(pool_ids):
            members = tuple(
                chosen[position + step * count] for step in range(pool_width)
            )
            pools[pool_id] = members
    if any(pool is None for pool in pools):
        raise MotorCalibrationError("routing pool was not assigned a neural population")
    return {"pool_indices": tuple(pools), "relaxed": relaxed}  # type: ignore[arg-type]


def _quality_report(
    *,
    pool_activity: NDArray[np.float64],
    normalized: NDArray[np.float64],
    routing: MotorRouting,
    limits: MotorCalibrationThresholds,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    worst_range = np.inf
    worst_effective = np.inf
    worst_silent = 0.0
    for group in routing.group_names:
        pool_ids = list(routing.group_pool_ids(group))
        raw = pool_activity[:, pool_ids]
        norm = normalized[:, pool_ids]
        correlation = _correlation_matrix(raw)
        off_diagonal = correlation[~np.eye(len(pool_ids), dtype=bool)]
        eigenvalues = np.linalg.eigvalsh(correlation)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        effective = float(
            eigenvalues.sum() ** 2 / max(float((eigenvalues**2).sum()), 1e-12)
        )
        ranges = np.ptp(norm, axis=0)
        silent = np.mean(raw <= limits.silence_hz, axis=0)
        groups[group] = {
            "pools": len(pool_ids),
            "baseline_hz": {
                "min": float(np.median(raw, axis=0).min()),
                "median": float(np.median(np.median(raw, axis=0))),
                "max": float(np.median(raw, axis=0).max()),
            },
            "dynamic_range_hz": {
                "min": float(np.ptp(raw, axis=0).min()),
                "median": float(np.median(np.ptp(raw, axis=0))),
                "max": float(np.ptp(raw, axis=0).max()),
            },
            "variance_hz2": {
                "min": float(raw.var(axis=0).min()),
                "median": float(np.median(raw.var(axis=0))),
                "max": float(raw.var(axis=0).max()),
            },
            "normalized_option_range": {
                "min": float(ranges.min()),
                "median": float(np.median(ranges)),
                "max": float(ranges.max()),
            },
            "pool_silent_fraction": {
                "min": float(silent.min()),
                "median": float(np.median(silent)),
                "max": float(silent.max()),
            },
            "pool_high_rate_fraction": {
                "max": float(np.mean(raw >= limits.high_rate_hz, axis=0).max()),
            },
            "competing_pool_correlation": {
                "mean_absolute": float(np.abs(off_diagonal).mean()) if off_diagonal.size else 0.0,
                "max_absolute": float(np.abs(off_diagonal).max()) if off_diagonal.size else 0.0,
            },
            "effective_distinct_signals": effective,
            "effective_signal_fraction": effective / max(len(pool_ids), 1),
        }
        worst_range = min(worst_range, float(ranges.min()))
        worst_effective = min(worst_effective, effective / max(len(pool_ids), 1))
        worst_silent = max(worst_silent, float(silent.max()))
    checks = {
        "reward_free": True,
        "pool_width_sufficient": True,
        "option_dynamic_range": worst_range >= limits.minimum_normalized_option_range,
        "group_signal_diversity": worst_effective >= limits.minimum_effective_signal_fraction,
        "pools_not_silent": worst_silent <= limits.maximum_pool_silent_fraction,
    }
    return {
        "by_group": groups,
        "minimum_normalized_option_range": worst_range,
        "minimum_effective_signal_fraction": worst_effective,
        "maximum_pool_silent_fraction": worst_silent,
        "checks": checks,
    }


def _correlation_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centred = values - values.mean(axis=0)
    std = values.std(axis=0)
    safe = np.where(std > 0, std, 1.0)
    standardized = centred / safe
    matrix = standardized.T @ standardized / max(values.shape[0] - 1, 1)
    constant = std <= 0
    if constant.any():
        matrix[constant, :] = 0.0
        matrix[:, constant] = 0.0
    np.fill_diagonal(matrix, 1.0)
    return np.clip(matrix, -1.0, 1.0)


def _encode(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unserializable calibration value: {type(value)!r}")
