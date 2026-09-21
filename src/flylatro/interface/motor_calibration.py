"""Reward-free selection of fixed motor pools and their fixed normalization.

Nothing in this module observes reward, win rate, game score, a correct action
or any strategy label.  Selection uses only structural and statistical
properties of neural activity recorded over a frozen, reward-free calibration
state corpus, plus the environment's own legality masks.

Selection is **context-aware** (v3).  A routing group is evaluated in the states
where its pools are actually read, not across the whole corpus: a neuron with a
large dynamic range in ``PLAYING`` states and none in the shop is not evidence
that the contextual slot pools carry shop signal.  The four contextual heads
(shop, pack, joker, consumable) deliberately share one pool group, so each of
them is assessed separately and a context with too little evidence is reported
as such instead of being absorbed into a whole-corpus average.
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
from flylatro.interface.motor_contexts import (
    MOTOR_CONTEXT_VERSION,
    MotorContextWindow,
    group_relevant_rows,
)


MOTOR_CALIBRATION_VERSION = "reward-free-neural-motor-calibration-v3"
MOTOR_SELECTION_METHOD = (
    "reward-free-context-conditioned-robust-scale-with-within-group-decorrelation"
)


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
    #: States in which a context is actually interpreted before its evidence
    #: counts.  Set to 0 only to deliberately opt out (development doubles).
    minimum_context_states: int = 4
    #: Of those, states offering more than one legal option: a context where a
    #: single option is always forced proves nothing about competition.
    minimum_competing_context_states: int = 2


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
    contexts: Mapping[str, MotorContextWindow] | None = None,
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
    if contexts is not None:
        for window in contexts.values():
            if window.relevant.shape[0] != values.shape[0]:
                raise MotorCalibrationError(
                    "motor context windows must align with the calibration samples"
                )

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

    group_evidence = _group_evidence(values, routing, contexts, limits, roots)
    assignment = _assign_pools(
        routing=routing,
        pool_width=pool_width,
        maximum_correlation=limits.maximum_within_group_correlation,
        group_evidence=group_evidence,
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
        contexts=contexts,
    )
    selected_positions = np.asarray(
        [index for pool in pool_indices for index in pool], dtype=np.int64
    )
    raw_stats: dict[str, Any] = {
        "calibration_version": MOTOR_CALIBRATION_VERSION,
        "context_version": MOTOR_CONTEXT_VERSION,
        "context_aware": contexts is not None,
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
        "selection_evidence_by_group": {
            group: evidence["summary"] for group, evidence in group_evidence.items()
        },
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


# --------------------------------------------------------------------------
# context-conditioned candidate evidence
# --------------------------------------------------------------------------


def _group_evidence(
    values: NDArray[np.float64],
    routing: MotorRouting,
    contexts: Mapping[str, MotorContextWindow] | None,
    limits: MotorCalibrationThresholds,
    roots: NDArray[np.int64],
) -> dict[str, dict[str, Any]]:
    """Rank and filter candidates inside each group's own interpretation states."""

    evidence: dict[str, dict[str, Any]] = {}
    all_rows = np.arange(values.shape[0], dtype=np.int64)
    for group in routing.group_names:
        rows = all_rows
        restricted = False
        note = "evaluated over every calibration state (no legality context supplied)"
        if contexts is not None:
            relevant = group_relevant_rows(contexts, group)
            if relevant.size >= 2:
                rows = relevant
                restricted = True
                note = "evaluated only in the states where this group is interpreted"
            else:
                note = (
                    f"only {int(relevant.size)} state(s) interpret this group; fell "
                    "back to the whole corpus and the context-evidence gate fails"
                )
        subset = values[rows]
        statistics = _candidate_statistics(subset, limits)
        evidence[group] = {
            "rows": rows,
            "eligible": statistics["eligible"],
            "order": _ranked_candidates(statistics, roots),
            "standardized": _standardize(subset, statistics["std"]),
            "summary": {
                "relevant_states": int(rows.size),
                "context_restricted": restricted,
                "note": note,
                "eligible_candidates": int(statistics["eligible"].sum()),
                "raw_pool_dynamic_range_hz": {
                    "median": float(np.median(statistics["dynamic_range"])),
                    "max": float(statistics["dynamic_range"].max(initial=0.0)),
                },
            },
        }
    return evidence


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
    routing: MotorRouting,
    pool_width: int,
    maximum_correlation: float,
    group_evidence: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    used: set[int] = set()
    pools: list[tuple[int, ...] | None] = [None] * routing.pool_count
    relaxed: list[dict[str, Any]] = []
    for group, count in zip(routing.group_names, routing.group_pool_counts, strict=True):
        evidence = group_evidence[group]
        standardized = evidence["standardized"]
        eligible = evidence["eligible"]
        samples = standardized.shape[0]
        ranked = [int(index) for index in evidence["order"] if bool(eligible[index])]
        pool_ids = routing.group_pool_ids(group)
        need = count * pool_width
        chosen: list[int] = []
        for slot in range(need):
            available = [index for index in ranked if index not in used]
            if not available:
                raise MotorCalibrationError(
                    f"group {group} exhausted usable reward-free motor candidates "
                    f"in the {samples} state(s) where it is interpreted"
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


# --------------------------------------------------------------------------
# quality evidence
# --------------------------------------------------------------------------


def _quality_report(
    *,
    pool_activity: NDArray[np.float64],
    normalized: NDArray[np.float64],
    routing: MotorRouting,
    limits: MotorCalibrationThresholds,
    contexts: Mapping[str, MotorContextWindow] | None,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in routing.group_names:
        pool_ids = list(routing.group_pool_ids(group))
        groups[group] = _block_metrics(
            pool_activity[:, pool_ids], normalized[:, pool_ids], limits
        )
        groups[group]["pools"] = len(pool_ids)
    by_context: dict[str, Any] = {}
    worst_range = np.inf
    worst_effective = np.inf
    worst_silent = 0.0
    insufficient: list[str] = []
    if contexts is None:
        by_context = {}
        insufficient = list(
            () if limits.minimum_context_states <= 0 else ("<no legality context supplied>",)
        )
        worst_range = min(
            (float(entry["normalized_option_range"]["min"]) for entry in groups.values()),
            default=np.inf,
        )
        worst_effective = min(
            (float(entry["effective_signal_fraction"]) for entry in groups.values()),
            default=np.inf,
        )
        worst_silent = max(
            (float(entry["pool_silent_fraction"]["max"]) for entry in groups.values()),
            default=0.0,
        )
    else:
        for name, window in contexts.items():
            entry = _context_metrics(
                window, pool_activity, normalized, routing, limits
            )
            by_context[name] = entry
            if entry["insufficient_evidence"]:
                # A zero requirement is an explicit declaration that this run
                # does not demand context evidence (development doubles only).
                # The per-context report still records exactly what was missing.
                if limits.minimum_context_states > 0:
                    insufficient.append(name)
                continue
            worst_range = min(worst_range, float(entry["normalized_option_range"]["min"]))
            worst_effective = min(worst_effective, float(entry["effective_signal_fraction"]))
            worst_silent = max(worst_silent, float(entry["pool_silent_fraction"]["max"]))
        if not by_context:
            insufficient.append("<no motor contexts>")
    if worst_range is np.inf:
        worst_range = 0.0
    if worst_effective is np.inf:
        worst_effective = 0.0
    checks = {
        "reward_free": True,
        "pool_width_sufficient": True,
        "context_evidence_sufficient": not insufficient,
        "option_dynamic_range": worst_range >= limits.minimum_normalized_option_range,
        "group_signal_diversity": worst_effective >= limits.minimum_effective_signal_fraction,
        "pools_not_silent": worst_silent <= limits.maximum_pool_silent_fraction,
    }
    return {
        "context_version": MOTOR_CONTEXT_VERSION,
        "by_group": groups,
        "by_context": by_context,
        "contexts_with_insufficient_evidence": insufficient,
        "minimum_normalized_option_range": worst_range,
        "minimum_effective_signal_fraction": worst_effective,
        "maximum_pool_silent_fraction": worst_silent,
        "evidence_rule": (
            "metrics are measured in the states where each head is actually "
            "read; a context with too few relevant or competing states is "
            "reported as insufficient and never averaged into a passing gate"
        ),
        "checks": checks,
    }


def _context_metrics(
    window: MotorContextWindow,
    pool_activity: NDArray[np.float64],
    normalized: NDArray[np.float64],
    routing: MotorRouting,
    limits: MotorCalibrationThresholds,
) -> dict[str, Any]:
    rows = window.relevant_indices
    route = np.asarray(routing.head_routes[window.spec.head], dtype=np.int64)
    active = window.active_options
    options = np.asarray(
        [int(option) for option in active if route[int(option)] >= 0], dtype=np.int64
    )
    competing = window.competing_states
    insufficient = (
        rows.size < limits.minimum_context_states
        or competing < limits.minimum_competing_context_states
        or options.size < 2
        or rows.size < 2
    )
    base: dict[str, Any] = {
        "group": window.spec.group,
        "head": window.spec.head,
        "description": window.spec.description,
        "relevant_states": int(rows.size),
        "states_with_competing_options": competing,
        "active_option_count": int(options.size),
        "option_width": int(route.size),
        "coverage_of_contract_options": float(options.size / max(route.size, 1)),
        "insufficient_evidence": bool(insufficient),
        "minimum_states_required": limits.minimum_context_states,
        "minimum_competing_states_required": limits.minimum_competing_context_states,
    }
    if insufficient:
        return {
            **base,
            "reason": (
                f"{int(rows.size)} relevant state(s), {competing} with competing "
                f"options and {int(options.size)} active option(s): not enough "
                "evidence that these pools carry usable signal in this context"
            ),
            "normalized_option_range": {"min": 0.0, "median": 0.0, "max": 0.0},
            "raw_pool_hz": {"min": 0.0, "median": 0.0, "max": 0.0},
            "dynamic_range_hz": {"min": 0.0, "median": 0.0, "max": 0.0},
            "pool_silent_fraction": {"min": 1.0, "median": 1.0, "max": 1.0},
            "competing_pool_correlation": {"mean_absolute": 0.0, "max_absolute": 0.0},
            "effective_distinct_signals": 0.0,
            "effective_signal_fraction": 0.0,
        }
    pool_ids = [int(route[int(option)]) for option in options]
    metrics = _block_metrics(
        pool_activity[np.ix_(rows, pool_ids)],
        normalized[np.ix_(rows, pool_ids)],
        limits,
    )
    return {**base, **metrics, "pools": len(pool_ids)}


def _block_metrics(
    raw: NDArray[np.float64],
    norm: NDArray[np.float64],
    limits: MotorCalibrationThresholds,
) -> dict[str, Any]:
    correlation = _correlation_matrix(raw)
    off_diagonal = correlation[~np.eye(raw.shape[1], dtype=bool)]
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    effective = float(
        eigenvalues.sum() ** 2 / max(float((eigenvalues**2).sum()), 1e-12)
    )
    ranges = np.ptp(norm, axis=0)
    silent = np.mean(raw <= limits.silence_hz, axis=0)
    return {
        "baseline_hz": {
            "min": float(np.median(raw, axis=0).min()),
            "median": float(np.median(np.median(raw, axis=0))),
            "max": float(np.median(raw, axis=0).max()),
        },
        "raw_pool_hz": {
            "min": float(raw.min()),
            "median": float(np.median(raw)),
            "max": float(raw.max()),
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
        "effective_signal_fraction": effective / max(raw.shape[1], 1),
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
