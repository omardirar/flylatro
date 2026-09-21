"""Does the configured sensory route and time window reach the plastic pool?

V1 injects synthetic drive only at annotated ALPNs, but the plastic population
is every KC->MBON pair in the connectome.  If most Kenyon cells never fire
inside the configured decision window, most plastic edges can never become
eligible and the experiment silently studies a much smaller circuit than it
claims.  These diagnostics measure that directly.

They never modify the biological population. A poor result is evidence for a
separate, versioned model decision (restrict the primary plastic population, or
add another anatomically legitimate sensory route), not an automatic change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.mushroom_body.plasticity import PlasticityConfig
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


REACHABILITY_REPORT_VERSION = "kc-subtype-reachability-v1"


@dataclass(frozen=True, slots=True)
class ReachabilityThresholds:
    silence_hz: float = 0.0
    minimum_reachable_edge_fraction: float = 0.10
    minimum_active_kc_fraction: float = 0.01
    minimum_plastic_mbon_active_fraction: float = 0.50
    minimum_motor_output_active_fraction: float = 0.80


def kc_reachability_report(
    *,
    kc_activity: NDArray[np.floating],
    kc_indices: NDArray[np.integer],
    kc_types: NDArray[np.str_] | tuple[str, ...],
    mbon_activity: NDArray[np.floating],
    mbon_indices: NDArray[np.integer],
    topology: PlasticEdgeTopology,
    plasticity: PlasticityConfig,
    motor_output_indices: NDArray[np.integer] | None = None,
    motor_activity: NDArray[np.floating] | None = None,
    thresholds: ReachabilityThresholds | None = None,
    chunk_size: int = 8,
) -> dict[str, Any]:
    limits = thresholds or ReachabilityThresholds()
    kc = np.asarray(kc_activity, dtype=np.float64)
    mbon = np.asarray(mbon_activity, dtype=np.float64)
    kc_index = np.asarray(kc_indices, dtype=np.int64)
    mbon_index = np.asarray(mbon_indices, dtype=np.int64)
    if kc.ndim != 2 or mbon.ndim != 2 or kc.shape[0] != mbon.shape[0]:
        raise ValueError("KC and MBON activity must be aligned sample-by-neuron")
    if kc.shape[1] != len(kc_index) or mbon.shape[1] != len(mbon_index):
        raise ValueError("activity columns must align with the supplied indices")
    types = np.asarray(kc_types, dtype=np.str_)
    if types.shape != kc_index.shape:
        raise ValueError("one KC type label is required per KC column")
    samples = kc.shape[0]

    edge_pre = _positions(kc_index, topology.pre_indices, "presynaptic KC")
    edge_post = _positions(mbon_index, topology.post_indices, "postsynaptic MBON")
    kc_active = kc > limits.silence_hz
    kc_ever = kc_active.any(axis=0)
    mbon_ever = (mbon > limits.silence_hz).any(axis=0)

    ever_eligible = np.zeros(topology.edge_count, dtype=np.bool_)
    eligibility_mass = np.zeros(topology.edge_count, dtype=np.float64)
    for start in range(0, samples, max(chunk_size, 1)):
        stop = min(start + max(chunk_size, 1), samples)
        pre = np.clip(kc[start:stop][:, edge_pre] / plasticity.kc_reference_hz, 0.0, 1.0)
        post = np.clip(
            mbon[start:stop][:, edge_post] / plasticity.mbon_reference_hz, 0.0, 1.0
        )
        coincidence = pre * post
        ever_eligible |= (coincidence > 0).any(axis=0)
        eligibility_mass += coincidence.sum(axis=0)

    edge_types = types[edge_pre]
    by_subtype: dict[str, Any] = {}
    total_mass = float(eligibility_mass.sum())
    for subtype in sorted(set(types.tolist())):
        columns = np.flatnonzero(types == subtype)
        edges = np.flatnonzero(edge_types == subtype)
        block = kc[:, columns]
        by_subtype[subtype] = {
            "kc_neurons": int(len(columns)),
            "fraction_active_samples": float(np.mean(block > limits.silence_hz)),
            "fraction_ever_active": float(np.mean(kc_ever[columns])),
            "mean_hz": float(block.mean()),
            "median_hz": float(np.median(block)),
            "p95_hz": float(np.quantile(block, 0.95)) if block.size else 0.0,
            "max_hz": float(block.max()) if block.size else 0.0,
            "plastic_edges": int(len(edges)),
            "plastic_edges_with_active_presynaptic_kc": int(
                np.count_nonzero(kc_ever[edge_pre[edges]])
            ),
            "plastic_edges_ever_eligible": int(np.count_nonzero(ever_eligible[edges])),
            "eligibility_mass": float(eligibility_mass[edges].sum()),
            "eligibility_mass_fraction": (
                float(eligibility_mass[edges].sum() / total_mass) if total_mass > 0 else 0.0
            ),
        }

    motor_coverage: dict[str, Any] = {"evaluated": False}
    if motor_output_indices is not None:
        outputs = np.asarray(motor_output_indices, dtype=np.int64)
        if motor_activity is not None:
            active = (np.asarray(motor_activity, dtype=np.float64) > limits.silence_hz).any(axis=0)
        else:
            lookup = {int(value): position for position, value in enumerate(mbon_index)}
            active = np.asarray(
                [
                    bool(mbon_ever[lookup[int(value)]]) if int(value) in lookup else False
                    for value in outputs
                ]
            )
        motor_coverage = {
            "evaluated": True,
            "motor_outputs": int(len(outputs)),
            "fraction_motor_outputs_active": float(active.mean()) if active.size else 0.0,
        }

    reachable_edge_fraction = float(np.mean(kc_ever[edge_pre]))
    eligible_edge_fraction = float(np.mean(ever_eligible))
    checks = {
        "kc_population_active": float(kc_ever.mean()) >= limits.minimum_active_kc_fraction,
        "plastic_edges_reachable": reachable_edge_fraction
        >= limits.minimum_reachable_edge_fraction,
        "plastic_edges_eligible": eligible_edge_fraction
        >= limits.minimum_reachable_edge_fraction,
        "plastic_mbons_active": float(mbon_ever.mean())
        >= limits.minimum_plastic_mbon_active_fraction,
        "motor_outputs_active": (
            not motor_coverage["evaluated"]
            or motor_coverage["fraction_motor_outputs_active"]
            >= limits.minimum_motor_output_active_fraction
        ),
    }
    return {
        "version": REACHABILITY_REPORT_VERSION,
        "samples": int(samples),
        "thresholds": {
            "silence_hz": limits.silence_hz,
            "minimum_reachable_edge_fraction": limits.minimum_reachable_edge_fraction,
            "minimum_active_kc_fraction": limits.minimum_active_kc_fraction,
            "minimum_plastic_mbon_active_fraction": limits.minimum_plastic_mbon_active_fraction,
            "minimum_motor_output_active_fraction": limits.minimum_motor_output_active_fraction,
        },
        "plastic_topology_sha256": topology.sha256,
        "kc_population": {
            "neurons": int(len(kc_index)),
            "fraction_ever_active": float(kc_ever.mean()),
            "fraction_active_samples": float(kc_active.mean()),
            "mean_hz": float(kc.mean()),
            "median_hz": float(np.median(kc)),
            "p95_hz": float(np.quantile(kc, 0.95)),
            "subtypes": int(len(by_subtype)),
        },
        "plastic_edges": {
            "edges": int(topology.edge_count),
            "fraction_with_ever_active_presynaptic_kc": reachable_edge_fraction,
            "fraction_ever_eligible": eligible_edge_fraction,
            "total_eligibility_mass": total_mass,
        },
        "mbon_coverage": {
            "plastic_mbons": int(len(mbon_index)),
            "fraction_plastic_mbons_active": float(mbon_ever.mean()),
        },
        "motor_coverage": motor_coverage,
        "by_kc_subtype": by_subtype,
        "interpretation": (
            "a low reachable or eligible fraction means V1's ALPN-only drive and "
            "decision window do not exercise most of the plastic population; that "
            "is a model decision for the experimenter, not an automatic change"
        ),
        "gates": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "failed": [name for name, passed in checks.items() if not passed],
        },
    }


def _positions(
    haystack: NDArray[np.int64], needles: NDArray[np.int64], label: str
) -> NDArray[np.int64]:
    positions = np.searchsorted(haystack, needles)
    if positions.size and (
        positions.max(initial=0) >= len(haystack)
        or not np.array_equal(haystack[np.clip(positions, 0, len(haystack) - 1)], needles)
    ):
        raise ValueError(f"{label} index is missing from the recorded activity")
    return positions.astype(np.int64)


def merge_identity(report: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(report), "evidence_identity": dict(identity)}
