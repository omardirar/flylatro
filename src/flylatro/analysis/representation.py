"""Cheap population diagnostics for naive and trained fly responses."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def representation_diagnostics(
    kc_activity: NDArray[np.floating],
    mbon_activity: NDArray[np.floating],
    descending_activity: NDArray[np.floating],
    *,
    state_labels: NDArray[np.integer] | None = None,
) -> dict[str, Any]:
    kc = _matrix(kc_activity, "KC")
    mbon = _matrix(mbon_activity, "MBON")
    descending = _matrix(descending_activity, "descending")
    result: dict[str, Any] = {
        "samples": kc.shape[0],
        "kc_active_fraction": float(np.mean(kc > 0)),
        "kc_mean_activity": float(kc.mean()),
        "mbon_mean_activity": float(mbon.mean()),
        "mbon_silent_fraction": float(np.mean(np.all(mbon == 0, axis=0))),
        "mbon_saturated_fraction": float(
            np.mean(np.all(mbon >= 0.99, axis=0))
        ),
        "descending_mean_activity": float(descending.mean()),
        "descending_silent_fraction": float(
            np.mean(np.all(descending == 0, axis=0))
        ),
        "same_state_variability": None,
        "different_state_separability": None,
    }
    if state_labels is not None:
        labels = np.asarray(state_labels)
        if labels.shape != (kc.shape[0],):
            raise ValueError("state_labels must identify every sample")
        same_distances = []
        different_distances = []
        for left in range(len(mbon)):
            for right in range(left + 1, len(mbon)):
                distance = float(np.linalg.norm(mbon[left] - mbon[right]))
                target = same_distances if labels[left] == labels[right] else different_distances
                target.append(distance)
        result["same_state_variability"] = (
            float(np.mean(same_distances)) if same_distances else None
        )
        result["different_state_separability"] = (
            float(np.mean(different_distances)) if different_distances else None
        )
    return result


def _matrix(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError(f"{name} activity must be a finite sample-by-neuron matrix")
    return array
