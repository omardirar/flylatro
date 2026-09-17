"""Versioned extraction of small fly representations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.backend import FlyActivity


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    version: str
    readout_neuron_ids: tuple[int, ...]
    rate_scale_hz: float = 200.0
    include_final_voltage: bool = True

    def __post_init__(self) -> None:
        if not self.readout_neuron_ids:
            raise ValueError("at least one readout neuron is required")
        if len(set(self.readout_neuron_ids)) != len(self.readout_neuron_ids):
            raise ValueError("readout neuron IDs must be unique")
        if min(self.readout_neuron_ids) < 0:
            raise ValueError("readout neuron IDs must be non-negative")
        if self.rate_scale_hz <= 0:
            raise ValueError("rate_scale_hz must be positive")

    @property
    def output_size(self) -> int:
        multiplier = 2 if self.include_final_voltage else 1
        return len(self.readout_neuron_ids) * multiplier

    @property
    def sha256(self) -> str:
        value = json.dumps(
            {
                "version": self.version,
                "readout_neuron_ids": list(self.readout_neuron_ids),
                "rate_scale_hz": self.rate_scale_hz,
                "include_final_voltage": self.include_final_voltage,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(value).hexdigest()


class RateFeatureExtractor:
    """Extracts bounded firing rates and optional terminal voltage."""

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec

    def extract(self, activity: FlyActivity) -> FloatArray:
        if activity.neuron_ids is None:
            maximum = activity.spike_counts.shape[1]
            if max(self.spec.readout_neuron_ids) >= maximum:
                raise ValueError("readout neuron ID is outside the fly activity")
            ids = self.spec.readout_neuron_ids
        else:
            positions = {
                int(neuron_id): index
                for index, neuron_id in enumerate(activity.neuron_ids)
            }
            missing = [
                neuron_id
                for neuron_id in self.spec.readout_neuron_ids
                if neuron_id not in positions
            ]
            if missing:
                raise ValueError(f"readout neuron IDs missing from activity: {missing[:3]}")
            ids = tuple(positions[neuron_id] for neuron_id in self.spec.readout_neuron_ids)
        duration_seconds = activity.duration_ms / 1_000.0
        firing_hz = activity.spike_counts[:, ids] / duration_seconds
        rate_features = np.clip(firing_hz / self.spec.rate_scale_hz, 0.0, 1.0)
        if not self.spec.include_final_voltage:
            return rate_features.astype(np.float64, copy=False)
        voltage = np.clip(activity.final_voltage[:, ids], -1.0, 1.0)
        return np.concatenate((rate_features, voltage), axis=1).astype(
            np.float64, copy=False
        )
