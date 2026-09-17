"""Fly-simulation abstraction and a tiny deterministic LIF-like backend."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.encoder import Stimulus


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FlyActivity:
    spike_counts: IntArray
    final_voltage: FloatArray
    duration_ms: float
    backend_version: str
    neuron_ids: IntArray | None = None

    def __post_init__(self) -> None:
        if self.spike_counts.ndim != 2:
            raise ValueError("spike counts must have shape [batch, neurons]")
        if self.final_voltage.shape != self.spike_counts.shape:
            raise ValueError("voltage and spike arrays must have matching shapes")
        if self.neuron_ids is not None and self.neuron_ids.shape != (
            self.spike_counts.shape[1],
        ):
            raise ValueError("neuron_ids must identify every activity column")


class FlyBackend(Protocol):
    neuron_count: int
    backend_version: str

    def reset(self, batch_size: int, seeds: tuple[int, ...] | None = None) -> None: ...

    def simulate(self, stimulus: Stimulus, duration_ms: float) -> FlyActivity: ...


class TinyGraphFlyBackend:
    """Cheap fixed graph for development; not a scientific fly model.

    The graph and dynamics are immutable after construction.  A batch shares
    one connectivity matrix, matching the intended execution model of the
    future adult-connectome backend.
    """

    backend_version = "tiny-lif-v1"

    def __init__(
        self,
        neuron_count: int = 96,
        *,
        graph_seed: int = 1701,
        edge_probability: float = 0.08,
        dt_ms: float = 1.0,
        leak: float = 0.82,
        threshold: float = 1.0,
        input_gain: float = 0.38,
        recurrent_gain: float = 0.55,
    ) -> None:
        if neuron_count < 2:
            raise ValueError("neuron_count must be at least 2")
        if not 0 < edge_probability <= 1:
            raise ValueError("edge_probability must be in (0, 1]")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        self.neuron_count = neuron_count
        self.graph_seed = graph_seed
        self.dt_ms = dt_ms
        self.leak = leak
        self.threshold = threshold
        self.input_gain = input_gain
        self.recurrent_gain = recurrent_gain
        self._weights = self._build_graph(edge_probability)
        canonical_weights = self._weights.astype("<f8", copy=False).tobytes(order="C")
        self.connectivity_hash = hashlib.sha256(canonical_weights).hexdigest()
        self._voltage: FloatArray | None = None

    @property
    def connectivity(self) -> FloatArray:
        view = self._weights.view()
        view.setflags(write=False)
        return view

    def reset(self, batch_size: int, seeds: tuple[int, ...] | None = None) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if seeds is not None and len(seeds) != batch_size:
            raise ValueError("one fly seed is required per batch row")
        self._voltage = np.zeros((batch_size, self.neuron_count), dtype=np.float64)

    def simulate(self, stimulus: Stimulus, duration_ms: float) -> FlyActivity:
        if self._voltage is None:
            raise RuntimeError("reset must be called before simulate")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if stimulus.rates_hz.shape != self._voltage.shape:
            raise ValueError(
                "stimulus shape must match the reset batch size and neuron count"
            )
        steps = max(1, int(np.ceil(duration_ms / self.dt_ms)))
        spike_counts = np.zeros_like(self._voltage, dtype=np.int64)
        prior_spikes = np.zeros_like(self._voltage)
        external_drive = np.clip(stimulus.rates_hz / 100.0, 0.0, 1.0)
        for _ in range(steps):
            recurrent = prior_spikes @ self._weights.T
            self._voltage = (
                self.leak * self._voltage
                + self.input_gain * external_drive
                + self.recurrent_gain * recurrent
            )
            spikes = self._voltage >= self.threshold
            spike_counts += spikes
            self._voltage = np.where(spikes, self._voltage - self.threshold, self._voltage)
            prior_spikes = spikes.astype(np.float64)
        return FlyActivity(
            spike_counts=spike_counts,
            final_voltage=self._voltage.copy(),
            duration_ms=duration_ms,
            backend_version=self.backend_version,
        )

    def _build_graph(self, edge_probability: float) -> FloatArray:
        rng = np.random.default_rng(self.graph_seed)
        adjacency = rng.random((self.neuron_count, self.neuron_count)) < edge_probability
        np.fill_diagonal(adjacency, False)
        neuron_sign = np.where(
            rng.random(self.neuron_count) < 0.8, 1.0, -1.0
        )  # fixed excitatory/inhibitory identity by presynaptic neuron
        magnitudes = rng.uniform(0.05, 0.25, size=adjacency.shape)
        weights = adjacency * magnitudes * neuron_sign[np.newaxis, :]
        incoming_scale = max(1.0, edge_probability * self.neuron_count)
        weights = weights / np.sqrt(incoming_scale)
        weights.setflags(write=False)
        return weights
