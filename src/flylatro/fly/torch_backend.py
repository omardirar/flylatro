"""Sparse batched PyTorch implementation of the fixed Shiu-style fly model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from flylatro.fly.backend import FlyActivity
from flylatro.fly.encoder import Stimulus
from flylatro.fly.flywire_artifact import FlyWireArtifact, connectivity_sha256


@dataclass(frozen=True, slots=True)
class ShiuLIFParameters:
    dt_ms: float = 0.1
    synaptic_tau_ms: float = 5.0
    synaptic_delay_ms: float = 1.8
    resting_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.2
    poisson_scale: float = 250.0
    synapse_scale_mv: float = 0.275


@dataclass(frozen=True, slots=True)
class TorchFlyActivity:
    spike_counts: Any
    final_voltage: Any
    duration_ms: float
    neuron_ids: Any
    event_batch: Any | None = None
    event_neuron_ids: Any | None = None
    event_times_ms: Any | None = None


class TorchFlyWireBackend:
    """Fixed real-connectome backend with independent batched fly states.

    The backend only returns the configured descending-neuron readout by
    default. Full spike events are opt-in for selected evaluation episodes.
    No parameter is registered as trainable.
    """

    backend_version = "flywire-shiu-torch-v1"

    def __init__(
        self,
        artifact: FlyWireArtifact,
        *,
        device: str = "cpu",
        parameters: ShiuLIFParameters | None = None,
        readout_indices: Sequence[int] | None = None,
        shuffle_seed: int | None = None,
        record_events: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "PyTorch is required for TorchFlyWireBackend; install the "
                "'training' extra"
            ) from error
        self.torch = torch
        self.artifact = artifact
        self.neuron_count = artifact.neuron_count
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.parameters = parameters or ShiuLIFParameters()
        self.record_events = record_events
        selected = np.asarray(
            artifact.descending_indices if readout_indices is None else readout_indices,
            dtype=np.int64,
        )
        if not selected.size:
            raise ValueError("at least one readout neuron is required")
        self.readout_indices = selected
        self.readout_root_ids = artifact.root_ids[selected]
        pre, post, weights, procedure = artifact.edge_arrays(shuffle_seed=shuffle_seed)
        sparse_indices = torch.from_numpy(np.stack((post, pre))).to(
            device=self.device, dtype=torch.int64
        )
        values = torch.from_numpy(weights).to(device=self.device, dtype=torch.float32)
        self.weights = torch.sparse_coo_tensor(
            sparse_indices,
            values,
            (self.neuron_count, self.neuron_count),
            device=self.device,
            check_invariants=False,
        ).coalesce()
        self.topology_procedure = procedure
        coalesced = self.weights.coalesce()
        coo = coalesced.indices().detach().cpu().numpy()
        vals = coalesced.values().detach().cpu().numpy()
        self.connectivity_hash = connectivity_sha256(coo[1], coo[0], vals)
        self._state: tuple[Any, ...] | None = None
        self._generators: list[Any] = []

    @property
    def dynamics_hash(self) -> str:
        payload = {
            "backend_version": self.backend_version,
            "parameters": asdict(self.parameters),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def reset(self, batch_size: int, seeds: tuple[int, ...] | None = None) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if seeds is None:
            seeds = tuple(range(batch_size))
        if len(seeds) != batch_size:
            raise ValueError("one fly seed is required per batch row")
        torch = self.torch
        p = self.parameters
        shape = (batch_size, self.neuron_count)
        voltage = torch.full(shape, p.resting_mv, dtype=torch.float32, device=self.device)
        conductance = torch.zeros(shape, dtype=torch.float32, device=self.device)
        spikes = torch.zeros(shape, dtype=torch.float32, device=self.device)
        refractory = torch.full(
            shape,
            int(round(p.refractory_ms / p.dt_ms)),
            dtype=torch.int16,
            device=self.device,
        )
        delay_steps = int(round(p.synaptic_delay_ms / p.dt_ms))
        delay = torch.zeros(
            (delay_steps + 1, *shape), dtype=torch.float32, device=self.device
        )
        counts = torch.zeros(shape, dtype=torch.int32, device=self.device)
        self._state = (conductance, delay, spikes, voltage, refractory, counts)
        self._generators = []
        for seed in seeds:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            self._generators.append(generator)

    def simulate_tensor(self, stimulus: Stimulus, duration_ms: float) -> TorchFlyActivity:
        if self._state is None:
            raise RuntimeError("reset must be called before simulate")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        torch = self.torch
        conductance, delay, spikes, voltage, refractory, counts = self._state
        if stimulus.rates_hz.shape != tuple(voltage.shape):
            raise ValueError("stimulus shape does not match fly batch state")
        rates = torch.as_tensor(
            np.asarray(stimulus.rates_hz), dtype=torch.float32, device=self.device
        )
        p = self.parameters
        step_count = int(round(duration_ms / p.dt_ms))
        if step_count < 1:
            raise ValueError("duration is shorter than one simulation step")
        delay_slot = 0
        event_batch: list[Any] = []
        event_neuron: list[Any] = []
        event_time: list[Any] = []
        refractory_steps = int(round(p.refractory_ms / p.dt_ms))

        with torch.no_grad():
            for step in range(step_count):
                poisson_rows = [
                    (
                        torch.rand(
                            self.neuron_count,
                            device=self.device,
                            generator=generator,
                        )
                        < (rates[row] * p.dt_ms / 1000.0).clamp(0.0, 1.0)
                    ).to(torch.float32)
                    for row, generator in enumerate(self._generators)
                ]
                poisson = torch.stack(poisson_rows) * p.poisson_scale
                recurrent = torch.sparse.mm(self.weights, spikes.T).T
                incoming = p.synapse_scale_mv * (poisson + recurrent)

                refractory = refractory * (1 - spikes.to(torch.int16)) + 1
                can_integrate = (refractory > refractory_steps).to(torch.float32)
                conductance_new = (
                    conductance * (1.0 - p.dt_ms / p.synaptic_tau_ms)
                    + delay[delay_slot] * can_integrate
                )
                target_slot = (delay_slot + delay.shape[0] - 1) % delay.shape[0]
                delay[target_slot] = incoming
                delay_slot = (delay_slot + 1) % delay.shape[0]

                voltage = voltage + (p.dt_ms / p.membrane_tau_ms) * (
                    conductance - (voltage - p.resting_mv)
                )
                spikes = (voltage > p.threshold_mv).to(torch.float32)
                voltage = torch.where(
                    spikes.bool(), torch.full_like(voltage, p.reset_mv), voltage
                )
                conductance = torch.where(
                    spikes.bool(), torch.zeros_like(conductance_new), conductance_new
                )
                counts += spikes.to(torch.int32)

                if self.record_events and bool(spikes.any()):
                    batch_ids, neuron_indices = spikes.nonzero(as_tuple=True)
                    event_batch.append(batch_ids)
                    event_neuron.append(neuron_indices)
                    event_time.append(
                        torch.full_like(
                            batch_ids,
                            fill_value=step * p.dt_ms,
                            dtype=torch.float32,
                        )
                    )

        self._state = (conductance, delay, spikes, voltage, refractory, counts)
        readout = torch.as_tensor(
            self.readout_indices, dtype=torch.int64, device=self.device
        )
        root_ids = torch.as_tensor(
            self.readout_root_ids.astype(np.int64),
            dtype=torch.int64,
            device=self.device,
        )
        return TorchFlyActivity(
            spike_counts=counts.index_select(1, readout),
            final_voltage=voltage.index_select(1, readout),
            duration_ms=duration_ms,
            neuron_ids=root_ids,
            event_batch=torch.cat(event_batch) if event_batch else None,
            event_neuron_ids=(
                self._event_root_ids(torch.cat(event_neuron)) if event_neuron else None
            ),
            event_times_ms=torch.cat(event_time) if event_time else None,
        )

    def simulate(self, stimulus: Stimulus, duration_ms: float) -> FlyActivity:
        activity = self.simulate_tensor(stimulus, duration_ms)
        return FlyActivity(
            spike_counts=activity.spike_counts.detach().cpu().numpy().astype(np.int64),
            final_voltage=activity.final_voltage.detach().cpu().numpy().astype(np.float64),
            duration_ms=duration_ms,
            backend_version=self.backend_version,
            neuron_ids=activity.neuron_ids.detach().cpu().numpy().astype(np.int64),
        )

    def _event_root_ids(self, neuron_indices: Any) -> Any:
        roots = self.torch.as_tensor(
            self.artifact.root_ids.astype(np.int64),
            dtype=self.torch.int64,
            device=self.device,
        )
        return roots.index_select(0, neuron_indices)
