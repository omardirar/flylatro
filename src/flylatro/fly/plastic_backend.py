"""Plastic KC->MBON execution over the otherwise fixed sparse FlyWire graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.encoder import Stimulus
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.fly.torch_backend import (
    ShiuLIFParameters,
    TorchFlyActivity,
    TorchFlyWireBackend,
)


class PlasticTorchFlyWireBackend(TorchFlyWireBackend):
    """Torch LIF backend whose only mutable weights are known KC->MBON pairs.

    Fixed recurrence is one shared sparse matrix.  The plastic contribution is
    gathered and scatter-added per batch row, so independent efficacy vectors
    run in one simulation without duplicating the 3.7M-edge fixed graph.
    """

    backend_version = "flywire-shiu-torch-plastic-kc-mbon-v1"

    def __init__(
        self,
        artifact: FlyWireArtifact,
        topology: PlasticEdgeTopology,
        *,
        device: str = "cpu",
        parameters: ShiuLIFParameters | None = None,
        readout_indices: Sequence[int] | None = None,
        record_events: bool = False,
        shuffle_seed: int | None = None,
        shuffle_scope: str = "whole_brain",
    ) -> None:
        super().__init__(
            artifact,
            device=device,
            parameters=parameters,
            readout_indices=readout_indices,
            record_events=record_events,
            shuffle_seed=shuffle_seed,
            shuffle_preserve_populations=shuffle_seed is not None,
            shuffle_scope=shuffle_scope,
        )
        self.plastic_topology = topology
        torch = self.torch
        coalesced = self.weights.coalesce()
        coo = coalesced.indices()
        keys = coo[0] * self.neuron_count + coo[1]
        plastic_keys = torch.as_tensor(
            topology.post_indices * self.neuron_count + topology.pre_indices,
            dtype=torch.int64,
            device=self.device,
        )
        positions = torch.searchsorted(keys, plastic_keys)
        if bool((positions >= len(keys)).any()) or not torch.equal(
            keys.index_select(0, positions), plastic_keys
        ):
            raise ValueError("plastic topology is not a subset of backend edges")
        self._plastic_positions = positions
        self._plastic_anatomical = torch.as_tensor(
            topology.anatomical_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self._plastic_pre = torch.as_tensor(
            topology.pre_indices, dtype=torch.int64, device=self.device
        )
        self._plastic_post = torch.as_tensor(
            topology.post_indices, dtype=torch.int64, device=self.device
        )
        with torch.no_grad():
            self.weights.values().index_fill_(0, self._plastic_positions, 0.0)
        self._efficacy = torch.ones(
            (1, topology.edge_count), dtype=torch.float32, device=self.device
        )

    @property
    def efficacy(self) -> NDArray[np.float32]:
        return self._efficacy.detach().cpu().numpy().copy()

    def set_plastic_efficacy(self, efficacy: NDArray[np.floating]) -> None:
        torch = self.torch
        if hasattr(efficacy, "detach"):
            tensor = efficacy.detach().to(device=self.device, dtype=self.torch.float32)
        else:
            tensor = self.torch.as_tensor(
                np.asarray(efficacy, dtype=np.float32),
                dtype=self.torch.float32,
                device=self.device,
            )
        if tensor.ndim == 1:
            tensor = tensor[None, :]
        if tensor.ndim != 2 or tensor.shape[1] != self.plastic_topology.edge_count:
            raise ValueError("efficacy batch does not match plastic topology")
        if not bool(self.torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
            raise ValueError("efficacy must be finite and non-negative")
        with torch.no_grad():
            self._efficacy = tensor.clone()

    def _recurrent(self, spikes: object) -> object:
        torch = self.torch
        fixed = torch.sparse.mm(self.weights, spikes.T).T
        if self._efficacy.shape[0] != spikes.shape[0]:
            if self._efficacy.shape[0] == 1:
                efficacy = self._efficacy.expand(spikes.shape[0], -1)
            else:
                raise ValueError("plastic efficacy batch differs from neural batch")
        else:
            efficacy = self._efficacy
        contribution = (
            spikes.index_select(1, self._plastic_pre)
            * efficacy
            * self._plastic_anatomical[None, :]
        )
        plastic = torch.zeros_like(fixed)
        plastic.scatter_add_(
            1, self._plastic_post[None, :].expand(spikes.shape[0], -1), contribution
        )
        return fixed + plastic

    def propagate_once(
        self, activity: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """One sparse linear propagation step for diagnostics/tests."""

        values = np.asarray(activity, dtype=np.float32)
        one_row = values.ndim == 1
        if one_row:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.neuron_count:
            raise ValueError("activity has the wrong neuron dimension")
        tensor = self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device)
        result = self._recurrent(tensor)
        array = result.detach().cpu().numpy().astype(np.float32, copy=False)
        return array[0] if one_row else array


@dataclass(frozen=True, slots=True)
class PlasticFlyDecisionActivity:
    output_activity: NDArray[np.float32]
    edge_pre_activity: object
    edge_post_activity: object
    kc_activity: NDArray[np.float32]
    mbon_activity: NDArray[np.float32]
    dan_activity: NDArray[np.float32]
    descending_activity: NDArray[np.float32]
    duration_ms: float
    activity_unit: str
    event_batch: NDArray[np.int64] | None = None
    event_neuron_ids: NDArray[np.int64] | None = None
    event_times_ms: NDArray[np.float32] | None = None
    stimulation_batch: NDArray[np.int64] | None = None
    stimulation_neuron_ids: NDArray[np.int64] | None = None
    stimulation_rates_hz: NDArray[np.float32] | None = None


class PlasticFlyProcessor:
    """Run independent plastic flies and expose edge-local learning activity."""

    version = "plastic-fly-processor-v1"

    def __init__(
        self,
        encoder: object,
        backend: PlasticTorchFlyWireBackend,
        topology: PlasticEdgeTopology,
        *,
        output_indices: NDArray[np.int64],
        mode: str,
        duration_ms: float,
        reset_fast_state_each_decision: bool = True,
    ) -> None:
        if mode not in {"mbon_direct", "whole_brain"}:
            raise ValueError("unknown plastic fly output mode")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if not reset_fast_state_each_decision:
            raise NotImplementedError(
                "persistent fast neural state requires per-learner backend state"
            )
        self.encoder = encoder
        self.backend = backend
        self.topology = topology
        self.output_indices = np.asarray(output_indices, dtype=np.int64)
        self.output_root_ids = backend.artifact.root_ids[self.output_indices].copy()
        self.mode = mode
        self.duration_ms = duration_ms
        self.reset_fast_state_each_decision = reset_fast_state_each_decision
        required = np.unique(
            np.concatenate(
                (
                    topology.pre_indices,
                    topology.post_indices,
                    self.output_indices,
                    backend.artifact.descending_indices,
                    backend.artifact.dan_indices,
                )
            )
        )
        if not np.array_equal(required, backend.readout_indices):
            raise ValueError(
                "backend readout must be the sorted union of KC, MBON, motor and descending indices"
            )
        self._readout_position = np.full(backend.neuron_count, -1, dtype=np.int64)
        self._readout_position[required] = np.arange(len(required), dtype=np.int64)
        self._readout_position_device = backend.torch.as_tensor(
            self._readout_position,
            dtype=backend.torch.int64,
            device=backend.device,
        )
        self._kc_indices = np.unique(topology.pre_indices)
        self._mbon_indices = np.unique(topology.post_indices)
        self._dan_indices = backend.artifact.dan_indices
        self.kc_root_ids = backend.artifact.root_ids[self._kc_indices].copy()
        self.mbon_root_ids = backend.artifact.root_ids[self._mbon_indices].copy()
        self.dan_root_ids = backend.artifact.root_ids[self._dan_indices].copy()
        self.descending_root_ids = backend.artifact.root_ids[
            backend.artifact.descending_indices
        ].copy()
        self.pam_root_ids = backend.artifact.root_ids[backend.artifact.pam_indices].copy()
        self.ppl1_root_ids = backend.artifact.root_ids[backend.artifact.ppl1_indices].copy()

    @staticmethod
    def required_readout_indices(
        artifact: FlyWireArtifact,
        topology: PlasticEdgeTopology,
        output_indices: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        return np.unique(
            np.concatenate(
                (
                    topology.pre_indices,
                    topology.post_indices,
                    np.asarray(output_indices, dtype=np.int64),
                    artifact.descending_indices,
                    artifact.dan_indices,
                )
            )
        )

    def process(
        self,
        observations: dict[str, NDArray[np.generic]],
        *,
        fly_seeds: Sequence[int],
        efficacy: NDArray[np.floating],
    ) -> PlasticFlyDecisionActivity:
        batch = next(iter(observations.values())).shape[0]
        efficacy_values = efficacy
        if tuple(efficacy_values.shape) != (batch, self.topology.edge_count):
            raise ValueError("one efficacy vector is required per independent fly")
        if len(fly_seeds) != batch:
            raise ValueError("one fly seed is required per independent fly")
        stimulus: Stimulus = self.encoder.encode(observations)
        self.backend.set_plastic_efficacy(efficacy_values)
        self.backend.reset(batch, tuple(int(seed) for seed in fly_seeds))
        activity: TorchFlyActivity = self.backend.simulate_tensor(stimulus, self.duration_ms)
        stimulation_batch, stimulation_indices = np.nonzero(stimulus.rates_hz > 0)
        rates_device = activity.spike_counts.to(dtype=self.backend.torch.float32)
        rates_device /= self.duration_ms / 1000.0
        rates = rates_device.detach().cpu().numpy()
        def selected(indices: NDArray[np.int64]) -> NDArray[np.float32]:
            return rates[:, self._readout_position[indices]]
        def selected_device(indices: NDArray[np.int64]) -> object:
            positions = self._readout_position_device.index_select(
                0,
                self.backend.torch.as_tensor(
                    indices, dtype=self.backend.torch.int64, device=self.backend.device
                ),
            )
            return rates_device.index_select(1, positions)
        return PlasticFlyDecisionActivity(
            output_activity=selected(self.output_indices),
            edge_pre_activity=selected_device(self.topology.pre_indices),
            edge_post_activity=selected_device(self.topology.post_indices),
            kc_activity=selected(self._kc_indices),
            mbon_activity=selected(self._mbon_indices),
            dan_activity=selected(self._dan_indices),
            descending_activity=selected(self.backend.artifact.descending_indices),
            duration_ms=self.duration_ms,
            activity_unit="spikes_per_second_hz",
            event_batch=(activity.event_batch.detach().cpu().numpy().astype(np.int64) if activity.event_batch is not None else None),
            event_neuron_ids=(activity.event_neuron_ids.detach().cpu().numpy().astype(np.int64) if activity.event_neuron_ids is not None else None),
            event_times_ms=(activity.event_times_ms.detach().cpu().numpy().astype(np.float32) if activity.event_times_ms is not None else None),
            stimulation_batch=stimulation_batch.astype(np.int64, copy=False),
            stimulation_neuron_ids=self.backend.artifact.root_ids[stimulation_indices],
            stimulation_rates_hz=stimulus.rates_hz[stimulation_batch, stimulation_indices].astype(np.float32, copy=False),
        )
