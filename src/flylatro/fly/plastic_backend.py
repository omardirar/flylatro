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

    A backend instance executes one efficacy vector at a time. Independent
    learners share the artifact/topology at the orchestrator level and are
    evaluated sequentially or by separate backend instances; weights are never
    mixed across learners.
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
    ) -> None:
        super().__init__(
            artifact,
            device=device,
            parameters=parameters,
            readout_indices=readout_indices,
            record_events=record_events,
            shuffle_seed=shuffle_seed,
            shuffle_preserve_populations=shuffle_seed is not None,
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
        self._efficacy = np.ones(topology.edge_count, dtype=np.float32)

    @property
    def efficacy(self) -> NDArray[np.float32]:
        return self._efficacy.copy()

    def set_plastic_efficacy(self, efficacy: NDArray[np.floating]) -> None:
        values = np.asarray(efficacy, dtype=np.float32)
        if values.shape != (self.plastic_topology.edge_count,):
            raise ValueError("efficacy vector does not match plastic topology")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("efficacy must be finite and non-negative")
        torch = self.torch
        effective = self._plastic_anatomical * torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        )
        # The COO structure and every fixed value remain shared. Updating the
        # existing coalesced value buffer touches only identified plastic
        # positions instead of copying/re-coalescing all ~3.7M whole-brain
        # connections for every decision.
        with torch.no_grad():
            self.weights.values().index_copy_(
                0, self._plastic_positions, effective
            )
        self._efficacy = values.copy()

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
        result = self.torch.sparse.mm(self.weights, tensor.T).T
        array = result.detach().cpu().numpy().astype(np.float32, copy=False)
        return array[0] if one_row else array


@dataclass(frozen=True, slots=True)
class PlasticFlyDecisionActivity:
    output_activity: NDArray[np.float32]
    edge_pre_activity: NDArray[np.float32]
    edge_post_activity: NDArray[np.float32]
    kc_activity: NDArray[np.float32]
    mbon_activity: NDArray[np.float32]
    dan_activity: NDArray[np.float32]
    descending_activity: NDArray[np.float32]
    duration_ms: float


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
        efficacy_values = np.asarray(efficacy, dtype=np.float32)
        if efficacy_values.shape != (batch, self.topology.edge_count):
            raise ValueError("one efficacy vector is required per independent fly")
        if len(fly_seeds) != batch:
            raise ValueError("one fly seed is required per independent fly")
        output_rows = []
        edge_pre_rows = []
        edge_post_rows = []
        kc_rows = []
        mbon_rows = []
        descending_rows = []
        dan_rows = []
        for row in range(batch):
            one = {key: value[row : row + 1] for key, value in observations.items()}
            stimulus: Stimulus = self.encoder.encode(one)
            self.backend.set_plastic_efficacy(efficacy_values[row])
            self.backend.reset(1, (int(fly_seeds[row]),))
            activity: TorchFlyActivity = self.backend.simulate_tensor(
                stimulus, self.duration_ms
            )
            rates = (
                activity.spike_counts.detach().cpu().numpy().astype(np.float32)
                / (self.duration_ms / 1000.0)
            )[0]
            output_rows.append(rates[self._readout_position[self.output_indices]])
            edge_pre_rows.append(rates[self._readout_position[self.topology.pre_indices]])
            edge_post_rows.append(rates[self._readout_position[self.topology.post_indices]])
            kc_rows.append(rates[self._readout_position[self._kc_indices]])
            mbon_rows.append(rates[self._readout_position[self._mbon_indices]])
            dan_rows.append(rates[self._readout_position[self._dan_indices]])
            descending_rows.append(
                rates[self._readout_position[self.backend.artifact.descending_indices]]
            )
        return PlasticFlyDecisionActivity(
            output_activity=np.stack(output_rows),
            edge_pre_activity=np.stack(edge_pre_rows),
            edge_post_activity=np.stack(edge_post_rows),
            kc_activity=np.stack(kc_rows),
            mbon_activity=np.stack(mbon_rows),
            dan_activity=np.stack(dan_rows),
            descending_activity=np.stack(descending_rows),
            duration_ms=self.duration_ms,
        )
