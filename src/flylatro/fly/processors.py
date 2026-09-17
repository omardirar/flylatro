"""Fixed processors used interchangeably by training and controls."""

from __future__ import annotations

from typing import Protocol, Sequence
import hashlib
import json

import numpy as np
import torch
from torch import nn

from flylatro.env.upstream_contract import ObsDict
from flylatro.fly.torch_backend import TorchFlyWireBackend
from flylatro.fly.upstream_encoder import (
    FixedUpstreamBalatroEncoder,
    observation_features,
)


class ObservationProcessor(Protocol):
    output_size: int
    version: str

    def process(
        self, observations: ObsDict, *, fly_seeds: Sequence[int]
    ) -> torch.Tensor: ...


class FixedReservoirProcessor(nn.Module):
    """Tiny fixed reservoir for PPO smoke tests; it has zero trainable params."""

    version = "fixed-synthetic-reservoir-v1"

    def __init__(
        self,
        input_size: int,
        *,
        reservoir_size: int = 64,
        output_size: int = 32,
        steps: int = 3,
        seed: int = 19,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if not 1 <= output_size <= reservoir_size:
            raise ValueError("invalid reservoir output size")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        input_weights = torch.randn(
            input_size, reservoir_size, generator=generator
        ) / np.sqrt(input_size)
        recurrent = torch.randn(
            reservoir_size, reservoir_size, generator=generator
        ) / np.sqrt(reservoir_size)
        self.register_buffer("input_weights", input_weights.to(device))
        self.register_buffer("recurrent", recurrent.to(device))
        self.output_size = output_size
        self.steps = steps
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def process(
        self, observations: ObsDict, *, fly_seeds: Sequence[int]
    ) -> torch.Tensor:
        del fly_seeds
        values = torch.as_tensor(
            observation_features(observations),
            dtype=torch.float32,
            device=self.input_weights.device,
        )
        state = torch.tanh(values @ self.input_weights)
        for _ in range(self.steps - 1):
            state = torch.tanh(values @ self.input_weights + state @ self.recurrent)
        return state[:, : self.output_size]


class DirectObservationProcessor:
    """No-connectome control exposing the same complete Balatro information."""

    version = "direct-observation-v1"

    def __init__(self, *, device: str = "cpu") -> None:
        from flylatro.fly.upstream_encoder import full_feature_names

        self.output_size = len(full_feature_names())
        self.device = torch.device(device)

    def process(
        self, observations: ObsDict, *, fly_seeds: Sequence[int]
    ) -> torch.Tensor:
        del fly_seeds
        return torch.as_tensor(
            observation_features(observations),
            dtype=torch.float32,
            device=self.device,
        )


class RealFlyProcessor:
    version = "real-fly-processor-v1"

    def __init__(
        self,
        encoder: FixedUpstreamBalatroEncoder,
        backend: TorchFlyWireBackend,
        *,
        duration_ms: float,
        microbatch_size: int,
        rate_scale_hz: float = 200.0,
    ) -> None:
        if duration_ms <= 0 or microbatch_size < 1 or rate_scale_hz <= 0:
            raise ValueError("processor timing/batching values must be positive")
        if encoder.spec.neuron_count != backend.neuron_count:
            raise ValueError("encoder and real fly neuron counts differ")
        self.encoder = encoder
        self.backend = backend
        self.duration_ms = duration_ms
        self.microbatch_size = microbatch_size
        self.rate_scale_hz = rate_scale_hz
        self.output_size = len(backend.readout_indices) * 2

    @property
    def feature_extractor_hash(self) -> str:
        p = self.backend.parameters
        header = {
            "version": self.version,
            "duration_ms": self.duration_ms,
            "rate_scale_hz": self.rate_scale_hz,
            "features_per_neuron": ["spike_rate", "final_voltage"],
            "resting_mv": p.resting_mv,
            "threshold_mv": p.threshold_mv,
        }
        digest = hashlib.sha256(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(
            np.asarray(self.backend.readout_root_ids, dtype="<i8").tobytes()
        )
        return digest.hexdigest()

    def process(
        self, observations: ObsDict, *, fly_seeds: Sequence[int]
    ) -> torch.Tensor:
        return self.process_with_events(observations, fly_seeds=fly_seeds)

    def process_with_events(
        self,
        observations: ObsDict,
        *,
        fly_seeds: Sequence[int],
        decision_ids: Sequence[int] | None = None,
        recorder: object | None = None,
    ) -> torch.Tensor:
        batch_size = next(iter(observations.values())).shape[0]
        if len(fly_seeds) != batch_size:
            raise ValueError("one fly seed is required per observation")
        if recorder is not None:
            if not self.backend.record_events:
                raise ValueError("fly backend event recording is not enabled")
            if decision_ids is None or len(decision_ids) != batch_size:
                raise ValueError("recording requires one decision ID per observation")
        outputs = []
        for start in range(0, batch_size, self.microbatch_size):
            stop = min(start + self.microbatch_size, batch_size)
            batch = {key: value[start:stop] for key, value in observations.items()}
            stimulus = self.encoder.encode(batch)
            if recorder is not None:
                self._record_stimulus(
                    stimulus,
                    start=start,
                    decision_ids=decision_ids,
                    recorder=recorder,
                )
            self.backend.reset(stop - start, tuple(fly_seeds[start:stop]))
            activity = self.backend.simulate_tensor(stimulus, self.duration_ms)
            if recorder is not None and activity.event_batch is not None:
                self._record_events(
                    activity, start=start, decision_ids=decision_ids, recorder=recorder
                )
            rate = (
                activity.spike_counts.to(torch.float32)
                / (self.duration_ms / 1000.0)
                / self.rate_scale_hz
            ).clamp(0.0, 1.0)
            p = self.backend.parameters
            voltage = (
                (activity.final_voltage - p.resting_mv)
                / (p.threshold_mv - p.resting_mv)
            ).clamp(-1.0, 1.0)
            outputs.append(torch.cat((rate, voltage), dim=1))
        return torch.cat(outputs, dim=0)

    def _record_events(
        self, activity: object, *, start: int, decision_ids: Sequence[int], recorder: object
    ) -> None:
        batch_ids = activity.event_batch.detach().cpu().numpy()
        neuron_ids = activity.event_neuron_ids.detach().cpu().numpy()
        times_ms = activity.event_times_ms.detach().cpu().numpy()
        input_ids = self.encoder.spec.population_root_ids.reshape(-1)
        readout_ids = self.backend.readout_root_ids
        roles = np.full(len(neuron_ids), "internal", dtype=object)
        roles[np.isin(neuron_ids, input_ids)] = "input"
        roles[np.isin(neuron_ids, readout_ids)] = "readout"
        for local_row in np.unique(batch_ids):
            selected = batch_ids == local_row
            recorder.record(
                decision_id=int(decision_ids[start + int(local_row)]),
                times_ms=times_ms[selected],
                neuron_ids=neuron_ids[selected],
                roles=roles[selected],
                event_kind="spike",
            )

    def _record_stimulus(
        self, stimulus: object, *, start: int,
        decision_ids: Sequence[int], recorder: object,
    ) -> None:
        rates = np.asarray(stimulus.rates_hz)
        roots = self.backend.artifact.root_ids
        for local_row in range(rates.shape[0]):
            indices = np.flatnonzero(rates[local_row] > 0)
            recorder.record(
                decision_id=int(decision_ids[start + local_row]),
                times_ms=np.zeros(len(indices), dtype=np.float32),
                neuron_ids=roots[indices],
                roles=np.full(len(indices), "input", dtype=object),
                activities=rates[local_row, indices],
                event_kind="stimulation",
            )
