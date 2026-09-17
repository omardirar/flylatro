"""Composition root for one Balatro-state-to-action decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from flylatro.env.types import ActionMask, BalatroObservation
from flylatro.fly.backend import FlyBackend
from flylatro.fly.encoder import FixedBalatroEncoder
from flylatro.fly.features import RateFeatureExtractor
from flylatro.policy.structured import PolicyDecision, StructuredLinearPolicy


@dataclass(frozen=True, slots=True)
class AgentDecision:
    policy: PolicyDecision
    fly_total_spikes: int
    fly_active_neurons: int


class FlyAgent:
    """Runs the fixed fly once per state, split into bounded microbatches."""

    def __init__(
        self,
        *,
        encoder: FixedBalatroEncoder,
        fly_backend: FlyBackend,
        feature_extractor: RateFeatureExtractor,
        policy: StructuredLinearPolicy,
        duration_ms: float,
        microbatch_size: int,
    ) -> None:
        if duration_ms <= 0 or microbatch_size < 1:
            raise ValueError("duration and microbatch size must be positive")
        if encoder.spec.neuron_count != fly_backend.neuron_count:
            raise ValueError("encoder and fly backend neuron counts differ")
        if feature_extractor.spec.output_size != policy.feature_size:
            raise ValueError("feature extractor and policy dimensions differ")
        self.encoder = encoder
        self.fly_backend = fly_backend
        self.feature_extractor = feature_extractor
        self.policy = policy
        self.duration_ms = duration_ms
        self.microbatch_size = microbatch_size

    def act(
        self,
        observations: Sequence[BalatroObservation],
        masks: Sequence[ActionMask],
        *,
        deterministic: bool = False,
    ) -> tuple[AgentDecision, ...]:
        if not observations:
            return ()
        if len(observations) != len(masks):
            raise ValueError("one action mask is required per observation")
        feature_batches = []
        spike_totals: list[int] = []
        active_counts: list[int] = []
        for start in range(0, len(observations), self.microbatch_size):
            stop = min(start + self.microbatch_size, len(observations))
            stimulus = self.encoder.encode(observations[start:stop])
            self.fly_backend.reset(stop - start)
            activity = self.fly_backend.simulate(stimulus, self.duration_ms)
            feature_batches.append(self.feature_extractor.extract(activity))
            spike_totals.extend(
                int(value) for value in activity.spike_counts.sum(axis=1)
            )
            active_counts.extend(
                int(value)
                for value in np.count_nonzero(activity.spike_counts, axis=1)
            )
        features = np.concatenate(feature_batches, axis=0)
        policy_decisions = self.policy.act(
            features, masks, deterministic=deterministic
        )
        return tuple(
            AgentDecision(
                policy=decision,
                fly_total_spikes=spike_totals[index],
                fly_active_neurons=active_counts[index],
            )
            for index, decision in enumerate(policy_decisions)
        )

