"""Primary plastic-fly agent: fixed interfaces around internal synaptic learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import ActionDict, MaskDict, ObsDict
from flylatro.fly.mushroom_body.plasticity import (
    PlasticityEvent,
    ThreeFactorPlasticity,
)
from flylatro.fly.plastic_backend import PlasticFlyDecisionActivity
from flylatro.interface.motor import FixedMotorInterface
from flylatro.learning.reinforcement import DopaminePulse, ReinforcementMapper


class PlasticFlyProcessorProtocol(Protocol):
    version: str
    output_root_ids: NDArray[np.int64]

    def process(
        self,
        observations: ObsDict,
        *,
        fly_seeds: Sequence[int],
        efficacy: NDArray[np.floating],
    ) -> PlasticFlyDecisionActivity: ...


@dataclass(frozen=True, slots=True)
class PlasticDecision:
    actions: ActionDict
    neural: PlasticFlyDecisionActivity


@dataclass(frozen=True, slots=True)
class LearningResult:
    pulses: tuple[DopaminePulse, ...]
    events: tuple[PlasticityEvent, ...]


class PlasticFlyAgent:
    """No external actor, decoder parameters, critic, gradients, or optimiser."""

    def __init__(
        self,
        processor: PlasticFlyProcessorProtocol,
        plasticity: ThreeFactorPlasticity,
        motor: FixedMotorInterface,
        reinforcement: ReinforcementMapper,
    ) -> None:
        if plasticity.state.learners < 1:
            raise ValueError("plastic fly agent needs at least one learner")
        if not np.array_equal(
            processor.output_root_ids, motor.mapping.output_root_ids
        ):
            raise ValueError("processor outputs and motor mapping differ")
        self.processor = processor
        self.plasticity = plasticity
        self.motor = motor
        self.reinforcement = reinforcement

    @property
    def trainable_parameter_count(self) -> int:
        # The motor/encoder have zero learned parameters. Plastic efficacy is
        # explicitly counted as biological learned state, not an external model.
        return 0

    @property
    def plastic_parameter_count(self) -> int:
        efficacy = self.plasticity.state.efficacy
        return int(efficacy.numel() if hasattr(efficacy, "numel") else efficacy.size)

    def act(
        self,
        observations: ObsDict,
        masks: MaskDict,
        *,
        fly_seeds: Sequence[int],
        deterministic_motor: bool,
        record_eligibility: bool = True,
        motor_learner_ids: Sequence[int] | None = None,
        motor_decision_ids: Sequence[int] | None = None,
    ) -> PlasticDecision:
        batch = next(iter(observations.values())).shape[0]
        if batch != self.plasticity.state.learners:
            raise ValueError(
                "each simultaneous environment must own an independent plastic fly"
            )
        neural = self.processor.process(
            observations,
            fly_seeds=fly_seeds,
            efficacy=self.plasticity.state.efficacy,
        )
        if record_eligibility:
            self.plasticity.record_activity(
                neural.edge_pre_activity, neural.edge_post_activity
            )
        actions = self.motor.decode(
            neural.output_activity,
            masks,
            deterministic=deterministic_motor,
            learner_ids=(None if motor_learner_ids is None else np.asarray(motor_learner_ids)),
            decision_ids=(None if motor_decision_ids is None else np.asarray(motor_decision_ids)),
        )
        return PlasticDecision(actions=actions, neural=neural)

    def learn(
        self,
        infos: Sequence[dict[str, Any]],
        *,
        plasticity_enabled: bool,
        override_pulses: Sequence[DopaminePulse] | None = None,
        detail: bool = False,
    ) -> LearningResult:
        if len(infos) != self.plasticity.state.learners:
            raise ValueError("one outcome is required per independent fly")
        pulses = tuple(
            override_pulses[index]
            if override_pulses is not None
            else self.reinforcement.map(
                dict(info.get("reward_components", {})), info
            )
            for index, info in enumerate(infos)
        )
        events = self.plasticity.apply_dopamine(
            np.asarray([pulse.appetitive for pulse in pulses], dtype=np.float32),
            np.asarray([pulse.aversive for pulse in pulses], dtype=np.float32),
            plasticity_enabled=plasticity_enabled,
            detail=detail,
        )
        for index, info in enumerate(infos):
            if isinstance(info.get("episode"), dict):
                self.plasticity.state.episode_count[index] += 1
                # The terminal outcome must first update the persistent
                # efficacy, then both fast traces are cleared for this learner
                # only. Auto-reset environments can otherwise leak terminal
                # credit into the next episode.
                self.plasticity.state.reset_fast_traces(
                    learners=(index,), eligibility=True
                )
        return LearningResult(pulses=pulses, events=events)
