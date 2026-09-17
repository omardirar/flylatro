"""Versioned, deterministic Balatro-to-neuron encoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.env.types import BalatroObservation, GamePhase, Suit


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    version: str
    neuron_count: int
    max_cards: int
    max_targets: int
    input_neuron_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    min_rate_hz: float = 0.0
    max_rate_hz: float = 100.0
    money_scale: float = 100.0
    max_ante: int = 8
    max_hands: int = 4
    max_discards: int = 3

    def __post_init__(self) -> None:
        if self.neuron_count < 1:
            raise ValueError("neuron_count must be positive")
        if len(self.feature_names) != len(self.input_neuron_ids):
            raise ValueError("each encoder feature needs one input neuron")
        if len(set(self.input_neuron_ids)) != len(self.input_neuron_ids):
            raise ValueError("input neuron IDs must be unique")
        if any(index < 0 or index >= self.neuron_count for index in self.input_neuron_ids):
            raise ValueError("input neuron ID is outside the backend graph")
        if not 0 <= self.min_rate_hz < self.max_rate_hz:
            raise ValueError("invalid stimulation rate range")

    @classmethod
    def development_default(
        cls, *, neuron_count: int = 96, max_cards: int = 8, max_targets: int = 8
    ) -> "EncoderSpec":
        names = [
            "score_progress",
            "money",
            "ante",
            "hands_remaining",
            "discards_remaining",
        ]
        names.extend(f"phase:{phase.value}" for phase in GamePhase)
        for card in range(max_cards):
            names.append(f"card:{card}:rank")
            names.extend(f"card:{card}:suit:{suit.value}" for suit in Suit)
        names.extend(f"shop_target:{target}:present" for target in range(max_targets))
        names.append("bias")
        if len(names) > neuron_count:
            raise ValueError(
                f"development encoder requires {len(names)} neurons, got {neuron_count}"
            )
        return cls(
            version="mock-sensory-v1",
            neuron_count=neuron_count,
            max_cards=max_cards,
            max_targets=max_targets,
            input_neuron_ids=tuple(range(len(names))),
            feature_names=tuple(names),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "neuron_count": self.neuron_count,
            "max_cards": self.max_cards,
            "max_targets": self.max_targets,
            "input_neuron_ids": list(self.input_neuron_ids),
            "feature_names": list(self.feature_names),
            "min_rate_hz": self.min_rate_hz,
            "max_rate_hz": self.max_rate_hz,
            "money_scale": self.money_scale,
            "max_ante": self.max_ante,
            "max_hands": self.max_hands,
            "max_discards": self.max_discards,
        }

    @property
    def sha256(self) -> str:
        value = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class Stimulus:
    rates_hz: FloatArray
    encoder_version: str
    encoder_hash: str

    def __post_init__(self) -> None:
        if self.rates_hz.ndim != 2:
            raise ValueError("stimulus rates must have shape [batch, neurons]")


class FixedBalatroEncoder:
    """Maps simulator-neutral observations to fixed stimulation rates."""

    def __init__(self, spec: EncoderSpec) -> None:
        self.spec = spec

    def encode(self, observations: Sequence[BalatroObservation]) -> Stimulus:
        if not observations:
            raise ValueError("at least one observation is required")
        features = np.stack([self._features(observation) for observation in observations])
        rates = np.zeros((len(observations), self.spec.neuron_count), dtype=np.float64)
        scaled = self.spec.min_rate_hz + features * (
            self.spec.max_rate_hz - self.spec.min_rate_hz
        )
        rates[:, self.spec.input_neuron_ids] = scaled
        rates.setflags(write=False)
        return Stimulus(
            rates_hz=rates,
            encoder_version=self.spec.version,
            encoder_hash=self.spec.sha256,
        )

    def _features(self, observation: BalatroObservation) -> FloatArray:
        target = max(observation.blind_target, 1.0)
        values = [
            _bounded(observation.score / target),
            _bounded(observation.money / self.spec.money_scale),
            _bounded(observation.ante / self.spec.max_ante),
            _bounded(observation.hands_remaining / self.spec.max_hands),
            _bounded(observation.discards_remaining / self.spec.max_discards),
        ]
        values.extend(float(observation.phase is phase) for phase in GamePhase)
        for index in range(self.spec.max_cards):
            if index >= len(observation.hand):
                values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
                continue
            card = observation.hand[index]
            values.append((card.rank - 2) / 12.0)
            values.extend(float(card.suit is suit) for suit in Suit)
        values.extend(
            float(index < observation.shop_slots)
            for index in range(self.spec.max_targets)
        )
        values.append(1.0)
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(self.spec.feature_names),):
            raise RuntimeError("encoder feature layout drifted from its specification")
        return result


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

