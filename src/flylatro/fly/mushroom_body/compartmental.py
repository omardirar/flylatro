"""Versioned edge-modulation contract for future compartmental DAN rules.

V1 deliberately uses one global synthetic reinforcement channel.  This file
defines the data needed by a defensible compartmental extension without
inventing DAN-to-compartment assignments absent from the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


@dataclass(frozen=True, slots=True)
class EdgeModulationAssignment:
    version: str
    channel_names: tuple[str, ...]
    edge_channel_indices: NDArray[np.int64]
    edge_plasticity_signs: NDArray[np.float32]
    edge_compartments: NDArray[np.str_]
    evidence: str

    def __post_init__(self) -> None:
        count = len(self.edge_channel_indices)
        if self.edge_plasticity_signs.shape != (count,) or self.edge_compartments.shape != (count,):
            raise ValueError("edge modulation arrays must align")
        if count and (self.edge_channel_indices.min() < 0 or self.edge_channel_indices.max() >= len(self.channel_names)):
            raise ValueError("edge modulation channel is outside channel table")
        if np.any(~np.isin(self.edge_plasticity_signs, (-1.0, 1.0))):
            raise ValueError("edge plasticity signs must be -1 or +1")

    @classmethod
    def global_v1(cls, topology: PlasticEdgeTopology) -> "EdgeModulationAssignment":
        return cls(
            version="three-factor-global-v1",
            channel_names=("synthetic_signed_global_reinforcement",),
            edge_channel_indices=np.zeros(topology.edge_count, dtype=np.int64),
            edge_plasticity_signs=np.ones(topology.edge_count, dtype=np.float32),
            edge_compartments=topology.compartments.copy(),
            evidence="engineering global-rule assumption; compartment labels are descriptive only",
        )

    @classmethod
    def compartmental_dan_v2(
        cls,
        topology: PlasticEdgeTopology,
        *,
        compartment_to_channel: Mapping[str, str],
        compartment_to_sign: Mapping[str, float],
        evidence: str,
    ) -> "EdgeModulationAssignment":
        compartments = topology.compartments.astype(str)
        missing = sorted(set(compartments) - set(compartment_to_channel))
        if missing:
            raise ValueError(
                "compartmental-dan-v2 requires evidence-backed assignments; missing "
                + ", ".join(missing)
            )
        channels = tuple(sorted(set(compartment_to_channel.values())))
        channel_lookup = {name: index for index, name in enumerate(channels)}
        return cls(
            version="compartmental-dan-v2",
            channel_names=channels,
            edge_channel_indices=np.asarray(
                [channel_lookup[compartment_to_channel[value]] for value in compartments],
                dtype=np.int64,
            ),
            edge_plasticity_signs=np.asarray(
                [compartment_to_sign[value] for value in compartments], dtype=np.float32
            ),
            edge_compartments=topology.compartments.copy(),
            evidence=evidence,
        )

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "version": self.version,
                    "channel_names": self.channel_names,
                    "evidence": self.evidence,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(self.edge_channel_indices.astype("<i8", copy=False).tobytes())
        digest.update(self.edge_plasticity_signs.astype("<f4", copy=False).tobytes())
        digest.update("\0".join(self.edge_compartments.tolist()).encode())
        return digest.hexdigest()
