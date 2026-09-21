"""Canonical motor output universe derived from real unshuffled anatomy.

For `mbon_direct` the motor candidates are exactly the MBONs that are
postsynaptic in the canonical real KC->MBON plastic edge set, so every motor
output is reachable by plasticity.  The universe is deliberately computed from
the *real unshuffled* topology at the canonical minimum synapse count, so it
cannot drift when a control shuffles topology or a sensitivity experiment
changes `minimum_synapse_count`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


#: The canonical motor universe never moves with a sensitivity experiment.
#: Changing this constant is a versioned motor-map sensitivity analysis and
#: must be declared as such.
CANONICAL_MINIMUM_SYNAPSE_COUNT = 1
CANONICAL_CANDIDATE_VERSION = "canonical-plastic-reachable-motor-candidates-v1"


@dataclass(frozen=True, slots=True)
class MotorCandidateSet:
    version: str
    mode: str
    rule: str
    indices: NDArray[np.int64]
    root_ids: NDArray[np.int64]
    plastic_input_edge_counts: NDArray[np.int64]
    plastic_input_synapse_weights: NDArray[np.float32]
    neuron_types: tuple[str, ...]
    compartments: tuple[str, ...]
    minimum_synapse_count: int

    def __post_init__(self) -> None:
        length = len(self.indices)
        if not length:
            raise ValueError("motor candidate set is empty")
        if any(
            len(values) != length
            for values in (
                self.root_ids,
                self.plastic_input_edge_counts,
                self.plastic_input_synapse_weights,
                self.neuron_types,
                self.compartments,
            )
        ):
            raise ValueError("motor candidate arrays must align")

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "version": self.version,
                    "mode": self.mode,
                    "rule": self.rule,
                    "minimum_synapse_count": self.minimum_synapse_count,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(self.root_ids.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    def describe(self, positions: NDArray[np.integer] | None = None) -> list[dict[str, Any]]:
        """Per-candidate provenance for the persisted motor artifact."""

        selected = (
            np.arange(len(self), dtype=np.int64)
            if positions is None
            else np.asarray(positions, dtype=np.int64)
        )
        return [
            {
                "candidate_position": int(position),
                "root_id": int(self.root_ids[position]),
                "kc_plastic_input_edges": int(self.plastic_input_edge_counts[position]),
                "kc_mbon_anatomical_synapse_weight": float(
                    self.plastic_input_synapse_weights[position]
                ),
                "neuron_type": self.neuron_types[position],
                "compartment": self.compartments[position],
            }
            for position in selected
        ]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "rule": self.rule,
            "minimum_synapse_count": self.minimum_synapse_count,
            "candidate_count": len(self),
            "candidates": self.describe(),
            "sha256": self.sha256,
        }


def canonical_motor_candidates(
    artifact: FlyWireArtifact, *, mode: str
) -> MotorCandidateSet:
    """Return the fixed motor output universe for one output mode."""

    if mode not in {"mbon_direct", "whole_brain"}:
        raise ValueError("mode must be mbon_direct or whole_brain")
    canonical = PlasticEdgeTopology.from_artifact(
        artifact,
        post_indices=None,
        version="flywire-v783-kc-mbon-neuron-pairs-v1",
        minimum_synapse_count=CANONICAL_MINIMUM_SYNAPSE_COUNT,
    )
    reachable, inverse = np.unique(canonical.post_indices, return_inverse=True)
    edge_counts = np.bincount(inverse, minlength=len(reachable)).astype(np.int64)
    weights = np.bincount(
        inverse, weights=canonical.anatomical_weights.astype(np.float64), minlength=len(reachable)
    ).astype(np.float32)
    if mode == "mbon_direct":
        indices = reachable
        rule = (
            "unique postsynaptic MBONs of the real unshuffled KC->MBON plastic "
            f"topology at minimum_synapse_count={CANONICAL_MINIMUM_SYNAPSE_COUNT}"
        )
        counts = edge_counts
        synapses = weights
    else:
        indices = np.asarray(artifact.descending_indices, dtype=np.int64)
        rule = "classification.super_class == descending (real unshuffled artifact)"
        lookup = {int(value): position for position, value in enumerate(reachable)}
        counts = np.asarray(
            [edge_counts[lookup[int(value)]] if int(value) in lookup else 0 for value in indices],
            dtype=np.int64,
        )
        synapses = np.asarray(
            [weights[lookup[int(value)]] if int(value) in lookup else 0.0 for value in indices],
            dtype=np.float32,
        )
    types = _text_column(artifact.primary_types, indices)
    compartments = tuple(_compartment_label(value) for value in types)
    return MotorCandidateSet(
        version=CANONICAL_CANDIDATE_VERSION,
        mode=mode,
        rule=rule,
        indices=np.asarray(indices, dtype=np.int64),
        root_ids=artifact.root_ids[indices].astype(np.int64, copy=True),
        plastic_input_edge_counts=counts,
        plastic_input_synapse_weights=synapses,
        neuron_types=types,
        compartments=compartments,
        minimum_synapse_count=CANONICAL_MINIMUM_SYNAPSE_COUNT,
    )


def _text_column(values: NDArray[np.str_], indices: NDArray[np.int64]) -> tuple[str, ...]:
    if values.shape[:1] != (0,) and len(values) > int(indices.max(initial=-1)):
        return tuple(str(value) for value in values[indices])
    return tuple("unknown" for _ in indices)


def _compartment_label(type_name: str) -> str:
    value = type_name.strip()
    if not value.upper().startswith("MBON"):
        return "unknown"
    suffix = value[4:].lstrip("-_")
    return suffix.split(">", 1)[0] if suffix else "unknown"
