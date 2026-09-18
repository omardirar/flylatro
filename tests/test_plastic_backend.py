from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.fly.plastic_backend import PlasticTorchFlyWireBackend


def tiny_chain(tmp_path: Path) -> tuple[FlyWireArtifact, PlasticEdgeTopology]:
    artifact = FlyWireArtifact(
        path=tmp_path / "chain.npz",
        manifest={"connectivity_sha256": "test"},
        root_ids=np.asarray([100, 200, 300], dtype=np.int64),
        pre_indices=np.asarray([0, 1], dtype=np.int64),
        post_indices=np.asarray([1, 2], dtype=np.int64),
        signed_synapse_counts=np.asarray([2.0, 3.0], dtype=np.float32),
        sensory_indices=np.asarray([0], dtype=np.int64),
        descending_indices=np.asarray([2], dtype=np.int64),
        coordinates_nm=np.zeros((3, 3), dtype=np.float32),
    )
    topology = PlasticEdgeTopology.synthetic(
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([2.0], dtype=np.float32),
    )
    return artifact, topology


def test_plastic_kc_mbon_change_propagates_to_mbon_and_descending(
    tmp_path: Path,
) -> None:
    artifact, topology = tiny_chain(tmp_path)
    backend = PlasticTorchFlyWireBackend(
        artifact, topology, readout_indices=np.asarray([0, 1, 2])
    )
    kc = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    backend.set_plastic_efficacy(np.asarray([0.5], dtype=np.float32))
    weak_mbon = backend.propagate_once(kc)
    weak_downstream = backend.propagate_once(weak_mbon)
    backend.set_plastic_efficacy(np.asarray([1.5], dtype=np.float32))
    strong_mbon = backend.propagate_once(kc)
    strong_downstream = backend.propagate_once(strong_mbon)

    assert strong_mbon[1] > weak_mbon[1]
    assert strong_downstream[2] > weak_downstream[2]
    assert backend.efficacy[0] == pytest.approx(1.5)
