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


def test_compact_health_preserves_deferred_population_diagnostics(tmp_path):
    from types import SimpleNamespace
    from flylatro.fly.encoder import Stimulus
    from flylatro.fly.plastic_backend import PlasticFlyProcessor, DeferredHostReadout
    artifact, topology = tiny_chain(tmp_path)
    backend = PlasticTorchFlyWireBackend(artifact, topology, readout_indices=[0, 1, 2])
    encoder = SimpleNamespace(encode=lambda observations: Stimulus(
        np.full((2, 3), 500, dtype=np.float32), 'test', 'test'))
    processor = PlasticFlyProcessor(encoder, backend, topology,
        output_indices=np.asarray([1]), mode='mbon_direct', duration_ms=20)
    first = processor.process({'test': np.zeros((2, 1))}, fly_seeds=(17, 29),
        efficacy=np.ones((2, 1), dtype=np.float32))
    assert isinstance(first.kc_activity, DeferredHostReadout)
    assert first.kc_activity._host is None
    expected = first.kc_activity.tensor.clone().numpy()
    # The next decision can reuse neural buffers without changing retained diagnostics.
    processor.process({'test': np.zeros((2, 1))}, fly_seeds=(31, 43),
        efficacy=np.full((2, 1), .5, dtype=np.float32))
    assert first.kc_activity._host is None
    np.testing.assert_array_equal(np.asarray(first.kc_activity), expected)
    np.testing.assert_array_equal(first.kc_activity > 0, expected > 0)
    assert first.health_summary['kc_active_fraction'] == pytest.approx(np.mean(expected > 0))
    assert first.health_summary['min_learner_kc_active_fraction'] == pytest.approx(
        np.mean(expected > 0, axis=1).min())
    assert first.health_summary['mbon_mean_activity'] == pytest.approx(np.asarray(first.mbon_activity).mean())
    assert first.health_summary['finite_activity'] == 1


@pytest.mark.parametrize('index', [0, 1, 3])
def test_health_rejects_nonfinite_fast_state_even_with_finite_readouts(tmp_path, index):
    from types import SimpleNamespace
    from flylatro.fly.encoder import Stimulus
    from flylatro.fly.plastic_backend import PlasticFlyProcessor
    artifact, topology = tiny_chain(tmp_path)
    backend = PlasticTorchFlyWireBackend(artifact, topology, readout_indices=[0, 1, 2])
    encoder = SimpleNamespace(encode=lambda observations: Stimulus(np.full((1, 3), 500, dtype=np.float32), 'test', 'test'))
    processor = PlasticFlyProcessor(encoder, backend, topology,
        output_indices=np.asarray([1]), mode='mbon_direct', duration_ms=20)
    simulate = backend.simulate_tensor
    def corrupt(*args):
        activity = simulate(*args)
        backend._state[index].reshape(-1)[-1] = float('nan')
        return activity
    backend.simulate_tensor = corrupt
    with pytest.raises(ValueError, match='fast state must be finite'):
        processor.process({'test': np.zeros((1, 1))}, fly_seeds=(17,),
            efficacy=np.ones((1, 1), dtype=np.float32))
