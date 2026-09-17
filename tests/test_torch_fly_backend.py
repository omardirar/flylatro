from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flylatro.fly.encoder import Stimulus
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.torch_backend import TorchFlyWireBackend


def artifact(tmp_path: Path) -> FlyWireArtifact:
    return FlyWireArtifact(
        path=tmp_path / "tiny.npz",
        manifest={"connectivity_sha256": "test"},
        root_ids=np.asarray([100, 101, 102, 103, 104, 105], dtype=np.int64),
        pre_indices=np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64),
        post_indices=np.asarray([1, 2, 2, 3, 4, 5], dtype=np.int64),
        signed_synapse_counts=np.asarray(
            [20, 15, 10, 30, 30, 30], dtype=np.float32
        ),
        sensory_indices=np.asarray([0, 1], dtype=np.int64),
        descending_indices=np.asarray([4, 5], dtype=np.int64),
        coordinates_nm=np.zeros((6, 3), dtype=np.float32),
    )


def input_stimulus() -> Stimulus:
    rates = np.zeros((2, 6), dtype=np.float32)
    rates[:, :2] = 150.0
    return Stimulus(rates, "test", "hash")


def test_torch_backend_is_batched_seeded_and_reset_deterministic(
    tmp_path: Path,
) -> None:
    backend = TorchFlyWireBackend(artifact(tmp_path), device="cpu")
    backend.reset(2, seeds=(10, 11))
    first = backend.simulate(input_stimulus(), duration_ms=25)
    backend.reset(2, seeds=(10, 11))
    second = backend.simulate(input_stimulus(), duration_ms=25)

    np.testing.assert_array_equal(first.spike_counts, second.spike_counts)
    np.testing.assert_array_equal(first.final_voltage, second.final_voltage)
    np.testing.assert_array_equal(first.neuron_ids, [104, 105])
    assert first.spike_counts.shape == (2, 2)
    assert backend.weights.layout == torch.sparse_coo
    assert len(backend.dynamics_hash) == 64


def test_torch_backend_control_changes_topology_hash(tmp_path: Path) -> None:
    real = TorchFlyWireBackend(artifact(tmp_path), device="cpu")
    shuffled = TorchFlyWireBackend(
        artifact(tmp_path), device="cpu", shuffle_seed=99
    )

    assert real.connectivity_hash != shuffled.connectivity_hash
    assert shuffled.topology_procedure.endswith("seed=99")


def test_detailed_spike_recording_is_opt_in(tmp_path: Path) -> None:
    plain = TorchFlyWireBackend(artifact(tmp_path), device="cpu")
    plain.reset(2, seeds=(4, 5))
    plain_activity = plain.simulate_tensor(input_stimulus(), duration_ms=10)
    recording = TorchFlyWireBackend(
        artifact(tmp_path), device="cpu", record_events=True
    )
    recording.reset(2, seeds=(4, 5))
    recorded_activity = recording.simulate_tensor(input_stimulus(), duration_ms=10)

    assert plain_activity.event_batch is None
    if recorded_activity.event_batch is not None:
        assert len(recorded_activity.event_batch) == len(
            recorded_activity.event_neuron_ids
        )
        assert len(recorded_activity.event_batch) == len(
            recorded_activity.event_times_ms
        )


def test_synaptic_delay_and_voltage_update_match_reference_step_order(
    tmp_path: Path,
) -> None:
    backend = TorchFlyWireBackend(
        artifact(tmp_path), device="cpu", readout_indices=[1]
    )
    backend.reset(1, seeds=(1,))
    conductance, delay, spikes, voltage, refractory, counts = backend._state
    spikes[0, 0] = 1.0
    zero = Stimulus(np.zeros((1, 6), dtype=np.float32), "test", "hash")

    activity = backend.simulate_tensor(zero, duration_ms=2.0)

    # Edge 0→1 has weight 20. At 1.8 ms its 20*0.275 mV drive enters
    # conductance; the reference AlphaLIF update feeds the previous
    # conductance into voltage, so membrane voltage changes one step later.
    assert float(activity.final_voltage[0, 0]) == pytest.approx(-51.9725, abs=1e-5)
