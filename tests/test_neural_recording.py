from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from flylatro.fly.flywire_artifact import EXPECTED_NEURONS, sha256_file
from flylatro.replay.bundle import ReplayBundleWriter, ReplayIdentity
from flylatro.replay.neural import NeuralEventRecorder
from flylatro.visualization.cli import main as visualize_main
from flylatro.visualization.cli import _read_selected_decisions


def _identity() -> ReplayIdentity:
    return ReplayIdentity(
        episode_id="visual-test",
        balatro_seed="TESTSEED",
        checkpoint_id="model.pt",
        checkpoint_sha256=None,
        simulator_version="test",
        encoder_hash="a" * 64,
        feature_extractor_hash="b" * 64,
        connectome_hash="c" * 64,
        policy_version="test",
        simulator_seed=4,
    )


def test_neural_parquet_and_activity_driven_renderer_round_trip(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle"
    with ReplayBundleWriter(bundle_path, _identity()) as writer:
        writer.record(
            {
                "decision_id": 0,
                "action": {"type": "play_hand", "cards": [0]},
                "action_probability": 0.8,
                "action_type_probabilities": [0.8, 0.2],
                "reward": 1.0,
                "state_hash_before": "a" * 64,
                "state_hash_after": "b" * 64,
                "done": True,
            }
        )
        neural_path = bundle_path / "neural-events.parquet"
        recorder = NeuralEventRecorder(neural_path)
        recorder.record(
            decision_id=0,
            times_ms=[0.0],
            neuron_ids=[1000],
            roles=["input"],
            activities=[150.0],
            event_kind="stimulation",
        )
        recorder.record(
            decision_id=0,
            times_ms=[10.0, 20.0],
            neuron_ids=[1001, 1002],
            roles=["internal", "readout"],
            event_kind="spike",
        )
        recorder.record(
            decision_id=0,
            times_ms=[50.0],
            neuron_ids=[-1],
            roles=["synthetic_appetitive"],
            activities=[0.4],
            event_kind="synthetic_reinforcement",
        )
        recorder.close()
        writer.attach_neural_file(neural_path)

    table = pq.read_table(neural_path)
    assert table.num_rows == 4
    assert table.column("event_kind").to_pylist() == [
        "stimulation", "spike", "spike", "synthetic_reinforcement"
    ]
    assert "dan" not in table.column("role").to_pylist()
    assert pq.ParquetFile(neural_path).num_row_groups == 1

    artifact_path = _artifact(tmp_path)
    output = tmp_path / "visual"
    assert visualize_main(
        [
            "--bundle", str(bundle_path),
            "--artifact", str(artifact_path),
            "--output-dir", str(output),
            "--playback-seconds", "1.0",
            "--fps", "1",
        ]
    ) == 0
    assert (output / "frame-0000000.svg").is_file()
    assert (output / "timeline.json").is_file()
    assert "20" in (output / "timeline.json").read_text(encoding="utf-8")


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "flywire.npz"
    root_ids = np.arange(1000, 1000 + EXPECTED_NEURONS, dtype=np.int64)
    coordinates = np.zeros((EXPECTED_NEURONS, 3), dtype=np.float32)
    coordinates[:3, :2] = [[0, 0], [1, 1], [2, 0]]
    np.savez(
        path,
        root_ids=root_ids,
        pre_indices=np.asarray([], dtype=np.int64),
        post_indices=np.asarray([], dtype=np.int64),
        signed_synapse_counts=np.asarray([], dtype=np.float32),
        sensory_indices=np.asarray([0], dtype=np.int64),
        descending_indices=np.asarray([2], dtype=np.int64),
        coordinates_nm=coordinates,
    )
    manifest = {
        "dataset": "FlyWire FAFB",
        "version": "783",
        "artifact_sha256": sha256_file(path),
        "connectivity_sha256": "test",
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def test_streaming_recorder_writes_decision_row_groups_and_filtered_reader(tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    recorder = NeuralEventRecorder(path)
    recorder.record(decision_id=0, times_ms=[1], neuron_ids=[10], roles=["kc"])
    recorder.record(decision_id=1, times_ms=[2], neuron_ids=[11], roles=["mbon"])
    recorder.close()
    source = pq.ParquetFile(path)
    assert source.num_row_groups == 2
    selected = _read_selected_decisions(source, {1})
    assert selected.column("decision_id").to_pylist() == [1]
