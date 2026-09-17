from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from flylatro.fly.backend import FlyActivity
from flylatro.fly.features import FeatureSpec, RateFeatureExtractor
from flylatro.fly.flywire_artifact import FlyWireArtifact


def tiny_artifact(tmp_path: Path) -> FlyWireArtifact:
    return FlyWireArtifact(
        path=tmp_path / "tiny.npz",
        manifest={"connectivity_sha256": "test"},
        root_ids=np.asarray([100, 101, 102, 103], dtype=np.int64),
        pre_indices=np.asarray([0, 0, 1, 2], dtype=np.int64),
        post_indices=np.asarray([1, 2, 2, 3], dtype=np.int64),
        signed_synapse_counts=np.asarray([1, 2, -3, 4], dtype=np.float32),
        sensory_indices=np.asarray([0, 1], dtype=np.int64),
        descending_indices=np.asarray([2, 3], dtype=np.int64),
        coordinates_nm=np.zeros((4, 3), dtype=np.float32),
    )


def test_control_shuffle_preserves_edges_degree_distributions_and_weights(
    tmp_path: Path,
) -> None:
    artifact = tiny_artifact(tmp_path)

    pre, post, weights, procedure = artifact.edge_arrays(shuffle_seed=42)

    np.testing.assert_array_equal(pre, artifact.pre_indices)
    np.testing.assert_array_equal(weights, artifact.signed_synapse_counts)
    assert not np.array_equal(post, artifact.post_indices)
    np.testing.assert_array_equal(
        np.bincount(pre, minlength=artifact.neuron_count),
        np.bincount(artifact.pre_indices, minlength=artifact.neuron_count),
    )
    np.testing.assert_array_equal(
        np.sort(np.bincount(post, minlength=artifact.neuron_count)),
        np.sort(
            np.bincount(
                artifact.post_indices, minlength=artifact.neuron_count
            )
        ),
    )
    assert len(set(zip(pre.tolist(), post.tolist()))) == len(pre)
    assert "seed=42" in procedure


def test_feature_extractor_resolves_real_neuron_ids() -> None:
    activity = FlyActivity(
        spike_counts=np.asarray([[1, 2]], dtype=np.int64),
        final_voltage=np.asarray([[0.1, 0.2]], dtype=np.float64),
        duration_ms=10,
        backend_version="test",
        neuron_ids=np.asarray([7201, 7202], dtype=np.int64),
    )
    extractor = RateFeatureExtractor(FeatureSpec("real-readout-v1", (7202,)))

    features = extractor.extract(activity)

    assert features.shape == (1, 2)
    assert features[0, 0] == 1.0
    assert features[0, 1] == 0.2


def test_full_artifact_build_refuses_implicit_execution(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flylatro.fly.build_flywire",
            "--source-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode != 0
    assert "requires explicit --full" in result.stderr
