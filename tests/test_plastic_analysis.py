from __future__ import annotations

import numpy as np

from flylatro.analysis.plasticity import plasticity_diagnostics, synaptic_change_report
from flylatro.analysis.representation import representation_diagnostics
from flylatro.analysis.representation_cli import main as representation_main
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology


def test_synaptic_report_groups_changes_by_biological_metadata() -> None:
    topology = PlasticEdgeTopology(
        version="test",
        edge_indices=np.arange(3, dtype=np.int64),
        pre_indices=np.asarray([0, 1, 2], dtype=np.int64),
        post_indices=np.asarray([3, 3, 4], dtype=np.int64),
        pre_root_ids=np.asarray([10, 11, 12], dtype=np.int64),
        post_root_ids=np.asarray([20, 20, 21], dtype=np.int64),
        anatomical_weights=np.ones(3, dtype=np.float32),
        kc_types=np.asarray(["KCab", "KCab", "KCg"], dtype=np.str_),
        mbon_types=np.asarray(["MBON-a", "MBON-a", "MBON-b"], dtype=np.str_),
        compartments=np.asarray(["a", "a", "b"], dtype=np.str_),
    )
    state = PlasticEdgeState.initialize(3)
    state.efficacy[0] = [1.2, 1.0, 0.8]

    report = synaptic_change_report(topology, state)

    assert report["modified_synapses"] == 2
    assert report["positive_changes"] == 1
    assert report["negative_changes"] == 1
    assert {row["group"] for row in report["by_compartment"]} == {"a", "b"}


def test_diagnostics_expose_silence_sparsity_and_failure_flags() -> None:
    state = PlasticEdgeState.initialize(2)
    plastic = plasticity_diagnostics(state, PlasticityConfig())
    assert plastic["finite"]
    assert plastic["no_synaptic_change"]

    mbon = np.asarray([[0.2, 0.0], [0.3, 0.0]])
    representation = representation_diagnostics(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        mbon,
        np.asarray([[0.1], [0.2]]),
        motor_activity=mbon,
        motor_activity_population="mbon",
        state_labels=np.asarray([0, 1]),
    )
    assert representation["motor_activity_population"] == "mbon"
    assert representation["kc_active_fraction"] == 0.5
    assert representation["mbon_silent_fraction"] == 0.5
    assert representation["different_state_separability"] is not None


def test_representation_cli_measures_repeated_identical_states(tmp_path) -> None:
    import json

    from helpers import build_mock_corpus, write_config

    corpus = tmp_path / "corpus.npz"
    build_mock_corpus(corpus, seeds=(11, 12), states_per_seed=2)
    config = write_config(tmp_path, corpus_path=corpus)
    output = tmp_path / "representation-pre.json"
    reachability = tmp_path / "reachability.json"
    assert representation_main(
        [
            "--config", str(config),
            "--stage", "pre",
            "--repeats", "2",
            "--output", str(output),
            "--reachability-output", str(reachability),
        ]
    ) in (0, 2)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["stage"] == "pre"
    assert report["samples"] == 8
    assert report["same_state_variability"] is not None
    assert report["different_state_separability"] is not None
    assert report["final_motor_interface_evaluated"] is False
    assert report["evidence_identity"]["calibration_corpus_sha256"]
    assert json.loads(reachability.read_text(encoding="utf-8"))["by_kc_subtype"]
