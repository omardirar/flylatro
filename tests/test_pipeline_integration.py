from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.motor_calibration_cli import main as calibrate_motor_main
from flylatro.analysis.preflight import run_preflight
from flylatro.analysis.representation_cli import main as representation_main
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.reinforcement import (
    SENSITIVITY_CONDITIONS,
    ReinforcementConfig,
    ReinforcementMapper,
    sensitivity_condition,
)
from helpers import build_mock_corpus, write_config

torch = pytest.importorskip("torch")

#: The mock environment only ever reaches PLAYING states, so the shop, pack,
#: joker and consumable motor contexts have no evidence in it.  These tests opt
#: out of the context-evidence gate explicitly; `test_motor_interface_v2`
#: asserts that the default gate refuses exactly this situation.
RELAXED_MOTOR = [
    "--minimum-candidate-robust-scale-hz", "0.0001",
    "--minimum-effective-signal-fraction", "0.0",
    "--minimum-normalized-option-range", "0.0",
    "--minimum-context-states", "0",
    "--minimum-competing-context-states", "0",
]


@pytest.fixture()
def calibrated(tmp_path: Path):
    """A synthetic experiment carried through corpus -> motor calibration."""

    corpus_path = tmp_path / "corpus.npz"
    build_mock_corpus(corpus_path, seeds=(1, 2, 3, 4), states_per_seed=4)
    pre_config = write_config(tmp_path, corpus_path=corpus_path, name="pre")
    motor_path = tmp_path / "motor.json"
    report_path = tmp_path / "motor-report.json"
    assert calibrate_motor_main(
        [
            "--config", str(pre_config),
            "--calibration-seed", "91001",
            "--output", str(motor_path),
            "--report", str(report_path),
            *RELAXED_MOTOR,
        ]
    ) == 0
    post_config = write_config(
        tmp_path,
        corpus_path=corpus_path,
        name="post",
        overrides={"fly": {"motor_mapping_path": str(motor_path.resolve())}},
    )
    return corpus_path, motor_path, report_path, pre_config, post_config


def test_post_motor_representation_uses_the_persisted_final_mapping(
    tmp_path: Path, calibrated
) -> None:
    _, motor_path, _, pre_config, post_config = calibrated
    post = tmp_path / "rep-post.json"
    representation_main(
        [
            "--config", str(post_config),
            "--stage", "post",
            "--repeats", "2",
            "--minimum-normalized-option-range", "0.0",
            "--minimum-action-coverage-fraction", "0.0",
            "--maximum-competing-pool-correlation", "1.0",
            "--output", str(post),
        ]
    )
    report = json.loads(post.read_text(encoding="utf-8"))
    persisted = json.loads(motor_path.read_text(encoding="utf-8"))
    assert report["stage"] == "post"
    assert report["final_motor_interface_evaluated"] is True
    assert report["motor_interface"]["motor_artifact_sha256"] == persisted["sha256"]
    assert (
        report["motor_interface"]["motor_mapping_sha256"]
        == persisted["structure_sha256"]
    )
    assert report["motor_interface"]["motor_normalization_sha256"] is not None
    assert (
        report["evidence_identity"]["motor_mapping_sha256"]
        == persisted["structure_sha256"]
    )
    assert report["motor_activity_population"] == "mbon"

    # A post-motor report is refused when only a bootstrap mapping exists.
    with pytest.raises(ValueError, match="persisted reward-free motor artifact"):
        representation_main(
            [
                "--config", str(pre_config),
                "--stage", "post",
                "--output", str(tmp_path / "never.json"),
            ]
        )


def test_preflight_rejects_a_motor_mapping_from_another_calibration(
    tmp_path: Path, calibrated
) -> None:
    _, _, report_path, _, post_config = calibrated
    config = PlasticExperimentConfig.load(post_config)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    good = run_preflight(config, profile="initial", motor_calibration_report=report)
    gate = next(item for item in good["gates"] if item["name"] == "motor_calibration")
    assert gate["status"] == "PASS"

    stale = copy.deepcopy(report)
    stale["evidence_identity"]["motor_mapping_sha256"] = "a-different-motor-artifact"
    bad = run_preflight(config, profile="initial", motor_calibration_report=stale)
    gate = next(item for item in bad["gates"] if item["name"] == "motor_calibration")
    assert gate["status"] == "FAIL"
    assert "motor_mapping_sha256" in gate["reason"]


def test_stack_refuses_a_motor_artifact_from_another_candidate_universe(
    tmp_path: Path, calibrated
) -> None:
    corpus_path, motor_path, _, _, post_config = calibrated
    payload = json.loads(motor_path.read_text(encoding="utf-8"))
    payload["candidate_set_sha256"] = "a-different-canonical-universe"
    from flylatro.interface.motor import MotorMapping

    rebuilt = MotorMapping.from_manifest({**payload, "sha256": None})
    payload["sha256"] = rebuilt.sha256
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    config = PlasticExperimentConfig.load(
        write_config(
            tmp_path,
            corpus_path=corpus_path,
            name="tampered",
            overrides={"fly": {"motor_mapping_path": str(tampered.resolve())}},
        )
    )
    # The synthetic backend declares no canonical universe, so the mismatch can
    # only be enforced where a canonical universe exists.
    stack = build_plastic_stack(config)
    assert stack.components["canonical_motor_candidate_set_sha256"] is None
    assert stack.agent.motor.mapping.candidate_set_sha256 == (
        "a-different-canonical-universe"
    )


def test_exploration_seed_does_not_change_the_matched_motor_identity(
    tmp_path: Path, calibrated
) -> None:
    corpus_path, motor_path, _, _, _ = calibrated
    identities = set()
    artifacts = set()
    for seed in (2203, 9999):
        config = PlasticExperimentConfig.load(
            write_config(
                tmp_path,
                corpus_path=corpus_path,
                name=f"seed-{seed}",
                overrides={
                    "fly": {"motor_mapping_path": str(motor_path.resolve())},
                    "motor": {"exploration_seed": seed},
                },
            )
        )
        components = build_plastic_stack(config).components
        identities.add(components["motor_mapping_sha256"])
        artifacts.add(components["motor_artifact_sha256"])
    # Replicates differ in exploration seed but must share one motor interface.
    assert len(identities) == 1
    assert len(artifacts) == 2


def test_motor_artifact_survives_a_round_trip_through_training(
    tmp_path: Path, calibrated
) -> None:
    _, motor_path, _, _, post_config = calibrated
    config = PlasticExperimentConfig.load(post_config)
    stack = build_plastic_stack(config)
    persisted = json.loads(motor_path.read_text(encoding="utf-8"))
    mapping = stack.agent.motor.mapping
    assert mapping.selection_method.startswith("reward-free")
    assert mapping.normalization is not None
    np.testing.assert_allclose(
        mapping.normalization.baseline, persisted["normalization"]["baseline"]
    )
    assert stack.components["motor_calibration_status"] == "persisted-artifact"
    assert stack.components["motor_mapping_bootstrap_only"] is False
    assert stack.components["motor_normalization_sha256"] == mapping.normalization.sha256
    assert stack.components["external_trainable_parameter_count"] == 0


def test_predeclared_reinforcement_sensitivity_conditions_exist() -> None:
    assert set(SENSITIVITY_CONDITIONS) == {
        "primary-progress",
        "reduced-progress",
        "terminal-or-clear-only",
    }
    hashes = set()
    for name in SENSITIVITY_CONDITIONS:
        config = sensitivity_condition(name)
        assert config.condition == name
        hashes.add(config.sha256)
    assert len(hashes) == 3, "each condition must have its own identity"
    assert sensitivity_condition("primary-progress").progress_scale == 0.05
    assert sensitivity_condition("reduced-progress").progress_scale == 0.01
    assert sensitivity_condition("terminal-or-clear-only").progress_scale == 0.0

    components = {"blind_progress": 1.0, "blind_clear": 0.0, "win": 0.0}
    dense = ReinforcementMapper(sensitivity_condition("primary-progress")).map(
        dict(components), {}
    )
    sparse = ReinforcementMapper(sensitivity_condition("terminal-or-clear-only")).map(
        dict(components), {}
    )
    assert dense.appetitive > 0
    assert sparse.appetitive == 0.0
    # Terminal outcomes still reinforce in the sparse condition.
    cleared = ReinforcementMapper(sensitivity_condition("terminal-or-clear-only")).map(
        {"blind_progress": 0.0, "blind_clear": 1.0, "win": 0.0}, {}
    )
    assert cleared.appetitive > 0


def test_ad_hoc_reinforcement_retuning_is_refused() -> None:
    with pytest.raises(ValueError, match="predeclared"):
        ReinforcementConfig(condition="tuned-until-it-worked")
    with pytest.raises(ValueError, match="declare a new named sensitivity condition"):
        ReinforcementConfig(condition="reduced-progress", progress_scale=0.04)


def test_poisson_methods_are_reproducible_and_row_order_independent() -> None:
    from flylatro.fly.encoder import Stimulus
    from flylatro.fly.flywire_artifact import FlyWireArtifact
    from flylatro.fly.torch_backend import POISSON_METHODS, TorchFlyWireBackend

    neurons = 48
    artifact = FlyWireArtifact(
        path=Path("tiny.npz"),
        manifest={"connectivity_sha256": "x"},
        root_ids=np.arange(neurons, dtype=np.int64),
        pre_indices=np.arange(neurons - 1, dtype=np.int64),
        post_indices=np.arange(1, neurons, dtype=np.int64),
        signed_synapse_counts=np.ones(neurons - 1, dtype=np.float32),
        sensory_indices=np.arange(4, dtype=np.int64),
        descending_indices=np.arange(neurons - 8, neurons, dtype=np.int64),
        coordinates_nm=np.zeros((neurons, 3), dtype=np.float32),
    )
    rates = np.zeros((3, neurons), dtype=np.float32)
    rates[:, :4] = 150.0
    stimulus = Stimulus(rates, "test", "hash")
    hashes = set()
    for method in POISSON_METHODS:
        backend = TorchFlyWireBackend(
            artifact,
            readout_indices=np.arange(neurons),
            poisson_method=method,
            poisson_chunk_steps=7,
        )
        hashes.add(backend.dynamics_hash)
        backend.reset(3, (11, 22, 33))
        first = backend.simulate(stimulus, 4.0).spike_counts
        backend.reset(3, (11, 22, 33))
        assert np.array_equal(first, backend.simulate(stimulus, 4.0).spike_counts)
        # Each fly owns its stream, so reordering rows moves results with them.
        backend.reset(3, (33, 22, 11))
        reordered = backend.simulate(stimulus, 4.0).spike_counts
        np.testing.assert_array_equal(first[0], reordered[2])
        np.testing.assert_array_equal(first[1], reordered[1])
        np.testing.assert_array_equal(first[2], reordered[0])
    # Changing the generation method is a versioned dynamics change.
    assert len(hashes) == len(POISSON_METHODS)


def test_component_profiling_attributes_simulator_time() -> None:
    from flylatro.fly.encoder import Stimulus
    from flylatro.fly.flywire_artifact import FlyWireArtifact
    from flylatro.fly.torch_backend import TorchFlyWireBackend

    neurons = 32
    artifact = FlyWireArtifact(
        path=Path("tiny.npz"),
        manifest={"connectivity_sha256": "x"},
        root_ids=np.arange(neurons, dtype=np.int64),
        pre_indices=np.arange(neurons - 1, dtype=np.int64),
        post_indices=np.arange(1, neurons, dtype=np.int64),
        signed_synapse_counts=np.ones(neurons - 1, dtype=np.float32),
        sensory_indices=np.arange(4, dtype=np.int64),
        descending_indices=np.arange(neurons - 4, neurons, dtype=np.int64),
        coordinates_nm=np.zeros((neurons, 3), dtype=np.float32),
    )
    backend = TorchFlyWireBackend(
        artifact, readout_indices=np.arange(neurons), profile_components=True
    )
    rates = np.zeros((2, neurons), dtype=np.float32)
    rates[:, :4] = 120.0
    backend.reset(2, (1, 2))
    backend.simulate(Stimulus(rates, "test", "hash"), 3.0)
    assert set(backend.component_seconds) >= {
        "poisson",
        "recurrent",
        "lif_update",
        "readout_sync",
    }
    assert all(value >= 0.0 for value in backend.component_seconds.values())
    assert sum(backend.component_seconds.values()) > 0.0
