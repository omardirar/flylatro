"""Shared fixtures for fast, artifact-free regression tests."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np

from flylatro.analysis.corpus import CalibrationCorpus, build_calibration_corpus
from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.learning.protocol_materialize import render_toml

SMOKE_CONFIG = Path("configs/plastic-smoke.toml")


def build_mock_corpus(
    path: Path,
    *,
    seeds: tuple[int, ...] = (900_001, 900_002, 900_003),
    states_per_seed: int = 3,
    navigation_seed: int = 4242,
) -> CalibrationCorpus:
    env = MockArrayBalatroEnv(1, blind_target=18.0, initial_hands=3, initial_discards=2)
    corpus = build_calibration_corpus(
        env,
        environment_seeds=seeds,
        states_per_seed=states_per_seed,
        navigation_seed=navigation_seed,
    )
    corpus.save(path)
    return corpus


def write_config(
    tmp_path: Path,
    *,
    corpus_path: Path | None = None,
    name: str = "test-smoke",
    overrides: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Write a smoke configuration into ``tmp_path`` with optional overrides."""

    raw = tomllib.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    document: dict[str, object] = {"name": name}
    for section in (
        "environment",
        "fly",
        "motor",
        "plasticity",
        "reinforcement",
        "training",
        "curriculum",
        "runtime",
        "calibration",
        "protocol",
    ):
        document[section] = dict(raw.get(section, {}))
    if corpus_path is not None:
        document["calibration"] = {
            **document["calibration"],  # type: ignore[dict-item]
            "corpus_path": str(corpus_path.resolve()),
        }
    for section, values in (overrides or {}).items():
        document[section] = {**document.get(section, {}), **values}  # type: ignore[dict-item]
    # The repository root is two levels above the config path, so keep the same
    # depth the loader expects for relative artefact paths.
    directory = tmp_path / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.toml"
    path.write_text(render_toml(document), encoding="utf-8")
    return path


def tiny_artifact(
    *,
    neurons: int = 24,
    kenyon: tuple[int, ...] = (0, 1, 2),
    mbon: tuple[int, ...] = (4, 5, 6, 7),
    weights: tuple[float, ...] = (3.0, 2.0, 5.0, 1.0),
) -> FlyWireArtifact:
    """A minimal artifact for canonical-candidate and topology unit tests."""

    pre = np.asarray([0, 1, 2, 0], dtype=np.int64)
    post = np.asarray([4, 5, 6, 5], dtype=np.int64)
    types = ["KC"] * 4 + ["MBON-gamma1pedc>a/b", "MBON-a2", "MBON-b1", "MBON-c1"]
    types += ["ALPN"] * 4 + ["DN"] * 4 + ["other"] * (neurons - len(types) - 8)
    return FlyWireArtifact(
        path=Path("tiny.npz"),
        manifest={"connectivity_sha256": "x", "artifact_sha256": "y"},
        root_ids=np.arange(1_000, 1_000 + neurons, dtype=np.int64),
        pre_indices=pre,
        post_indices=post,
        signed_synapse_counts=np.asarray(weights, dtype=np.float32),
        sensory_indices=np.asarray([8, 9, 10, 11], dtype=np.int64),
        descending_indices=np.asarray([12, 13, 14, 15], dtype=np.int64),
        coordinates_nm=np.zeros((neurons, 3), dtype=np.float32),
        kenyon_indices=np.asarray(kenyon, dtype=np.int64),
        mbon_indices=np.asarray(mbon, dtype=np.int64),
        dan_indices=np.asarray([16], dtype=np.int64),
        projection_indices=np.asarray([8, 9, 10, 11], dtype=np.int64),
        kc_mbon_edge_indices=np.arange(4, dtype=np.int64),
        primary_types=np.asarray(types, dtype=np.str_),
    )
