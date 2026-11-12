"""Hash-verified reward-free neural samples for iterative motor calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from flylatro.analysis.evidence import CorpusActivity, record_corpus_activity


ACTIVITY_CACHE_VERSION = "reward-free-corpus-activity-cache-v1"
IDENTITY_FIELDS = (
    "artifact_sha256", "population_sha256", "fly_connectivity_sha256",
    "fly_dynamics_sha256", "plastic_topology_sha256", "minimum_synapse_count",
    "sensory_mapping_sha256", "sensory_feature_contract_sha256",
    "motor_candidate_set_sha256", "output_mode", "decision_duration_ms",
    "device", "backend_evidence", "calibration_corpus_sha256", "simulator_version",
)


def _digest(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name, values in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def cached_corpus_activity(
    path: Path | None,
    stack: Any,
    corpus: Any,
    *,
    identity: Mapping[str, Any],
    seed: int,
    repeats: int,
    batch_size: int,
    limit: int | None,
) -> tuple[CorpusActivity, str | None]:
    if path is not None and path.suffix != ".npz":
        raise ValueError("neural activity cache path must end in .npz")
    expected = {
        "version": ACTIVITY_CACHE_VERSION,
        "identity": {key: identity.get(key) for key in IDENTITY_FIELDS},
        "seed": seed, "repeats": repeats, "batch_size": batch_size, "limit": limit,
        "efficacy_policy": "fixed-initial-unit-efficacy-no-learning",
    }
    if identity.get('fly_implementation_sha256') is not None:
        expected['identity']['fly_implementation_sha256'] = identity['fly_implementation_sha256']
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            claimed = metadata.pop("sha256")
            arrays = {name: data[name].copy() for name in ("kc", "mbon", "descending", "output", "state_labels")}
        if _digest(metadata, arrays) != claimed:
            raise ValueError("neural activity cache SHA-256 mismatch")
        if metadata != expected:
            raise ValueError("neural activity cache identity or sampling settings differ")
        labels = arrays["state_labels"]
        activity = CorpusActivity(
            **arrays,
            state_hashes=tuple(corpus.state_hashes[int(index)] for index in labels),
            repeats=repeats, duration_ms=float(identity["decision_duration_ms"]),
        )
        return activity, claimed
    activity = record_corpus_activity(
        stack, corpus, seed=seed, repeats=repeats, batch_size=batch_size, limit=limit,
    )
    if path is None:
        return activity, None
    arrays = {name: np.asarray(getattr(activity, name)) for name in ("kc", "mbon", "descending", "output", "state_labels")}
    digest = _digest(expected, arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata=json.dumps({**expected, "sha256": digest}, sort_keys=True))
    return activity, digest
