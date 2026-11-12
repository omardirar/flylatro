from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flylatro.analysis import activity_cache
from flylatro.analysis.evidence import CorpusActivity


@pytest.fixture()
def cache_inputs(monkeypatch):
    activity = CorpusActivity(
        kc=np.arange(12, dtype=float).reshape(4, 3),
        mbon=np.arange(8, dtype=float).reshape(4, 2),
        descending=np.empty((4, 0)),
        output=np.arange(8, dtype=float).reshape(4, 2),
        state_labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
        state_hashes=("first", "second", "first", "second"),
        repeats=2, duration_ms=500.0,
    )
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return activity

    monkeypatch.setattr(activity_cache, "record_corpus_activity", record)
    corpus = SimpleNamespace(state_hashes=("first", "second"))
    settings = dict(
        identity={"decision_duration_ms": 500.0, "artifact_sha256": "graph",
                  "calibration_corpus_sha256": "corpus"},
        seed=91, repeats=2, batch_size=2, limit=None,
    )
    return activity, calls, corpus, settings


def test_verified_cache_reuses_exact_activity_without_resimulation(tmp_path, cache_inputs):
    original, calls, corpus, settings = cache_inputs
    path = tmp_path / "activity.npz"
    first, digest = activity_cache.cached_corpus_activity(path, None, corpus, **settings)
    second, second_digest = activity_cache.cached_corpus_activity(path, None, corpus, **settings)
    assert first is original
    assert len(calls) == 1
    assert second_digest == digest and len(digest) == 64
    assert second.state_hashes == original.state_hashes
    assert second.duration_ms == original.duration_ms
    for name in ("kc", "mbon", "descending", "output", "state_labels"):
        np.testing.assert_array_equal(getattr(second, name), getattr(original, name))


@pytest.mark.parametrize("field,value", [("seed", 92), ("repeats", 3), ("batch_size", 1), ("limit", 1)])
def test_cache_refuses_other_sampling_settings(tmp_path, cache_inputs, field, value):
    _, calls, corpus, settings = cache_inputs
    path = tmp_path / "activity.npz"
    activity_cache.cached_corpus_activity(path, None, corpus, **settings)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="identity or sampling"):
        activity_cache.cached_corpus_activity(path, None, corpus, **{**settings, field: value})
    assert len(calls) == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("artifact_sha256", "other-graph"), ("calibration_corpus_sha256", "other-corpus"),
    ("decision_duration_ms", 50.0), ("sensory_mapping_sha256", "other-inputs"),
])
def test_cache_refuses_other_neural_identity(tmp_path, cache_inputs, field, value):
    _, calls, corpus, settings = cache_inputs
    path = tmp_path / "activity.npz"
    activity_cache.cached_corpus_activity(path, None, corpus, **settings)
    changed = {**settings, "identity": {**settings["identity"], field: value}}
    with pytest.raises(ValueError, match="identity or sampling"):
        activity_cache.cached_corpus_activity(path, None, corpus, **changed)
    assert len(calls) == 1


def test_cache_detects_modified_activity(tmp_path, cache_inputs):
    _, calls, corpus, settings = cache_inputs
    path = tmp_path / "activity.npz"
    activity_cache.cached_corpus_activity(path, None, corpus, **settings)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["output"][0, 0] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        activity_cache.cached_corpus_activity(path, None, corpus, **settings)
    assert len(calls) == 1


def test_bad_cache_suffix_is_rejected_before_simulating(tmp_path, cache_inputs):
    _, calls, corpus, settings = cache_inputs
    with pytest.raises(ValueError, match="end in .npz"):
        activity_cache.cached_corpus_activity(tmp_path / "wrong.json", None, corpus, **settings)
    assert not calls


def test_no_cache_retains_existing_simulation_behavior(cache_inputs):
    original, calls, corpus, settings = cache_inputs
    result, digest = activity_cache.cached_corpus_activity(None, None, corpus, **settings)
    assert result is original and digest is None and len(calls) == 1


def test_cache_native_build_identity_is_required_and_invalidated(tmp_path, cache_inputs):
    _, calls, corpus, settings = cache_inputs
    path = tmp_path / 'activity.npz'
    activity_cache.cached_corpus_activity(path, None, corpus, **settings)
    native = {**settings, 'identity': {**settings['identity'], 'fly_implementation_sha256': 'build-one'}}
    with pytest.raises(ValueError, match='identity or sampling'):
        activity_cache.cached_corpus_activity(path, None, corpus, **native)
    native_path = tmp_path / 'native.npz'
    activity_cache.cached_corpus_activity(native_path, None, corpus, **native)
    changed = {**native, 'identity': {**native['identity'], 'fly_implementation_sha256': 'build-two'}}
    with pytest.raises(ValueError, match='identity or sampling'):
        activity_cache.cached_corpus_activity(native_path, None, corpus, **changed)
    assert len(calls) == 2
