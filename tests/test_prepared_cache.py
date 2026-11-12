"""Preparation caching must preserve anatomy and reject stale/tampered payloads."""
import json
from pathlib import Path

import numpy as np
import pytest

from flylatro.fly import flywire_artifact
from flylatro.fly.flywire_artifact import FlyWireArtifact, sha256_file
from flylatro.fly.prepared_cache import load_prepared_artifact, read_layout, write_layout, validate_csr_layout


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(flywire_artifact, 'EXPECTED_NEURONS', 3)
    path = tmp_path / 'source.npz'
    np.savez(path, root_ids=np.asarray([100, 200, 300], dtype=np.int64),
        pre_indices=np.asarray([0, 1], dtype=np.int64), post_indices=np.asarray([1, 2], dtype=np.int64),
        signed_synapse_counts=np.asarray([2., -3.], dtype=np.float32),
        sensory_indices=np.asarray([0], dtype=np.int64), descending_indices=np.asarray([2], dtype=np.int64),
        coordinates_nm=np.zeros((3, 3), dtype=np.float32))
    manifest = {'dataset': 'FlyWire FAFB', 'version': '783', 'schema_version': 1,
        'artifact_sha256': sha256_file(path), 'connectivity_sha256': 'source-connectivity'}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    return path


def test_prepared_artifact_is_verified_read_only_and_preserves_source(source, tmp_path):
    original = FlyWireArtifact.load(source)
    first = load_prepared_artifact(source, tmp_path / 'prepared')
    second = load_prepared_artifact(source, tmp_path / 'prepared')
    for name in ('root_ids', 'pre_indices', 'post_indices', 'signed_synapse_counts', 'coordinates_nm'):
        np.testing.assert_array_equal(getattr(first, name), getattr(original, name))
        np.testing.assert_array_equal(getattr(second, name), getattr(original, name))
        assert not getattr(first, name).flags.writeable
    with pytest.raises(ValueError, match='read-only'):
        first.signed_synapse_counts[0] = 100.
    first.manifest['dataset'] = 'changed by consumer'
    assert second.manifest['dataset'] == 'FlyWire FAFB'


def test_prepared_artifact_rejects_changed_cache_payload(source, tmp_path):
    directory = tmp_path / 'prepared'
    load_prepared_artifact(source, directory)
    target = next(directory.glob('*/signed_synapse_counts.npy'))
    np.save(target, np.asarray([99., -3.], dtype=np.float32))
    with pytest.raises(ValueError, match='cache SHA-256'):
        load_prepared_artifact(source, directory)


def test_prepared_artifact_rejects_changed_original_bytes(source, tmp_path):
    load_prepared_artifact(source, tmp_path / 'prepared')
    with source.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='artifact SHA-256'):
        load_prepared_artifact(source, tmp_path / 'prepared')


def test_layout_cache_is_immutable_content_bound_and_detects_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv('FLYLATRO_LAYOUT_CACHE_DIR', str(tmp_path / 'layout'))
    identity = {'source': 'one', 'backend': 'one', 'precision': 'float32', 'index_bits': 32}
    arrays = {'crow': np.asarray([0, 1, 2], dtype=np.int32),
        'col': np.asarray([1, 0], dtype=np.int32), 'values': np.asarray([2., -3.], dtype=np.float32)}
    assert read_layout(identity) is None
    write_layout(identity, arrays, connectivity_hash='canonical', neurons=2)
    metadata, loaded = read_layout(identity)
    assert metadata['connectivity_hash'] == 'canonical'
    for name in arrays:
        np.testing.assert_array_equal(loaded[name], arrays[name])
        assert not loaded[name].flags.writeable
    for field in ('source', 'backend', 'precision', 'index_bits'):
        assert read_layout({**identity, field: 'different'}) is None
    file = next((tmp_path / 'layout').glob('*.npz'))
    with np.load(file, allow_pickle=False) as data:
        changed = {name: data[name].copy() for name in data.files}
    changed['values'][0] += 1
    np.savez(file, **changed)
    with pytest.raises(ValueError, match='SHA-256'):
        read_layout(identity)


def test_cached_layout_checks_ranges_width_and_plastic_separation():
    arrays = {'crow': np.asarray([0, 1, 2], dtype=np.int32),
        'col': np.asarray([1, 0], dtype=np.int32), 'values': np.asarray([0., -3.], dtype=np.float32),
        'positions': np.asarray([0], dtype=np.int64)}
    validate_csr_layout(arrays, 2, plastic_edges=1)
    for field, changed, message in [
        ('col', np.asarray([2, 0], dtype=np.int32), 'outside neuron'),
        ('crow', np.asarray([0, 2, 1], dtype=np.int32), 'row offsets'),
        ('col', arrays['col'].astype(np.int64), 'index width'),
        ('values', np.asarray([2., -3.], dtype=np.float32), 'double counts'),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_csr_layout({**arrays, field: changed}, 2, plastic_edges=1)


def test_cached_backend_skips_coo_preparation_and_keeps_anatomy_private(source, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from flylatro.fly.execution import build_backend
    from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
    from flylatro.fly.torch_backend import TorchFlyWireBackend
    monkeypatch.setenv('FLYLATRO_LAYOUT_CACHE_DIR', str(tmp_path / 'layout'))
    artifact = load_prepared_artifact(source, tmp_path / 'prepared')
    topology = PlasticEdgeTopology.synthetic(np.asarray([0]), np.asarray([1]), np.asarray([2.], dtype=np.float32))
    cold = build_backend('torch_csr', artifact, topology, readout_indices=[0, 1, 2])
    expected_hash = cold.connectivity_hash
    cold.set_plastic_efficacy(np.asarray([.75], dtype=np.float32))
    expected = cold.propagate_once(np.asarray([1., 1., 0.], dtype=np.float32))
    cold.weights.values().add_(100.)
    def forbidden(*args, **kwargs):
        raise AssertionError('warm layout must not construct COO or coalesce again')
    monkeypatch.setattr(TorchFlyWireBackend, '_initialize_connectivity', forbidden)
    warm = build_backend('torch_csr', artifact, topology, readout_indices=[0, 1, 2])
    warm.set_plastic_efficacy(np.asarray([.75], dtype=np.float32))
    assert warm.connectivity_hash == expected_hash
    assert warm.layout_preparation['incoming_hit']
    np.testing.assert_array_equal(warm.propagate_once(np.asarray([1., 1., 0.], dtype=np.float32)), expected)
    assert warm.weights.crow_indices().dtype == torch.int32


def test_event_csr_bound_handles_empty_rows_and_matches_original_check():
    from flylatro.fly.execution import check_integral_event_csr, check_integral_event_weights
    crow = np.asarray([0, 0, 2, 2, 3], dtype=np.int32)
    values = np.asarray([2., -3., 5.], dtype=np.float32)
    assert check_integral_event_csr(crow, values) == check_integral_event_weights(
        np.asarray([1, 1, 3]), values, 4)
    assert check_integral_event_csr(np.zeros(4, dtype=np.int32), np.empty(0, dtype=np.float32)) == 0
    with pytest.raises(ValueError, match='integral'):
        check_integral_event_csr(crow, values + .1)
    with pytest.raises(ValueError, match='accumulation bound'):
        check_integral_event_csr(crow, np.asarray([2**24, 0., 0.], dtype=np.float32))
