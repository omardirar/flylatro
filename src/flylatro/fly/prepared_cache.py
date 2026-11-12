"""Hash-verified immutable artifact and graph-layout preparation caches.

Both are opt-in. Cache payloads contain source anatomy and zeroed fixed plastic
edges, never learned efficacies, stochastic activity or independent learner
state. Every load verifies source identity and payload hashes.
"""
from __future__ import annotations

from dataclasses import fields
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from flylatro.fly.flywire_artifact import FlyWireArtifact, sha256_file

PREPARATION_VERSION = 'immutable-flywire-preparation-v1'
LAYOUT_VERSION = 'fixed-plastic-separated-csr32-v1'


def _identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_prepared_artifact(path: Path, cache_dir: Path | None = None) -> FlyWireArtifact:
    """Reuse read-only mapped arrays after verifying original bytes and manifest."""
    directory = cache_dir or os.environ.get('FLYLATRO_ARTIFACT_CACHE_DIR')
    if not directory:
        artifact = FlyWireArtifact.load(path)
        for field in fields(FlyWireArtifact):
            array = getattr(artifact, field.name)
            if isinstance(array, np.ndarray):
                array.flags.writeable = False
        return artifact
    path = path.resolve()
    manifest_bytes = path.with_suffix('.manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    source_sha = sha256_file(path)
    if source_sha != manifest.get('artifact_sha256'):
        raise ValueError('artifact SHA-256 mismatch before preparation cache')
    identity = {'version': PREPARATION_VERSION, 'artifact_sha256': source_sha,
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}
    destination = Path(directory) / _identity_digest(identity)
    names = [field.name for field in fields(FlyWireArtifact) if field.name not in ('path', 'manifest')]
    if not destination.exists():
        artifact = FlyWireArtifact.load(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=destination.name + '-', dir=destination.parent))
        try:
            metadata = {'identity': identity, 'arrays': {}}
            for name in names:
                array = np.asarray(getattr(artifact, name))
                file = staging / (name + '.npy')
                np.save(file, array, allow_pickle=False)
                metadata['arrays'][name] = {'sha256': sha256_file(file),
                    'shape': list(array.shape), 'dtype': array.dtype.str}
            if sha256_file(path) != source_sha or path.with_suffix('.manifest.json').read_bytes() != manifest_bytes:
                raise ValueError('source changed during artifact preparation')
            (staging / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True) + '\n')
            try:
                staging.rename(destination)
            except OSError as error:
                # Another process may publish a nonempty directory first.
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not destination.is_dir():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    metadata = json.loads((destination / 'metadata.json').read_text())
    if metadata.get('identity') != identity or set(metadata.get('arrays', {})) != set(names):
        raise ValueError('prepared artifact cache identity differs')
    arrays = {}
    for name in names:
        file = destination / (name + '.npy')
        expected = metadata['arrays'][name]
        if sha256_file(file) != expected['sha256']:
            raise ValueError('prepared artifact cache SHA-256 mismatch')
        array = np.load(file, mmap_mode='r', allow_pickle=False)
        if list(array.shape) != expected['shape'] or array.dtype.str != expected['dtype']:
            raise ValueError('prepared artifact cache array schema differs')
        arrays[name] = array
    result = FlyWireArtifact(path=path, manifest=manifest, **arrays)
    result.validate()
    return result


def layout_identity(backend, *, direction='incoming'):
    """Relevant immutable geometry identity; no duration/RNG/learned values."""
    if not os.environ.get('FLYLATRO_LAYOUT_CACHE_DIR'):
        return None
    source = backend.artifact.manifest.get('artifact_sha256')
    if any(array.flags.writeable for array in (backend.artifact.root_ids,
            backend.artifact.pre_indices, backend.artifact.post_indices,
            backend.artifact.signed_synapse_counts)):
        # Mutable hand-built/source arrays may no longer represent disk bytes.
        return None
    if not source or not backend.artifact.path.exists():
        # Hand-built development circuits have no verified immutable source.
        return None
    if sha256_file(backend.artifact.path) != source:
        raise ValueError('artifact SHA-256 mismatch before layout preparation')
    return {'version': LAYOUT_VERSION, 'artifact_sha256': source,
        'population_sha256': backend.artifact.population_hash,
        'source_connectivity_sha256': backend.artifact.connectivity_hash,
        'plastic_topology_sha256': backend.plastic_topology.sha256,
        'minimum_synapse_count': backend.plastic_topology.minimum_synapse_count,
        'topology_procedure': backend.topology_procedure,
        'backend': backend.backend_version, 'precision': 'float32', 'index_bits': 32,
        'torch': backend.torch.__version__, 'cuda': backend.torch.version.cuda,
        'numpy': np.__version__,
        'gpu_capability': (list(backend.torch.cuda.get_device_capability(backend.device))
            if backend.device.type == 'cuda' else None),
        'native_build': getattr(backend, 'native_identity', None),
        'device_kind': backend.device.type, 'direction': direction,
        'fixed_weight_policy': 'anatomical-values-with-plastic-positions-zeroed'}


def _array_digest(metadata, arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name, array in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_layout(identity):
    if identity is None:
        return None
    file = Path(os.environ['FLYLATRO_LAYOUT_CACHE_DIR']) / (_identity_digest(identity) + '.npz')
    if not file.exists():
        return None
    with np.load(file, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
        claimed = metadata.pop('sha256')
        arrays = {name: data[name].copy() for name in data.files if name != 'metadata'}
    if metadata.get('identity') != identity:
        raise ValueError('graph layout cache identity differs')
    if _array_digest(metadata, arrays) != claimed:
        raise ValueError('graph layout cache SHA-256 mismatch')
    for array in arrays.values():
        array.flags.writeable = False
    return metadata, arrays


def validate_csr_layout(arrays, neurons, *, plastic_edges=None):
    """Check every loaded index range and fixed/plastic separation."""
    crow, col, values = (arrays[name] for name in ('crow', 'col', 'values'))
    if not 0 < neurons <= 2**31 - 1 or len(values) > 2**31 - 1:
        raise ValueError('cached CSR dimensions exceed signed int32')
    if crow.dtype != np.int32 or col.dtype != np.int32 or values.dtype != np.float32:
        raise ValueError('cached CSR precision or index width differs')
    if crow.shape != (neurons + 1,) or col.ndim != 1 or values.shape != col.shape:
        raise ValueError('cached CSR dimensions differ')
    if crow[0] != 0 or crow[-1] != len(values) or np.any(crow[1:] < crow[:-1]):
        raise ValueError('cached CSR row offsets are invalid')
    if col.size and (col.min() < 0 or col.max() >= neurons):
        raise ValueError('cached CSR column index outside neuron range')
    if not np.isfinite(values).all():
        raise ValueError('cached CSR weights are not finite')
    if plastic_edges is not None:
        positions = arrays['positions']
        if positions.dtype != np.int64 or positions.shape != (plastic_edges,):
            raise ValueError('cached plastic positions differ from topology')
        if positions.size and (positions.min() < 0 or positions.max() >= len(values)):
            raise ValueError('cached plastic position outside fixed graph')
        if np.any(values[positions] != 0):
            raise ValueError('cached fixed graph double counts plastic edges')


def write_layout(identity, arrays, *, connectivity_hash, neurons):
    if identity is None:
        return
    directory = Path(os.environ['FLYLATRO_LAYOUT_CACHE_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (_identity_digest(identity) + '.npz')
    metadata = {'identity': identity, 'connectivity_hash': connectivity_hash, 'neurons': neurons}
    metadata['sha256'] = _array_digest(metadata, arrays)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.stem + '-', suffix='.npz', dir=directory)
    os.close(descriptor)
    staging = Path(temporary)
    try:
        np.savez(staging, **arrays, metadata=json.dumps(metadata, sort_keys=True))
        staging.replace(destination)
    finally:
        if staging.exists():
            staging.unlink()
