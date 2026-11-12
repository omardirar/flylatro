"""Opt-in outgoing-event CUDA extension with a separate implementation identity."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from flylatro.fly.native_build import describe_native


def describe_event():
    toolkit, native = describe_native()
    source = Path(__file__).parent / 'native/event_probe.cu'
    return toolkit, {'version': 'flylatro-outgoing-integral-events-v1',
        'toolchain': native, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'cuda_flags': ['-O3', '--fmad=false'], 'fast_math': False}


def load_event(*, verbose=False):
    from torch.utils import cpp_extension
    toolkit, identity = describe_event()
    cpp_extension.CUDA_HOME = toolkit
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(os.environ.get('FLYLATRO_NATIVE_BUILD_DIR', 'artefacts/native-build')) / digest
    directory.mkdir(parents=True, exist_ok=True)
    runtime = Path(toolkit) / 'lib/libcudart.so'
    if not runtime.exists() and not (directory / 'libcudart.so').exists():
        (directory / 'libcudart.so').symlink_to((Path(toolkit) / 'lib/libcudart.so.13').resolve())
    module = cpp_extension.load(name='flylatro_event_' + digest[:12],
        sources=[str(Path(__file__).parent / 'native/event_probe.cu')],
        extra_cflags=['-O3'], extra_cuda_cflags=identity['cuda_flags'],
        extra_ldflags=['-L' + str(directory.resolve())],
        build_directory=str(directory.resolve()), verbose=verbose)
    (directory / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    return module, identity


if __name__ == '__main__':
    _, identity = load_event(verbose=True)
    print(json.dumps(identity, indent=2))
