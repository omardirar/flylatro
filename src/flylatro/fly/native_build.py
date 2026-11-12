"""Opt-in build of the native CUDA extension; never silently falls back."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def describe_native():
    import torch
    root = Path(__file__).parent / 'native'
    toolkit = os.environ.get('CUDA_HOME')
    if not toolkit:
        candidate = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cu13'
        if (candidate / 'bin/nvcc').exists():
            toolkit = str(candidate)
    if not toolkit or not (Path(toolkit) / 'bin/nvcc').exists():
        raise RuntimeError('cuda_native requires a compatible CUDA toolkit; set CUDA_HOME and run python -m flylatro.fly.native_build')
    nvcc_version = subprocess.check_output([str(Path(toolkit) / 'bin/nvcc'), '--version'], text=True)
    if f'release {torch.version.cuda},' not in nvcc_version:
        raise RuntimeError(f'nvcc must match Torch CUDA {torch.version.cuda}: {nvcc_version}')
    compiler = os.environ.get('CXX', shutil.which('c++') or 'c++')
    identity = {'version': 'flylatro-native-lif-plastic-v2', 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'nvcc': nvcc_version,
        'compiler': subprocess.check_output([compiler, '--version'], text=True),
        'source_sha256': hashlib.sha256((root / 'lif.cu').read_bytes()).hexdigest(),
        'cuda_flags': ['-O3', '--fmad=false'], 'fast_math': False,
        'architectures': os.environ.get('TORCH_CUDA_ARCH_LIST', 'automatic')}
    return toolkit, identity


def load_native(*, verbose=False):
    import torch
    from torch.utils import cpp_extension
    root = Path(__file__).parent / 'native'
    toolkit, identity = describe_native()
    cpp_extension.CUDA_HOME = toolkit
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    build_dir = Path(os.environ.get('FLYLATRO_NATIVE_BUILD_DIR', 'artefacts/native-build')) / digest
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '2')
    runtime = Path(toolkit) / 'lib/libcudart.so'
    if not runtime.exists():
        versioned = Path(toolkit) / f'lib/libcudart.so.{torch.version.cuda.split(".")[0]}'
        if versioned.exists() and not (build_dir / 'libcudart.so').exists():
            (build_dir / 'libcudart.so').symlink_to(versioned.resolve())
    module = cpp_extension.load(name='flylatro_native_' + digest[:12],
        sources=[str(root / 'lif.cu')], extra_cuda_cflags=identity['cuda_flags'],
        extra_cflags=['-O3'], extra_ldflags=['-L' + str(build_dir.resolve())], build_directory=str(build_dir.resolve()), verbose=verbose)
    (build_dir / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    return module, identity


if __name__ == '__main__':
    _, identity = load_native(verbose=True)
    print(json.dumps(identity, indent=2))
