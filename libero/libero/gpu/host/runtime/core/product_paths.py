"""User-owned cold build locations and the tested external MathDx SDK."""
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import uuid

_CURRENT = ContextVar('libero_gpu_product_build', default=None)
_SDK = ContextVar('libero_gpu_product_sdk', default=None)
_MATHDX_URL = ('https://developer.nvidia.com/downloads/compute/cublasdx/redist/'
               'cublasdx/cuda12/nvidia-mathdx-25.12.1-cuda12.tar.gz')
_MATHDX_SHA = 'd905fb32e4baafe7f6cd3ad94ac194d86dcf463481f37221da972b2784fa73ac'


def cache_root():
    explicit = os.environ.get('LIBERO_GPU_CACHE_DIR')
    root = Path(explicit).expanduser() if explicit else Path(
        os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'libero-gpu'
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_directory(label):
    """A unique mutable owner directory; no source/module authority is inherited."""
    current = _CURRENT.get()
    parent = current if current is not None else cache_root() / 'builds'
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=label + '-', dir=parent))


def current_build_root():
    current = _CURRENT.get()
    if current is not None:
        return current
    return build_directory('assembly')


def _sdk_valid(root):
    return all((root / relative).is_file() for relative in (
        'include/cublasdx.hpp', 'include/cusolverdx.hpp', 'lib/libcusolverdx.a'))


def mathdx_root(*, download=True):
    explicit = os.environ.get('LIBERO_MATHDX_ROOT')
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not _sdk_valid(root):
            raise FileNotFoundError('Incomplete LIBERO_MATHDX_ROOT: ' + str(root))
        return root
    roots = []
    for entry in sys.path:
        roots.extend(Path(entry or '.').glob('nvidia/mathdx/25.12'))
    sdk = cache_root() / 'toolchains' / 'mathdx-25.12.1-cuda12'
    roots.extend(sdk.rglob('nvidia/mathdx/25.12') if sdk.exists() else ())
    for root in roots:
        if _sdk_valid(root):
            return root.resolve()
    if not download or os.environ.get('LIBERO_AUTO_DOWNLOAD', '1') == '0':
        raise FileNotFoundError('MathDx 25.12.1 CUDA12 is absent; set LIBERO_MATHDX_ROOT.')
    parent = sdk.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Cross-process serialization; failures never publish a completed SDK.
    import fcntl
    with (parent / '.mathdx.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for root in sdk.rglob('nvidia/mathdx/25.12') if sdk.exists() else ():
            if _sdk_valid(root):
                return root.resolve()
        archive = parent / ('mathdx-download-' + uuid.uuid4().hex + '.tar.gz')
        stage = Path(tempfile.mkdtemp(prefix='mathdx-stage-', dir=parent))
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(_MATHDX_URL, timeout=120) as response, archive.open('wb') as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != _MATHDX_SHA:
                raise ValueError('MathDx archive digest differs from the tested official SDK')
            with tarfile.open(archive, 'r:gz') as bundle:
                for member in bundle.getmembers():
                    destination = (stage / member.name).resolve()
                    if not destination.is_relative_to(stage.resolve()) or member.issym() or member.islnk():
                        raise ValueError('Unsafe MathDx archive member: ' + member.name)
                bundle.extractall(stage)
            root = next((p for p in stage.rglob('nvidia/mathdx/25.12') if _sdk_valid(p)), None)
            if root is None:
                raise ValueError('MathDx archive does not contain the required tested SDK layout')
            if sdk.exists():
                raise FileExistsError('Incomplete SDK directory preserved: ' + str(sdk))
            relative = root.relative_to(stage)
            stage.rename(sdk)
            return sdk / relative
        finally:
            archive.unlink(missing_ok=True)
            if stage.exists():
                # Only this invocation's verified newly-created temporary stage.
                shutil.rmtree(stage)


@contextmanager
def product_build():
    """Bind cold constructors to one private arena without changing process env."""
    previous = _CURRENT.get()
    if previous is not None:
        yield previous
        return
    root = build_directory('environment')
    token = _CURRENT.set(root)
    try:
        yield root
    finally:
        _CURRENT.reset(token)


@contextmanager
def toolchain_scope(sdk):
    sdk=Path(sdk).resolve()
    if not _sdk_valid(sdk):
        raise FileNotFoundError('Incomplete MathDx SDK: '+str(sdk))
    token=_SDK.set(sdk)
    try:yield
    finally:_SDK.reset(token)


def selected_mathdx_root(fallback):
    explicit=os.environ.get('LIBERO_MATHDX_ROOT')
    return Path(explicit).expanduser().resolve() if explicit else (_SDK.get() or Path(fallback))


def nvcc_path():
    explicit=os.environ.get('LIBERO_NVCC')
    selected=explicit or shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    result=Path(selected).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError('CUDA nvcc is absent; install the CUDA Toolkit or set LIBERO_NVCC.')
    return str(result)
