"""Private local build authority for exact, completed native build products.

This authorizes reuse of compiler output, never numerical qualification or an
owner/binding. The local user's private signing key lives outside the product
cache. The trust boundary is the user's account, not hostile code in that same
account. Self-hashed legacy receipts are deliberately not accepted here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import time
import uuid
import base64


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


_scope = ContextVar('libero_product_cache_authority', default=None)


@contextmanager
def product_cache_scope(cache_root, *, authority_root=None, max_bytes=4*1024**3,
                        max_entries=4096, lock_timeout=1800):
    """Enable exact build reuse only for the enclosed public cold constructor.

    Combine with native.strict_build_receipts_scope(True). This is deliberately
    independent of LIBERO_BUILD_RECEIPTS and the historical diagnostic cache.
    """
    authority = ProductBuildAuthority(cache_root, authority_root=authority_root,
        max_bytes=max_bytes, max_entries=max_entries, lock_timeout=lock_timeout)
    token = _scope.set(authority)
    try:
        yield authority
    finally:
        _scope.reset(token)


def current_product_authority():
    return _scope.get()


def compiler_environment_record(environment=None):
    """Bind compiler/linker inputs without unrelated worker-specific variables."""
    names = {'PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH',
        'LD_LIBRARY_PATH', 'LD_PRELOAD', 'COMPILER_PATH', 'GCC_EXEC_PREFIX',
        'SOURCE_DATE_EPOCH', 'CUDA_HOME', 'CUDA_PATH', 'CUDAHOSTCXX', 'CUDACXX',
        'LIBERO_NVCC', 'LIBERO_MATHDX_ROOT', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ'}
    runtime_cache_names = {'CUDA_CACHE_PATH', 'CUDA_CACHE_MAXSIZE', 'CUDA_CACHE_DISABLE'}
    selected = {k:v for k,v in (os.environ if environment is None else environment).items()
                if k not in runtime_cache_names and
                    (k in names or k.startswith(('NVCC_', 'CUDA_', 'LC_')))}
    return {'schema':'compiler-input-environment-v1', 'sha256':_digest(_canonical(selected))}


@contextmanager
def _lock(path, timeout):
    # flock releases on process death; no stale PID/lock-file recovery heuristic.
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()-start >= timeout:
                    raise TimeoutError('native product build lock timeout: '+str(path))
                time.sleep(.05)
        yield
    finally:
        os.close(fd)


def _private_directory(path):
    path = Path(path).expanduser().absolute()
    if not path.exists():
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError('build signing authority directory must be owned/private: '+str(path))
    return path


class ProductBuildAuthority:
    """Publish only through a successful builder and current-input validator.

    ``obtain`` runs both callbacks under a per-key process lock. ``validate``
    must verify actual source/dependencies/toolchain/archive, and ``exports``
    must load the *new private product* and check the requested symbols.
    Unsigned, incomplete, corrupt or mismatched cache entries are misses. No
    successful build receipt exists until both validations succeed.
    """
    def __init__(self, cache_root, *, authority_root=None, max_bytes=4*1024**3,
                 max_entries=4096, lock_timeout=1800):
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.root = self.cache_root/'signed-v1'
        self.root.mkdir(parents=True, exist_ok=True)
        self.authority_root = _private_directory(authority_root or
            Path.home()/'.local/share/libero-gpu/build-authority')
        authority_location = self.authority_root.resolve()
        if authority_location == self.cache_root or authority_location.is_relative_to(self.cache_root):
            raise ValueError('signing key must live outside the product cache')
        if type(max_bytes) is not int or max_bytes < 1 or type(max_entries) is not int or max_entries < 1:
            raise ValueError('positive native product cache bounds required')
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.lock_timeout = float(lock_timeout)
        key_path = self.authority_root/'signing.key'
        with _lock(self.authority_root/'key.lock', self.lock_timeout):
            if not key_path.exists():
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                    getattr(os, 'O_NOFOLLOW', 0), 0o600)
                try:
                    os.write(fd, os.urandom(32)); os.fsync(fd)
                finally:
                    os.close(fd)
            fd = os.open(key_path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise PermissionError('build signing key must be owned/private')
                self._key = os.read(fd, 33)
            finally:
                os.close(fd)
        if len(self._key) != 32:
            raise ValueError('invalid build signing key')

    def _signature(self, receipt):
        return hmac.new(self._key, _canonical(receipt), hashlib.sha256).hexdigest()

    def _read(self, folder, key, identity, required_symbols):
        try:
            envelope = json.loads((folder/'receipt.json').read_bytes())
            receipt = envelope['receipt']; signature = envelope['signature']
            if type(receipt) is not dict:
                return None
            if type(signature) is not str or not hmac.compare_digest(self._signature(receipt), signature):
                return None
            raw = (folder/'product.so').read_bytes()
            if (receipt.get('schema') != 'authorized-native-product-v1' or
                receipt.get('build_key') != key or receipt.get('identity') != identity or
                receipt.get('required_symbols') != list(required_symbols) or
                receipt.get('binary_sha256') != _digest(raw) or receipt.get('binary_bytes') != len(raw) or
                receipt.get('completed') is not True):
                return None
            return raw, receipt
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _publish(self, folder, key, identity, symbols, raw):
        receipt = {'schema':'authorized-native-product-v1', 'build_key':key,
            'identity':identity, 'required_symbols':list(symbols),
            'binary_sha256':_digest(raw), 'binary_bytes':len(raw), 'completed':True}
        envelope = {'receipt':receipt, 'signature':self._signature(receipt)}
        encoded = _canonical(envelope)
        # Bounded storage without deleting any existing evidence or old cache.
        # Full cache still allows the private build to run, but grants no future
        # persistent reuse until the user explicitly makes room.
        with _lock(self.root/'catalog.lock', self.lock_timeout):
            products = list(self.root.glob('*/product.so'))
            occupied = sum(p.stat().st_size+p.with_name('receipt.json').stat().st_size
                           for p in products if p.with_name('receipt.json').is_file())
            prior = (folder/'product.so')
            replacing = prior.exists()
            old = prior.stat().st_size if replacing else 0
            if replacing and (folder/'receipt.json').exists():
                old += (folder/'receipt.json').stat().st_size
            if occupied-old+len(raw)+len(encoded) > self.max_bytes or len(products)+(not replacing) > self.max_entries:
                return receipt, False
            nonce = uuid.uuid4().hex
            for name, content in ((nonce+'.so',raw),(nonce+'.json',encoded)):
                with (folder/name).open('xb') as stream:
                    stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.replace(folder/(nonce+'.so'), folder/'product.so')
            os.replace(folder/(nonce+'.json'), folder/'receipt.json')
        return receipt, True

    def obtain(self, identity, required_symbols, *, build, validate, exports):
        """Return bytes and signed receipt after exact current cold admission.

        build() returns bytes after actual compiler/linker commands succeed;
        validate() rejects changed inputs; exports(bytes) materializes and loads
        a fresh private path, and must reject missing ABI symbols.
        """
        # Export contracts participate in the cache address even if the source
        # accidentally omits a public symbol from a caller's expected ABI.
        identity = json.loads(_canonical(identity))
        symbols = tuple(required_symbols)
        if any(type(s) is not str or not s for s in symbols):
            raise ValueError('native product symbols must be nonempty strings')
        key = _digest(_canonical({'identity':identity, 'required_symbols':list(symbols)}))
        folder = self.root/key; folder.mkdir(exist_ok=True)
        with _lock(folder/'build.lock', self.lock_timeout):
            validate()
            found = self._read(folder, key, identity, symbols)
            if found is not None:
                raw, receipt = found
                exports(raw)
                validate()
                return raw, receipt, {'hit':True, 'persisted':True, 'key':key}
            raw = build()
            if type(raw) is not bytes:
                raise TypeError('native builder must return completed product bytes')
            exports(raw); validate()
            receipt, persisted = self._publish(folder, key, identity, symbols, raw)
            return raw, receipt, {'hit':False, 'persisted':persisted, 'key':key}

    def obtain_artifacts(self, identity, required_symbols, *, names, build, validate, exports):
        """Authorize an indivisible set of completed compiler artifacts.

        All members (for example the actual .so and actual PTX) are signed as
        one bundle. An unsigned sidecar or missing member cannot be substituted.
        ``exports`` receives bytes by basename and must validate new private
        files, including the existing owner/module/ABI/resource gates.
        """
        names=tuple(sorted(names))
        if not names or any(type(n) is not str or Path(n).name!=n or n in ('','.','..') for n in names):
            raise ValueError('native artifact names must be safe basenames')
        def encode_build():
            artifacts=build()
            if type(artifacts) is not dict or tuple(sorted(artifacts))!=names or any(type(v) is not bytes for v in artifacts.values()):
                raise ValueError('incomplete native build artifact set')
            return _canonical({'schema':'native-artifact-bundle-v1',
                'artifacts':{n:base64.b64encode(artifacts[n]).decode('ascii') for n in names}})
        def decode(raw):
            bundle=json.loads(raw)
            if bundle.get('schema')!='native-artifact-bundle-v1' or tuple(sorted(bundle['artifacts']))!=names:
                raise ValueError('native artifact bundle identity mismatch')
            return {n:base64.b64decode(bundle['artifacts'][n],validate=True) for n in names}
        identity={'artifact_names':list(names),'build':identity,'schema':'native-multi-product-v1'}
        raw,receipt,status=self.obtain(identity,required_symbols,build=encode_build,
            validate=validate,exports=lambda raw:exports(decode(raw)))
        return decode(raw),receipt,status
