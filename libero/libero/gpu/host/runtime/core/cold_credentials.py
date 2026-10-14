"""OS protected digest leases; unsupported or broken leases use strict reads.

No timestamp is a content certificate. Windows handles deny write/delete.
Linux read leases block a conflicting writer, signal a private service thread,
and report the break synchronously through F_GETLEASE. Every reuse also checks
the actual path's inode: rename, unlink and symlink replacement are not leases.
This module imports no Warp/CUDA code and installs no process signal handler.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import queue
import signal
import stat
import struct
import sys
import threading
from contextlib import contextmanager


def _identity(s):
    return s.st_dev, s.st_ino


def _read_digest(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


class _FileLease:
    def __init__(self, path, fd, backend, service=None):
        self.path = path
        self.fd = fd
        self.backend = backend
        self.service = service
        self.identity = _identity(os.fstat(fd))
        self.lock = threading.RLock()
        self.active = True
        self.digest = _read_digest(fd)

    def valid(self):
        with self.lock:
            if not self.active:
                return False
            if self.backend == 'linux-read-lease':
                import fcntl
                try:
                    if fcntl.fcntl(self.fd, fcntl.F_GETLEASE) != fcntl.F_RDLCK:
                        return False
                except OSError:
                    return False
            try:
                return _identity(os.stat(self.path)) == self.identity
            except OSError:
                return False

    def _close(self):
        with self.lock:
            if not self.active:
                return
            # Publish revocation BEFORE permitting the waiting writer.
            self.active = False
            if self.backend == 'linux-read-lease':
                import fcntl
                try:
                    fcntl.fcntl(self.fd, fcntl.F_SETLEASE, fcntl.F_UNLCK)
                except OSError:
                    pass
            os.close(self.fd)

    def close(self):
        if self.service is None:
            self._close()
        else:
            self.service.call('close', self)


class _LinuxLeaseService:
    """Thread-directed blocked SIGIO; application handlers/masks are untouched."""
    def __init__(self):
        self.requests = queue.Queue()
        self.leases = {}
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='source-read-leases', daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.error is not None:
            raise OSError('Linux lease signal isolation unavailable') from self.error

    def call(self, operation, value):
        result = {}
        done = threading.Event()
        self.requests.put((operation, value, result, done))
        done.wait()
        if 'error' in result:
            raise result['error']
        return result.get('value')

    def _create(self, path):
        import fcntl
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError('digest leases require a regular file')
            # Linux f_owner_ex: F_OWNER_TID=0, targeting only this blocked thread.
            fcntl.fcntl(fd, getattr(fcntl, 'F_SETOWN_EX', 15),
                        struct.pack('ii', 0, threading.get_native_id()))
            fcntl.fcntl(fd, fcntl.F_SETSIG, signal.SIGIO)
            fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_RDLCK)
            lease = _FileLease(path, fd, 'linux-read-lease', self)
            if not lease.valid():
                lease._close()
                raise OSError('source lease was broken during cold authentication')
            self.leases[fd] = lease
            return lease
        except BaseException:
            # _close may already have closed it.
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def _drain_breaks(self):
        import fcntl
        signaled = False
        while signal.sigtimedwait({signal.SIGIO}, 0) is not None:
            signaled = True
        if not signaled:
            return
        for fd, lease in tuple(self.leases.items()):
            with lease.lock:
                if not lease.active:
                    self.leases.pop(fd, None)
                    continue
                try:
                    broken = fcntl.fcntl(fd, fcntl.F_GETLEASE) != fcntl.F_RDLCK
                except OSError:
                    broken = True
                if broken:
                    lease._close()
                    self.leases.pop(fd, None)

    def _run(self):
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGIO})
        except BaseException as error:
            self.error = error
            self.ready.set()
            return
        self.ready.set()
        while True:
            self._drain_breaks()
            try:
                operation, value, result, done = self.requests.get(timeout=0.02)
            except queue.Empty:
                continue
            try:
                if operation == 'create':
                    result['value'] = self._create(value)
                else:
                    fd = value.fd
                    value._close()
                    self.leases.pop(fd, None)
            except BaseException as error:
                result['error'] = error
            finally:
                done.set()


_service = None
_service_lock = threading.Lock()


class _WindowsBridge:
    """WSL interop helper owns Windows FileShare.Read handles until stdin EOF."""
    def __init__(self):
        import base64
        import subprocess
        script = Path(__file__).with_name('winlease_bridge.ps1').read_text()
        encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        executable = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
        arguments = [executable, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded]
        options = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       text=True, encoding='utf-8', bufsize=1)
        try:
            self.process = subprocess.Popen(arguments, **options)
        except OSError as error:
            if error.errno != 8:
                raise
            # Same /init interpreter and preserved argv0 as WSLInterop binfmt
            # P registration. A transient missing binfmt entry needs no global
            # configuration change. All unsupported interop still falls back.
            self.process = subprocess.Popen(['/init', executable] + arguments, **options)
        self.lock = threading.RLock()
        self.serial = 0
        self.request(dict(op='ping'))
        self._namespace = Path('/proc/self/mountinfo').read_bytes()
        self._scope = threading.local()

    def namespace_unchanged(self):
        if getattr(self._scope, 'depth', 0):
            return self._scope.valid
        return Path('/proc/self/mountinfo').read_bytes() == self._namespace

    @contextmanager
    def namespace_scope(self):
        depth = getattr(self._scope, 'depth', 0)
        if not depth:
            self._scope.valid = Path('/proc/self/mountinfo').read_bytes() == self._namespace
        self._scope.depth = depth + 1
        try:
            yield
        finally:
            self._scope.depth = depth

    def request(self, payload):
        import json
        import select
        with self.lock:
            if self.process.poll() is not None:
                raise OSError('Windows source lease helper exited')
            try:
                self.process.stdin.write(json.dumps(payload) + '\n')
                self.process.stdin.flush()
                if not select.select([self.process.stdout], [], [], 10)[0]:
                    self.process.kill()
                    raise OSError('Windows source lease helper did not acknowledge')
                line = self.process.stdout.readline()
                result = json.loads(line)
            except (BrokenPipeError, ValueError) as error:
                raise OSError('Windows source lease helper protocol failed') from error
            if not result.get('ok'):
                raise OSError(result.get('error', 'Windows source lease failed'))
            return result

    def acquire(self, path):
        import re
        matched = re.fullmatch(r'/mnt/([a-zA-Z])/(.*)', path)
        if matched is None:
            raise OSError('Windows bridge requires a drive-mounted path')
        # A newly observed namespace is not permission to trust an arbitrary
        # overmount. Prove this path still resolves to the real drive's drvfs
        # root. Bind mounts or a different filesystem use strict current bytes.
        mounts = []
        for line in self._namespace.decode().splitlines():
            fields = line.split()
            mount = fields[4]
            if path == mount or path.startswith(mount.rstrip('/') + '/'):
                separator = fields.index('-')
                mounts.append((len(mount), mount, fields[separator + 1],
                               fields[separator + 2], fields[separator + 3]))
        actual = max(mounts, default=None)
        expected_mount = '/mnt/' + matched[1].lower()
        if (actual is None or actual[1] != expected_mount or actual[2] != '9p'
                or not actual[3].upper().startswith(matched[1].upper() + ':')
                or 'aname=drvfs' not in actual[4]):
            raise OSError('source path is not the proven drive root; strict authentication required')
        windows_path = matched[1].upper() + ':\\' + matched[2].replace('/', '\\')
        with self.lock:
            self.serial += 1
            identifier = str(self.serial)
            result = self.request(dict(op='open', id=identifier, path=windows_path))
        try:
            lease = _BridgeLease(path, identifier, result['digest'], self)
            if not lease.valid():
                lease.close()
                raise OSError('source path changed during Windows lease acquisition')
            return lease
        except BaseException:
            self.request(dict(op='close', id=identifier))
            raise

    def shutdown(self):
        with self.lock:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    self.process.kill()


class _BridgeLease:
    def __init__(self, path, identifier, digest, bridge):
        self.path = path
        self.identifier = identifier
        self.digest = digest
        self.bridge = bridge
        self.identity = _identity(os.stat(path))
        self.backend = 'wsl-windows-share-read'
        self.service = None
        self.active = True
        self.lock = threading.RLock()

    def valid(self):
        with self.lock:
            if not self.active or self.bridge.process.poll() is not None:
                return False
            # Helper holds non-reparse file AND all ancestors without write or
            # delete sharing. mountinfo catches any Linux-side overmount; the
            # exact namespace check is shared only within this validation call.
            return self.bridge.namespace_unchanged()

    def _close(self):
        with self.lock:
            if self.active:
                self.active = False
                try:
                    self.bridge.request(dict(op='close', id=self.identifier))
                except OSError:
                    pass

    close = _close


_bridge = None
_bridge_lock = threading.Lock()


def _bridge_acquire(path):
    global _bridge
    with _bridge_lock:
        if _bridge is not None and not _bridge.namespace_unchanged():
            _bridge.shutdown()
        if _bridge is None or _bridge.process.poll() is not None:
            _bridge = _WindowsBridge()
    return _bridge.acquire(path)

def _acquire(path):
    if os.name == 'nt':
        import ctypes as ct
        import msvcrt
        kernel = ct.WinDLL('kernel32', use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [ct.c_wchar_p, ct.c_uint32, ct.c_uint32, ct.c_void_p,
                           ct.c_uint32, ct.c_uint32, ct.c_void_p]
        create.restype = ct.c_void_p
        handle = create(path, 0x80000000, 1, None, 3, 0x80, None)
        if handle == ct.c_void_p(-1).value:
            raise ct.WinError(ct.get_last_error())
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            kernel.CloseHandle.argtypes = [ct.c_void_p]
            kernel.CloseHandle(handle)
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError('digest leases require a regular file')
            lease = _FileLease(path, fd, 'windows-share-read')
            if not lease.valid():
                lease._close()
                raise OSError('source path changed during cold authentication')
            return lease
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    if sys.platform.startswith('linux'):
        # drvfs/9p leases are not assumed to work. The Windows-side handle
        # demonstrably denies write/rename through both Windows and WSL.
        if path.startswith('/mnt/') and os.environ.get('WSL_DISTRO_NAME'):
            return _bridge_acquire(path)
        global _service
        with _service_lock:
            if _service is None:
                _service = _LinuxLeaseService()
        return _service.call('create', path)
    raise OSError('no proven OS digest lease on this platform')


class FileDigestLeasePool:
    """A process-local, explicitly revocable source/report/binary credential.

    Fallback always reads current bytes. A failed/broken acquisition is NOT
    memoized as a successful certificate. Locks, descriptors and digests never
    serialize into acceptance documents or grant numerical/owner/ABI rights.
    """
    def __init__(self, *, enabled=True):
        self.enabled = bool(enabled)
        self.generation = 0
        self._leases = {}
        self._unprotected = set()
        self._lock = threading.RLock()
        self.stats = dict(cold_seals=0, hits=0, strict_reads=0, revocations=0)

    @contextmanager
    def verification(self):
        with self._lock:
            bridge = _bridge
            if bridge is None:
                yield
            else:
                with bridge.namespace_scope():
                    yield

    def digest(self, path, expected):
        key = os.path.abspath(os.fspath(path))
        with self._lock:
            lease = self._leases.get(key)
            if lease is not None:
                if lease.digest == expected and lease.valid():
                    self.stats['hits'] += 1
                    return lease.digest
                lease.close()
                del self._leases[key]
                self.generation += 1
            if self.enabled and key not in self._unprotected:
                try:
                    lease = _acquire(key)
                    if lease.digest == expected and lease.valid():
                        self._leases[key] = lease
                        self.stats['cold_seals'] += 1
                        return lease.digest
                    actual = lease.digest
                    lease.close()
                    return actual
                except OSError:
                    # Cache only inability to protect, NEVER a content digest.
                    # Retrying a broken/unsupported interop on every hot call
                    # can be slower than legacy validation. Explicit revoke
                    # clears this capability result for a later cold retry.
                    self._unprotected.add(key)
            self.stats['strict_reads'] += 1
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def invalidate(self, reason='explicit update', *, disable=False):
        with self._lock:
            for lease in tuple(self._leases.values()):
                lease.close()
            self._leases.clear()
            self._unprotected.clear()
            self.generation += 1
            self.stats['revocations'] += 1
            if disable:
                self.enabled = False
            return self.generation

    def state(self):
        with self._lock:
            return dict(generation=self.generation, enabled=self.enabled,
                        protected_files=len(self._leases), backends=sorted(
                            {x.backend for x in self._leases.values() if x.active}),
                        **self.stats)

    def __del__(self):
        # Windows handles can be closed directly; avoid waiting on a daemon
        # during interpreter shutdown. Linux releases leases when descriptors
        # close at process exit. Normal owners MUST explicitly invalidate.
        try:
            if not sys.is_finalizing():
                self.invalidate('owner collected')
                return
            for lease in tuple(self._leases.values()):
                if lease.service is None:
                    lease._close()
        except Exception:
            pass
