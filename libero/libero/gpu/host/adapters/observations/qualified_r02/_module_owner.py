"""Explicit compiled objects owned by one prepared job; standard library only.

This is not a compiler cache. No outside bundle is searched, and a ready object
is reused without asking what another compiler invocation would produce.
"""
from dataclasses import dataclass, field
import ctypes
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import uuid


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def _relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in path.parts) or '\\' in name:
        raise ValueError('Module files must have relative paths: ' + name)
    return path


@dataclass(frozen=True)
class ModuleSpec:
    files: dict
    commands: list
    defines: dict = field(default_factory=dict)
    abi: object = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    binary_name: str = 'probe.so'

    def description(self):
        if not self.files or not self.commands:
            raise ValueError('Explicit source files and original command sequence required')
        _relative(self.binary_name)
        files = {}
        for name, data in self.files.items():
            _relative(name)
            if not isinstance(data, bytes):
                raise TypeError('Source files are immutable bytes')
            files[name] = dict(sha256=sha(data), bytes=len(data))
        if any(not command or any(not isinstance(a, str) for a in command) for command in self.commands):
            raise TypeError('Commands must be explicit argv arrays')
        value = dict(files=files, commands=self.commands, defines=self.defines,
                     abi=self.abi, configuration=self.configuration, binary_name=self.binary_name)
        return json.loads(json.dumps(value))


class BuildFailure(RuntimeError):
    pass


def module_spec_id(spec):
    """Public deterministic name within a bundle, not a cross-job cache query."""
    return sha(json.dumps(spec.description(), sort_keys=True, separators=(',', ':')).encode())


@dataclass
class PrivateLibrary:
    library: object
    path: Path
    receipt: dict
    runtime_owners: list = field(default_factory=list)

    def hold(self, *owners):
        self.runtime_owners.extend(owners)
        return self

    @contextmanager
    def counter_guard(self, symbol='configure_counters'):
        """Retire the original optional instrumentation token even on an error."""
        function = getattr(self.library, symbol)
        initial = function(None)
        self.receipt['counter_retirement'] = dict(initial_returncode=initial, retired=initial == 0)
        _write_json(self.path.parent / 'OWNER.json', self.receipt)
        if initial:
            raise RuntimeError('Counter initial retirement failed')
        try:
            yield self.library
        finally:
            code = function(None)
            self.receipt['counter_retirement'] = dict(initial_returncode=initial, final_returncode=code, retired=code == 0)
            _write_json(self.path.parent / 'OWNER.json', self.receipt)
            if code:
                raise RuntimeError('Counter final retirement failed')


class FixedJobModules:
    def __init__(self, bundle_dir, job_id=None, *, runner=subprocess.run, loader=ctypes.CDLL):
        self.root = Path(bundle_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'modules_manifest.json'
        self.runner, self.loader = runner, loader
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
            if self.manifest['bundle_dir'] != str(self.root):
                raise ValueError('Prepared bundle was moved; fixed source paths must stay fixed')
            if job_id is not None and self.manifest['job_id'] != job_id:
                raise ValueError('This module owner belongs to a different job')
        else:
            self.manifest = dict(schema=1, job_id=job_id or uuid.uuid4().hex,
                                 bundle_dir=str(self.root), modules={})
            self._save()
        self.job_id = self.manifest['job_id']

    def _save(self):
        _write_json(self.path, self.manifest)

    def register(self, spec):
        description = spec.description()
        # A name within this owner, never a lookup key for another job/cache.
        identity = module_spec_id(spec)
        modules = self.manifest['modules']
        if identity in modules:
            if modules[identity]['description'] != description:
                raise ValueError('Ambiguous explicit module specification')
            self._verify_sources(modules[identity])
            return identity
        directory = self.root / 'compiled' / identity
        directory.mkdir(parents=True, exist_ok=False)
        for name, data in spec.files.items():
            path = directory.joinpath(*_relative(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        commands = [[a.replace('${MODULE_DIR}', str(directory)) for a in c] for c in description['commands']]
        modules[identity] = dict(spec_id=identity, description=description, directory=str(directory),
                                 actual_commands=commands, status='registered', build_count=0)
        self._save()
        return identity

    def _verify_sources(self, record):
        directory = Path(record['directory'])
        if directory != self.root / 'compiled' / record['spec_id']:
            raise ValueError('Module source path is outside its fixed owner')
        for name, expected in record['description']['files'].items():
            data = directory.joinpath(*_relative(name).parts).read_bytes()
            if sha(data) != expected['sha256'] or len(data) != expected['bytes']:
                raise ValueError('Fixed source content changed: ' + name)

    def record_for(self, spec_id):
        # Return a detached receipt so caller metadata cannot mutate the owner.
        return json.loads(json.dumps(self.manifest['modules'][spec_id]))

    def verify_ready(self, spec_id):
        record = self.manifest['modules'][spec_id]
        self._verify_sources(record)
        if record['status'] != 'ready' or record['build_count'] != 1:
            raise BuildFailure('Module is not a successful prepared object: ' + spec_id)
        path = Path(record['directory']) / record['description']['binary_name']
        data = path.read_bytes()
        if sha(data) != record['binary']['sha256'] or len(data) != record['binary']['bytes']:
            raise ValueError('Prepared binary content changed')
        return self.record_for(spec_id)

    def build(self, spec_id):
        record = self.manifest['modules'][spec_id]
        if record['status'] == 'ready':
            return self.verify_ready(spec_id)
        if record['status'] != 'registered':
            raise BuildFailure('Failed/interrupted build is not a reusable object; use a new explicit bundle')
        self._verify_sources(record)
        record.update(status='building', build_count=1, command_results=[])
        self._save()
        directory = Path(record['directory'])
        binary = directory / record['description']['binary_name']
        start = time.monotonic()
        try:
            for command in record['actual_commands']:
                phase_start = time.monotonic()
                result = self.runner(command, cwd=str(directory), capture_output=True, text=True)
                with (directory / 'compile.log').open('a', encoding='utf-8') as log:
                    log.write(result.stdout or '')
                    log.write(result.stderr or '')
                record['command_results'].append(dict(argv=command, returncode=result.returncode,
                                                       wall_seconds=time.monotonic()-phase_start))
                if result.returncode:
                    raise BuildFailure(str(directory / 'compile.log'))
            self._verify_sources(record)
            data = binary.read_bytes()
            if not data:
                raise BuildFailure('Compiler did not produce a nonempty binary')
            record.update(status='ready', binary=dict(path=str(binary), sha256=sha(data), bytes=len(data)))
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            if binary.exists():
                data = binary.read_bytes()
                record['failed_binary_evidence'] = dict(path=str(binary), sha256=sha(data), bytes=len(data))
            raise
        finally:
            record['build_wall_seconds'] = time.monotonic()-start
            self._save()
        return self.record_for(spec_id)

    def private_library(self, spec_id, tag):
        record = self.verify_ready(spec_id)
        if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in tag):
            raise ValueError('Explicit simple tag required')
        directory = self.root / 'instances' / tag / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        target = directory / 'private.so'
        start = time.monotonic()
        # Physical copies, never a hardlink or the source path used by CDLL.
        shutil.copyfile(record['binary']['path'], target)
        if sha(target.read_bytes()) != record['binary']['sha256']:
            raise ValueError('Private copy does not belong to the prepared binary')
        receipt = dict(job_id=self.job_id, tag=tag, spec_id=spec_id,
                       binary_sha256=record['binary']['sha256'], private_path=str(target),
                       copy_wall_seconds=time.monotonic()-start, status='copied',
                       physical_file=dict(device=target.stat().st_dev, inode=target.stat().st_ino,
                                          hardlink_count=target.stat().st_nlink, size_bytes=target.stat().st_size))
        start = time.monotonic()
        try:
            library = self.loader(str(target))
            receipt.update(status='loaded', load_wall_seconds=time.monotonic()-start)
        except Exception as error:
            receipt.update(status='load_failed', load_wall_seconds=time.monotonic()-start,
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            _write_json(directory / 'OWNER.json', receipt)
        return PrivateLibrary(library, target, receipt)
