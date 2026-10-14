"""Explicit prepared native products for this one cold-installer proof job.

Never searches another job or predicts a future compilation. The original
strict compiler builds each exact generated-source/configuration once in its
fixed canonical directory. Runtime plans receive distinct physical copies and
CDLL owners. Existing receipts, dependency checks and plan validators remain.
"""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager
import sys
import threading
import struct


def _binding_defaults(value):
    """Immutable content snapshot of mutable defaults; opaque values stay refs."""
    kind=type(value)
    if value is None:return ('none',)
    if kind in (str,bool,int):return (kind.__name__,value)
    if kind is float:return ('float64',struct.pack('!d',value).hex())
    if kind in (tuple,list):return (kind.__name__,tuple(_binding_defaults(x) for x in value))
    if kind is dict:return ('dict',tuple(sorted((_binding_defaults(k),_binding_defaults(v)) for k,v in value.items())))
    return ('reference',kind,id(value))


def _compiler_binding(function):
    defaults=getattr(function,'__defaults__',None);keywords=getattr(function,'__kwdefaults__',None)
    return (function,getattr(function,'__code__',None),defaults,keywords,
        _binding_defaults(defaults),_binding_defaults(keywords))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    path = Path(path)
    raw = path.read_bytes()
    info=path.stat()
    return dict(path=str(path), resolved=str(path.resolve()), bytes=len(raw), sha256=sha(raw),
        device=info.st_dev,inode=info.st_ino,hardlink_count=info.st_nlink)


def encoded(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Path):
        return dict(path=str(value))
    if isinstance(value, (list, tuple)):
        return [encoded(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encoded(item) for key, item in sorted(value.items())}
    raise TypeError('opaque compile configuration prohibited: ' + type(value).__name__)


class HeldNativeJob:
    def __init__(self, root, *, phase, loader=ctypes.CDLL):
        if phase not in ('prepare', 'run'):
            raise ValueError('explicit prepare/run required')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.loader = loader
        self.path = self.root / 'HELD_NATIVE_JOB.json'
        self.owners = []
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text())
            if self.manifest['root'] != str(self.root):
                raise ValueError('fixed prepared module directory moved')
            if phase == 'run' and self.manifest['status'] != 'prepared':
                raise ValueError('held native job not prepared')
        elif phase == 'run':
            raise ValueError('run cannot prepare missing native job')
        else:
            self.manifest = dict(schema='explicit-held-native-job-v1', root=str(self.root),
                job_id=uuid.uuid4().hex, status='preparing', modules={}, builds=0, private_loads=0)
            self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest, indent=2))

    def validate(self, entry):
        manifest = entry['manifest']
        receipt = manifest.get('build_receipt')
        if not isinstance(receipt, dict) or receipt.get('schema') != 'complete-build-product-v1':
            raise ValueError('original complete build product receipt required')
        identity = receipt['identity']
        if identity.get('schema')!='compiler-dependency-build-v1':
            raise ValueError('actual compiler dependency identity required')
        key = sha(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        if key != receipt['build_key'] or key != receipt['identity_sha256'] or key != manifest['sha256']:
            raise ValueError('complete native build identity mismatch')
        if identity.get('policy') != dict(fma=True, precise_div=True, precise_sqrt=True, ftz=False):
            raise ValueError('native arithmetic policy mismatch')
        if manifest['architecture'] != identity['architecture']:
            raise ValueError('native architecture mismatch')
        if record(manifest['source'])!=entry['source']:
            raise ValueError('fixed canonical generated source path/image changed')
        rows = identity['dependencies'] + identity['toolchain']['components']
        if identity.get('archive') is not None:
            rows = rows + [identity['archive']]
        for row in rows:
            actual = record(row['path'])
            if actual['bytes'] != row['bytes'] or actual['sha256'] != row['sha256']:
                raise ValueError('held module dependency changed: ' + row['path'])
        if sha(Path(manifest['source']).read_text().encode()) != identity['source_sha256']:
            raise ValueError('held generated source changed')
        if record(manifest['library']) != entry['binary']:
            raise ValueError('held module physical binary changed')
        if (entry['binary']['sha256'] != receipt['binary_sha256'] or
                entry['binary']['bytes'] != receipt['binary_bytes']):
            raise ValueError('held binary differs from original receipt')

    def compile(self, original, name, source, build_root, **kwargs):
        if name not in ('collision_before_detect', 'collision_after_detect', 'collision_reset_errors',
                'collision_detect', 'collision_complete'):
            raise ValueError('unregistered cold collision native role')
        if kwargs.get('mathdx', False) or kwargs.get('bridges', ()):
            raise ValueError('this cold collision adapter has no MathDx/bridge qualification')
        # Only device metadata is normalized; each byte/define/flag/include is
        # otherwise exact. The build_root is a private owner location, not code.
        options = dict(kwargs)
        device = options.get('device', 'cuda:0')
        options['device'] = str(device)
        if hasattr(device, 'arch'):
            options['device_arch'] = int(device.arch)
        description = dict(name=name, source_sha256=sha(source.encode()), options=encoded(options))
        key = sha(json.dumps(description, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        entry = self.manifest['modules'].get(key)
        if entry is None:
            if self.phase != 'prepare' or self.manifest['status'] != 'preparing':
                raise ValueError('unknown native specification cannot compile in run')
            folder = self.root / 'modules' / key
            folder.mkdir(parents=True, exist_ok=False)
            entry = dict(status='building', description=description)
            self.manifest['modules'][key] = entry
            self.save()
            try:
                library, manifest = original(name, source, folder, **kwargs)
                self.owners.append(library)
                entry.update(status='ready', manifest=copy.deepcopy(manifest), binary=record(manifest['library']),source=record(manifest['source']))
                self.validate(entry)
                self.manifest['builds'] += 1
                self.save()
            except BaseException as error:
                entry.update(status='failed_preserved', error_type=type(error).__name__, error=str(error))
                self.save()
                raise
        if entry['status'] != 'ready' or entry['description'] != description:
            raise ValueError('failed/incompatible native specification cannot be reused')
        self.validate(entry)
        destination = Path(build_root).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / (name + '_private_' + uuid.uuid4().hex + '.so')
        shutil.copyfile(entry['manifest']['library'], target)
        if target.stat().st_nlink != 1 or sha(target.read_bytes()) != entry['binary']['sha256']:
            raise ValueError('private native physical copy mismatch')
        library = self.loader(str(target))
        self.owners.append(library)
        manifest = copy.deepcopy(entry['manifest'])
        manifest['library'] = str(target)
        manifest['held_native_job'] = dict(job_id=self.manifest['job_id'], module_key=key,
            physical_inode=target.stat().st_ino, hardlink_count=1)
        self.manifest['private_loads'] += 1
        self.save()
        return library, manifest

    def freeze(self):
        if self.phase != 'prepare':
            raise ValueError('only prepare may publish held native job')
        for entry in self.manifest['modules'].values():
            if entry['status'] != 'ready':
                raise ValueError('partial native job cannot publish')
            self.validate(entry)
        self.manifest['status'] = 'prepared'
        self.save()
        self.phase = 'run'


class LeasedHeldNativeJob(HeldNativeJob):
    """Opt-in source authentication for one fresh, same-process prepared job.

    H01 proves current file content. It never proves a numerical grant or
    substitutes a generic receipt. The original strict complete-product checks
    below run on every load. Unknown OS protection keeps current-byte reads.
    Mutable job metadata remains small and is checked strictly on every call.
    """
    def __init__(self, root, *, phase, source_lease_pool, loader=ctypes.CDLL, _existing_scope_pins=None):
        from libero.gpu.host.runtime.core.cold_credentials import FileDigestLeasePool
        if type(source_lease_pool) is not FileDigestLeasePool:
            raise TypeError('actual H01 FileDigestLeasePool required')
        if source_lease_pool.state()['protected_files']:
            raise ValueError('dedicated unused H01 source lease pool required')
        existing_path=Path(root).resolve()/'HELD_NATIVE_JOB.json'
        if _existing_scope_pins is not None:
            raw=existing_path.read_bytes();document=json.loads(raw)
            pins=_existing_scope_pins
            modules_digest=sha(json.dumps(document['modules'],sort_keys=True,separators=(',',':'),allow_nan=False).encode())
            if (phase!='run' or sha(raw)!=pins['manifest_sha256'] or
                    document['job_id']!=pins['job_id'] or document['root']!=str(Path(root).resolve()) or
                    document['status']!='prepared' or modules_digest!=pins['modules_sha256'] or
                    sorted(document['modules'])!=sorted(pins['module_keys'])):
                raise ValueError('explicit original held job/source/build manifest pins differ')
        elif phase!='prepare' or existing_path.exists():
            raise ValueError('source lease scope requires a fresh same-process prepared job')
        self._auth_pool=source_lease_pool
        self._pool_owner=source_lease_pool;self._pool_type=type(source_lease_pool)
        self._auth_enabled=True;self._closed=False;self._boot=True;self._operation_depth=0
        self._auth_lock=threading.RLock()
        self._entry_images={};self._file_identities={};self._headers={};self._text_hashes={}
        self._private_records=[]
        self._compiler_tokens={};self._guard_files=[]
        super().__init__(root,phase=phase,loader=loader)
        if not hasattr(self,'_manifest_image'):
            self._manifest_image=self.path.read_bytes();self._manifest_record=record(self.path)
        self._admitted_keys=frozenset(self.manifest['modules']);self._pending_spec_key=None
        self._scope_root=str(self.root);self._scope_path=str(self.path)
        self._scope_job_id=self.manifest['job_id'];self._scope_phase=self.phase
        self._loader_token=self.loader
        self._method_tokens=[]
        for cls in (HeldNativeJob,LeasedHeldNativeJob):
            for name,descriptor in cls.__dict__.items():
                value=descriptor.__func__ if isinstance(descriptor,(classmethod,staticmethod)) else descriptor
                if callable(value) and hasattr(value,'__code__'):
                    self._method_tokens.append((cls,name,descriptor,_compiler_binding(value)))
        self._global_tokens=[(name,globals()[name],getattr(globals()[name],'__code__',None)) for name in ('record','sha','encoded','_binding_defaults','_compiler_binding')]
        self._provider_tokens=[(name,getattr(type(self._auth_pool),name),getattr(type(self._auth_pool),name).__code__) for name in ('digest','verification','invalidate','state')]
        provider=Path(sys.modules[type(self._auth_pool).__module__].__file__)
        for path in (Path(__file__),provider,provider.with_name('winlease_bridge.ps1')):
            if path.exists():self._add_guard_file(path)
        self._boot=False
        if _existing_scope_pins is not None:
            if sha(self._manifest_image)!=_existing_scope_pins['manifest_sha256']:
                self.close();raise ValueError('original held manifest changed during explicit admission')
            try:
                with self._operation():
                    roles={entry['description']['name'] for entry in self.manifest['modules'].values()}
                    if set(_existing_scope_pins['original_compilers'])!=roles:
                        raise ValueError('actual original complete compiler role bindings required')
                    for name,original in _existing_scope_pins['original_compilers'].items():
                        if not callable(original) or not hasattr(original,'__code__'):
                            raise ValueError('actual original complete compiler function required')
                        self._compiler_tokens[name]=_compiler_binding(original)
                        self._add_guard_file(original.__code__.co_filename)
                    for key,entry in self.manifest['modules'].items():
                        description_digest=sha(json.dumps(entry['description'],sort_keys=True,separators=(',',':'),allow_nan=False).encode())
                        if entry['status']!='ready' or description_digest!=key:
                            raise ValueError('original held complete module specification differs')
                        # Mandatory once per exact pinned module. This is not a
                        # prepared/generic receipt or future compilation claim.
                        HeldNativeJob.validate(self,entry)
                        self.validate(entry)
            except BaseException:
                self.close();raise

    @classmethod
    def admit_existing(cls,root,*,expected_job_id,expected_manifest_sha256,expected_modules_sha256,
            expected_module_keys,original_compilers,source_lease_pool,loader=ctypes.CDLL):
        """Root-authorized one-job re-admission of exact existing compiled products.

        The pins are supplied from the preserved, stopped original job boundary.
        All original complete receipts and current source/dependency/product
        bytes are verified once before any private runtime load is permitted.
        """
        for value in (expected_manifest_sha256,expected_modules_sha256):
            if type(value) is not str or len(value)!=64 or any(x not in '0123456789abcdef' for x in value):
                raise ValueError('exact original held manifest/modules SHA256 required')
        if not expected_job_id or not expected_module_keys or len(set(expected_module_keys))!=len(expected_module_keys):
            raise ValueError('explicit original held job and complete module set required')
        pins=dict(job_id=expected_job_id,manifest_sha256=expected_manifest_sha256,modules_sha256=expected_modules_sha256,
            module_keys=tuple(expected_module_keys),original_compilers=dict(original_compilers))
        return cls(root,phase='run',source_lease_pool=source_lease_pool,loader=loader,_existing_scope_pins=pins)

    def _add_guard_file(self,path):
        row=record(path)
        if not any(x['path']==row['path'] for x in self._guard_files):
            self._guard_files.append(row)
            if self._auth_pool.digest(row['path'],row['sha256'])!=row['sha256']:
                raise ValueError('source authentication helper changed')

    def _assert_scope(self):
        if self._closed:raise ValueError('held source authentication scope closed')
        if self._auth_pool is not self._pool_owner or type(self._auth_pool) is not self._pool_type:
            raise ValueError('held source authentication pool owner changed')
        if str(self.root)!=self._scope_root or str(self.path)!=self._scope_path or self.manifest['job_id']!=self._scope_job_id or self.phase!=self._scope_phase:
            raise ValueError('fixed prepared source authentication job changed')
        if self.loader is not self._loader_token:
            raise ValueError('held private module loader changed')
        if frozenset(self.manifest['modules'])!=self._admitted_keys:
            raise ValueError('held original complete module set changed')
        for cls,name,descriptor,prior in self._method_tokens:
            actual=cls.__dict__.get(name)
            function=actual.__func__ if isinstance(actual,(classmethod,staticmethod)) else actual
            current=_compiler_binding(function)
            if actual is not descriptor or any(x is not y for x,y in zip(current[:4],prior[:4])) or current[4:]!=prior[4:]:
                raise ValueError('held source authentication method changed: '+name)
        for name in ('compile','validate','freeze','save'):
            if getattr(getattr(self,name),'__func__',None) is not getattr(LeasedHeldNativeJob,name):
                raise ValueError('held source authentication method changed: '+name)
        for name,value,code in self._global_tokens:
            actual=globals()[name]
            if actual is not value or getattr(actual,'__code__',None) is not code:
                raise ValueError('held source authentication function changed: '+name)
        for name,value,code in self._provider_tokens:
            actual=getattr(self._auth_pool,name)
            if getattr(actual,'__func__',None) is not value or value.__code__ is not code:
                raise ValueError('held source authentication provider changed: '+name)
        # The mutable small manifest is never licensed by mtime/cached hash.
        if record(self.path)!=self._manifest_record or self.path.read_bytes()!=self._manifest_image:
            raise ValueError('held source authentication manifest changed')
        current={key:value for key,value in self.manifest.items() if key!='modules'}
        original={key:value for key,value in json.loads(self._manifest_image).items() if key!='modules'}
        if current!=original:
            raise ValueError('held source authentication manifest header changed')
        for row in self._guard_files:
            if self._record(row['path'],row)!=row:
                raise ValueError('source authentication helper changed')

    @contextmanager
    def _operation(self):
        with self._auth_lock:
            with self._pool_owner.verification():
                self._assert_scope()
                self._operation_depth+=1
                try:yield
                finally:self._operation_depth-=1

    def _record(self,path,expected,*,header=False):
        if not self._auth_enabled:
            actual=record(path)
        else:
            path=Path(path);before=path.stat()
            content=self._auth_pool.digest(path,expected['sha256'])
            after=path.stat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_nlink)!=(after.st_dev,after.st_ino,after.st_size,after.st_nlink):
                raise ValueError('held source authentication file identity changed during verification')
            actual=dict(path=str(path),resolved=str(path.resolve()),bytes=after.st_size,sha256=content,
                device=after.st_dev,inode=after.st_ino,hardlink_count=after.st_nlink)
        token=(actual['resolved'],actual['device'],actual['inode'],actual['hardlink_count'])
        prior=self._file_identities.setdefault(actual['path'],token)
        if token!=prior:raise ValueError('held source authentication file identity changed: '+actual['path'])
        if header:
            with Path(path).open('rb') as stream:prefix=stream.read(64)
            prior=self._headers.setdefault(actual['path'],prefix)
            if prefix!=prior:raise ValueError('held canonical binary header changed')
        return actual

    def save(self):
        if not self._boot and not self._operation_depth:
            raise ValueError('held job save requires controlled source authentication transaction')
        super().save()
        self._manifest_image=self.path.read_bytes();self._manifest_record=record(self.path)

    def validate(self,entry):
        if not self._operation_depth:
            with self._operation():return self.validate(entry)
        entry_keys=[key for key,value in self.manifest['modules'].items() if value is entry]
        if len(entry_keys)!=1:
            raise ValueError('held module is outside this source authentication job')
        entry_key=entry_keys[0]
        if entry_key not in self._admitted_keys and not (self.phase=='prepare' and entry_key==self._pending_spec_key):
            raise ValueError('held module lacks original complete specification admission')
        manifest=entry['manifest'];receipt=manifest.get('build_receipt')
        if not isinstance(receipt,dict) or receipt.get('schema')!='complete-build-product-v1':
            raise ValueError('original complete build product receipt required')
        identity=receipt['identity']
        if identity.get('schema')!='compiler-dependency-build-v1':
            raise ValueError('actual compiler dependency identity required')
        key=sha(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        if key!=receipt['build_key'] or key!=receipt['identity_sha256'] or key!=manifest['sha256']:
            raise ValueError('complete native build identity mismatch')
        if identity.get('policy')!=dict(fma=True,precise_div=True,precise_sqrt=True,ftz=False):
            raise ValueError('native arithmetic policy mismatch')
        if manifest['architecture']!=identity['architecture']:
            raise ValueError('native architecture mismatch')
        if self._record(manifest['source'],entry['source'])!=entry['source']:
            raise ValueError('fixed canonical generated source path/image changed')
        rows=identity['dependencies']+identity['toolchain']['components']
        if identity.get('archive') is not None:rows=rows+[identity['archive']]
        for row in rows:
            actual=self._record(row['path'],row)
            if actual['bytes']!=row['bytes'] or actual['sha256']!=row['sha256']:
                raise ValueError('held module dependency changed: '+row['path'])
        # Derived text hash is bound to a currently authenticated exact raw
        # source digest and identity, never source path/id/mtime alone.
        text_key=(manifest['source'],entry['source']['sha256'])
        if text_key not in self._text_hashes:
            self._text_hashes[text_key]=sha(Path(manifest['source']).read_text().encode())
        if self._text_hashes[text_key]!=identity['source_sha256']:
            raise ValueError('held generated source changed')
        if self._record(manifest['library'],entry['binary'],header=True)!=entry['binary']:
            raise ValueError('held module physical binary changed')
        if entry['binary']['sha256']!=receipt['binary_sha256'] or entry['binary']['bytes']!=receipt['binary_bytes']:
            raise ValueError('held binary differs from original receipt')
        image=json.dumps(entry,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        previous=self._entry_images.setdefault(entry_key,image)
        if image!=previous:raise ValueError('held complete source/build/policy entry changed')

    def compile(self,original,name,source,build_root,**kwargs):
        with self._operation():
            token=_compiler_binding(original)
            prior=self._compiler_tokens.get(name)
            if prior is not None and (any(x is not y for x,y in zip(token[:4],prior[:4])) or token[4:]!=prior[4:]):
                raise ValueError('held original complete compiler method changed')
            if prior is None:
                if self.phase!='prepare':raise ValueError('unknown held complete compiler in run')
                self._compiler_tokens[name]=token
                filename=getattr(token[1],'co_filename',None)
                if filename and Path(filename).is_file():self._add_guard_file(filename)
            options=dict(kwargs);device=options.get('device','cuda:0');options['device']=str(device)
            if hasattr(device,'arch'):options['device_arch']=int(device.arch)
            description=dict(name=name,source_sha256=sha(source.encode()),options=encoded(options))
            requested_key=sha(json.dumps(description,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
            if self.phase=='prepare':self._pending_spec_key=requested_key
            try:library,manifest=super().compile(original,name,source,build_root,**kwargs)
            finally:self._pending_spec_key=None
            self._admitted_keys=self._admitted_keys|frozenset((requested_key,))
            physical=record(manifest['library'])
            declared=manifest['held_native_job']
            handle=getattr(library,'_handle',None)
            if (physical['hardlink_count']!=1 or physical['sha256']!=manifest['build_receipt']['binary_sha256'] or
                    physical['bytes']!=manifest['build_receipt']['binary_bytes'] or
                    physical['inode']!=declared['physical_inode'] or declared['job_id']!=self._scope_job_id or
                    str(Path(getattr(library,'_name','')).resolve())!=physical['resolved'] or not handle):
                raise ValueError('actual private native copy/CDLL owner changed after load')
            for previous in self._private_records:
                if (library is previous['owner'] or handle==previous['handle'] or
                        (physical['device'],physical['inode'])==(previous['file']['device'],previous['file']['inode'])):
                    raise ValueError('private native physical/module owner was shared')
            self._private_records.append(dict(file=physical,owner=library,handle=handle))
            return library,manifest

    def freeze(self):
        with self._operation():
            super().freeze()
            self._scope_phase=self.phase

    def invalidate_source_leases(self,reason='explicit source authentication revoke'):
        """Release this scope's handles; subsequent validations are strict reads."""
        with self._auth_lock:
            self._pool_owner.invalidate(reason)
            self._auth_enabled=False

    def close(self):
        """Revoke authentication only. Existing private module owners stay alive."""
        with self._auth_lock:
            self.invalidate_source_leases('held job authentication cleanup')
            self._closed=True

    @property
    def source_auth_state(self):
        return dict(job_id=self._scope_job_id,phase=self._scope_phase,closed=self._closed,
            source_lease_enabled=self._auth_enabled,pool=self._auth_pool.state(),
            private_loader_owners=len(self.owners),scope='this fresh same-process job only')
