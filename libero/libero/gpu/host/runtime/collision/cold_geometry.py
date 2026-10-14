"""Cold admission for the explicit handwritten_only-v1 prototype.

Host readset reuse never grants GPU write, upload, owner, or binary permission.
Every source/build check uses an H01 live file lease or current file bytes.
The existing Shared/DetectorPlan/NativePlan validators remain mandatory.
"""
from contextlib import nullcontext,contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

_JOB_SOURCE_AUTHORITY=ContextVar('collision_job_source_authority',default=None)

@contextmanager
def collision_job_source_scope(authority=None):
    """Explicit job-local construction only. None is the original strict route."""
    token=_JOB_SOURCE_AUTHORITY.set(authority)
    try:yield authority
    finally:_JOB_SOURCE_AUTHORITY.reset(token)


def _encoded(value):
    """Strict typed byte encoding used both before and after sealing."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:raise ValueError('object static field is unsupported')
        meta=json.dumps((value.dtype.str,value.dtype.descr if value.dtype.fields else None,
            tuple(value.shape)),separators=(',',':')).encode()
        raw=np.ascontiguousarray(value).tobytes()
        return b'a'+struct.pack('<QQ',len(meta),len(raw))+meta+raw
    if isinstance(value,np.generic):return b'n'+_encoded(np.asarray(value))
    if value is None:return b'z'
    if type(value) is bool:return b'b'+bytes([value])
    if type(value) is int:return b'i'+_encoded(str(value))
    if type(value) is float:return b'f'+struct.pack('<d',value)
    if type(value) is str:
        raw=value.encode();return b's'+struct.pack('<Q',len(raw))+raw
    if type(value) is bytes:return b'r'+struct.pack('<Q',len(value))+value
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('static/context keys must be strings')
        return b'd'+struct.pack('<Q',len(value))+b''.join(_encoded(k)+_encoded(value[k]) for k in sorted(value))
    if isinstance(value,(tuple,list)):
        return (b't' if type(value) is tuple else b'l')+struct.pack('<Q',len(value))+b''.join(map(_encoded,value))
    raise TypeError('opaque static/context field: '+type(value).__name__)


def strict_static_digest(value):return hashlib.sha256(_encoded(value)).hexdigest()


def _freeze(value,immutable_array):
    if isinstance(value,np.ndarray):return immutable_array(value)
    if isinstance(value,dict):return {k:_freeze(v,immutable_array) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_freeze(v,immutable_array) for v in value)
    if isinstance(value,list):return [_freeze(v,immutable_array) for v in value]
    _encoded(value)
    return value


def _arrays(value,prefix='',result=None):
    result={} if result is None else result
    if isinstance(value,np.ndarray):result[prefix]=value
    elif isinstance(value,dict):
        for k in sorted(value):_arrays(value[k],prefix+'/'+k,result)
    elif isinstance(value,(tuple,list)):
        for k,v in enumerate(value):_arrays(v,prefix+'/'+str(k),result)
    return result


def _array_skeleton(value):
    if isinstance(value,np.ndarray):return ('array',value.dtype.str,value.dtype.descr if value.dtype.fields else None,tuple(value.shape))
    if isinstance(value,dict):return {k:_array_skeleton(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_array_skeleton(v) for v in value)
    if isinstance(value,list):return [_array_skeleton(v) for v in value]
    return value


class _StrictFiles:
    """Missing H01 lease capability always reads current bytes."""
    def verification(self):return nullcontext()
    def digest(self,path,expected=None):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def state(self):return {'capability':'strict_current_bytes_only'}
    def invalidate(self,reason,disable=False):return None


class CollisionColdAdmission:
    def __init__(self,static,*,gpu_root,context=None,options=None,maintained_sources=()):
        try:
            from ..core.static_readset import immutable_array,StaticReadsetSeal
        except ImportError:
            # Pure bytes fallback remains immutable, but cannot reuse a digest.
            def immutable_array(a):
                a=np.asarray(a,order='C')
                if a.dtype.hasobject:raise ValueError('object static field')
                return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
            StaticReadsetSeal=None
        self._job_source_authority=_JOB_SOURCE_AUTHORITY.get()
        if self._job_source_authority is not None:self.files=self._job_source_authority.borrow('collision admission')
        else:
            try:
                from ..core.cold_credentials import FileDigestLeasePool
                self.files=FileDigestLeasePool()
            except ImportError:self.files=_StrictFiles()
        self.static=_freeze(static,immutable_array)
        self.context_source={} if context is None else context
        self.options={} if options is None else dict(options)
        self.context_digest=strict_static_digest(self.context_source)
        self.skeleton_digest=strict_static_digest(_array_skeleton(self.static))
        self.static_digest=strict_static_digest(self.static)
        self.gpu_root=Path(gpu_root).resolve();self.rows={};self.plans=();self.workspace=None
        self.sealed=False;self.seal_class=StaticReadsetSeal;self.static_seal=None
        self.revoked=False
        self._seal_identity=None;self.build_digests=()
        self.complete_context=(isinstance(self.context_source,dict) and all(k in self.context_source for k in
            ('model_identity','spec_identity','generation','field_permissions')))
        paths=[p for p in self.gpu_root.rglob('*') if p.suffix in ('.py','.cu','.cuh','.h','.c','.cpp') and p.is_file()]
        paths.extend(Path(p) for p in maintained_sources)
        if not paths:raise ValueError('maintained collision source readset is empty')
        self.pin(paths)

    def pin(self,paths):
        if self._job_source_authority is not None:
            with self.files.verification():return self._pin_records(paths)
        return self._pin_records(paths)

    def _pin_records(self,paths):
        for selector in paths:
            selector=Path(selector).absolute();key=str(selector)
            if self._job_source_authority is not None:
                previous=self.rows.get(key)
                actual=self.files.pin_record(selector,None if previous is None else previous[2]);row=actual[:3]
            else:
                path=selector.resolve();raw=path.read_bytes();row=(str(path),len(raw),hashlib.sha256(raw).hexdigest())
            if key in self.rows and self.rows[key]!=row:raise ValueError('cold source changed during construction: '+key)
            self.rows[key]=row

    def attach(self,workspace):
        if self.workspace is not None:raise ValueError('admission owner is one-shot')
        self.workspace=workspace
        workspace._collision_cold_admission=self
        return workspace

    def _identity(self):
        ws=self.workspace
        dimensions={} if ws is None else {k:getattr(ws,k) for k in ('nworld','ngeom','nbody','slots',
            'call_capacity','pair_capacity','stage_slots','req_slots','contact_capacity') if hasattr(ws,k)}
        device=None if ws is None else str(ws.device)
        return strict_static_digest({'schema':'handwritten-collision-static-readset-v1',
            'context':self.context_source,'dimensions':dimensions,'device':device,'options':self.options,
            'layout':'Params400/public-raw500/contact-f65-i5','field_permissions':self.context_source.get('field_permissions')
                if isinstance(self.context_source,dict) else None,
            'static_skeleton':_array_skeleton(self.static),'candidate_sources':self.rows,
            'active_builds':[p.manifest for p in self.plans]})

    def _verify_files(self):
        with self.files.verification():
            for selector,(resolved,count,expected) in self.rows.items():
                if self._job_source_authority is not None:
                    actual=self.files.pin_record(selector,expected)
                    if actual[:3]!=(resolved,count,expected):raise ValueError('collision source/build changed: '+selector)
                    continue
                path=Path(selector)
                if str(path.resolve())!=resolved or path.stat().st_size!=count or self.files.digest(path,expected)!=expected:
                    raise ValueError('collision source/build changed: '+selector)

    def validate(self,*,require_installed=True):
        if self.revoked:raise ValueError('collision cold admission was revoked')
        if require_installed and not self.sealed:raise ValueError('collision cold admission is not sealed')
        if strict_static_digest(self.context_source)!=self.context_digest:raise ValueError('collision model/spec/context changed')
        if strict_static_digest(_array_skeleton(self.static))!=self.skeleton_digest:raise ValueError('collision static metadata/scalars changed')
        identity=self._identity()
        current=(self.static_seal.current_digest(_arrays(self.static),identity,strict=lambda:strict_static_digest(self.static))
            if self.static_seal is not None else strict_static_digest(self.static))
        if current!=self.static_digest:raise ValueError('collision host static readset changed')
        if self.sealed and (identity!=self._seal_identity or tuple(strict_static_digest(p.manifest) for p in self.plans)!=self.build_digests):
            raise ValueError('collision installed plan/context changed')
        self._verify_files()
        return True

    def seal(self,plans):
        if self.sealed:raise ValueError('collision admission seal is one-shot')
        if self.revoked:raise ValueError('collision cold admission was revoked')
        if self._job_source_authority is not None:
            with self.files.verification():return self._seal_checked(plans)
        return self._seal_checked(plans)

    def _seal_checked(self,plans):
        if self.sealed:raise ValueError('collision admission seal is one-shot')
        self.validate(require_installed=False)
        self.plans=tuple(plans)
        if not self.plans:raise ValueError('actual installed plans required')
        for plan in self.plans:
            manifest=plan.manifest;receipt=manifest.get('build_receipt')
            if not isinstance(receipt,dict) or receipt.get('schema')!='complete-build-product-v1':
                raise ValueError('handwritten_only prototype requires explicit complete H01 build receipt')
            identity=receipt['identity']
            canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            key=hashlib.sha256(canonical(identity)).hexdigest()
            if receipt['build_key']!=key or receipt['identity_sha256']!=key or manifest['sha256']!=key:
                raise ValueError('collision build receipt/key mismatch')
            if identity.get('schema')!='compiler-dependency-build-v1':raise ValueError('compiler dependency receipt required')
            if identity.get('policy')!={'fma':True,'precise_div':True,'precise_sqrt':True,'ftz':False} or manifest.get('architecture')!=identity['architecture']:
                raise ValueError('collision build arithmetic/architecture policy changed')
            paths=[manifest['source'],manifest['library']]
            rows=list(identity['dependencies'])+list(identity['toolchain']['components'])
            if identity.get('archive') is not None:rows.append(identity['archive'])
            for row in rows:
                path=Path(row['path'])
                if self._job_source_authority is not None:
                    actual=self.files.pin_record(path,row['sha256']);count,digest=actual[1:3]
                else:
                    raw=path.read_bytes();count,digest=len(raw),hashlib.sha256(raw).hexdigest()
                if count!=row['bytes'] or digest!=row['sha256']:
                    raise ValueError('collision receipt dependency changed: '+str(path))
                paths.append(path)
            text_digest=(self.files.text_digest(manifest['source']) if self._job_source_authority is not None else
                hashlib.sha256(Path(manifest['source']).read_text().encode()).hexdigest())
            if text_digest!=identity['source_sha256']:
                raise ValueError('collision generated source changed')
            bridge_units=[]
            for command in manifest['compile_commands']:
                for item in command:
                    path=Path(str(item))
                    if path.suffix=='.cu' and path.is_file() and path.resolve()!=Path(manifest['source']).resolve() and path not in bridge_units:
                        bridge_units.append(path)
            bridge_digests=([self.files.text_digest(p) for p in bridge_units] if self._job_source_authority is not None else
                [hashlib.sha256(p.read_text().encode()).hexdigest() for p in bridge_units])
            if bridge_digests!=identity['bridges_sha256']:
                raise ValueError('collision generated bridge changed or missing')
            paths.extend(bridge_units)
            if self._job_source_authority is not None:self.files.bind_plan(plan)
            if self._job_source_authority is not None:
                actual=self.files.pin_record(manifest['library'],receipt['binary_sha256']);count,digest=actual[1:3]
            else:
                binary=Path(manifest['library']).read_bytes();count,digest=len(binary),hashlib.sha256(binary).hexdigest()
            if count!=receipt['binary_bytes'] or digest!=receipt['binary_sha256']:
                raise ValueError('collision installed binary differs from receipt')
            self.pin(paths)
        # Retain actual loaded host source modules consumed by the factory/ABI.
        self.pin([m.__file__ for name,m in tuple(sys.modules.items()) if name.startswith(('libero.','scripts.'))
            and getattr(m,'__file__',None) and Path(m.__file__).is_file()])
        self._seal_identity=self._identity();self.build_digests=tuple(strict_static_digest(p.manifest) for p in self.plans)
        if self.complete_context and self.seal_class is not None:
            self.static_seal=self.seal_class(_arrays(self.static),self._seal_identity,self.static_digest)
        self.sealed=True
        self.validate()
        return self.manifest()

    def manifest(self):
        return {'schema':'handwritten-collision-cold-admission-v1','sealed':self.sealed,
            'host_static_digest':self.static_digest,'host_static_reuse':bool(self.static_seal and self.static_seal.enabled),
            'complete_context':self.complete_context,'source_files':len(self.rows),
            'source_readset_digest':strict_static_digest(self.rows),'build_receipts':len(self.plans),
            'file_verification':self.files.state(),'gpu_static_write_or_upload_permission':False,
            'enqueue_policy':'existing complete validators; opaque callable graphs receive no H03 skip'}

    def authority_token(self):
        """Snapshot expected proof data; callers compare before invoking validators."""
        seal=self.static_seal
        return (id(self),id(self.files),id(self.static),id(self.context_source),id(self.options),
            self.context_digest,self.skeleton_digest,self.static_digest,self._seal_identity,
            self.sealed,self.revoked,self.complete_context,self.build_digests,tuple(sorted(self.rows.items())),
            id(seal),None if seal is None else (seal.identity,seal.digest,seal.enabled))

    def revoke(self,reason='resident arena released'):
        """Explicit arena close drops file handles and forbids future submission."""
        self.revoked=True;self.sealed=False
        self.files.invalidate(reason,disable=True)
