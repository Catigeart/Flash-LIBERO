"""Narrow opt-in existing RP factory wiring to the actual adopted H01 pool.

No cached SourceVersion/admission result, no raw resident ownership grant, no
default activation. Provider, source/build receipt and policy files stay bound.
"""
from pathlib import Path
from .private_payload import assemble_private_payloads as _private_factory
from types import FunctionType,ModuleType,SimpleNamespace
import hashlib,sys,threading

_providers={}

def load_provider(path,expected_sha256):
    path=Path(path).absolute();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('H01 provider source differs')
    name='_rp_explicit_h01_'+expected_sha256
    key=(str(path),expected_sha256)
    if key not in _providers:
        # Execute the exact cold-read pinned bytes, avoiding a second loader
        # read; no module/code is borrowed from the public GPU namespace.
        module=ModuleType(name);module.__file__=str(path);module.__package__=''
        exec(compile(raw,str(path),'exec',dont_inherit=True),module.__dict__)
        sys.modules[name]=module
        _providers[key]=(module,_provider_methods(module))
    module,methods=_providers[key]
    for owner,member,fn,code in methods:
        if getattr(owner,member,None) is not fn or fn.__code__ is not code:raise ValueError('Cold H01 provider loaded method changed')
    if Path(module.__file__).absolute()!=path:raise ValueError('H01 provider filename differs')
    return module

def _provider_methods(module):
    out=[]
    for name,value in vars(module).items():
        if isinstance(value,FunctionType) and value.__module__==module.__name__:out.append((module,name,value,value.__code__))
        elif isinstance(value,type) and value.__module__==module.__name__:
            for member,fn in vars(value).items():
                if isinstance(fn,FunctionType):out.append((value,member,fn,fn.__code__))
    return tuple(out)

def _assemble_source_leased_private_payloads(original_assemble,bundle,*,dynamic_fields,
        static_fields,source_records,provider_path,provider_sha256,
        build_policy_records=(),capture_policy='eager'):
    """Cold wrapper: own H01 pool, same RP ABI and explicit source invalidation.

    build_policy_records are real (path,SHA) records for this job's receipts or
    policies, not asserted epochs. Existing original bundle/model/stream/buffer
    and current backend guards remain inside every operation. Caller must revoke
    source leases before artifact rewrites and construct a fresh qualified owner
    for changed pins. Unsupported paths/providers retain H01 current-byte reads.
    """
    provider=load_provider(provider_path,provider_sha256)
    pool=provider.FileDigestLeasePool();lock=threading.RLock()
    trusted_invalidate=pool.invalidate;trusted_state=pool.state
    trusted_invalidate_code=trusted_invalidate.__func__.__code__
    trusted_state_code=trusted_state.__func__.__code__
    def close_pool(reason,*,disable=False):
        # Instance replacement cannot replace terminal cleanup authority. A
        # mutated underlying code body must fail visibly rather than pretend
        # to close the real source handles.
        if trusted_invalidate.__func__.__code__ is not trusted_invalidate_code or trusted_state.__func__.__code__ is not trusted_state_code:
            raise RuntimeError('Trusted RP source cleanup code changed; lease release not certified')
        result=trusted_invalidate(reason,disable=disable)
        if trusted_state()['protected_files']!=0:raise RuntimeError('RP source cleanup retained protected leases')
        return result
    methods=_provider_methods(provider)
    classes=tuple((name,value) for name,value in vars(provider).items() if isinstance(value,type) and value.__module__==provider.__name__)
    build_policy_records=tuple(build_policy_records)
    records=list(source_records)+list(build_policy_records)
    records.append((Path(provider_path).absolute(),provider_sha256))
    helper=Path(provider_path).with_name('winlease_bridge.ps1')
    if helper.exists():records.append((helper,hashlib.sha256(helper.read_bytes()).hexdigest()))
    own=Path(__file__).absolute();records.append((own,hashlib.sha256(own.read_bytes()).hexdigest()))
    original=None;original_release=None
    try:
        original=original_assemble(bundle,dynamic_fields=dynamic_fields,static_fields=static_fields,
            source_records=records,digest_pool=pool,capture_policy=capture_policy)
        original_release=getattr(original,'release',None)
        for name in ('begin','write','materialize_error','discard','invalidate','reset','rebind','release'):
            if not callable(getattr(original,name,None)):raise ValueError('New opt-in RP ABI method missing: '+name)
    except BaseException:
        try:
            if original is not None:
                if not callable(original_release):raise RuntimeError('Rejected RP ABI has no trustworthy storage release')
                original_release()
        finally:close_pool('cold RP source owner failed')
        raise
    quarantine=original.invalidate
    bindings=tuple((name,getattr(pool,name).__func__,getattr(pool,name).__func__.__code__) for name in ('digest','verification','invalidate'))
    def guard():
        if type(pool) is not provider.FileDigestLeasePool:raise ValueError('RP H01 source pool type changed')
        if any(getattr(provider,name,None) is not value for name,value in classes):raise ValueError('RP H01 source provider type changed')
        for owner,name,fn,code in methods:
            current=getattr(owner,name)
            if current is not fn or current.__code__ is not code:raise ValueError('RP H01 source provider method changed')
        for name,fn,code in bindings:
            current=getattr(pool,name)
            if getattr(current,'__self__',None) is not pool or getattr(current,'__func__',None) is not fn or fn.__code__ is not code:
                raise ValueError('RP H01 source pool method changed')
    def expose(name):
        bound=getattr(original,name)
        def call(*args,**kwargs):
            with lock:
                if name not in ('invalidate','release'):
                    try:guard()
                    except BaseException:
                        quarantine('source provider changed');raise
                return bound(*args,**kwargs)
        return call
    members={name:expose(name) for name in ('begin','write','materialize_error','discard','invalidate','reset','rebind')}
    def invalidate_source_credentials(reason='explicit source/build/policy mutation',*,strict=False):
        with lock:
            try:quarantine(reason)
            finally:result=close_pool(reason,disable=strict)
            return result
    def release():
        with lock:
            try:return original_release()
            finally:close_pool('RP source owner released')
    members.update(invalidate_source_credentials=invalidate_source_credentials,release=release)
    return SimpleNamespace(**members,manifest=dict(original.manifest,source_auth='dynamic-v2-explicit-owner-H01-os-lease-or-current-byte-fallback-r02',
        default_installed=False,provider_path=str(Path(provider_path).absolute()),provider_sha256=provider_sha256,
        build_policy_records=tuple((str(Path(p).absolute()),h) for p,h in build_policy_records),source_files=len(records),cold_pool_state=pool.state()))

_factory_binding=(_private_factory,_private_factory.__code__)
_H01_SOURCE_SHA256='faf182ef35a49d6857c97f2b2e2cbd1b97d478bee74893c333ce507b621564dd'

def assemble_source_leased_private_payloads(bundle,*,dynamic_fields,static_fields,
        source_records,build_policy_records=(),capture_policy='eager'):
    """Explicit NEW privately allocated dynamic v2 ABI; source leases only.

    Inputs are typed host NumPy payloads that the actual fixed factory freezes
    and copies into new storage. Public resident arrays/NativePlans/callbacks,
    arbitrary source pools and factory/provider overrides are not this API.
    Static values are immutable host descriptors; there is no static device
    consumer. Source/build/policy writes require explicit source invalidation;
    existing public science witnesses/static/history remain unchanged.
    """
    if _private_factory is not _factory_binding[0] or _private_factory.__code__ is not _factory_binding[1]:
        raise ValueError('Private dynamic factory source binding changed')
    provider=Path(__file__).parents[1]/'runtime/core/cold_credentials.py'
    return _assemble_source_leased_private_payloads(_private_factory,bundle,
        dynamic_fields=dynamic_fields,static_fields=static_fields,
        source_records=source_records,build_policy_records=build_policy_records,
        provider_path=provider,provider_sha256=_H01_SOURCE_SHA256,
        capture_policy=capture_policy)
