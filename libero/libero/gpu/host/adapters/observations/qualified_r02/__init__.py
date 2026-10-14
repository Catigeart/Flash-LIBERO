"""Explicit independent primitive ABI; existing observation defaults stay intact.

The CUDA source bank is the complete source of the measured r02 functions.
Inputs are typed snapshots; CUDA buffers/static metadata/outputs have one native
owner. This API does not issue F BODY success, replace a renderer, or claim a
complete application's speed. Root assembly owns upstream epoch/lifecycle guards.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from . import _module_owner as core
from ._contexts import MatrixContext,SamplerContext,SceneContext


class QualifiedPrimitiveUnavailable(RuntimeError):
    pass


def _source_root():
    return Path(__file__).resolve().parents[4]/'cuda/observations/qualified_r02'


_CATALOG_SHA256='632ecf636ec53bf9c4f2eb4c0bae86bda2999a8d78286fe69cf8650121d68af7'

def _catalog():
    path=_source_root()/'catalog.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=_CATALOG_SHA256:
        raise QualifiedPrimitiveUnavailable('qualified catalog changed')
    return json.loads(path.read_text())


def selected_role(kind,domain=None):
    """Unknown/default domains physically use the caller's original route."""
    table={
      ('O03','actual-captured-retained-pose'):'private',
      ('O03','actual-captured-reset-action-alternating'):'private',
      ('O04','actual-captured-periodic-private'):'owned',
      ('O04','actual-captured-forced-private'):'owned',
      ('O05','controlled-retained-hit'):'memo',
      ('O05','controlled-forced-miss'):'original',
      ('O07','actual-captured-periodic-raw'):'sensor',
      ('O07','actual-captured-forced-sample'):'flat',
      ('O07','ordered'):'existing'}
    return table.get((kind,domain))


def _spec(entry):
    desc=entry['description'];files={}
    gpu=_source_root().parents[2]
    for name,path in entry['assets'].items():
        data=(gpu/path).read_bytes();proof=desc['files'][name]
        if len(data)!=proof['bytes'] or hashlib.sha256(data).hexdigest()!=proof['sha256']:
            raise QualifiedPrimitiveUnavailable('qualified source bytes changed')
        files[name]=data
    value=core.ModuleSpec(files=files,commands=desc['commands'],defines=desc['defines'],
        abi=desc['abi'],configuration=desc['configuration'],binary_name=desc['binary_name'])
    if core.module_spec_id(value)!=entry['tested_spec_id']:
        raise QualifiedPrimitiveUnavailable('qualified complete source/ABI changed')
    return value


def prepare(kind,bundle_dir,*,scope_guard):
    """Explicit cold build once; guard verifies actual sm120/toolchain/source scope.

scope_guard raises if the actual platform/source/lifecycle scope is unavailable;
an epoch or user-supplied bool is not a writer certificate. No prepare occurs on
the default legacy branch. Failed/old banks are preserved, not silently rebuilt.
"""
    if not callable(scope_guard):raise QualifiedPrimitiveUnavailable('actual scope guard required')
    scope_guard();catalog=_catalog()
    if kind not in catalog['catalog']:raise QualifiedPrimitiveUnavailable('unknown primitive')
    root=Path(bundle_dir)
    if (root/'qualified.json').exists():raise QualifiedPrimitiveUnavailable('preserve prepared primitive owner')
    job=core.FixedJobModules(root)
    roles={role:job.register(_spec(entry)) for role,entry in catalog['catalog'][kind]['roles'].items()}
    for identity in roles.values():job.build(identity)
    receipt=dict(kind=kind,roles=roles,catalog_sha256=hashlib.sha256((_source_root()/'catalog.json').read_bytes()).hexdigest(),
       scope='independent r02 primitive; default legacy; upload/seed/readback excluded from measured device region')
    (root/'qualified.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def _array(value,dtype,size):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype(dtype) or value.shape!=(size,):
        raise ValueError('qualified primitive exact one-dimensional dtype/extent required')
    return np.ascontiguousarray(value).copy()


def _fingerprint(value):return hashlib.sha256(value.tobytes()).hexdigest()


class _Controller:
    def __init__(self,kind,domain,context,owner,guard,catalog):
        self._kind,self._domain,self._context,self._owner,self._guard=kind,domain,context,owner,guard
        self._catalog=catalog;self._closed=False
    def _live(self):
        if self._closed:raise RuntimeError('qualified primitive released/revoked')
        self._guard()
    def close(self):
        if not self._closed:
            self._closed=True
            try:self._context.close()
            finally:
                if hasattr(self,'_original_context'):self._original_context.close()
    def sample(self,raw,state,flags):
        self._live()
        if self._kind not in ('O04','O07'):raise TypeError('sampler primitive required')
        cfg=self._catalog['sampler']
        raw=_array(raw,np.float64,cfg['raw_stride']);state=_array(state,np.float64,cfg['NSTATE'])
        flags=_array(flags,np.int32,cfg['NFLAGS'])
        self._context.upload(raw,state,flags)
        force=int('forced' in self._domain)
        self._context.step(force,cfg['timestep']);return self._context.read()
    def quaternion(self,matrix,mask,*,reset=False):
        self._live()
        if self._kind!='O05':raise TypeError('matrix primitive required')
        count=self._context.worlds
        matrix=_array(matrix,np.float32,9*count);mask=_array(mask,np.int32,count)
        key=matrix.tobytes()
        if self._domain=='controlled-retained-hit':
            # This hot profile has immutable input and is cold-primed once.
            # Changes/reset must use the original role or a fresh cold owner.
            if reset or key!=self._matrix_key:self._use_original=True
        context=self._original_context if getattr(self,'_use_original',False) else self._context
        context.upload(matrix,mask);context.step(int(reset));return context.read()
    def scene(self,frame_a,frame_b,*,publish=True,mask=1,valid=1,reset=False):
        self._live()
        if self._kind!='O03':raise TypeError('scene primitive required')
        a=_array(frame_a,np.float64,312);b=_array(frame_b,np.float64,312)
        if mask not in (0,1) or valid not in (0,1):raise ValueError('scene mask/valid are controlled binary flags')
        if publish:self._context.upload(a,b,1)
        self._context.step(mask,valid,int(reset));return self._context.read()
    def mutate_metadata(self,integers):
        self._live()
        if self._kind!='O04':raise TypeError('private immutable sampler required')
        values=_array(integers,np.int32,self._catalog['sampler']['integer_count'])
        # Native owned mutate revokes BEFORE CUDA for any changed structural word.
        self._context.mutate(values)


def open_primitive(kind,bundle_dir,*,domain=None,legacy_factory,scope_guard=None,
        worlds=1,integers=None,periods=None,local_static=None,matrix=None):
    """Cold dispatcher: opt-in known local domain only, otherwise original API.

O05 changing/reset/miss callers select controlled-forced-miss (original role).
O07 ordered callers select existing. These choices happen before native loading.
Metadata/static byte fingerprints and source spec are checked independently from
the caller guard. Every returned operation checks guard/lifecycle before CUDA.
"""
    role=selected_role(kind,domain)
    if role is None:return legacy_factory()
    if not callable(scope_guard):raise QualifiedPrimitiveUnavailable('actual scope/lifecycle guard required')
    scope_guard();catalog=_catalog();cfg=catalog['sampler']
    if kind=='O05':
        if type(worlds) is not int or worlds not in (1,2):return legacy_factory()
        if domain=='controlled-retained-hit':matrix=_array(matrix,np.float32,worlds*9)
    else:
        if worlds!=1:return legacy_factory()
    if kind in ('O04','O07'):
        integers=_array(integers,np.int32,cfg['integer_count']);periods=_array(periods,np.float64,cfg['ns'])
        if _fingerprint(integers)!=cfg['integer_sha256'] or _fingerprint(periods)!=cfg['period_sha256']:
            return legacy_factory()
    if kind=='O03':
        cfg=catalog['scene'];integers=_array(integers,np.int32,cfg['integer_count'])
        local_static=_array(local_static,np.float64,cfg['local_count'])
        if _fingerprint(integers)!=cfg['integer_sha256'] or _fingerprint(local_static)!=cfg['local_sha256']:
            return legacy_factory()
    receipt=json.loads((Path(bundle_dir)/'qualified.json').read_text())
    if receipt['kind']!=kind or receipt['catalog_sha256']!=hashlib.sha256((_source_root()/'catalog.json').read_bytes()).hexdigest():
        raise QualifiedPrimitiveUnavailable('foreign/changed prepared primitive owner')
    _spec(catalog['catalog'][kind]['roles'][role])
    def no_build(*a,**kw):raise RuntimeError('run-only primitive compiler forbidden')
    job=core.FixedJobModules(bundle_dir,runner=no_build)
    owner=job.private_library(receipt['roles'][role],kind+'-'+role)
    context=None
    try:
        if kind=='O05':context=MatrixContext(owner,worlds)
        elif kind=='O03':context=SceneContext(owner,integers,local_static)
        else:context=SamplerContext(owner,dict(NSTATE=cfg['NSTATE'],NFLAGS=cfg['NFLAGS'],integers=integers,periods=periods))
        owner.hold(context)
        result=_Controller(kind,domain,context,owner,scope_guard,catalog)
        if kind=='O05' and domain=='controlled-retained-hit':
            context.upload(matrix,np.ones(worlds,dtype=np.int32));context.step(1)
            result._matrix_key=matrix.tobytes()  # cold prime excluded from measured hot gain
            original_owner=job.private_library(receipt['roles']['original'],kind+'-original-fallback')
            original_context=MatrixContext(original_owner,worlds)
            result._original_context=original_context
            original_owner.hold(original_context);result._original_owner=original_owner
            original_context.upload(matrix,np.ones(worlds,dtype=np.int32));original_context.step(0)
            # Both outputs start at the same old result, so masked worlds stay
            # held even on a change/reset transition. Original selection sticks
            # until cold reprepare; no silent memo return after a miss domain.
            result._use_original=False
        return result
    except BaseException:
        if context is not None:
            try:context.close()
            finally:
                if 'result' in locals() and hasattr(result,'_original_context'):result._original_context.close()
        raise
