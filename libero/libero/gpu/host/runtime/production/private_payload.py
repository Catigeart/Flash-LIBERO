"""Opaque owned payload consumer, distinct from public scientific witnesses.

The backend allocates independent storage and never publishes an array. Every
dynamic writer enters this owner under its lock. Source credentials and physical
bindings remain checked; a generation alone cannot authorize public storage.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import threading
from contextlib import nullcontext

from .retained_witness import RetainedWitness
from .static_domain import ImmutableStaticDomain


class SourceVersion:
    """Exact source reads, or an explicitly supplied H01 digest lease pool."""
    def __init__(self, records, *, digest_pool=None):
        # Keep the declared source name. Resolving a symlink once would permit
        # later retargeting while checks kept reading the old physical target.
        self._records=tuple((Path(p).absolute(),h) for p,h in records)
        if not self._records:raise ValueError('actual source version records required')
        if any(type(h) is not str or len(h)!=64 or any(c not in '0123456789abcdef' for c in h)
               for _,h in self._records):raise ValueError('exact SHA256 source pin required')
        self._pool=digest_pool;self._closed=False
        self.validate()

    def validate(self):
        if self._closed:raise ValueError('private payload source credential revoked')
        scope=self._pool.verification() if self._pool is not None else nullcontext()
        with scope:
            for path,expected in self._records:
                # H01 keys use lexical abspath. For a declared symlink/../name,
                # lexical normalization can differ from OS path traversal; keep
                # strict reads of the actual declared path in that case.
                actual=(hashlib.sha256(path.read_bytes()).hexdigest() if self._pool is None or '..' in path.parts
                        else self._pool.digest(path,expected))
                if actual!=expected:raise ValueError('private payload source changed: '+str(path))

    def revoke(self):self._closed=True


@dataclass(frozen=True)
class PayloadReceipt:
    owner: object
    generation: int
    ordinal: int


def _method(value):
    return getattr(value,'__func__',value)


class PrivatePayloadOwner:
    """Internal controller; public assembly returns only bound opaque methods.

    Backend capture/read preserve exact bits. Capture is ordered on the same
    stream before overwrite; materialize_error waits at the existing error
    boundary. A failed copy/write never silently discards the pending request.
    """
    _GUARDED=('begin','write','materialize_error','discard',
              'invalidate','reset','rebind','release','_validate','_validate_bound','_require','_new_witness')
    _BACKEND=('validate','capture','validate_write','write','read_capture','replace','release')

    def __init__(self,backend,credential,binding_guard,model_identity,static_assets,
                 *,capture_policy='eager'):
        if capture_policy not in ('eager','copy_on_error'):
            raise ValueError('unknown owned witness policy')
        if not callable(binding_guard):raise TypeError('actual binding guard required')
        self._backend=backend;self._credential=credential;self._binding_guard=binding_guard
        self._binding=(backend,credential,binding_guard)
        self._static=ImmutableStaticDomain(model_identity,static_assets)
        self._lease=self._static.lease();self._policy=capture_policy
        self._lock=threading.RLock();self._generation=0;self._closed=False;self._release_complete=False
        self._needs_reset=False;self._bad_state=False
        self._receipt=None;self._ordinal=-1
        self._methods=tuple((name,_method(getattr(self,name))) for name in self._GUARDED)
        self._backend_methods=tuple((name,_method(getattr(backend,name))) for name in self._BACKEND)
        self._credential_method=_method(credential.validate)
        self._witness_methods=tuple((n,getattr(RetainedWitness,n)) for n in
            ('begin','before_write','materialize','discard','_copy','_require'))
        self._witness=self._new_witness()
        self._validate()

    def _new_witness(self):
        # These actual captures close over the sole storage owner; no caller
        # provided boolean or arbitrary public-array callback proves writers.
        return RetainedWitness(self,self._generation,
            {name:(lambda n=name:self._backend.capture(n)) for name in self._backend.names},
            controlled_writers=True)

    def _validate(self):
        if self._closed:raise ValueError('private payload owner released')
        if self._bad_state:raise ValueError('failed private transaction; fresh cold owner required')
        try:
            if any(_method(getattr(self,n)) is not f for n,f in self._methods):
                raise ValueError('private payload consumer method changed')
            self._validate_bound()
        except BaseException:
            # No failed admission may leave an older receipt consumable after
            # a source/method is restored. Fresh requests revalidate all guards.
            self._receipt=None;self._generation+=1
            self._needs_reset=True
            raise
        if self._needs_reset:
            self._witness=self._new_witness();self._needs_reset=False

    def _validate_bound(self):
        if any(a is not b for a,b in zip((self._backend,self._credential,self._binding_guard),self._binding)):
            raise ValueError('private payload controller binding changed')
        if any(getattr(RetainedWitness,n) is not f for n,f in self._witness_methods):
            raise ValueError('private retained witness method changed')
        if any(_method(getattr(self,n)) is not f for n,f in self._methods):
            raise ValueError('private payload consumer method changed')
        if any(_method(getattr(self._backend,n)) is not f for n,f in self._backend_methods):
            raise ValueError('private payload backend method changed')
        if _method(self._credential.validate) is not self._credential_method:
            raise ValueError('private payload source method changed')
        self._credential.validate();self._binding_guard();self._backend.validate()
        self._static.assert_current(self._lease)

    def begin(self):
        with self._lock:
            self._validate()
            if self._receipt is not None:raise ValueError('previous private payload request must be discarded')
            if self._ordinal>=2**63-1:raise ValueError('private payload request overflow')
            ordinal=self._ordinal+1
            self._ordinal=ordinal
            try:request=self._witness.begin(ordinal,eager=self._backend.names if self._policy=='eager' else ())
            except BaseException:
                self.invalidate('entry capture failed');raise
            self._receipt=PayloadReceipt(self,self._generation,ordinal)
            self._request=request
            return self._receipt

    def _require(self,receipt):
        if (type(receipt) is not PayloadReceipt or receipt is not self._receipt or receipt.owner is not self or
                receipt.generation!=self._generation):
            raise ValueError('foreign or revoked private payload receipt')

    def write(self,name,payload):
        with self._lock:
            self._validate()
            if name not in self._backend.names:raise ValueError('writer outside private dynamic domain')
            self._backend.validate_write(name,payload)
            self._witness.before_write((name,))
            self._backend.write(name,payload)

    def materialize_error(self,receipt):
        with self._lock:
            self._validate();self._require(receipt)
            captures=self._witness.materialize(self._request)
            # Returned bytes have no writable alias of device storage.
            return {name:self._backend.read_capture(capture) for name,capture in captures.items()}

    def discard(self,receipt):
        with self._lock:
            self._validate();self._require(receipt)
            self._witness.discard(self._request);self._receipt=None

    def invalidate(self,reason='explicit invalidation',*,graph=False):
        with self._lock:
            # Failures/source drift still revoke. No saved device handle escapes
            # from this terminal operation; old receipts cannot be consumed.
            self._receipt=None;self._generation+=1
            self._needs_reset=True
            return str(reason)

    def reset(self,payloads):
        with self._lock:
            self._validate()
            if set(payloads)!=set(self._backend.names):raise ValueError('complete private reset schema required')
            for name in self._backend.names:self._backend.validate_write(name,payloads[name])
            self.invalidate('reset')
            try:
                for name in self._backend.names:self._backend.write(name,payloads[name])
            except BaseException:
                self._bad_state=True;raise

    def rebind(self,payloads):
        with self._lock:
            self._validate()
            if set(payloads)!=set(self._backend.names):raise ValueError('complete private rebind schema required')
            for name in self._backend.names:self._backend.validate_write(name,payloads[name])
            self.invalidate('private storage rebound')
            try:self._backend.replace(payloads)
            except BaseException:
                self._bad_state=True;raise

    def release(self):
        with self._lock:
            if self._closed and self._release_complete:return
            if not self._closed:
                self.invalidate('private storage released',graph=True)
                self._closed=True;self._static.revoke()
            # A failed backend wait must not reopen the owner or silently
            # certify release. Retain closed/pending state and retry this cold
            # cleanup on the next release; only success makes it idempotent.
            self._backend.release()
            self._release_complete=True
