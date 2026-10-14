"""Opt-in CPU parameter/static-input receiver. No native/device permission.

The actual destination is the complete original S static mapping, accepted by
generate_specialization. It contains no baked model identity: the original
generator still consumes and validates the fresh complete manifest separately.
"""
from dataclasses import dataclass
import copy
import hashlib
import marshal
import json
from threading import RLock
from types import MappingProxyType, SimpleNamespace
import numpy as np


@dataclass(frozen=True)
class TypedStaticArtifact:
    arrays: object
    readset: tuple
    source_build_policy: tuple
    owner_scope: str


@dataclass(frozen=True)
class PreparedTransaction:
    owner: object
    lease: object
    base: object
    descriptor: tuple
    snapshot: object
    static: TypedStaticArtifact
    artifact_dependency_plan: object
    reused: bool


class ColdParameterArtifactOwner:
    """Own immutable source/destination bytes, an exact descriptor and a lease.

    Every prepare includes original payload/token and descriptor admission.
    Every commit rechecks the complete original descriptor and actual base
    snapshot/owner lease. Publication swaps one private state tuple, so an
    exception never exposes partially written body/static arrays. Revoke/close
    replace the lease and reject all prior prepared transactions.
    """
    def __init__(self, api, snapshot, model_generation, *, allowed_body_ids,
                 owner_scope, build_policy, input_metadata, enabled=False):
        if not enabled:
            raise ValueError('typed parameter artifact consumer requires explicit opt-in')
        self._api = api
        self._lock = RLock()
        self._scope = owner_scope
        if not isinstance(owner_scope, str) or not owner_scope:
            raise ValueError('nonempty physical CPU owner scope required')
        if not isinstance(build_policy, tuple) or not all(type(x) is str and x for x in build_policy):
            raise ValueError('complete immutable CPU build policy required')
        self._program = api.program_binding()
        self._code_digests = tuple(hashlib.sha256(marshal.dumps(row[1])).hexdigest() for row in self._program)
        self._allowed = tuple(sorted(set(int(i) for i in allowed_body_ids)))
        self._source_descriptor = json.dumps(input_metadata,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()
        metadata = json.loads(self._source_descriptor)
        source_spec = api.contracts.ModelSpec.from_factory_metadata(metadata,
            features=model_generation.spec.features,mutable_parameters=model_generation.spec.mutable_parameters)
        if source_spec.identity_key != model_generation.spec.identity_key:
            raise ValueError('typed artifact source descriptor differs from generation')
        self._source_pin = hashlib.sha256(self._source_descriptor).hexdigest()
        self._policy = (api.binding, self._code_digests, np.__version__, build_policy,
                        self._source_pin, self._allowed)
        dims = model_generation.spec.to_factory_metadata()['dimensions']
        self._schema = api.static_schema(dims)
        required = set(self._schema) | {'body_pos', 'body_quat'}
        if set(snapshot.fields) != required:
            raise ValueError('complete scoped body/S static inventory required')
        expected = dict(self._schema, body_pos=((dims['nbody'], 3), np.float64),
                        body_quat=((dims['nbody'], 4), np.float64))
        for name, (shape, dtype) in expected.items():
            field = snapshot.fields[name]
            if field.shape != shape or np.dtype(field.dtype) != np.dtype(dtype):
                raise ValueError('typed artifact layout mismatch: ' + name)
            if metadata['static_array_inventory'].get(name) != {'shape':list(field.shape),'dtype':field.dtype}:
                raise ValueError('typed artifact source inventory differs: ' + name)
            if metadata['array_sha256'].get(name) != field.encoded_digest:
                raise ValueError('typed artifact source field bytes differ: ' + name)
        if any(i < 0 or i >= dims['nbody'] for i in self._allowed):
            raise ValueError('allowed body ID outside actual receiver')
        self._lease = object()
        self._closed = False
        self._pending = None
        self._state = (snapshot, self._clone_generation(model_generation), self._materialize(snapshot))

    def _admit_program(self):
        if self._api.program_binding() != self._program:
            raise ValueError('typed artifact frozen source execution changed')

    def _clone_generation(self, generation):
        C = self._api.contracts
        spec = C.ModelSpec.from_factory_metadata(generation.spec.to_factory_metadata(),
            features=generation.spec.features, mutable_parameters=generation.spec.mutable_parameters)
        return C.ModelGeneration(spec, generation.generation, generation.parameters_sha256)

    def _materialize(self, snapshot):
        # Original cold_model prepares a C-contiguous private copy of every S
        # field. Keep the same exact typed bytes, with immutable ultimate owners.
        fields = tuple(snapshot.fields[name] for name in self._schema)
        arrays = {}
        for field in fields:
            copied = np.ascontiguousarray(field.array()).copy()
            arrays[field.name] = self._api.snapshots.ArrayBytes.capture(field.name, copied).array()
        return TypedStaticArtifact(MappingProxyType(arrays), fields, self._policy, self._scope)

    def lease(self):
        with self._lock:
            if self._closed: raise ValueError('typed artifact owner closed')
            return self._lease

    def read(self):
        with self._lock:
            if self._closed: raise ValueError('typed artifact owner closed')
            snapshot, generation, static = self._state
            return snapshot, self._clone_generation(generation), static

    def token(self, changes):
        import hashlib
        digests = {name: hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode() +
                    np.ascontiguousarray(a).tobytes()).hexdigest() for name, a in changes.items()}
        with self._lock:
            if self._closed: raise ValueError('typed artifact owner closed')
            return self._state[1].prepare_parameter_update(digests)

    def prepare(self, changes, token, derived_spec, *, metadata_preparation, lease,
                reuse_static=True):
        with self._lock:
            if self._closed: raise ValueError('typed artifact owner closed')
            self._admit_program()
            base, generation, old_static = self._state
            built = SimpleNamespace(native_model=SimpleNamespace(**base.arrays()),
                domain_shell=SimpleNamespace(obj_body_id={str(i): i for i in self._allowed}))
            payload, _ = self._api.payload(None, built, changes, token, generation)
            cloned = self._clone_generation(self._api.contracts.ModelGeneration(derived_spec))
            descriptor = self._api.generation.prepare_parameter_descriptor(generation, token,
                cloned.spec, metadata_preparation=metadata_preparation)
            if lease is not self._lease:
                raise ValueError('typed artifact owner lease revoked or foreign')
            # Actual original body assignment loop, applied to private writable
            # destinations. No caller payload or live source array is aliased.
            private = SimpleNamespace(**{name: np.array(field.array(), copy=True, order='C')
                for name, field in base.fields.items() if name in ('body_pos', 'body_quat')})
            self._api.write_body_payload(private, payload)
            fields = dict(base.fields)
            for name in payload:
                fields[name] = self._api.snapshots.ArrayBytes.capture(name, getattr(private, name))
            derived = self._api.snapshots.ReadOnlyModelSnapshot(fields)
            dependency_plan = self._api.snapshots.parameter_artifact_plan(base, derived,
                tuple(changes), {'constraint_static':tuple(self._schema)})
            readset = tuple(derived.fields[name] for name in self._schema)
            reused = (reuse_static and 'constraint_static' in dependency_plan['reusable'] and
                readset == old_static.readset and
                old_static.source_build_policy == self._policy and old_static.owner_scope == self._scope)
            static = old_static if reused else self._materialize(derived)
            fresh = self._clone_generation(generation)
            self._api.generation.commit_parameter_descriptor(fresh, descriptor)
            public_descriptor = (descriptor[0], self._clone_generation(fresh).spec,
                                 copy.deepcopy(descriptor[2]))
            prepared = PreparedTransaction(self, self._lease, base, public_descriptor, derived, static,
                                           dependency_plan, reused)
            # Exactly one outstanding private transaction. The caller receives
            # no mutable generation instance that could forge publication.
            self._pending = (prepared, descriptor, derived, static, fresh)
            return prepared

    def commit(self, prepared):
        with self._lock:
            if self._closed: raise ValueError('typed artifact owner closed')
            self._admit_program()
            base, generation, _ = self._state
            # Keep the original descriptor's first-error order, then require the
            # actual immutable base and owner/lease rather than a metadata epoch.
            self._api.generation.prepare_parameter_descriptor(generation, *prepared.descriptor[:2],
                metadata_preparation=prepared.descriptor[2])
            if prepared.owner is not self or prepared.lease is not self._lease or prepared.base is not base:
                raise ValueError('typed artifact prepared owner/base lifetime changed')
            if self._pending is None or self._pending[0] is not prepared:
                raise ValueError('typed artifact transaction is not the current prepared owner')
            if prepared.static.source_build_policy != self._policy or prepared.static.owner_scope != self._scope:
                raise ValueError('typed artifact source/build/policy changed')
            if prepared.static.readset != tuple(prepared.snapshot.fields[name] for name in self._schema):
                raise ValueError('typed artifact destination readset changed')
            _, descriptor, snapshot, static, fresh = self._pending
            self._api.generation.prepare_parameter_descriptor(generation, *descriptor[:2],
                metadata_preparation=descriptor[2])
            self._state = (snapshot, fresh, static)
            self._pending = None
            return static

    def revoke(self, *, build_policy=None):
        with self._lock:
            if build_policy is not None:
                if not isinstance(build_policy, tuple) or not all(type(x) is str and x for x in build_policy):
                    raise ValueError('complete immutable CPU build policy required')
                self._policy = (self._api.binding, self._code_digests, np.__version__, build_policy,
                                self._source_pin, self._allowed)
            self._lease = object()
            self._pending = None

    def close(self):
        with self._lock:
            self._closed = True
            self._lease = object()
            self._state = None
            self._pending = None
