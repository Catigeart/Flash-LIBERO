"""CPU-side descriptors for GPU runtime assembly.

Descriptors never license a kernel: original source/model guards remain mandatory.
A compatible specialization permits code reuse, never static parameter reuse.
Publication records a producing phase, not numerical acceptance or a CUDA fence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


class ContractError(ValueError):
    pass


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2147483647:
        raise ContractError(f'{name}: expected nonnegative int32 count')
    return value


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ContractError(f'{name}: expected lowercase SHA256')
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _hash(payload):
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ContractError('descriptor is not finite JSON') from exc
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1024)
def _specialization_digest(dimensions, extra_dimensions, topology, options_json, features, layout):
    # Complete immutable value key, never model object identity or a generation
    # asserted by an external mutable array owner. Encoding is exactly legacy.
    return _hash({'dimensions': dict(dimensions), 'topology': topology,
                  'extra_dimensions': dict(extra_dimensions),
                  'options': json.loads(options_json), 'features': features,
                  'layout': layout})


@lru_cache(maxsize=1024)
def _model_identity_digest(specialization, model, static_arrays, mutable_parameters):
    return _hash({'specialization': specialization, 'model': model,
                  'static_arrays': static_arrays, 'mutable_parameters': mutable_parameters})


@dataclass(frozen=True)
class ModelDimensions:
    nq: int
    nv: int
    nu: int
    na: int
    nM: int
    nbody: int
    ngeom: int
    nsite: int
    njnt: int
    ntendon: int = 0
    neq: int = 0
    nplugin: int = 0
    nmocap: int = 0
    nsensordata: int = 0

    def __post_init__(self):
        for name, value in vars(self).items():
            _count(value, name)
        if self.nbody == 0:
            raise ContractError('model must contain world body')

    def as_dict(self):
        return dict(vars(self))


@dataclass(frozen=True)
class ModelSpec:
    dimensions: ModelDimensions
    model_sha256: str
    topology_sha256: str
    static_arrays_sha256: str
    options: Mapping[str, Any]
    features: tuple[str, ...] = ()
    layout_version: str = 'gpu-runtime-v1'
    mutable_parameters: tuple[str, ...] = ('body_pos', 'body_quat')
    extra_dimensions: Mapping[str, int] = field(default_factory=dict)
    _options_json: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.dimensions, ModelDimensions):
            raise ContractError('dimensions must be ModelDimensions')
        for name in ('model_sha256', 'topology_sha256', 'static_arrays_sha256'):
            _digest(getattr(self, name), name)
        extra = dict(self.extra_dimensions)
        if set(extra) & set(self.dimensions.as_dict()):
            raise ContractError('extra_dimensions duplicates primary dimensions')
        for name, value in extra.items():
            if not isinstance(name, str) or not name:
                raise ContractError('extra dimension name required')
            _count(value, name)
        object.__setattr__(self, 'extra_dimensions', MappingProxyType(extra))
        if not self.layout_version:
            raise ContractError('layout_version is required')
        # Defensive deep serialization prevents caller mutation changing cache identity.
        _hash(dict(self.options))
        packed = json.dumps(dict(self.options), sort_keys=True, allow_nan=False)
        object.__setattr__(self, '_options_json', packed)
        object.__setattr__(self, 'options', _freeze(json.loads(packed)))
        object.__setattr__(self, 'features', tuple(sorted(set(self.features))))
        object.__setattr__(self, 'mutable_parameters', tuple(sorted(set(self.mutable_parameters))))
        if any(not isinstance(x, str) or not x for x in self.features + self.mutable_parameters):
            raise ContractError('feature and parameter names must be nonempty strings')

    @property
    def specialization_key(self):
        dimensions = self.dimensions.as_dict()
        extra = dict(self.extra_dimensions)
        features = tuple(self.features)
        # Re-read every identity input. Even object.__setattr__/__dict__ edits
        # produce a different complete key. Unusual mutable/custom value types
        # keep the original strict JSON encoder and error contract.
        if (all(type(k) is str and type(v) is int for k,v in dimensions.items()) and
                all(type(k) is str and type(v) is int for k,v in extra.items()) and
                all(type(x) is str for x in features) and
                all(type(x) is str for x in (self.topology_sha256,self._options_json,self.layout_version))):
            return _specialization_digest(tuple(sorted(dimensions.items())),tuple(sorted(extra.items())),
                                          self.topology_sha256,self._options_json,features,self.layout_version)
        return _hash({'dimensions': dimensions, 'topology': self.topology_sha256,
                      'extra_dimensions': extra, 'options': json.loads(self._options_json),
                      'features': self.features, 'layout': self.layout_version})

    @property
    def identity_key(self):
        if (type(self.model_sha256) is str and type(self.static_arrays_sha256) is str and
                type(self.mutable_parameters) is tuple and all(type(x) is str for x in self.mutable_parameters)):
            return _model_identity_digest(self.specialization_key,self.model_sha256,
                                          self.static_arrays_sha256,self.mutable_parameters)
        return _hash({'specialization': self.specialization_key, 'model': self.model_sha256,
                      'static_arrays': self.static_arrays_sha256,
                      'mutable_parameters': self.mutable_parameters})

    @classmethod
    def from_factory_metadata(cls, metadata: Mapping[str, Any], *, features,
                              mutable_parameters=None):
        """Consume native factory descriptors; no XML hash substitution or licensing."""
        if mutable_parameters is None:
            mutable_parameters = metadata.get('mutable_parameters', ('body_pos', 'body_quat'))
        dimensions = dict(metadata['dimensions'])
        primary_names = set(ModelDimensions.__dataclass_fields__)
        primary = {key: value for key, value in dimensions.items() if key in primary_names}
        extra = {key: value for key, value in dimensions.items() if key not in primary_names}
        return cls(ModelDimensions(**primary), metadata['model_sha256'],
                   metadata['topology_sha256'], metadata['static_arrays_sha256'],
                   metadata['options'], tuple(features), layout_version=metadata.get('layout_version', 'gpu-runtime-v1'),
                   mutable_parameters=tuple(mutable_parameters),
                   extra_dimensions=extra)

    def to_factory_metadata(self):
        """Round-trip plain JSON data, preserving exact native options and counts."""
        return {'dimensions': dict(self.dimensions.as_dict(), **dict(self.extra_dimensions)),
                'model_sha256': self.model_sha256, 'topology_sha256': self.topology_sha256,
                'static_arrays_sha256': self.static_arrays_sha256,
                'options': json.loads(self._options_json), 'layout_version': self.layout_version,
                'mutable_parameters': list(self.mutable_parameters)}

    def shares_specialization(self, other):
        return isinstance(other, ModelSpec) and self.specialization_key == other.specialization_key


@dataclass(frozen=True)
class ParameterUpdate:
    model_identity: str
    base_generation: int
    base_digest: str
    next_generation: int
    next_digest: str
    fields: tuple[str, ...]


@dataclass
class ModelGeneration:
    spec: ModelSpec
    generation: int = 0
    parameters_sha256: str = ''

    def __post_init__(self):
        _count(self.generation, 'generation')
        if not self.parameters_sha256:
            self.parameters_sha256 = self.spec.static_arrays_sha256
        _digest(self.parameters_sha256, 'parameters_sha256')

    def prepare_parameter_update(self, changes: Mapping[str, str]):
        """Validate digest metadata before any upload; caller validates payload."""
        unknown = set(changes) - set(self.spec.mutable_parameters)
        if unknown:
            raise ContractError(f'model rebuild required for changes: {sorted(unknown)}')
        for name, value in changes.items():
            _digest(value, name)
        next_generation = _count(self.generation + bool(changes), 'generation')
        new_digest = (_hash({'previous': self.parameters_sha256, 'changes': dict(changes)})
                      if changes else self.parameters_sha256)
        return ParameterUpdate(self.spec.identity_key, self.generation, self.parameters_sha256,
                               next_generation, new_digest, tuple(sorted(changes)))

    def commit_parameter_update(self, update: ParameterUpdate):
        """Commit only after successful upload. Caller must quarantine partial failure.

        The token detects concurrent/reentrant metadata changes. It is not an
        upload capability or proof that the device transfer completed.
        """
        if (update.model_identity != self.spec.identity_key or
                update.base_generation != self.generation or
                update.base_digest != self.parameters_sha256):
            raise ContractError('stale parameter update token; upload must be quarantined')
        self.generation, self.parameters_sha256 = update.next_generation, update.next_digest
        return self.generation

    def apply_parameter_update(self, changes: Mapping[str, str]):
        """Convenience for already validated/uploaded changes; use two-phase API in adapters."""
        return self.commit_parameter_update(self.prepare_parameter_update(changes))

DTYPE_BYTES = {'float64': 8, 'float32': 4, 'int32': 4, 'int64': 8, 'uint8': 1, 'bool': 1}


@dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    owner: str
    scope: str = 'world'

    def __post_init__(self):
        if not self.name or not self.owner or self.dtype not in DTYPE_BYTES:
            raise ContractError('buffer name, owner and supported dtype required')
        if self.scope not in ('world', 'model', 'workspace'):
            raise ContractError('unknown buffer scope')
        object.__setattr__(self, 'shape', tuple(self.shape))
        for i, length in enumerate(self.shape):
            _count(length, f'{self.name}.shape[{i}]')
        if self.element_count > 2147483647:
            raise ContractError('buffer element count exceeds int32 indexing')

    @property
    def element_count(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.element_count * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class BufferBinding:
    spec: BufferSpec
    handle: Any = field(compare=False, repr=False)
    allocation_id: str

    def __post_init__(self):
        if not self.allocation_id:
            raise ContractError('stable allocation identity required')


@dataclass
class FirstError:
    code: int = 0
    stage: str = ''
    detail: str = ''

    def record(self, code: int, stage: str, detail: str = ''):
        if isinstance(code, bool) or not isinstance(code, int) or code <= 0:
            raise ContractError('error code must be positive integer')
        if self.code == 0:
            self.code, self.stage, self.detail = code, stage, detail
        return self.code

    def clear(self):
        self.code, self.stage, self.detail = 0, '', ''


@dataclass(frozen=True)
class Publication:
    phase: str
    state_epoch: int
    model_generation: int
    sequence: int


@dataclass(frozen=True)
class StateWrite:
    world_id: int
    model_identity: str
    model_generation: int
    base_epoch: int
    next_epoch: int
    fields: tuple[str, ...]


@dataclass
class DeviceWorldState:
    world_id: int
    model: ModelGeneration
    bindings: dict[str, BufferBinding] = field(default_factory=dict)
    state_epoch: int = 0
    publications: dict[str, Publication] = field(default_factory=dict)
    error: FirstError = field(default_factory=FirstError)
    closed: bool = False
    _sequence: int = 0

    def __post_init__(self):
        _count(self.world_id, 'world_id')
        _count(self.state_epoch, 'state_epoch')
        initial = dict(self.bindings)
        self.bindings = {}
        for binding in initial.values():
            self.bind(binding)

    def _open(self):
        if self.closed:
            raise ContractError('world is closed')

    def bind(self, binding: BufferBinding):
        self._open()
        dimensions = self.model.spec.dimensions
        required = {'qpos': ((dimensions.nq,), 'float64'),
                    'qvel': ((dimensions.nv,), 'float64'),
                    'ctrl': ((dimensions.nu,), 'float64'),
                    'qfrc_applied': ((dimensions.nv,), 'float64'),
                    'xfrc_applied': ((dimensions.nbody,6), 'float64'),
                    'qacc_warmstart': ((dimensions.nv,), 'float64'),
                    'time': ((1,), 'float64')}
        expected = required.get(binding.spec.name)
        if expected and (binding.spec.shape, binding.spec.dtype) != expected:
            raise ContractError(f'{binding.spec.name}: incompatible state shape/dtype')
        self.bindings[binding.spec.name] = binding
        self.publications.pop(binding.spec.name, None)

    def prepare_state_write(self, fields):
        """Validate metadata before boundary upload; no mutation or device transfer."""
        self._open()
        names = tuple(fields)
        if not names or set(names) - {'qpos', 'qvel', 'time', 'act', 'mocap_pos', 'mocap_quat', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied'}:
            raise ContractError('expected physical state write fields')
        if set(names) - set(self.bindings):
            raise ContractError('state write references unbound buffer')
        return StateWrite(self.world_id, self.model.spec.identity_key, self.model.generation,
                          self.state_epoch, _count(self.state_epoch + 1, 'state_epoch'), names)

    def commit_state_write(self, update: StateWrite):
        self._open()
        if (update.world_id != self.world_id or update.model_identity != self.model.spec.identity_key
                or update.model_generation != self.model.generation or update.base_epoch != self.state_epoch):
            raise ContractError('stale state write token; upload must be quarantined')
        if set(update.fields) - set(self.bindings):
            raise ContractError('state write lost binding; upload must be quarantined')
        self.state_epoch = update.next_epoch
        self.publish_phase('state_write', update.fields)

    def mark_state_write(self, fields):
        """Convenience for known successful writes; adapters use prepare/commit.

        Old derived publications survive, intentionally readable after Euler via
        require_publication(current_state=False), retaining their actual epoch.
        This is boundary metadata, not a per-substep device clock substitute.
        """
        self.commit_state_write(self.prepare_state_write(fields))
    def publish_phase(self, phase: str, fields, held=()):
        self._open()
        written, kept = set(fields), set(held)
        if not phase or written & kept:
            raise ContractError('phase required and written/held sets must be disjoint')
        if (written | kept) - set(self.bindings):
            raise ContractError('publication references unbound buffer')
        self._sequence += 1
        publication = Publication(phase, self.state_epoch, self.model.generation, self._sequence)
        for name in written:
            self.publications[name] = publication
        return publication

    def require_publication(self, field: str, current_state=True):
        self._open()
        publication = self.publications.get(field)
        if publication is None:
            raise ContractError(f'{field}: no producer publication')
        if publication.model_generation != self.model.generation:
            raise ContractError(f'{field}: stale model generation')
        if current_state and publication.state_epoch != self.state_epoch:
            raise ContractError(f'{field}: stale physical state epoch')
        return publication

    def validate_counts(self, counts: Mapping[str, int], capacities: Mapping[str, int]):
        self._open()
        for name, value in counts.items():
            _count(value, name)
            if name not in capacities or value > _count(capacities[name], f'{name}.capacity'):
                raise ContractError(f'{name}: capacity exceeded or missing; truncation forbidden')

    def reset_metadata(self, clear_errors=True):
        """Invalidate metadata only. Caller owns CPU-equivalent reset writes."""
        self._open()
        self.state_epoch = _count(self.state_epoch + 1, 'state_epoch')
        self.publications.clear()
        if clear_errors:
            self.error.clear()

    def close(self):
        self.publications.clear()
        self.bindings.clear()
        self.closed = True


@dataclass(frozen=True)
class GraphBinding:
    world_id: int
    model_identity: str
    model_generation: int
    parameters_sha256: str
    allocation_keys: tuple

    @classmethod
    def capture(cls, state: DeviceWorldState):
        state._open()
        return cls(state.world_id, state.model.spec.identity_key, state.model.generation,
                   state.model.parameters_sha256,
                   tuple(sorted((name, b.allocation_id, b.spec.shape, b.spec.dtype)
                                for name, b in state.bindings.items())))

    def validate(self, state: DeviceWorldState):
        if self != type(self).capture(state):
            raise ContractError('graph recapture required: model generation/layout/allocation changed')
        return True


@dataclass(frozen=True)
class HostBoundaryPolicy:
    allowed: tuple[str, ...] = (
        'compile_model', 'sample_placement', 'upload_initial_state',
        'upload_action', 'upload_model_parameters', 'restore_state', 'read_action_observation',
        'task_predicate', 'render_scene_bridge', 'serialize_state', 'schedule',
    )

    def require(self, action: str, boundary: str):
        if boundary not in ('prepare', 'reset', 'action', 'close'):
            raise ContractError('host action requires declared lifecycle/action boundary')
        if action not in self.allowed:
            raise ContractError(f'host action prohibited in production: {action}')
        if action in ('compile_model', 'sample_placement', 'upload_initial_state', 'upload_model_parameters') and boundary not in ('prepare', 'reset'):
            raise ContractError(f'{action}: requires prepare/reset boundary')
        return True







