"""Explicit structural capability and independent numerical admission registry.

No qpos/trajectory PRE whitelist is an admission condition. Structural readiness
is separate from numerical qualification; metadata never grants kernel acceptance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping
from types import MappingProxyType
from .contracts import ContractError, ModelSpec, _count, _digest, _hash
from .cold_credentials import FileDigestLeasePool


@dataclass(frozen=True)
class KernelCandidate:
    candidate_id: str
    component: str
    source_sha256: str
    algorithm: str
    layout_version: str
    architecture: str
    math_profile: str
    author: str
    specialization_keys: tuple[str, ...] = ()
    model_sha256s: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()
    forbidden_features: tuple[str, ...] = ()
    dimension_equalities: Mapping[str, int] = field(default_factory=dict)
    maximum_counts: Mapping[str, int] = field(default_factory=dict)
    mutable_parameters: tuple[str, ...] = ()
    source_files: Mapping[str, str] = field(default_factory=dict)
    execution_scope: str = 'component-runtime'
    generation_rebind_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if not all((self.candidate_id, self.component, self.algorithm, self.layout_version,
                    self.architecture, self.math_profile, self.author)):
            raise ContractError('candidate identity, component and algorithm provenance required')
        for name in ('specialization_keys', 'model_sha256s', 'required_features', 'forbidden_features', 'mutable_parameters', 'generation_rebind_fields'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        source_files = dict(self.source_files)
        for path, digest in source_files.items():
            if not path:
                raise ContractError('source path required')
            _digest(digest, 'source file')
        if source_files and _hash(source_files) != self.source_sha256:
            raise ContractError('candidate source bundle digest differs from file manifest')
        object.__setattr__(self, 'source_files', MappingProxyType(source_files))
        if not self.execution_scope:
            raise ContractError('candidate execution scope required')
        _digest(self.source_sha256, 'candidate source')
        if not (self.specialization_keys or self.model_sha256s):
            raise ContractError('candidate needs explicit model/specialization scope')
        for key in self.specialization_keys + self.model_sha256s:
            _digest(key, 'candidate scope')
        for name, count in dict(self.dimension_equalities, **MappingProxyType(dict(self.maximum_counts))).items():
            _count(count, name)
        for mapping in (self.dimension_equalities, self.maximum_counts):
            if any(name in {'qpos', 'qvel', 'qacc', 'time', 'ctrl', 'warmstart', 'nefc', 'ncon'} for name in mapping):
                raise ContractError('live values/counts are not structural admission predicates')
        object.__setattr__(self, 'dimension_equalities', MappingProxyType(dict(self.dimension_equalities)))
        object.__setattr__(self, 'maximum_counts', MappingProxyType(dict(self.maximum_counts)))

    def to_dict(self):
        result = {}
        for name, value in vars(self).items():
            if isinstance(value, Mapping):
                result[name] = dict(value)
            elif isinstance(value, tuple):
                result[name] = list(value)
            else:
                result[name] = value
        return result

    def verify_source_files(self, leases=None):
        """Verify bundle immediately before diagnostic/production execution.

        Empty manifest means legacy external source guards are still required.
        """
        for path, expected in self.source_files.items():
            try:
                actual = (leases.digest(path, expected) if leases is not None else
                          hashlib.sha256(Path(path).read_bytes()).hexdigest())
            except OSError as exc:
                raise ContractError(f'candidate source unavailable: {path}') from exc
            if actual != expected:
                raise ContractError(f'candidate source changed: {path}')
        return bool(self.source_files)

    def structural_reasons(self, model: ModelSpec):
        reasons = []
        if model.layout_version != self.layout_version:
            reasons.append('layout version differs')
        if self.specialization_keys and model.specialization_key not in self.specialization_keys:
            reasons.append('specialization outside candidate scope')
        if self.model_sha256s and model.model_sha256 not in self.model_sha256s:
            reasons.append('model outside candidate scope')
        if set(self.required_features) - set(model.features):
            reasons.append('required structural features absent')
        if set(self.forbidden_features) & set(model.features):
            reasons.append('unsupported structural feature present')
        dimensions = dict(model.dimensions.as_dict(), **dict(model.extra_dimensions))
        for name, value in self.dimension_equalities.items():
            if dimensions.get(name) != value:
                reasons.append(f'{name}: dimension differs')
        for name, maximum in self.maximum_counts.items():
            if name not in dimensions or dimensions[name] > maximum:
                reasons.append(f'{name}: structural capacity insufficient')
        return tuple(reasons)


@dataclass(frozen=True)
class AcceptanceRecord:
    candidate_id: str
    source_sha256: str
    model_identity_key: str
    status: str
    report_sha256: str
    report_path: str
    reviewer: str
    numerical_pass: bool
    actual_writes_pass: bool
    provenance_pass: bool
    scope: str
    permitted_parameter_fields: tuple[str, ...] = ()
    execution_scope: str = 'component-runtime'
    generation_rebind_fields: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('source_sha256', 'model_identity_key', 'report_sha256'):
            _digest(getattr(self, name), name)
        if not all(isinstance(value, bool) for value in (self.numerical_pass, self.actual_writes_pass, self.provenance_pass)):
            raise ContractError('acceptance verdicts must be booleans')
        object.__setattr__(self, 'permitted_parameter_fields', tuple(self.permitted_parameter_fields))
        if self.status not in ('pending', 'accepted', 'rejected'):
            raise ContractError('unknown acceptance status')
        if not all((self.report_path, self.reviewer, self.scope)):
            raise ContractError('independent report path/reviewer/scope required')
        if self.status == 'accepted' and not all((self.numerical_pass, self.actual_writes_pass, self.provenance_pass)):
            raise ContractError('acceptance requires numerical, writes and provenance qualification')


@dataclass(frozen=True)
class QualifiedGenerationBridge:
    """Independent native immutable/readset equivalence and legal mutation witness."""
    candidate_id: str
    source_sha256: str
    base_model_identity: str
    derived_model_identity: str
    changed_fields: tuple[str, ...]
    report_path: str
    report_sha256: str
    reviewer: str
    immutable_arrays_equal: bool
    topology_equal: bool
    options_equal: bool
    parameter_rules_pass: bool

    def __post_init__(self):
        for name in ('source_sha256', 'base_model_identity', 'derived_model_identity', 'report_sha256'):
            _digest(getattr(self, name), name)
        if not self.report_path or not self.reviewer:
            raise ContractError('generation bridge requires independent report')
        object.__setattr__(self, 'changed_fields', tuple(sorted(set(self.changed_fields))))
        verdicts = (self.immutable_arrays_equal, self.topology_equal, self.options_equal, self.parameter_rules_pass)
        if not all(isinstance(value, bool) and value for value in verdicts):
            raise ContractError('generation bridge requires exact immutable/topology/options equality and legal source rules')

    def verify_report(self, leases=None):
        try:
            actual = (leases.digest(self.report_path, self.report_sha256) if leases is not None else
                      hashlib.sha256(Path(self.report_path).read_bytes()).hexdigest())
        except OSError as exc:
            raise ContractError('generation bridge report unavailable') from exc
        if actual != self.report_sha256:
            raise ContractError('generation bridge report changed')


@dataclass(frozen=True)
class Admission:
    candidate_id: str
    model_identity_key: str
    structural_ready: bool
    execution_allowed: bool
    numerically_accepted: bool
    mode: str
    reasons: tuple[str, ...]
    acceptance: AcceptanceRecord | None = None


class CapabilityRegistry:
    def __init__(self, *, source_leases=True):
        self.candidates = {}
        self.acceptances = {}
        self.generation_bridges = {}
        self._source_leases = FileDigestLeasePool(enabled=source_leases)

    def prepare_source_leases(self):
        """Cold protect current source/report bytes; unsupported OS uses strict reads.

        This grants no model/scope/numerical/owner/ABI rights. admit still checks
        those on every call. Before writing/replacing protected artifacts call
        invalidate_source_leases, then use normal independent qualification.
        """
        for candidate in self.candidates.values():
            candidate.verify_source_files(self._source_leases)
        for record in self.acceptances.values():
            if self._source_leases.digest(record.report_path, record.report_sha256) != record.report_sha256:
                raise ContractError('independent report changed or unavailable')
        for bridge in self.generation_bridges.values():
            bridge.verify_report(self._source_leases)
        return self._source_leases.state()

    def invalidate_source_leases(self, reason='explicit update', *, strict=False):
        """Revoke/close OS protection BEFORE source/report/build mutation/release."""
        return self._source_leases.invalidate(reason, disable=strict)

    @property
    def source_lease_state(self):
        return self._source_leases.state()

    @classmethod
    def from_document(cls, document):
        """Load candidates, never trust cached execution/accepted flags in a receipt."""
        registry = cls()
        entries = document.get('entries', [document])
        for row in entries:
            payload = row.get('candidate')
            if payload is None:
                raise ContractError('full candidate descriptor required, receipt alone cannot license execution')
            try:
                candidate = KernelCandidate(**payload)
            except TypeError as exc:
                raise ContractError('candidate descriptor fields differ from schema') from exc
            if row.get('candidate_id', candidate.candidate_id) != candidate.candidate_id:
                raise ContractError('candidate receipt id differs from descriptor')
            registry.register_candidate(candidate)
        # Acceptance is deliberately a separate source/report-verified operation.
        return registry

    @classmethod
    def from_file(cls, path, expected_sha256=None):
        raw = Path(path).read_bytes()
        if expected_sha256 is not None:
            _digest(expected_sha256, 'registry manifest')
            if hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise ContractError('registry manifest content differs from sealed SHA256')
        return cls.from_document(json.loads(raw))

    def register_candidate(self, candidate: KernelCandidate):
        prior = self.candidates.get(candidate.candidate_id)
        if prior is not None and prior != candidate:
            raise ContractError('candidate id cannot change source/capability; use new id')
        self.candidates[candidate.candidate_id] = candidate
        # from_document/from_file and component cold registration use this
        # existing entrance. Seal eagerly, but do NOT move execution errors to
        # registration: missing/changed/unsupported sources remain strict at
        # the original admit boundary. Empty manifests retain external guards.
        for path, expected in candidate.source_files.items():
            try:
                self._source_leases.digest(path, expected)
            except OSError:
                pass

    def record_acceptance(self, record: AcceptanceRecord):
        try:
            digest = hashlib.sha256(Path(record.report_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ContractError('independent report file is unavailable') from exc
        if digest != record.report_sha256:
            raise ContractError('independent report content differs from bound SHA256')
        candidate = self.candidates.get(record.candidate_id)
        if candidate is None or candidate.source_sha256 != record.source_sha256:
            raise ContractError('acceptance source does not match registered candidate')
        if record.execution_scope != candidate.execution_scope:
            raise ContractError('acceptance execution scope differs from candidate')
        if record.status == 'accepted' and record.reviewer == candidate.author:
            raise ContractError('candidate author cannot independently accept own candidate')
        if set(record.permitted_parameter_fields) - set(candidate.mutable_parameters):
            raise ContractError('acceptance claims parameter changes outside candidate capability')
        # Only retain protection after all original qualification checks pass.
        try:
            digest = self._source_leases.digest(record.report_path, record.report_sha256)
        except OSError as exc:
            raise ContractError('independent report file is unavailable') from exc
        if digest != record.report_sha256:
            raise ContractError('independent report content differs from bound SHA256')
        self.acceptances[(record.candidate_id, record.model_identity_key)] = record

    def admit(self, candidate_id, model: ModelSpec, *, mode='production', parameter_fields=(), required_scope=None):
        # A cold all-unsupported/disabled registry uses the original byte-read
        # path, without per-file pool locks, abspath or bridge namespace work.
        # New registrations or explicit prepare_source_leases retry cold seals.
        if not self._source_leases._leases:
            return self._admit(candidate_id, model, mode=mode, parameter_fields=parameter_fields,
                               required_scope=required_scope)
        with self._source_leases.verification():
            return self._admit(candidate_id, model, mode=mode, parameter_fields=parameter_fields,
                               required_scope=required_scope, source_leases=self._source_leases)

    def _admit(self, candidate_id, model: ModelSpec, *, mode='production', parameter_fields=(), required_scope=None, source_leases=None):
        if mode not in ('production', 'diagnostic'):
            raise ContractError('admission mode must be production/diagnostic')
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            return Admission(candidate_id, model.identity_key, False, False, False, mode,
                             ('candidate not registered',))
        reasons = list(candidate.structural_reasons(model))
        if required_scope is not None and candidate.execution_scope != required_scope:
            reasons.append('candidate execution scope is insufficient')
        if candidate.source_files:
            try:
                candidate.verify_source_files(source_leases)
            except ContractError as exc:
                reasons.append(str(exc))
        if set(parameter_fields) - set(candidate.mutable_parameters):
            reasons.append('parameter change outside candidate capability')
        structural = not reasons
        record = self.acceptances.get((candidate_id, model.identity_key))
        accepted = bool(record and record.status == 'accepted')
        if accepted:
            try:
                report_hash = (source_leases.digest(record.report_path, record.report_sha256) if source_leases is not None else
                               hashlib.sha256(Path(record.report_path).read_bytes()).hexdigest())
            except OSError:
                report_hash = None
            if report_hash != record.report_sha256:
                accepted = False
                reasons.append('independent report changed or unavailable')
        if parameter_fields and (not accepted or set(parameter_fields) - set(record.permitted_parameter_fields)):
            accepted = False
            if mode == 'production':
                reasons.append('parameter-change numerical qualification absent')
        if record and record.status == 'rejected':
            reasons.append('independent acceptance rejected')
        if mode == 'production' and not accepted:
            reasons.append('independent model qualification pending')
        allowed = structural and (accepted if mode == 'production' else not (record and record.status == 'rejected'))
        return Admission(candidate_id, model.identity_key, structural, allowed,
                         structural and accepted, mode, tuple(reasons), record)

    def record_generation_bridge(self, bridge: QualifiedGenerationBridge):
        candidate = self.candidates.get(bridge.candidate_id)
        if candidate is None or candidate.source_sha256 != bridge.source_sha256:
            raise ContractError('generation bridge candidate/source differs')
        if bridge.reviewer == candidate.author:
            raise ContractError('candidate author cannot independently qualify generation bridge')
        if set(bridge.changed_fields) - set(candidate.generation_rebind_fields):
            raise ContractError('candidate does not support these generation source rebindings')
        bridge.verify_report(self._source_leases)
        self.generation_bridges[(bridge.candidate_id, bridge.base_model_identity,
                                 bridge.derived_model_identity)] = bridge

    def admit_generation(self, candidate_id, base: ModelSpec, derived: ModelSpec, *, required_scope='component-runtime'):
        """Transfer qualification only through explicit generic capability and proof.

        Original fixed-model guards remain mandatory: this method cannot bypass
        an exact-MJB-only candidate scope or a kernel's own static validation.
        """
        base_admission = self.admit(candidate_id, base, required_scope=required_scope)
        candidate = self.candidates.get(candidate_id)
        reasons = list(base_admission.reasons)
        bridge = self.generation_bridges.get((candidate_id, base.identity_key, derived.identity_key))
        if candidate is not None:
            reasons.extend(candidate.structural_reasons(derived))
        if bridge is None:
            reasons.append('independent generation bridge absent')
        else:
            if (base_admission.acceptance is None or
                    set(bridge.changed_fields) - set(base_admission.acceptance.permitted_parameter_fields)):
                reasons.append('base numerical qualification does not cover generation fields')
            try:
                bridge.verify_report(self._source_leases if self._source_leases._leases else None)
            except ContractError as exc:
                reasons.append(str(exc))
        allowed = base_admission.execution_allowed and bridge is not None and not reasons
        return Admission(candidate_id, derived.identity_key, not reasons, allowed, allowed,
                         'production-generation', tuple(reasons), base_admission.acceptance)

    def support_row(self, task_id, model, components: Mapping[str, str],
                    required_components=('F', 'D', 'C', 'S', 'N', 'Euler', 'observables', 'environment', 'renderer')):
        entries = {component: self.admit(candidate_id, model, required_scope='component-runtime')
                   for component, candidate_id in components.items()}
        missing = tuple(sorted(set(required_components) - set(components)))
        return {'task_id': task_id, 'model_identity_key': model.identity_key,
                'specialization_key': model.specialization_key,
                'status': 'accepted' if entries and not missing and all(a.numerically_accepted for a in entries.values()) else 'pending',
                'missing_components': list(missing),
                'components': {name: {'candidate_id': a.candidate_id, 'structural_ready': a.structural_ready,
                                     'execution_allowed': a.execution_allowed,
                                     'numerically_accepted': a.numerically_accepted, 'reasons': list(a.reasons)}
                               for name, a in entries.items()}}











