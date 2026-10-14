"""Explicit jar implementation with whole-backend admission and guarded entries.

Inner diagnostic component manifests retain their scope. Only an independently
accepted whole-backend admission authorizes the outer supported source domain.
"""
import hashlib
import json
from types import MappingProxyType
ENTRY = "libero.gpu.host.adapters.environment.production_factory_r02:create"

def _authorization(admission):
    from libero.gpu.host.runtime.production.domain_lease import SupportedDomainAdmission
    from libero.gpu.host.runtime.production.admission import QualificationError, _json
    if not isinstance(admission, SupportedDomainAdmission):
        raise TypeError("verified supported-domain admission object required")
    candidate = json.loads(admission._candidate_json)
    mapping = candidate.get("implementation_mapping", {})
    if (mapping.get("schema") != "supported-jar-implementation-v1" or
        mapping.get("combined_factory_entry") != ENTRY or
        candidate["domain"]["combined_factory_entry"] != ENTRY):
        raise QualificationError("jar whole-backend implementation entry differs")
    selection_record = mapping.get("jar_selection_receipt")
    selection = _json(selection_record)
    if selection.get("verdict") != "PASS_SCOPED":
        raise QualificationError("jar implementation selection is not independently accepted")
    # The outer admission loader binds source-equivalence and selection records
    # into its independently reviewed candidate. Recheck these real bytes here.
    _json(mapping["source_equivalence_review"])
    implementation_digest = hashlib.sha256(json.dumps(candidate["source_files"],
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return MappingProxyType({"schema":"whole-backend-domain-delegation-v1",
        "combined_factory_entry":ENTRY,"candidate_identity":admission._candidate_identity,
        "implementation_digest":implementation_digest,
        "actual_selection_receipt":MappingProxyType(dict(selection_record)),
        "delegated_inner_mode":"diagnostic","component_production_receipt":False})

def _provider(admission):
    from libero.gpu.host.runtime.core.namespace import require_canonical_gpu_namespace
    require_canonical_gpu_namespace()
    from libero.gpu.host.runtime.step import recovery_cleanup_factory_r02 as physical
    from libero.gpu.host.runtime.step.jar_recovery_factory_r01 import prepare_jar_recovery_physics
    from libero.gpu.model.sampler_history import wrap_runtime_factory
    authorization = None if admission is None else _authorization(admission)
    def selected_prepare(*args, **kwargs):
        if kwargs.get("mode", "diagnostic") != "diagnostic":
            raise ValueError("inner jar component delegates diagnostic mode only")
        bundle = prepare_jar_recovery_physics(*args, **kwargs)
        bundle.backend_domain_authorization = authorization
        return bundle
    def create(built):
        return physical.create(built, admission=admission,
            physics_factory=selected_prepare, factory_entry=ENTRY)
    def rebuild(built, previous):
        return physical.rebuild_components(built, previous, admission=admission,
            physics_factory=selected_prepare, factory_entry=ENTRY)
    create.rebuild_components = rebuild
    provider = wrap_runtime_factory(create)
    provider.production_entry = ENTRY
    provider.admission = admission
    provider.backend_domain_authorization = authorization
    return provider

def factory_for_admission(admission):
    # No external physics override: this ENTRY names one explicit implementation.
    if admission is None:
        raise TypeError("verified supported-domain admission object required")
    return _provider(admission)

def create(built, *, admission):
    return factory_for_admission(admission)(built)

def create_diagnostic(built):
    return _provider(None)(built)

def rebuild_diagnostic(built, previous):
    return _provider(None).rebuild_components(built, previous)

create_diagnostic.rebuild_components = rebuild_diagnostic
create_diagnostic.production_entry = ENTRY
create_diagnostic.admission = None
