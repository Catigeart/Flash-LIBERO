"""One explicit composite provider: source history, wide recovery and cleanup.

Diagnostic execution and supported-domain admission use the identical physical
route. A formal admission must be a verified SupportedDomainAdmission object.
This module never globally replaces another provider or changes kernel owners.
"""
from .facade import UnsupportedGpuOperation
ENTRY='libero.gpu.host.adapters.environment.production_factory_r01:create'

def _physical_provider(admission, physics_factory=None):
    from libero.gpu.host.runtime.core.namespace import require_canonical_gpu_namespace
    require_canonical_gpu_namespace()
    from libero.gpu.host.runtime.step import recovery_cleanup_factory as physical
    from libero.gpu.model.sampler_history import wrap_runtime_factory
    if physics_factory is None:
        from libero.gpu.host.runtime.step.recovery_full_angle_factory import prepare_full_angle_recovery_physics
        physics_factory=prepare_full_angle_recovery_physics
    if not callable(physics_factory):raise TypeError('explicit physical prepare callable required')
    def create(built):return physical.create(built,admission=admission,physics_factory=physics_factory)
    def rebuild_components(built,previous):return physical.rebuild_components(built,previous,admission=admission,physics_factory=physics_factory)
    create.rebuild_components=rebuild_components
    # Exactly one history wrapper; it owns no native or GPU arena itself.
    return wrap_runtime_factory(create)

def factory_for_admission(admission, *, physics_factory=None):
    from libero.gpu.host.runtime.production.domain_lease import SupportedDomainAdmission
    if not isinstance(admission,SupportedDomainAdmission):
        raise TypeError('verified supported-domain admission object required')
    provider=_physical_provider(admission,physics_factory)
    provider.production_entry=ENTRY
    provider.admission=admission
    return provider

_DIAGNOSTIC_PROVIDER=None

def _diagnostic_provider():
    global _DIAGNOSTIC_PROVIDER
    if _DIAGNOSTIC_PROVIDER is None:_DIAGNOSTIC_PROVIDER=_physical_provider(None)
    return _DIAGNOSTIC_PROVIDER

def create_diagnostic(built):
    return _diagnostic_provider()(built)

def rebuild_diagnostic(built,previous):
    return _diagnostic_provider().rebuild_components(built,previous)

create_diagnostic.rebuild_components=rebuild_diagnostic
create_diagnostic.production_entry=ENTRY
create_diagnostic.admission=None

def _admission_from_environment():
    import os
    from libero.gpu.host.runtime.production.domain_lease import SupportedDomainAdmission
    path=os.environ.get('LIBERO_GPU_SUPPORTED_DOMAIN_RECEIPT')
    digest=os.environ.get('LIBERO_GPU_SUPPORTED_DOMAIN_RECEIPT_SHA256')
    if not path or not digest:
        raise UnsupportedGpuOperation('production creation requires a pinned independently accepted supported-domain receipt')
    return SupportedDomainAdmission({'path':path,'sha256':digest})

def create(built):
    # Each environment receives a private, immutable receipt-bound closure.
    # The public loader itself validates the receipt; a boolean cannot grant it.
    return factory_for_admission(_admission_from_environment())(built)

def rebuild_components(built,previous):
    binding=getattr(previous.observation_fields,'__self__',None)
    prepared=getattr(binding,'prepared',None)
    if prepared is None or prepared.runtime is not previous.physics or binding.observer is not previous.observations or prepared.stream is not previous.stream:
        raise UnsupportedGpuOperation('production rebuild has no exact current prepared owner')
    admission=getattr(prepared,'domain_admission',None)
    return factory_for_admission(admission).rebuild_components(built,previous)

create.rebuild_components=rebuild_components
