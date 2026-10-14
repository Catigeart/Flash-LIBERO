"""Source-owner supported-action gate before any GPU step or task mutation.

The Task subclass retains its original admission owner across replacement.
An absent/foreign current admission cannot silently select diagnostic mode.
This helper imports no GPU/model runtime and performs no host/device writes.
"""
from .admission import QualificationError
from .domain_lease import SupportedDomainAdmission, DomainLease

class SupportedOwnerError(QualificationError):
    """Prepared arena differs from the retained admission/physical owner."""

def validate_supported_owner_action(task, action):
    if not hasattr(task, '_domain_admission_owner'):
        raise SupportedOwnerError('supported Task has no retained original admission owner')
    admission=task._domain_admission_owner
    c=task.components
    binding=getattr(c.observation_fields, '__self__', None)
    prepared=getattr(binding, 'prepared', None)
    # Diagnostic selection is retained at creation, never inferred from a
    # subsequently missing current arena field. Its original clip path remains.
    if admission is None:
        if prepared is not None and getattr(prepared, 'domain_admission', None) is not None:
            raise SupportedOwnerError('diagnostic Task acquired foreign supported admission')
        return None
    if not isinstance(admission, SupportedDomainAdmission):
        raise SupportedOwnerError('retained original admission is not receipt qualified')
    if (prepared is None or prepared.runtime is not c.physics or
        prepared.stream is not c.stream or binding.observer is not c.observations or
        getattr(prepared, 'domain_admission', None) is not admission):
        raise SupportedOwnerError('current prepared arena differs from retained supported owner')
    lease=getattr(prepared, 'domain_lease', None)
    if (not isinstance(lease, DomainLease) or
        lease.candidate_identity!=admission._candidate_identity or
        lease.model_sha256!=c.built.metadata['model_sha256'] or
        lease.model_sha256!=c.sim._model_generation.spec.model_sha256):
        raise SupportedOwnerError('current physical model lease is foreign or stale')
    # Returns validation only. The original Task consumes the original action,
    # retaining its dtype conversion/controller scale/clip arithmetic unchanged.
    # Invalid actions raise QualificationError without quarantining the owner.
    admission.check_action(lease, action)
    return None
