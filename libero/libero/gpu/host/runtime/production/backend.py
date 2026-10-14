"""Receipt-bound public GPU backend lifecycle for source-supported domains.

No action trace allow-list, GPU import at module import, environment-variable
receipt replacement, or global factory registration occurs here. Real owners
perform cold publication and hot protocol/quality checks; the wrapper retains
those owners through reset failures and retryable close failures.
"""
import hashlib
import json
from pathlib import Path
from threading import RLock
from .admission import QualificationError, candidate_id
from .domain_lease import SupportedDomainAdmission, DomainLease

ENTRY='libero.gpu.host.adapters.environment.production_factory_r01:create'

def _placement_error_type():
    from robosuite.utils.errors import RandomizationError
    return RandomizationError

class SupportedCreationError(RuntimeError):
    """Creation failure with retained actual owner cleanup available to caller."""
    def __init__(self,primary,cleanup):
        super().__init__('supported GPU creation failed: '+str(primary))
        self.primary=primary
        self._cleanup=cleanup
    def close(self):
        if self._cleanup is not None:
            self._cleanup()  # Keep the owner on failure, so caller can retry.
            self._cleanup=None

class SupportedGpuEnvironment:
    physics_backend='gpu'
    def __init__(self,task,admission):
        if not isinstance(admission,SupportedDomainAdmission):
            raise TypeError('receipt-qualified domain admission required')
        self._task=task;self._admission=admission;self._lock=RLock()
        self._closed=False;self._quarantined=False
        self._domain=json.loads(admission._candidate_json)['domain']
        self._assert_source_epoch()
        self._lease()

    def _assert_source_epoch(self):
        for record in json.loads(self._admission._candidate_json)['source_files']:
            if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:
                self._quarantined=True
                raise QualificationError('supported cold implementation source epoch changed')

    def _open(self,*,reset_retry=False):
        if self._closed:raise RuntimeError('supported GPU environment is closed')
        if self._quarantined or getattr(self._task,'_unusable',False):
            raise RuntimeError('supported GPU owner quarantined; close and rebuild required')
        if getattr(self._task,'_needs_reset',False) and not reset_retry:
            raise RuntimeError('original placement needs reset retry')

    def _lease(self):
        c=self._task.components
        binding=getattr(c.observation_fields,'__self__',None)
        p=getattr(binding,'prepared',None)
        if (p is None or p.runtime is not c.physics or p.stream is not c.stream or
            binding.observer is not c.observations or
            getattr(p,'domain_admission',None) is not self._admission):
            self._quarantined=True
            raise QualificationError('public owner differs from receipt-bound prepared arena')
        l=getattr(p,'domain_lease',None)
        if (not isinstance(l,DomainLease) or l.candidate_identity!=self._admission._candidate_identity or
            l.model_sha256!=c.built.metadata['model_sha256'] or
            l.model_sha256!=c.sim._model_generation.spec.model_sha256):
            self._quarantined=True
            raise QualificationError('public arena has foreign or stale model lease')
        return l

    def _call(self,name,*args,reset_retry=False):
        self._open(reset_retry=reset_retry)
        try:
            self._lease()
            value=getattr(self._task,name)(*args)
            self._lease()  # Reset/cold update may replace the complete arena.
            return value
        except BaseException:
            # Original RandomizationError placement retry remains usable for
            # reset. Other failures retain actual owner and allow only close.
            if not getattr(self._task,'_needs_reset',False) or getattr(self._task,'_unusable',False):
                self._quarantined=True
            raise

    @property
    def action_spec(self):
        with self._lock:
            self._open();box=self._lease().action_box
            import numpy as np
            return np.asarray(box['lower'],dtype=np.float64),np.asarray(box['upper'],dtype=np.float64)
    @property
    def action_dim(self):return len(self.action_spec[0])
    def seed(self,seed=None):
        with self._lock:
            self._open()
            if seed is not None:
                # Same original NumPy prevalidation, no draw/owner mutation.
                # None is always valid and must not request entropy twice.
                import numpy as np
                np.random.RandomState(seed)
            return self._call('seed',seed)
    def reset(self):
        with self._lock:
            self._open(reset_retry=True)
            self._assert_source_epoch()  # Before old release or fresh imports.
            while True:
                try:return self._call('reset',reset_retry=True)
                except BaseException as error:
                    if (self._quarantined or not getattr(self._task,'_needs_reset',False) or
                        getattr(self._task,'_unusable',False) or not isinstance(error,_placement_error_type())):
                        self._quarantined=True
                        raise
                    # Exactly original ControlEnv placement retry semantics.
                    # Unsupported/numeric/capacity/unknown exceptions escape.
    def step(self,action):
        with self._lock:
            self._open()
            # Invalid external action fails before touching runtime and does not
            # quarantine an otherwise healthy owner. New values are permitted.
            checked=self._admission.check_action(self._lease(),action)
            return self._call('step',checked)
    def get_sim_state(self):
        with self._lock:return self._call('get_sim_state')
    def set_init_state(self,state):
        with self._lock:
            self._open()
            if 'set_init_state' not in self._domain.get('supported_operations',[]):
                raise QualificationError('official state upload outside this accepted domain')
            return self._call('regenerate_obs_from_state',state)
    def gpu_physics_evidence(self):
        with self._lock:
            actual=self._call('gpu_physics_evidence')
            return dict(actual,supported_domain_candidate_id=self._admission._candidate_identity,
                        support_scope=self._domain,every_future_field_residual_proved=False)
    def close(self):
        with self._lock:
            if self._closed:return
            try:self._task.close()
            except BaseException:
                self._quarantined=True
                raise  # Retain task and all pending actual cleanup leases.
            self._closed=True;self._task=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def _construct_task(admission,bddl_file_name,options):
    """Delegate the exact non-global constructor/cleanup to its source owner."""
    from libero.gpu.host.runtime.core.namespace import require_canonical_gpu_namespace
    require_canonical_gpu_namespace()
    from libero.gpu.host.adapters.environment.production_factory_r01 import factory_for_admission
    from libero.gpu.host.adapters.environment.task_creation_r01 import create_task_for_provider
    provider=factory_for_admission(admission)
    problem=json.loads(admission._candidate_json)['domain']['task']['problem_name']
    return create_task_for_provider(problem,provider,bddl_file_name,**options)


def create_supported_environment(receipt_record,*,bddl_file_name,**options):
    """Create a usable backend only after exact independent domain admission.

    Qualification is loaded once before any builder/GPU import. Construction
    options are those actually accepted by the integrated source domain; seeds,
    repeated original resets and every new legal action are lifecycle operations,
    rather than a sequence of replay permissions.
    """
    admission=SupportedDomainAdmission(receipt_record)
    domain=json.loads(admission._candidate_json)['domain']
    if domain.get('combined_factory_entry')!=ENTRY:
        raise QualificationError('supported domain uses a different combined route')
    if hashlib.sha256(Path(bddl_file_name).read_bytes()).hexdigest()!=domain['task']['bddl_sha256']:
        raise QualificationError('requested BDDL source differs')
    required_operations={'seed','reset','step','close','get_sim_state'}
    if not required_operations.issubset(set(domain.get('supported_operations',[]))):
        raise QualificationError('supported backend lifecycle is not accepted by this domain')
    expected=domain['environment_options']
    if options and options!=expected:
        raise QualificationError('requested construction options outside supported source domain')
    try:task=_construct_task(admission,bddl_file_name,expected)
    except SupportedCreationError:
        raise
    except BaseException as primary:
        pending=list(getattr(primary,'pending_gpu_cleanup',()))
        if pending:
            def retry(error=primary):
                # Except targets are cleared on exit; retain the real error.
                from libero.gpu.host.adapters.environment.task_creation_r01 import retry_creation_cleanup
                retry_creation_cleanup(error)
            raise SupportedCreationError(primary,retry) from primary
        raise
    try:return SupportedGpuEnvironment(task,admission)
    except BaseException as primary:
        try:task.close()
        except BaseException as cleanup_error:
            raise SupportedCreationError(primary,task.close) from cleanup_error
        raise
