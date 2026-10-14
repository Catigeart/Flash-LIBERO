"""Original GPU lifecycle with publicly retryable failed cleanup owners."""
import gc
import numpy as np
from .task_driver import GpuTaskEnvironment as _OriginalTask
from .facade import UnsupportedGpuOperation, CompletedColdModelUpdate
from .cleanup_ownership_r02 import ReleaseOwner, attach_cleanup
class GpuTaskEnvironment(_OriginalTask):
    def __init__(self,*args,**kwargs):
        self._pending_cleanup=[]
        super().__init__(*args,**kwargs)
    def _adopt_cleanup(self,error):
        for lease in getattr(error,'pending_gpu_cleanup',()):
            if isinstance(lease,ReleaseOwner) and not lease.released and not any(x is lease or (not x.released and x.owner is lease.owner and x.method_name==lease.method_name) for x in self._pending_cleanup):
                self._pending_cleanup.append(lease)
    def _quarantine_replacement(self,detail):
        super()._quarantine_replacement(detail)
        self._adopt_cleanup(detail)
    def close(self):
        with self._lock:
            # Public operations stay quarantined until every retained owner has
            # released; each successfully released lease is removed immediately.
            while self._pending_cleanup:
                lease=self._pending_cleanup[0]
                try:lease.release()
                except BaseException as exc:
                    self._unusable=True
                    self.last_cleanup_failure={'type':type(exc).__name__,'message':str(exc)}
                    raise
                self._pending_cleanup.pop(0)
            super().close()

    def replace_components_for_model_update(self,build_new,token,model_generation):
        """Install restored actual owners atomically after cold model replacement.

        The source owner prepares a descriptor plan before old release, captures
        and restores actual device data, and returns (fresh_components, plan).
        No constructor, robot reset, implicit forward, or observation sample runs.
        """
        from libero.gpu.host.runtime.core.generation import commit_parameter_descriptor
        with self._lock:
            self._open()
            old=self.components
            if old.sim._model_generation is not model_generation:
                raise ValueError('foreign model generation replacement request')
            if not callable(build_new):raise TypeError('actual cold model builder required')
            if (token.model_identity != model_generation.spec.identity_key or
                    token.base_generation != model_generation.generation or
                    token.base_digest != model_generation.parameters_sha256 or
                    token.next_generation != model_generation.generation+1):
                raise ValueError('stale or foreign cold model update token')
            visual_metadata=getattr(old.sim.model,'_metadata',{})
            visual_snapshot={name:np.array(visual_metadata[name],copy=True)
                             for name in ('site_rgba','geom_rgba') if name in visual_metadata}
            visual_generation=getattr(old.sim,'visual_generation',0)
            state=getattr(old.sim,'_world_state',None)
            host_snapshot=None if state is None else {
                'state_epoch':state.state_epoch,'publications':dict(state.publications),
                'sequence':state._sequence,
            }
            self._replacing=True
            fresh=None
            try:
                fresh,plan=build_new(old)
                if not old.released:raise RuntimeError('old arena must release before new allocation')
                fresh.validate()
                if fresh.built.domain_shell is not old.built.domain_shell:
                    raise ValueError('parameter replacement must preserve the active task shell')
                if plan[0] != token or fresh.sim._model_generation.spec.identity_key != plan[1].identity_key:
                    raise ValueError('fresh native descriptor differs from parameter token/plan')
                # Visual state is independent of the fresh physical MJB/spec.
                # The observer owner has already restored actual visual bytes;
                # preserve this facade mirror without another GPU upload or gen.
                fresh_metadata=getattr(fresh.sim.model,'_metadata',{})
                for name,value in visual_snapshot.items():
                    target=fresh_metadata.get(name)
                    if not isinstance(target,np.ndarray) or target.shape!=value.shape or target.dtype!=value.dtype:
                        raise ValueError('fresh visual mirror schema differs: '+name)
                    value.flags.writeable=False
                    fresh_metadata[name]=value
                fresh.sim.visual_generation=visual_generation
                commit_parameter_descriptor(model_generation,plan)
                fresh.sim._model_generation=model_generation
                newstate=getattr(fresh.sim,'_world_state',None)
                if newstate is not None:
                    newstate.model=model_generation
                    if host_snapshot is not None:
                        newstate.state_epoch=host_snapshot['state_epoch']
                        # Old publication tags remain stale after generation+1;
                        # this host copy never manufactures a new device phase.
                        newstate.publications=dict(host_snapshot['publications'])
                        newstate._sequence=host_snapshot['sequence']
                self._replace_components(fresh)
                return CompletedColdModelUpdate(
                    model_generation,model_generation.spec.identity_key,
                    model_generation.generation,model_generation.parameters_sha256,
                    self._quarantine_replacement,
                )
            except BaseException as exc:
                self._quarantine_replacement(exc)
                if fresh is not None and fresh is not self.components:
                    quarantine=getattr(fresh.sim,'_quarantine',None)
                    if callable(quarantine):quarantine('model_replacement')
                    try:fresh.release()
                    except BaseException as close_error:
                        exc.gpu_cleanup_errors=list(getattr(exc,'gpu_cleanup_errors',()))+[repr(close_error)]
                        attach_cleanup(exc,ReleaseOwner(fresh,'release'))
                self._adopt_cleanup(exc)
                raise
            finally:
                self._replacing=False
