"""Strict assembly of actual single-world runtime and task lifecycle.

This performs no physics, allocation, or acceptance. The runtime owner must
first prepare source-qualified kernels and all real lifecycle callbacks.
"""
from dataclasses import dataclass
import numpy as np
from .facade import GpuSimFacade, UnsupportedGpuOperation
from .task_driver import GpuTaskComponents
from libero.gpu.host.runtime.core.contracts import ModelSpec, ModelGeneration, DeviceWorldState, BufferSpec, BufferBinding


@dataclass(frozen=True)
class ResidentWorldSlice:
    array: object
    offset: int
    size: int


def assemble_single_world_task(built, prepared, boundary, observation_binding, spec):
    """Bind the exact compiled model to true device roots; reject stale packs.

    ``boundary`` is ResidentPhysicsFacadeRuntime with actual reset, step1,
    state-write, parameter-upload and forensic callbacks. No callback is
    replaced with a no-op or a full-forward substitute.
    """
    if not isinstance(spec, ModelSpec):raise TypeError('exact native ModelSpec required')
    if built.native_model is not prepared.model_pack.model:
        raise UnsupportedGpuOperation('task and runtime must share exact native model owner')
    actual = ModelSpec.from_factory_metadata(built.metadata, features=spec.features,
                                             mutable_parameters=spec.mutable_parameters)
    if actual.identity_key != spec.identity_key:
        raise UnsupportedGpuOperation('task descriptor differs from runtime source identity')
    if boundary.runtime is not prepared.runtime or boundary.stream is not prepared.stream:
        raise ValueError('foreign physics owner or stream')
    if boundary.nworld != 1 or boundary.world != 0:
        raise UnsupportedGpuOperation('single-world assembly requires one resident world')
    if len(built.domain_shell.robots) != 1:
        raise UnsupportedGpuOperation('single-robot controller assembly required')
    if observation_binding.prepared is not prepared or observation_binding.observer.model is not built.native_model:
        raise ValueError('foreign observation model or runtime')
    for callback,name in ((boundary._reset,'reset'),(boundary._step1,'step1'),
                          (boundary._state_written,'state_written'),
                          (boundary._parameters,'parameter_upload'),
                          (boundary._visual,'visual_upload'),(boundary._evidence,'evidence')):
        if not callable(callback):raise UnsupportedGpuOperation('actual lifecycle callback missing: '+name)
    controller=prepared.controller
    if controller is None:raise UnsupportedGpuOperation('actual GPU controller required')
    for name in ('stage_action','bootstrap_from_current_physics'):
        if not callable(getattr(controller,name,None)):
            raise UnsupportedGpuOperation('actual controller callback missing: '+name)
    generation=ModelGeneration(spec)
    world=DeviceWorldState(0,generation)
    for name in ('qpos','qvel','ctrl','qacc_warmstart','time','qfrc_applied','xfrc_applied'):
        field=boundary.fields[name];array=field.array
        if str(array.dtype) != 'float64':
            # Warp names may render as <class 'warp._src.types.float64'>.
            if getattr(array.dtype,'__name__',None) != 'float64':raise TypeError('FP64 actual root required: '+name)
        shape=(1,) if name=='time' else field.shape
        handle=ResidentWorldSlice(array,0,field.size)
        allocation_id=f'{boundary.allocation_id}:{array.device}:{int(array.ptr)}:{array.size}:{name}:world0'
        world.bind(BufferBinding(BufferSpec(name,shape,'float64','resident-physics'),handle,allocation_id))
    sim=GpuSimFacade(boundary,built.facade_metadata(),world_state=world,model_generation=generation)
    observation_binding.install()
    def bootstrap(robot,mask):
        controller.bootstrap_from_current_physics(mask,stream=prepared.stream)
    def stage_action(action):
        controller.stage_action(np.asarray(action,dtype=np.float64).reshape(1,-1))
    components = GpuTaskComponents(
        built,sim,prepared.runtime,controller,observation_binding.observer,
        prepared.runtime.intent,observation_fields=observation_binding.fields,
        synchronize_and_check=boundary._check,bootstrap_controller=bootstrap,
        stage_action=stage_action,close_runtime=sim.close,
        observation_hook=observation_binding.observation_hook,stream=prepared.stream,
    )

    # Initial creation and native/parameter replacement share this assembly.
    from .public_spatial import PublicSpatialQueries
    queries = PublicSpatialQueries(built, prepared, validate_owner=boundary._check)
    try:
        boundary._bind_spatial_queries(queries)
    except BaseException as primary:
        try: queries.close()
        except BaseException as error:
            from .cleanup_ownership_r02 import ReleaseOwner, attach_cleanup
            attach_cleanup(primary, ReleaseOwner(queries, 'close'))
            primary.gpu_cleanup_errors = [repr(error)]
        raise
    return components
