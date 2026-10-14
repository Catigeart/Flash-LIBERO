"""Cold opt-in dataflow entry, including grid-based native world batching."""
from libero.gpu.host.assembly.components import prepare_cuda_physics as _prepare


def prepare_cuda_physics(*args,dataflow_profile="fast",profile=None,**kwargs):
    if profile is not None:dataflow_profile=profile
    return _prepare(*args,dataflow_profile=dataflow_profile,optimized_dataflow=True,**kwargs)

prepare_fused_physics=prepare_cuda_physics


def create(built):
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    return assemble(built,physics_factory=prepare_cuda_physics,
        factory_entry="libero.gpu.host.factory:create")


def rebuild_components(built,previous):
    from libero.gpu.host.runtime.step.shape_schema import RuntimeShape
    from libero.gpu.host.runtime.step import recovery_final_event_environment_factory as source
    RuntimeShape.from_native_metadata(built.metadata)
    source.default_integration_state(built.native_model)
    source.actual_decoder_record(built)
    previous.release()
    return create(built)

create.rebuild_components=rebuild_components
