"""Diagnostic same-combination provider with passive consumption witnesses."""
from libero.gpu.host.adapters.environment.production_factory_r01 import _physical_provider
from .quality_factory_r01 import prepare_quality_recovery_physics
_PROVIDER=None

def _provider():
    global _PROVIDER
    if _PROVIDER is None:_PROVIDER=_physical_provider(None,prepare_quality_recovery_physics)
    return _PROVIDER

def create(built):return _provider()(built)
def rebuild_components(built,previous):return _provider().rebuild_components(built,previous)
create.rebuild_components=rebuild_components

def capture_task_quality(task):
    binding=getattr(task.components.observation_fields,"__self__",None)
    bundle=getattr(binding,"prepared",None)
    if bundle is None or bundle.runtime is not task.components.physics:
        raise ValueError("current combined physical owner absent")
    packet=bundle.runtime.forward.download_quality()
    if bundle.controller is not None:
        packet.update({"preOSC__"+k:v for k,v in bundle.controller.quality_control.download().items()})
        packet.update({"OSC_public__"+k:a.numpy() for k,a in bundle.controller.public_arrays().items()})
    return packet
