"""Diagnostic one-route history/cleanup/wide recovery with sparse event copies."""
from libero.gpu.host.adapters.environment.production_factory_r01 import _physical_provider
from .jar_recovery_factory_r01 import prepare_jar_recovery_physics
_PROVIDER=None

def _provider():
    global _PROVIDER
    if _PROVIDER is None:_PROVIDER=_physical_provider(None,prepare_jar_recovery_physics)
    return _PROVIDER

def create(built):return _provider()(built)
def rebuild_components(built,previous):return _provider().rebuild_components(built,previous)
create.rebuild_components=rebuild_components
