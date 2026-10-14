"""Cold receipt/admission boundaries for explicit independent C construction."""
from contextlib import contextmanager


def prepare_collision_admission(static,*,context=None,options=None):
    from libero.gpu.host.paths import GPU_ROOT
    from .cold_geometry import CollisionColdAdmission
    maintained=() if not isinstance(context,dict) else context.get('maintained_source_paths',())
    return CollisionColdAdmission(static,gpu_root=GPU_ROOT,context=context,options=options,maintained_sources=maintained)


@contextmanager
def collision_build_receipts_scope():
    # H04 remains globally off. This ABI requests complete receipts explicitly.
    from libero.gpu.host.build.native import strict_build_receipts_scope
    with strict_build_receipts_scope(enabled=True):yield
