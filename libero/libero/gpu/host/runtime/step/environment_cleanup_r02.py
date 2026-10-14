"""Cold failure cleanup ownership; original boundary kernels remain identical."""
import ast
import inspect
from . import environment as original
from libero.gpu.host.adapters.environment.cleanup_ownership_r02 import ReleaseOwner, attach_cleanup


def _derive():
    tree=ast.parse(inspect.getsource(original.make_boundary_callbacks))
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=="libero.gpu.host.adapters.environment.raw_factory":
            node.module="libero.gpu.host.adapters.environment.cleanup_factory_r02"
        if isinstance(node,ast.ExceptHandler) and isinstance(node.name,str) and node.name=="cleanup":
            addition=ast.parse("attach_cleanup(error,ReleaseOwner(fresh,method_name=\"close\"))").body[0]
            node.body.append(addition);changed+=1
    if changed!=1:
        raise ValueError("exactly one fresh-close failure boundary required")
    ast.fix_missing_locations(tree)
    namespace=dict(vars(original));namespace.update(ReleaseOwner=ReleaseOwner,attach_cleanup=attach_cleanup)
    exec(compile(tree,original.__file__+":typed-retryable-cleanup","exec"),namespace)
    return namespace["make_boundary_callbacks"]

make_boundary_callbacks=_derive()
# These exact original Warp objects are used by the derived function globals.
_written=original._written
_model_commit=original._model_commit
