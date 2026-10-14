"""Shape-wide recovery and retryable cleanup, one recursive cold route."""
import ast
import inspect
from . import recovery_final_event_environment as original
from libero.gpu.host.adapters.environment.cleanup_ownership_r02 import ReleaseOwner, attach_cleanup

def _derive():
    tree=ast.parse(inspect.getsource(original.make_boundary_callbacks))
    route=0;cleanup=0
    function=tree.body[0]
    function.body.insert(1,ast.parse("domain_admission=getattr(bundle,\"domain_admission\",None)").body[0])
    function.body.insert(2,ast.parse("from .recovery_final_event_factory import prepare_final_event_recovery_physics as _default_prepare").body[0])
    function.body.insert(3,ast.parse("_prepare_source=getattr(bundle,"+repr("cold_recovery_prepare")+",_default_prepare)").body[0])
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="prepare_shape_physics":
            node.func.id="_prepare_source"
        if isinstance(node,ast.ImportFrom) and node.module=="recovery_final_event_environment_factory":
            node.module="recovery_cleanup_factory";route+=1
        if isinstance(node,ast.ExceptHandler) and node.name=="cleanup":
            node.body.append(ast.parse("attach_cleanup(error,ReleaseOwner(fresh,method_name="+repr("close")+"))").body[0])
            cleanup+=1
    if (route,cleanup)!=(1,1):raise ValueError("exact recursive route and cleanup boundary required")
    class DomainBoundary(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            targets=[x.id for x in node.targets if isinstance(x,ast.Name)]
            if "preparation" in targets:
                return [node]+ast.parse("domain_lease=None if domain_admission is None else domain_admission.admission_domain_for_built(preparation.built,factory_entry="+repr("libero.gpu.host.adapters.environment.production_factory_r01:create")+")").body
            if "fresh" in targets and isinstance(node.value,ast.Call):
                return [node]+ast.parse("fresh.domain_admission=domain_admission\nfresh.domain_lease=domain_lease\nfresh.cold_recovery_prepare=_prepare_source").body
            if "components" in targets:
                check=ast.parse("if domain_lease is not None:"+chr(10)+"    from .recovery_cleanup_factory import _publication"+chr(10)+"    _publication(fresh,domain_lease,token.next_generation)").body
                return check+[node]
            return node
    tree=DomainBoundary().visit(tree)
    ast.fix_missing_locations(tree)
    namespace=dict(vars(original))
    namespace.update(ReleaseOwner=ReleaseOwner,attach_cleanup=attach_cleanup)
    exec(compile(tree,original.__file__+":recovery-typed-cleanup","exec"),namespace)
    return namespace["make_boundary_callbacks"]
make_boundary_callbacks=_derive()
_written=original._written
_model_commit=original._model_commit
