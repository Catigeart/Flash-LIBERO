"""Cold second-batch fusion wiring; original and first-batch factories retained."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

from . import factory as first
from .newton import FusedFinalEventAdapter
from libero.gpu.host.runtime.newton import live_copy_api
from libero.gpu.host.runtime.step import recovery_event_factory, s04_jar_specialization_r01


def _load_candidate(nv, nm, build_root):
    from .newton_iteration import fused_solver_class
    package, manifest = s04_jar_specialization_r01.load_candidate(nv, nm, build_root)
    solver_type = fused_solver_class(package)
    # Private cold view: never mutate the shared generated candidate module.
    view = SimpleNamespace(workspace=package.workspace,
                           solver=SimpleNamespace(FwdConstraintSolver=solver_type))
    return view, dict(manifest, second_batch_fusion=True)


def _newton_factory(**kwargs):
    adapter = first._derive(live_copy_api.prepare_live_copy_runtime,
        {"load_candidate": _load_candidate,
         "ColdNewtonRuntimeAdapter": FusedFinalEventAdapter})(**kwargs)
    adapter.final_event = adapter.solver
    return adapter


def _cold_factories():
    collision, unused, choice = first._cold_factories()
    return collision, _newton_factory, choice


class _SecondBatchImports(first._RecoveryImports):
    inserted = 0

    def visit_For(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.iter) == "ast.walk(tree)":
            node.body.extend(ast.parse('''
if isinstance(node, ast.ImportFrom) and node.module == "shape_physics":
    node.module = "libero.gpu.host.assembly.compose.backup"
    node.level = 0
    node.names = [ast.alias(name="FusedShapePhysicsForward", asname="ShapePhysicsForward")]
if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "sm"
        and node.func.attr == "PositionReferenceWorkspace"):
    node.func = ast.Call(func=ast.Name(id="_fused_s_workspace_type", ctx=ast.Load()),
                         args=[node.func], keywords=[])
''').body)
            self.inserted += 1
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "dict" and any(
                k.arg == "_recovery_loop" for k in node.keywords):
            node.keywords.append(ast.keyword(arg="_fused_s_workspace_type",
                                value=ast.Name(id="_fused_s_workspace_type", ctx=ast.Load())))
        return node


def prepare_fused_physics(*args, **kwargs):
    if kwargs.get("mode", "diagnostic") != "diagnostic":
        raise ValueError("second fusion batch requires explicit diagnostic mode")
    from .backup import FusedEventRecoveringRuntime
    from .s_rows import fused_workspace_type
    recovery_loop = first._derive(recovery_event_factory._recovery_loop,
        {"RecoveringShapeRuntime": FusedEventRecoveringRuntime})
    transform = _SecondBatchImports()
    prepare = first._derive(recovery_event_factory.prepare_recovery_event_physics,
        {"_event_cold_factories": _cold_factories, "_recovery_loop": recovery_loop,
         "_fused_s_workspace_type": fused_workspace_type}, transform)
    if (transform.dynamics, transform.euler, transform.inserted) != (1, 1, 1):
        raise ValueError("exact cold constructor transformation required")
    bundle = prepare(*args, **kwargs)
    bundle.fusion_source_manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (GPU_ROOT / 'host/assembly/compose').glob("*.py")}
    bundle.fusion_batch = 2
    return bundle


def create(built):
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    return assemble(built, physics_factory=prepare_fused_physics,
        factory_entry="libero.gpu.host.assembly.compose.factory_r02:create")
