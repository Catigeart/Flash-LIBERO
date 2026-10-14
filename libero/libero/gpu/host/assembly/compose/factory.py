"""Cold opt-in wiring for D/Euler/N fusion; no original factory mutation."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

from libero.gpu.host.runtime.dynamics import wide_runtime, wide_factory
from libero.gpu.host.runtime.step import recovery_event_factory, newton_jar_candidate_r01
from .dynamics import FusedSmoothWorkspace


def _derive(function, replacements, transform=None):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if transform is not None:
        tree = transform.visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    exec(compile(tree, function.__code__.co_filename + ":fusion-r01-cold", "exec"), namespace)
    return namespace[function.__name__]


class FusedWideDynamicsAdapter(wide_runtime.GenericWideRuntimeDynamicsAdapter):
    __init__ = _derive(wide_runtime.GenericWideRuntimeDynamicsAdapter.__init__,
                       {"GenericSmoothWorkspace": FusedSmoothWorkspace})


class _DynamicsSources(ast.NodeTransformer):
    count = 0

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "names" for t in node.targets):
            if not isinstance(node.value, ast.Tuple):
                raise ValueError("original dynamics source bundle tuple required")
            node.value.elts.extend(ast.Constant(value="host/assembly/compose/" + name) for name in
                ("__init__.py", "compose.py", "dynamics.py", "euler.py", "newton.py", "factory.py"))
            self.count += 1
        return self.generic_visit(node)


def prepare_from_built(*args, **kwargs):
    transform = _DynamicsSources()
    function = _derive(wide_factory.prepare_from_built,
        {"GenericRuntimeDynamicsAdapter": FusedWideDynamicsAdapter}, transform)
    if transform.count != 1:
        raise ValueError("one original dynamics source bundle required")
    return function(*args, **kwargs)


prepare_post = wide_factory.prepare_post


def _newton_factory(**kwargs):
    # This substitution precedes allocation, live binding, seal and S07 setup.
    tree = ast.parse(inspect.getsource(newton_jar_candidate_r01.prepare_jar_final_event_runtime))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "libero.gpu.host.runtime.step.newton_final_event":
            node.module = "libero.gpu.host.assembly.compose.newton"
            node.names = [ast.alias(name="FusedFinalEventAdapter", asname="FinalEventAdapter")]
            changes += 1
    if changes != 1:
        raise ValueError("one cold final-event adapter import required")
    ast.fix_missing_locations(tree)
    namespace = dict(newton_jar_candidate_r01.prepare_jar_final_event_runtime.__globals__)
    exec(compile(tree, __file__ + ":newton-fusion-cold", "exec"), namespace)
    adapter = namespace["prepare_jar_final_event_runtime"](**kwargs)
    adapter.source_manifest["fusion_source_files"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), (GPU_ROOT / 'host/assembly/compose/newton.py'),
                  (GPU_ROOT / 'host/assembly/compose/compose.py'))}
    return adapter


def _cold_factories():
    from libero.gpu.host.runtime.step.jar_recovery_factory_r01 import _jar_cold_factories
    collision, unused, choice = _jar_cold_factories()
    return collision, _newton_factory, choice


class _RecoveryImports(ast.NodeTransformer):
    dynamics = 0
    euler = 0

    def visit_Constant(self, node):
        if node.value == "libero.gpu.host.runtime.dynamics.wide_factory":
            self.dynamics += 1
            return ast.Constant(value="libero.gpu.host.assembly.compose.factory")
        return node

    def visit_ImportFrom(self, node):
        if node.module == "libero.gpu.reference.integration.euler_integrator.wide_angle":
            node.module = "libero.gpu.host.assembly.compose.euler"
            node.names = [ast.alias(name="FusedEulerWorkspace", asname="WideAngleEulerWorkspace")]
            self.euler += 1
        return node


def prepare_fused_physics(*args, **kwargs):
    if kwargs.get("mode", "diagnostic") != "diagnostic":
        raise ValueError("fusion candidate requires its own qualification")
    transform = _RecoveryImports()
    function = _derive(recovery_event_factory.prepare_recovery_event_physics,
                       {"_event_cold_factories": _cold_factories}, transform)
    if (transform.dynamics, transform.euler) != (1, 1):
        raise ValueError("exact original cold dynamics and Euler imports required")
    bundle = function(*args, **kwargs)
    paths = tuple((GPU_ROOT / 'host/assembly/compose').glob("*.py"))
    bundle.fusion_source_manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return bundle


def create(built):
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    return assemble(built, physics_factory=prepare_fused_physics,
                    factory_entry="libero.gpu.host.assembly.compose.factory:create")
