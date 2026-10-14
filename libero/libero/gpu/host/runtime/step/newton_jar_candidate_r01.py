"""Unselected cold generic jar-only candidate; no hot solver replacement."""
import ast
import hashlib
import inspect
from pathlib import Path


def prepare_jar_final_event_runtime(**kwargs):
    # This candidate is not independently registered for production yet.
    if kwargs.get("mode", "diagnostic") != "diagnostic":
        raise ValueError("jar-only candidate requires explicit diagnostic mode")
    from libero.gpu.host.runtime.newton import live_copy_api as original
    from libero.gpu.host.runtime.step.newton_final_event import FinalEventAdapter
    from libero.gpu.host.runtime.step.s04_jar_specialization_r01 import load_candidate
    tree = ast.parse(inspect.getsource(original.prepare_live_copy_runtime))
    constructors = loaders = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "ColdNewtonRuntimeAdapter":
                node.func.id = "FinalEventAdapter"
                constructors += 1
            elif node.func.id == "load_candidate":
                node.func.id = "load_jar_candidate"
                loaders += 1
    if (constructors, loaders) != (1, 1):
        raise ValueError("one original cold loader and adapter required")
    ast.fix_missing_locations(tree)
    namespace = dict(vars(original))
    namespace.update(FinalEventAdapter=FinalEventAdapter, load_jar_candidate=load_candidate)
    exec(compile(tree, original.__file__ + ":jar-final-event-cold", "exec"), namespace)
    adapter = namespace["prepare_live_copy_runtime"](**kwargs)
    adapter.source_manifest["jar_cold_api_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    adapter.source_manifest["final_event_diagnostic_source_sha256"] = hashlib.sha256(Path(inspect.getfile(FinalEventAdapter)).read_bytes()).hexdigest()
    adapter.source_manifest["jar_candidate_selected_by_actual_predicate"] = False
    adapter.final_event = adapter.solver
    return adapter
