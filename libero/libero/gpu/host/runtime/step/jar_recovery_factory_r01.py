"""Unselected cold jar-only candidate for the same wide recovery route."""
import ast
import hashlib
import inspect
from pathlib import Path


def _jar_cold_factories():
    from .generic_prepare import _cold_factories
    from .newton_jar_candidate_r01 import prepare_jar_final_event_runtime
    collision, unused_newton, choice = _cold_factories()
    if choice != "cn":
        raise ValueError("jar-only candidate requires explicit CN cold scope")
    return collision, prepare_jar_final_event_runtime, choice


def prepare_jar_recovery_physics(*args, **kwargs):
    if kwargs.get("mode", "diagnostic") != "diagnostic":
        raise ValueError("jar-only candidate is not production qualified")
    from . import recovery_event_factory as original
    tree = ast.parse(inspect.getsource(original.prepare_recovery_event_physics))
    # The complete existing cold prepare routine is copied without AST edits.
    # Its retained cold callable is selected before any arena/bind/seal.
    namespace = dict(vars(original))
    namespace["_event_cold_factories"] = _jar_cold_factories
    exec(compile(tree, original.__file__ + ":jar-recovery-cold", "exec"), namespace)
    bundle = namespace["prepare_recovery_event_physics"](*args, **kwargs)
    bundle.recovery_source_manifest["jar_candidate_cold_factory_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle.recovery_source_manifest["unselected_jar_only_candidate"] = True
    return bundle
