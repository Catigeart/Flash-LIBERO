"""Cold layout and Newton estimates from the current compiled model.

This module reads no fixture inventory and performs no simulation. A plan is
not a source, owner, resource, or numerical admission: the actual producer must
still validate its source/build policy, complete bindings and linked resources.
"""
from math import prod

from libero.gpu.host.bindings.capacity import qualify
from libero.gpu.host.runtime.step.shape_schema import RuntimeShape


def _capacity(name, value, *, optional=False):
    if value is None and optional:
        return None
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise ValueError('positive native int32 capacity required: ' + name)
    return value


def plan_native_model(metadata, *, nworld=1, rows=None, contacts=None,
                      shared_limit_bytes=48 * 1024):
    """Plan exact declared layouts without granting execution permission.

    Row/contact capacities must come from the current collision/constraint
    planner. Observed active counts and historical maxima cannot substitute for
    these bounds. Without them, only the model/state and Newton estimates are
    returned. Standard LIBERO models fit the parameterized small Newton route;
    larger models still need the complete tiled/global solver implementation.
    """
    shape = RuntimeShape.from_native_metadata(metadata)
    nworld = _capacity('nworld', nworld)
    rows = _capacity('rows', rows, optional=True)
    contacts = _capacity('contacts', contacts, optional=True)
    shared_limit_bytes = _capacity('shared_limit_bytes', shared_limit_bytes)
    if (rows is None) != (contacts is None):
        raise ValueError('row and contact capacities must be supplied together')
    states = shape.state_shapes(nworld)
    solver = {} if rows is None else shape.solver_shapes(nworld, rows, contacts)
    for name, dimensions in {**states, **solver}.items():
        if prod(dimensions) > 2**31 - 1:
            raise ValueError('int32 device offset overflow: ' + name)
    newton = qualify(shape.nv, shape.nM, rows=rows, contacts=contacts,
                     nworld=nworld, shared_limit_bytes=shared_limit_bytes)
    missing = []
    if rows is None:
        missing.append('current collision and constraint capacity plan')
    if not newton['admitted_by_formula']:
        missing.append('complete tiled Hessian/global Newton arena and lifecycle')
    return dict(
        schema='native-model-capacity-plan-v1',
        model_sha256=shape.model_sha256,
        dimensions={name: getattr(shape, name)
                    for name in shape.__dataclass_fields__ if name != 'model_sha256'},
        nworld=nworld, state_shapes=states, solver_shapes=solver,
        state_bytes=sum(prod(dimensions) * 8 for dimensions in states.values()),
        newton=newton, missing_prerequisites=missing,
        execution_admitted=False,
        scope='cold declared layout and shared-memory estimate only',
        required_runtime_checks=(
            'current source/build/policy and native model identity',
            'actual complete static options and supported component families',
            'complete collision/constraint and total device memory capacity',
            'linked CUDA descriptors and actual static/dynamic resources',
            'allocation/owner/lifetime and complete launch bindings'))
