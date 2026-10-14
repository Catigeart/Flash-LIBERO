"""Independent cold same-model batched recovery assembly, with unchanged shape physics order."""
import ast
import hashlib
import inspect
import numpy as np
import warp as wp
from . import shape_prepare
from .recovery_runtime_r02 import CurrentWrittenRecoveringRuntime as RecoveringShapeRuntime
from libero.gpu.host.runtime.dynamics.recovery import ROOT_WIDTHS, ResetBinding, model_root_defaults, DataRecoveryWorkspace


def _recovery_loop(*, model, **kwargs):
    e, forward = kwargs['euler'], kwargs['forward']
    dimensions = {key: int(getattr(model, key)) for key in
                  ('nq', 'nv', 'nM', 'njnt', 'nu', 'na', 'nbody', 'nmocap',
                   'nuserdata', 'nsensordata', 'nplugin', 'npluginstate')}
    roots = dict(qpos=e.qpos, qvel=e.qvel, qacc=e.qacc,
                 qacc_warmstart=e.qacc_warmstart, ctrl=e.ctrl, time=e.time,
                 sensordata=e.sensordata, qfrc_applied=forward.d.qfrc_applied,
                 xfrc_applied=forward.d.xfrc_applied)
    defaults = model_root_defaults(dimensions, qpos0=np.asarray(model.qpos0),
        body_mocapid=np.asarray(model.body_mocapid), body_pos=np.asarray(model.body_pos),
        body_quat=np.asarray(model.body_quat))
    # Current shape intake rejects activation/mocap/plugin state. Nonempty
    # userdata without a real resident owner cannot be guessed or recreated.
    required = {name for name, width in ROOT_WIDTHS.items() if width(dimensions)}
    if not required <= roots.keys():
        raise NotImplementedError('missing current native root owner: ' + str(required - roots.keys()))
    bindings = {name: ResetBinding(roots[name], defaults[name], 'root') for name in required}
    W = e.capacity
    stream = kwargs['stream']
    with wp.ScopedStream(stream):
        active = wp.zeros(W, dtype=wp.int32, device=stream.device)
        warnings = wp.zeros((W, 8), dtype=wp.int32, device=stream.device)
        lastinfo = wp.zeros((W, 8), dtype=wp.int32, device=stream.device)
    recovery = DataRecoveryWorkspace(dimensions, defaults, bindings, nworld=W,
        active=active, warning_number=warnings, warning_lastinfo=lastinfo,
        stream=stream, current_bindings=lambda: bindings, mode='diagnostic')
    loop = RecoveringShapeRuntime(recovery=recovery, **kwargs)
    loop.recovery_inventory_scope = 'native required roots only; derived arena rebuilt by physical forward'
    loop.recovery_scope_gaps = ('full native header/derived inventory qualification',
                               'numeric-origin failed N work vs canonical-held classification',
                               'actual task115/116 spontaneous recovery')
    return loop



def _event_cold_factories():
    from .generic_prepare import _cold_factories
    from .newton_final_event import prepare_final_event_runtime
    collision, unused_newton, choice=_cold_factories()
    if choice!="cn":raise ValueError("Final event candidate requires explicit CN cold scope")
    return collision,prepare_final_event_runtime,choice


def prepare_final_event_recovery_physics(*args, **kwargs):
    """Replace only the cold loop constructor; never mutate a sealed instance."""
    source = inspect.getsource(shape_prepare)
    tree = ast.parse(source)
    changed = 0
    wide_changed = 0
    event_changed = 0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="_cold_factories":
            node.func.id="_event_cold_factories"
            event_changed+=1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GenericEulerWorkspace":
            node.func.id = "_wide_workspace"
            wide_changed += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ShapeCountedResidentStepRuntime':
            node.func.id = '_recovery_loop'
            node.keywords.append(ast.keyword(arg='model', value=ast.Name(id='model', ctx=ast.Load())))
            changed += 1
    if changed != 1 or wide_changed != 1 or event_changed != 1:
        raise ValueError('one exact cold runtime constructor required')
    ast.fix_missing_locations(tree)
    from libero.gpu.reference.integration.euler_integrator.wide_angle import WideAngleEulerWorkspace
    namespace = dict(_event_cold_factories=_event_cold_factories, _wide_workspace=WideAngleEulerWorkspace, __name__=__name__ + '.derived', __package__=__package__,
                     __file__=shape_prepare.__file__, _recovery_loop=_recovery_loop)
    exec(compile(tree, shape_prepare.__file__ + ':cold-recovery', 'exec'), namespace)
    bundle = namespace['prepare_shape_physics'](*args, **kwargs)
    bundle.recovery_source_manifest = dict(original_shape_prepare_sha256=
        hashlib.sha256(source.encode()).hexdigest(), cold_constructor_replacements=changed, wide_angle_constructor_replacements=wide_changed,
        physics_math_changed=False, controller_reset_invoked=False,
        qualification='diagnostic roots/conditional-graph forward/Euler only')
    return bundle
