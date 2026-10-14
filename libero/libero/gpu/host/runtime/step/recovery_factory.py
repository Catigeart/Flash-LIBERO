"""Independent cold N1 recovery assembly, with unchanged shape physics order."""
import ast
import hashlib
import inspect
import numpy as np
import warp as wp
from . import shape_prepare
from .recovery_runtime import RecoveringShapeRuntime
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
    stream = kwargs['stream']
    with wp.ScopedStream(stream):
        active = wp.zeros(1, dtype=wp.int32, device=stream.device)
        warnings = wp.zeros((1, 8), dtype=wp.int32, device=stream.device)
        lastinfo = wp.zeros((1, 8), dtype=wp.int32, device=stream.device)
    recovery = DataRecoveryWorkspace(dimensions, defaults, bindings, nworld=1,
        active=active, warning_number=warnings, warning_lastinfo=lastinfo,
        stream=stream, current_bindings=lambda: bindings, mode='diagnostic')
    loop = RecoveringShapeRuntime(recovery=recovery, **kwargs)
    loop.recovery_inventory_scope = 'native required roots only; derived arena rebuilt by physical forward'
    loop.recovery_scope_gaps = ('full native header/derived inventory qualification',
                               'numeric-origin failed N work vs canonical-held classification',
                               'actual task115/116 spontaneous recovery')
    return loop


def prepare_recovery_physics(*args, **kwargs):
    """Replace only the cold loop constructor; never mutate a sealed instance."""
    source = inspect.getsource(shape_prepare)
    tree = ast.parse(source)
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ShapeCountedResidentStepRuntime':
            node.func.id = '_recovery_loop'
            node.keywords.append(ast.keyword(arg='model', value=ast.Name(id='model', ctx=ast.Load())))
            changed += 1
    if changed != 1:
        raise ValueError('one exact cold runtime constructor required')
    ast.fix_missing_locations(tree)
    namespace = dict(__name__=__name__ + '.derived', __package__=__package__,
                     __file__=shape_prepare.__file__, _recovery_loop=_recovery_loop)
    exec(compile(tree, shape_prepare.__file__ + ':cold-recovery', 'exec'), namespace)
    bundle = namespace['prepare_shape_physics'](*args, **kwargs)
    bundle.recovery_source_manifest = dict(original_shape_prepare_sha256=
        hashlib.sha256(source.encode()).hexdigest(), cold_constructor_replacements=changed,
        physics_math_changed=False, controller_reset_invoked=False,
        qualification='diagnostic roots/conditional-graph forward/Euler only')
    return bundle
