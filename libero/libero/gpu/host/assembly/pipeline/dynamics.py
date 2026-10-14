"""Cold-planned fusion of native-count D phases, with explicit grid barriers.

Original host validation and device arithmetic are retained. A local cloned
launcher records the source launch schedule; it never monkey-patches Warp or
the original classes. Plans are built before the first physics dispatch.
"""
import ast
import inspect
import textwrap
import types
import warp as wp
from libero.gpu.host.assembly.compose.compose import compose
from libero.gpu.host.runtime.dynamics.generic_runtime import GenericRuntimeDynamicsAdapter
from libero.gpu.host.runtime.dynamics.runtime_adapter import RuntimePostAdapter


class _Calls(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in (
            'launch', 'launch_position', 'launch_velocity', '_launch', '_gate',
            'snapshot', 'restore_unrequested') and not (
                isinstance(f.value, ast.Name) and f.value.id == 'wp'):
            return ast.copy_location(ast.Call(ast.Name('_invoke_method', ast.Load()),
                [f.value, ast.Constant(f.attr), *node.args], node.keywords), node)
        if isinstance(f, ast.Name) and f.id in ('launch', 'compos_launch', 'crb_launch'):
            return ast.copy_location(ast.Call(ast.Name('_invoke_function', ast.Load()),
                [f, *node.args], node.keywords), node)
        return node


class _Warp:
    def __init__(self, context): self.context = context
    def __getattr__(self, name): return getattr(wp, name)
    def launch(self, kernel, dim, inputs=None, outputs=None, **kwargs):
        self.context.record(kernel, dim, list(inputs or ()) + list(outputs or ()), kwargs)
    def copy(self, *args, **kwargs):
        self.context.flush()
        if not self.context.cold: return wp.copy(*args, **kwargs)


class _Context:
    def __init__(self, capacity, *, cold=False, plans=None, label='D'):
        self.capacity, self.cold, self.label = capacity, cold, label
        self.plans = [] if plans is None else plans
        self.cursor, self.pending, self.clones = 0, [], {}
        self.warp = _Warp(self)

    def clone(self, function):
        function = getattr(function, '__func__', function)
        if function in self.clones: return self.clones[function]
        if not self.cold: raise ValueError('D cold launcher function drift')
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        tree = _Calls().visit(tree)
        tree.body[0].decorator_list = []
        # Several original host methods (_gate, launch) have names shared by
        # a kernel/global helper. Do not overwrite that binding when exec'ing.
        tree.body[0].name = '_d_cloned_host'
        ast.fix_missing_locations(tree)
        namespace = dict(function.__globals__)
        namespace.update(wp=self.warp, _invoke_method=self.method,
                         _invoke_function=self.function)
        exec(compile(tree, function.__code__.co_filename + ':D-fusion-r03', 'exec'), namespace)
        cloned = namespace['_d_cloned_host']
        self.clones[function] = cloned
        return cloned

    def function(self, function, *args, **kwargs):
        return self.clone(function)(*args, **kwargs)

    def method(self, owner, name, *args, **kwargs):
        # History preservation remains its original dependency boundary.
        function = getattr(owner, name)
        module = getattr(getattr(function, '__func__', function), '__module__', '')
        if name in ('snapshot', 'restore_unrequested'):
            self.flush()
            if not self.cold: return function(*args, **kwargs)
            return None
        if name == 'launch' and 'host.runtime.foundation' in module:
            # Retain prepared admission, then use the unmodified base frame
            # launch source. Its per-world status joins D1; the body/object
            # grids always retain barriers, even when an object count is 1.
            from libero.gpu.reference.position.kinematics_frames.frames import FrameWorkspace
            owner.validate_bindings()
            return self.clone(FrameWorkspace.launch)(owner, *args, **kwargs)
        return self.clone(function)(owner, *args, **kwargs)

    def record(self, kernel, dim, arguments, kwargs):
        frame_grid = kernel.func.__module__.endswith('kinematics_frames.frames') and kernel.key in ('_body', '_objects')
        if dim == self.capacity and not frame_grid:
            # compose rejects non-scalar tid and the reviewed D kernels own
            # disjoint world rows; original grid is exactly capacity.
            self.pending.append((kernel, arguments, kwargs))
        else:
            self.flush()
            if not self.cold: wp.launch(kernel, dim=dim, inputs=arguments, **kwargs)

    def flush(self):
        if not self.pending: return
        sequence = tuple(stage[0] for stage in self.pending)
        if self.cold:
            stages = []
            for i, kernel in enumerate(sequence):
                mapping = {name: 's%d_%s' % (i, name)
                           for name in inspect.signature(kernel.func).parameters}
                stages.append((kernel, mapping))
            fused = compose('%s_group%d' % (self.label, len(self.plans)), stages)
            self.plans.append((sequence, fused))
        else:
            if self.cursor >= len(self.plans): raise ValueError('D fusion schedule grew')
            expected, fused = self.plans[self.cursor]
            if sequence != expected: raise ValueError('D fusion launch schedule drift')
            options = self.pending[0][2]
            for _, _, other in self.pending[1:]:
                if other.get('stream') is not options.get('stream'):
                    raise ValueError('D fusion stream boundary drift')
            # Arrays are deliberately not deduplicated: an admitted external
            # input can alias differently on a subsequent call.
            wp.launch(fused, dim=self.capacity,
                      inputs=[value for _, values, _ in self.pending for value in values],
                      **options)
            self.cursor += 1
        self.pending.clear()

    def finish(self):
        self.flush()
        if not self.cold and self.cursor != len(self.plans):
            raise ValueError('D fusion launch schedule shrank')


_METHODS = ('launch_position', 'launch_velocity', 'launch_bias', 'launch_force',
            'gather_controller_inputs')


def _array_ranges(owner):
    """Cold collect owner rows for conservative external shifted-alias fallback."""
    ranges, seen = [], set()
    def walk(value, depth):
        if id(value) in seen or depth > 6: return
        seen.add(id(value))
        if isinstance(value, wp.array):
            if value.ptr is None or value.size is None: return
            size = value.size * wp.types.type_size_in_bytes(value.dtype)
            ranges.append((int(value.ptr), int(value.ptr) + size, value.shape, value.dtype))
        elif isinstance(value, dict):
            for child in value.values(): walk(child, depth + 1)
        elif isinstance(value, (tuple, list)):
            for child in value: walk(child, depth + 1)
        elif hasattr(value, '__dict__') and not isinstance(value, type) and not callable(value) and not hasattr(value, 'func') and (
            type(value).__module__.startswith(('libero.', 'warp.')) or
            type(value).__name__ == 'SimpleNamespace'):
            for key, child in vars(value).items():
                if key not in ('registry', 'spec', 'metadata', 'robot', '_d_fusion_plans', '_d_fusion_contexts'):
                    walk(child, depth + 1)
    walk(owner, 0)
    return tuple(ranges)


def _shifted_external_alias(adapter, values):
    for value in values:
        if not isinstance(value, wp.array): continue
        start = int(value.ptr)
        end = start + value.size * wp.types.type_size_in_bytes(value.dtype)
        for a, b, shape, dtype in adapter._d_fusion_ranges:
            if max(start, a) < min(end, b) and (
                start != a or end != b or value.shape != shape or value.dtype != dtype):
                return True
    return False


def prepare_dynamics(adapter):
    """Cold create the five API plans without issuing copies/device work."""
    calls = {
        'launch_position': {}, 'launch_velocity': {},
        'launch_bias': dict(reference_done=adapter.gate_bias,
            reference_error=adapter.gate_bias, reference_written=adapter.forward_epoch,
            reference_expected=adapter.forward_epoch),
        'launch_force': dict(ctrl=adapter.actuation.ctrl, ctrl_valid=adapter.actuation_active),
        'gather_controller_inputs': dict(mask=adapter.gate_control),
    }
    plans, contexts = {}, {}
    for name in _METHODS:
        context = _Context(adapter.capacity, cold=True, label='D_' + name)
        context.function(getattr(GenericRuntimeDynamicsAdapter, name), adapter, **calls[name])
        context.finish()
        plans[name] = context.plans
        context.cold = False
        contexts[name] = context
    adapter._d_fusion_plans = plans
    adapter._d_fusion_contexts = contexts
    adapter._d_fusion_ranges = _array_ranges(adapter)
    return adapter


def _wrapper(name, original):
    def fused(self, *args, **kwargs):
        contexts = getattr(self, '_d_fusion_contexts', None)
        if contexts is None: raise ValueError('D fusion requires cold preparation')
        if name != 'enqueue' and _shifted_external_alias(self, (*args, *kwargs.values())):
            return original(self, *args, **kwargs)
        context = contexts[name]
        if getattr(context, 'running', False): raise ValueError('D fusion reentrant dispatch')
        context.running = True
        context.cursor = 0
        try:
            result = context.function(original, self, *args, **kwargs)
            context.finish()
            return result
        finally:
            context.pending.clear()
            context.running = False
    fused.__name__ = name
    return fused


def runtime_methods():
    """Methods for root's cold runtime subclass, without changing owners."""
    return {name: _wrapper(name, getattr(GenericRuntimeDynamicsAdapter, name))
            for name in _METHODS}


def fused_runtime_type(base):
    return type('FusedDStagesRuntime', (base,), dict(runtime_methods(), __module__=__name__))


def prepare_post(adapter):
    context = _Context(adapter.capacity, cold=True, label='D_post')
    context.function(RuntimePostAdapter.enqueue, adapter)
    context.finish()
    context.cold = False
    adapter._d_fusion_plans = {'enqueue': context.plans}
    adapter._d_fusion_contexts = {'enqueue': context}
    return adapter


class FusedRuntimePostAdapter(RuntimePostAdapter):
    enqueue = _wrapper('enqueue', RuntimePostAdapter.enqueue)


def plan_catalog(adapter):
    return {name: [dict(stages=len(stages), kernels=[k.key for k in stages],
                       fused=fused.key) for stages, fused in groups]
            for name, groups in adapter._d_fusion_plans.items()}
