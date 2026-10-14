"""Cold OSC composition; callbacks and history snapshot boundaries retained."""
import inspect
import warp as wp
from libero.gpu.host.runtime.dynamics import runtime_adapter as original
from libero.gpu.reference.control.osc import controller as kernels
from libero.gpu.host.assembly.compose.compose import compose
from libero.gpu.host.assembly.compose.backup import MultiFieldCopyPlan, _overlap


class _Dynamic:
    def __init__(self, name, initial):
        self.name, self.initial = name, initial


class _Group:
    def __init__(self, name, stages):
        bindings, inputs, dynamic, composed = {}, [], {}, []
        for kernel, values in stages:
            parameters = tuple(inspect.signature(kernel.func).parameters.items())
            if len(parameters) != len(values):
                raise ValueError("OSC stage parameter extent differs")
            mapping = {}
            for (argument, parameter), value in zip(parameters, values):
                annotation = str(parameter.annotation)
                identity = ("dynamic", value.name) if isinstance(value, _Dynamic) else ("value", id(value))
                key = identity, annotation
                if key not in bindings:
                    index = len(inputs)
                    bindings[key] = "a" + str(index)
                    inputs.append(value.initial if isinstance(value, _Dynamic) else value)
                    if isinstance(value, _Dynamic):
                        dynamic.setdefault(value.name, []).append(index)
                mapping[argument] = bindings[key]
            composed.append((kernel, mapping))
        self.kernel = compose(name, composed)
        # compose argument insertion order is the same as the retained bindings.
        self.inputs, self.dynamic = inputs, dynamic

    def launch(self, owner, **values):
        for name, value in values.items():
            for index in self.dynamic[name]:
                self.inputs[index] = value
        wp.launch(self.kernel, owner.capacity, inputs=self.inputs,
                  device=owner.device, stream=owner.stream)


def _refresh(c):
    return kernels.refresh_cache, [getattr(c, name) for name in
        inspect.signature(kernels.refresh_cache.func).parameters]


def _control(c):
    rename = {"raw_tau": "torques_raw", "emit_diagnostics": "diagnostics"}
    return kernels.fused_osc_control, [getattr(c, rename.get(name, name)) for name in
        inspect.signature(kernels.fused_osc_control.func).parameters]


class FusedRuntimeControllerAdapter(original.RuntimeControllerAdapter):
    """O1-O4 opt-in adapter, constructed before the factory seals bindings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.controller
        self._seed = MultiFieldCopyPlan([(array, getattr(c, name)) for name, array in
                                        {**self.public, **self.public_int}.items()])
        policy = _Dynamic("policy", 0)
        gate = lambda mask, policy_value: (original._control_gate,
            [mask, self.parents.valid, self.parents.written, self.parents.expected,
             self.first_error, c.error, c.policy_step, policy_value])
        def commits(mask):
            stages = [(original._copy_admitted, [mask, getattr(c, name), array, array.shape[1]])
                      for name, array in self.public.items()]
            stages += [(original._copy_i32, [mask, getattr(c, name), array])
                       for name, array in self.public_int.items()]
            return stages
        def reset(mask, clear, update):
            return kernels.reset_controller, [mask, c.phys_ee_pos, c.phys_ee_rot,
                c.phys_q, c.goal_pos, c.goal_rot, c.q0, c.gripper, c.error,
                c.cache_valid, c.needs_update, clear, update]
        self._run = _Group("osc_control_publish_r03", [gate(c.active, policy),
            _refresh(c), _control(c), (original._control_commit_mask,
                [c.active, c.error, self.first_error, self.commit_mask, self.event, self.valid]),
            *commits(self.commit_mask), (original._publish_control,
                [self.commit_mask, self.event, self.parents.expected, self.written,
                 self.consumed_parent, self.valid])])
        self._reset = _Group("osc_reset_publish_r03", [gate(self.call_mask, 0),
            reset(self.call_mask, _Dynamic("clear", 1), _Dynamic("update", 1)),
            *commits(self.call_mask)])
        self._bootstrap_begin = _Group("osc_bootstrap_reset_r03", [gate(self.call_mask, 0),
            reset(self.call_mask, 1, 1)])
        self._bootstrap_end = _Group("osc_bootstrap_cache_publish_r03", [_refresh(c),
            (original._bootstrap_finish, [self.call_mask, c.error, c.cache_valid,
                c.needs_update, self.first_error]), *commits(self.call_mask)])
        # Exact aliases with matching world partitions are local to that world.
        # Shifted views or different row strides can cross world ownership;
        # retain the original grid synchronization for those admitted bindings.
        arrays = {id(a): a for group in (self._run, self._reset,
            self._bootstrap_begin, self._bootstrap_end) for a in group.inputs
            if isinstance(a, wp.array)}
        arrays = tuple(arrays.values())
        stride = lambda a: a.size * wp.types.type_size_in_bytes(a.dtype) // self.capacity
        self._world_alias_fallback = any(_overlap(a, b) and
            (a.ptr != b.ptr or stride(a) != stride(b))
            for i, a in enumerate(arrays) for b in arrays[i + 1:])

    def enqueue(self, *, policy_step=False, force_refresh=False, stream=None):
        self.validate(stream)
        if self._world_alias_fallback:
            return super().enqueue(policy_step=policy_step,
                force_refresh=force_refresh, stream=stream)
        c, st = self.controller, self.stream
        wp.launch(original._plan_control, self.capacity,
            inputs=[self.active, self.first_error, self.public_int["needs_update"],
                    self.refresh_mask, self.call_mask, int(bool(force_refresh))],
            device=self.device, stream=st)
        self.forward_refresh(self.refresh_mask, st)
        self.gather(self.refresh_mask, st)
        self.live_bias(self.call_mask, st)
        self._seed.launch(st)
        wp.copy(c.active, self.call_mask, stream=st)
        wp.copy(c.needs_update, self.refresh_mask, stream=st)
        self._run.launch(self, policy=int(bool(policy_step)))

    def reset_selected(self, mask, *, clear_gripper=True,
                       update_initial_joints=True, stream=None):
        self.validate(stream)
        if self._world_alias_fallback:
            return super().reset_selected(mask, clear_gripper=clear_gripper,
                update_initial_joints=update_initial_joints, stream=stream)
        original._array(mask, (self.capacity,), wp.int32, self.device)
        self._seed.launch(self.stream)
        wp.copy(self.call_mask, mask, stream=self.stream)
        self._reset.launch(self, clear=int(clear_gripper), update=int(update_initial_joints))

    def bootstrap_from_current_physics(self, mask, *, stream=None):
        self.validate(stream)
        if self._world_alias_fallback:
            return super().bootstrap_from_current_physics(mask, stream=stream)
        original._array(mask, (self.capacity,), wp.int32, self.device)
        self.forward_refresh(mask, self.stream)
        self.forward_refresh(mask, self.stream)
        self.gather(mask, self.stream)
        self.live_bias(mask, self.stream)
        wp.copy(self.call_mask, mask, stream=self.stream)
        self._bootstrap_begin.launch(self)
        wp.copy(self.controller.active, self.call_mask, stream=self.stream)
        self._bootstrap_end.launch(self)
