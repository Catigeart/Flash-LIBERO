"""Diagnostic original mj_step warning recovery for a W-major world arena.

A native data warning is separate from source/owner/sticky errors. The recovery
helper must bind its complete declared native reset inventory. This scheduler
never resets the host horizon, controller goal/action/cache, model placement,
or observations. CUDA conditional graphs avoid an extra forward on clean steps.
World masks isolate canonical recovery and integration; event history is world-major.
"""
import warp as wp
from .shape_counter import ShapeCountedResidentStepRuntime
from .shape_runtime import _token
from libero.gpu.host.runtime.dynamics.recovery import DataRecoveryWorkspace
wp.set_module_options({"enable_backward":False,"fuse_fp":True,"fast_math":False})


@wp.kernel(enable_backward=False)
def _warning_admission(intent:wp.array(dtype=wp.int32), first:wp.array(dtype=wp.int32),
                       active:wp.array(dtype=wp.int32)):
    w=wp.tid()
    active[w]=0
    if first[w]==0:
        active[w]=intent[w]


@wp.kernel(enable_backward=False)
def _reset_protocol(mask:wp.array(dtype=wp.int32),first:wp.array(dtype=wp.int32),
                    epoch:wp.array(dtype=wp.int64),expected:wp.array(dtype=wp.int64),
                    forward_valid:wp.array(dtype=wp.int32),step_valid:wp.array(dtype=wp.int32),
                    events:wp.array(dtype=wp.int64)):
    w=wp.tid()
    if mask[w]==1 and first[w]==0:
        if epoch[w]==wp.int64(9223372036854775807) or expected[w]==wp.int64(9223372036854775807):
            first[w]=9
            return
        # A real device data reset is a state writer. Independent planned epoch
        # advances by the same event, without copying producer output counters.
        epoch[w]=epoch[w]+wp.int64(1)
        expected[w]=expected[w]+wp.int64(1)
        forward_valid[w]=0
        step_valid[w]=0
        events[w]=events[w]+wp.int64(1)



@wp.kernel(enable_backward=False)
def _current_acceleration_admission(intent:wp.array(dtype=wp.int32), first:wp.array(dtype=wp.int32),
        forward_valid:wp.array(dtype=wp.int32), forward_written:wp.array(dtype=wp.int64),
        state_epoch:wp.array(dtype=wp.int64), n_valid:wp.array(dtype=wp.int32),
        n_written:wp.array(dtype=wp.int64), n_expected:wp.array(dtype=wp.int64),
        active:wp.array(dtype=wp.int32)):
    w=wp.tid()
    active[w]=0
    if intent[w]==1 and first[w]==0 and forward_valid[w]==1 and forward_written[w]==state_epoch[w]:
        if n_valid[w]==1 and n_written[w]==n_expected[w]:
            active[w]=1


def enqueue_source_mj_step(recovery, reset_protocol, forward, integrate,
                           conditional_forward, intent, stream, acceleration_admission):
    """Original checkPos -> checkVel -> forward -> checkAcc -> Euler order.

    reset_protocol only updates backend writer epochs/invalidations. The data
    helper clears native roots/warnings. Conditional forward is a prepared CUDA
    graph; a callback that host-reads a device mask is not an accepted backend.
    """
    reset_protocol(recovery.check_position(stream=stream))
    reset_protocol(recovery.check_velocity(stream=stream))
    forward.enqueue(intent,stream)
    acceleration_admission()
    reset_protocol(recovery.check_acceleration(stream=stream))
    conditional_forward()
    integrate()


@wp.kernel(enable_backward=False)
def _any_reset(mask:wp.array(dtype=wp.int32), predicate:wp.array(dtype=wp.int32), count:int):
    predicate[0]=0
    for w in range(count):
        if mask[w]==1:
            predicate[0]=1


class CurrentWrittenRecoveringRuntime(ShapeCountedResidentStepRuntime):
    """Explicit pending runtime candidate, not an automatic production grant."""
    def __init__(self,*,recovery,**kwargs):
        super().__init__(**kwargs)
        if not isinstance(recovery,DataRecoveryWorkspace):
            raise TypeError("actual prepared native recovery workspace required")
        if recovery.active.ptr in (self.intent.ptr,self.first_error.ptr,self.forward_valid.ptr,self.valid.ptr):
            raise ValueError("warning admission must own a private mask")
        for name in ("nq","nv","nM","njnt","nu","nbody"):
            if recovery.dimensions[name]!=getattr(self.runtime_shape,name):
                raise ValueError("recovery native dimensions differ: "+name)
        expected_roots={"qpos":self.euler.qpos,"qvel":self.euler.qvel,"qacc":self.euler.qacc,
            "qacc_warmstart":self.euler.qacc_warmstart,"ctrl":self.euler.ctrl,"time":self.euler.time,
            "sensordata":self.euler.sensordata,"qfrc_applied":self.forward.d.qfrc_applied,
            "xfrc_applied":self.forward.d.xfrc_applied}
        for name,array in expected_roots.items():
            if recovery.bindings[name].array is not array:
                raise ValueError("recovery canonical root differs: "+name)
        if self.euler.capacity!=recovery.N:
            raise ValueError("recovery world capacity differs from Euler arena")
        if not self.stream.device.is_cuda:
            raise ValueError("runtime recovery needs captured device conditional graph")
        if self.expected_state_epoch is None:
            raise ValueError("independent state epoch plan required")
        if recovery.stream is not self.stream:
            raise ValueError("foreign recovery stream")
        if not wp.is_conditional_graph_supported():
            raise RuntimeError("CUDA conditional graph support required for no-invocation history")
        self.recovery=recovery
        self._recovery_ref=recovery
        self.reset_events=wp.zeros(self.euler.capacity,dtype=wp.int64,device=self.stream.device)
        self.reset_predicate=wp.zeros(1,dtype=wp.int32,device=self.stream.device)
        self._recovery_roots=(recovery.active,recovery.reset_mask,self.reset_events,self.reset_predicate)
        self._recovery_tokens=tuple(_token(a) for a in self._recovery_roots)
        # No capture_if call is allowed outside CUDA capture: Warp otherwise
        # reads the condition on the host. This graph always branches on device.
        with wp.ScopedCapture(stream=self.stream,force_module_load=True) as capture:
            wp.capture_if(self.reset_predicate,on_true=self._reset_forward,
                          stream=self.stream)
        self._reset_forward_graph=capture.graph

    def _reset_forward(self):
        # The physical forward consumes reset ctrl=0. No robot/controller hook.
        preservation=getattr(self,"_reset_preserver",None)
        if preservation is None:
            self.forward.enqueue(self.recovery.reset_mask,self.stream)
            return
        preservation.save(self.stream)
        self.forward._reset_preservation_scope=True
        try:
            self.forward.enqueue(self.recovery.reset_mask,self.stream)
        finally:
            self.forward._reset_preservation_scope=False
        preservation.restore(self.stream)

    def _protocol(self,mask):
        wp.launch(_reset_protocol,dim=self.euler.capacity,inputs=[mask,self.first_error,self.state_epoch,
            self.expected_state_epoch,self.forward_valid,self.valid,self.reset_events],
            device=self.stream.device,stream=self.stream)

    def _recovery_validate(self):
        self.validate()
        if self.recovery is not self._recovery_ref:
            raise ValueError("prepared recovery workspace changed")
        if tuple(_token(a) for a in (self.recovery.active,self.recovery.reset_mask,self.reset_events,self.reset_predicate))!=self._recovery_tokens:
            raise ValueError("prepared recovery mask/counter changed")
        self.recovery.validate_bindings()
        preservation=getattr(self,"_reset_preserver",None)
        if preservation is not None:preservation.validate()

    def enqueue_physics_step(self):
        self._recovery_validate()
        wp.launch(_warning_admission,dim=self.euler.capacity,inputs=[self.intent,self.first_error,self.recovery.active],
                  device=self.stream.device,stream=self.stream)
        enqueue_source_mj_step(self.recovery,self._protocol,self.forward,self._integrate,
            self._conditional_reset_forward,
            self.intent,self.stream,self._admit_current_acceleration)

    def _update_reset_forward_condition(self):
        wp.launch(_any_reset,dim=1,inputs=[self.recovery.reset_mask,self.reset_predicate,self.euler.capacity],
                  device=self.stream.device,stream=self.stream)
    def _conditional_reset_forward(self):
        self._update_reset_forward_condition()
        # Conditional-node graphs cannot be launched as child graphs during
        # an outer capture. Insert the same device branch directly instead.
        if self.stream.is_capturing:
            wp.capture_if(self.reset_predicate,on_true=self._reset_forward,
                          stream=self.stream)
        else:
            wp.capture_launch(self._reset_forward_graph,stream=self.stream)

    def _admit_current_acceleration(self):
        n=self.forward.n
        wp.launch(_current_acceleration_admission,dim=self.euler.capacity,inputs=[self.intent,self.first_error,
            self.forward_valid,self.forward_written,self.state_epoch,n.valid,n.n_written,
            self.forward.expected_newton,self.recovery.active],device=self.stream.device,stream=self.stream)

    def enqueue_substep(self,*,policy_step=False,force_refresh=False):
        self._recovery_validate()
        # Original application sim.forward/control precedes native mj_step.
        # No warning/reset check is added to those explicit forward callbacks.
        self.forward.enqueue(self.intent,self.stream)
        self.controller.enqueue(policy_step=policy_step,force_refresh=force_refresh)
        self.enqueue_physics_step()
        self.observe()
