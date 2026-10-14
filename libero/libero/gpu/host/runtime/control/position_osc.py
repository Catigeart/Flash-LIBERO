"""Original OSC_POSITION action boundary over the unchanged six-axis OSC math."""
import numpy as np
import warp as wp
from libero.gpu.host.assembly.pipeline.osc import FusedRuntimeControllerAdapter

class RuntimePositionOSCControllerAdapter(FusedRuntimeControllerAdapter):
    action_dim=4
    control_limits=(np.full(4,-1.),np.full(4,1.))

    def validate(self,stream=None):
        super().validate(stream)
        if getattr(self.controller,"position_only",False) is not True:
            raise ValueError("prepared OSC_POSITION specialization changed")

    def stage_action(self,action):
        self.validate();a=np.asarray(action)
        if a.dtype!=np.float64 or a.shape!=(self.capacity,4) or not np.isfinite(a).all():
            raise ValueError("OSC_POSITION action must be finite float64 (worlds,4)")
        packed=np.zeros((self.capacity,7),dtype=np.float64)
        packed[:,:3]=a[:,:3];packed[:,6]=a[:,3]
        with wp.ScopedStream(self.stream):self.controller.action.assign(packed)
