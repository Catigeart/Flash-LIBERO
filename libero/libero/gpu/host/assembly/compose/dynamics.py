"""Three-RN force + ordered external projection + RHS in one launch."""
import warp as wp
from libero.gpu.host.runtime.dynamics.acceleration import _three_rn, _seed_rhs, _route
from libero.gpu.reference.dynamics.external_force_projection.kernel import _project
from libero.gpu.host.runtime.dynamics.generic_stages import GenericSmoothWorkspace
from .compose import compose, identity_mapping

prepare_smooth = compose("prepare_smooth", (
    (_three_rn, identity_mapping(_three_rn)),
    (_project, identity_mapping(_project, body_rootid="rootid", status="d02_status")),
    (_seed_rhs, identity_mapping(_seed_rhs)),
))


class FusedSmoothWorkspace(GenericSmoothWorkspace):
    def launch(self, stream=None):
        self.validate_bindings()
        values = {name: getattr(self, name) for name in prepare_smooth.fusion_argument_names
                  if hasattr(self, name)}
        values.update(rhs=self.solver.rhs, diagnostic=0)
        wp.launch(prepare_smooth, self.capacity,
                  inputs=[values[name] for name in prepare_smooth.fusion_argument_names],
                  device=self.device, stream=stream)
        # Retain the factor/solve as a separate dependency boundary.
        self.solver.launch(stream=stream)
        wp.launch(_route, self.capacity, inputs=[self.entry_status, self.d02_status,
            self.solver.status, self.active, self.solver.solution, self.qfrc,
            self.smooth, self.qacc_smooth, self.status, self.nv],
            device=self.device, stream=stream)
