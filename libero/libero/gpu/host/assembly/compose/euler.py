"""Wide-angle Euler: retain prepare/solve, fuse labels/integrate/commit."""
import warp as wp
from libero.gpu.reference.integration.euler_integrator.kernel import _prepare, _source_solve_labels, _commit
from libero.gpu.reference.integration.euler_integrator.wide_angle import _integrate_wide, WideAngleEulerWorkspace
from .compose import compose, identity_mapping

finish_euler = compose("finish_euler", (
    (_source_solve_labels, identity_mapping(_source_solve_labels,
        rhs="mass_rhs", ld="mass_ld", madr="dof_Madr", parent="dof_parentid",
        simple="dof_simplenum", work="zero_work")),
    (_integrate_wide, identity_mapping(_integrate_wide, solution="mass_solution",
        jtype="jnt_type", padr="jnt_qposadr", vadr="jnt_dofadr")),
    (_commit, identity_mapping(_commit)),
))


class FusedEulerWorkspace(WideAngleEulerWorkspace):
    def launch(self, stream=None):
        m = self.mass
        wp.launch(_prepare, self.capacity, inputs=[self.active, self.qpos, self.qvel,
            self.time, self.qM, self.qfrc_smooth, self.qfrc_constraint, self.qacc,
            self.dof_damping, m.dof_Madr, m.qm, m.rhs, m.count, m.active,
            self.damped, self.status, self.dt, self.nq, self.nv, self.nM],
            device=self.device, stream=stream)
        m.launch(stream=stream)
        values = {name: getattr(self, name) for name in finish_euler.fusion_argument_names
                  if hasattr(self, name)}
        values.update(mass_status=m.status, dof_Madr=m.dof_Madr,
            dof_parentid=m.dof_parentid, dof_simplenum=m.dof_simplenum,
            scratch_pos=self.qpos_after, scratch_vel=self.qvel_after,
            used_acc=self.qacc_used, scratch_time=self.time_after, nm=self.nM)
        for name in ("qm", "rhs", "ld", "inv", "sqrt_inv", "solution", "pivot",
                     "warning_count", "warning_last"):
            values["mass_" + name] = getattr(m, name)
        wp.launch(finish_euler, self.capacity,
                  inputs=[values[name] for name in finish_euler.fusion_argument_names],
                  device=self.device, stream=stream)
