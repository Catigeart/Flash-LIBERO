"""Private original ControlEnv GPU construction for the qualified finite-case API.

This module is cold-only and has no import side effects. Qualification is checked
by admission.create_qualified_case before this private function can be reached.
"""
import os

def create_private_environment(*,bddl_file_name,**options):
    if os.environ.get("P06_STEP_COLD_COPY")!="cn":
        raise ValueError("reviewed finite case requires explicit original cn cold choice")
    from libero.gpu.host.runtime.core.namespace import require_canonical_gpu_namespace
    require_canonical_gpu_namespace()
    from libero.libero.envs.bddl_utils import get_problem_info
    from libero.gpu.host.adapters.environment import install_model_only_gpu_factory
    from libero.gpu.host.runtime.step.raw_copy_factory import create
    from libero.libero.envs.env_wrapper import ControlEnv
    info=get_problem_info(bddl_file_name)
    install_model_only_gpu_factory(info["problem_name"],create,replace=True)
    env=ControlEnv(bddl_file_name,physics_backend="gpu",**options)
    return _PrivateControlEnv(env)

class _PrivateControlEnv:
    def __init__(self,env):self._env=env
    def seed(self,seed):return self._env.seed(seed)
    def reset(self):return self._env.reset()
    def step(self,action):return self._env.step(action)
    def get_sim_state(self):return self._env.get_sim_state()
    def gpu_physics_evidence(self):return self._env.env.gpu_physics_evidence()
    def close(self):return self._env.close()
    @property
    def action_spec(self):return self._env.env.action_spec
