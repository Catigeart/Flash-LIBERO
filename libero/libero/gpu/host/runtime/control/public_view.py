"""Original robot.controller inspection over successfully published GPU state."""
import numpy as np
import warp as wp

class GpuControllerView:
    def __init__(self,adapter,world=0,owner_guard=None):
        self._adapter=adapter;self._world=int(world);self._alive=True;self._owner_guard=owner_guard
        if not 0<=self._world<adapter.capacity:raise ValueError("controller world outside resident owner")

    def __getattribute__(self,name):
        if not name.startswith('_') and name not in ('invalidate','rebind'):
            if not object.__getattribute__(self,'_alive'):raise RuntimeError('retired GPU controller owner')
            guard=object.__getattribute__(self,'_owner_guard')
            if guard is not None and guard() is not True:raise RuntimeError('closed GPU controller owner')
        return object.__getattribute__(self,name)

    def invalidate(self):self._alive=False

    def rebind(self,adapter,world=0,owner_guard=None):
        """Keep the original controller view identity across parameter rebuild.

        The old lifecycle owner must first retire this view. New controls then
        come exclusively from the validated successor's actual public roots.
        """
        if self._alive and (self._owner_guard is None or self._owner_guard() is True):
            raise ValueError('retire the old controller owner before successor rebind')
        if not callable(getattr(adapter,'validate',None)) or not getattr(getattr(adapter,'device',None),'is_cuda',False):
            raise ValueError('actual prepared CUDA controller successor required')
        world=int(world)
        if not 0<=world<adapter.capacity:raise ValueError('controller successor world outside resident owner')
        adapter.validate()
        if owner_guard is not None and owner_guard() is not True:raise ValueError('new controller owner is closed')
        self._adapter,self._world,self._owner_guard=adapter,world,owner_guard
        self._alive=True
        return self

    @property
    def name(self):return getattr(self._adapter.controller,"name","OSC_POSE")

    @property
    def control_dim(self):
        c=self._adapter.controller
        return self._adapter.action_dim-getattr(c,"gripper_dof",1)

    @property
    def joint_dim(self):
        c=self._adapter.controller
        return getattr(c,"actual_joint_dim",getattr(c,"joint_dim",7))

    @property
    def control_limits(self):
        c=self._adapter.controller
        if hasattr(c,"control_limits"):return tuple(np.array(x,copy=True) for x in c.control_limits)
        return np.full(self.control_dim,-1.),np.full(self.control_dim,1.)

    def _read(self,name):
        self._adapter.validate();wp.synchronize_stream(self._adapter.stream)
        a=self._adapter.public.get(name)
        if a is None:a=getattr(self._adapter.controller,name)
        return a.numpy()[self._world].copy()

    @property
    def joint_pos(self):return self._read("q")[:self.joint_dim]
    @property
    def joint_vel(self):return self._read("qvel")[:self.joint_dim]
    @property
    def mass_matrix(self):
        a=self._read("mass");n=int(np.sqrt(a.size))
        return a.reshape(n,n)[:self.joint_dim,:self.joint_dim].copy()
    @property
    def torque_compensation(self):return self._read("phys_bias")[:self.joint_dim]
    @property
    def actuator_limits(self):
        c=self._adapter.controller
        if hasattr(c,"actuator_ranges"):
            wp.synchronize_stream(self._adapter.stream)
            r=c.actuator_ranges.numpy().reshape(-1,2)[:self.joint_dim]
        else:r=self._read("ctrl_range").reshape(-1,2)[:self.joint_dim]
        return r[:,0].copy(),r[:,1].copy()
    @property
    def goal_qpos(self):return self._read("goal")
    @property
    def goal_torque(self):return self._read("goal")
    @property
    def goal_vel(self):return self._read("goal")
    @property
    def goal_pos(self):return self._read("goal_pos")
    @property
    def goal_ori(self):return self._read("goal_rot").reshape(3,3)
    @property
    def ee_pos(self):return self._read("ee_pos")
    @property
    def ee_ori_mat(self):return self._read("ee_rot").reshape(3,3)
    @property
    def kp(self):
        c=self._adapter.controller
        return self._read("current_gains" if hasattr(c,"current_gains") else "gains")[:self.joint_dim if hasattr(c,"family") else 6]
    @property
    def kd(self):
        c=self._adapter.controller;n=self.joint_dim if hasattr(c,"family") else 6
        return self._read("current_gains" if hasattr(c,"current_gains") else "gains")[n:2*n]
    @property
    def torques(self):
        c=self._adapter.controller
        if getattr(c,"family",None)==2:
            wp.synchronize_stream(self._adapter.stream)
            return self._read("ctrl")[c.actuator_ids.numpy()[:c.joint_dim]]
        return self._read("torques_raw")[:self.joint_dim]

def controller_public_view(adapter,world=0,owner_guard=None):return GpuControllerView(adapter,world,owner_guard)
