"""Ordered robosuite observation descriptors and action-boundary packaging.

Input fields intentionally preserve their original epochs. In particular qpos and
qvel may be post-integrator while body/site poses remain pre-integrator. This
module must never request an extra forward to make their times agree.
"""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObservableSpec:
    name: str
    modality: str
    operation: str
    width: int
    indices: tuple = ()
    dependencies: tuple = ()
    period: float = 0.05
    enabled: bool = True
    active: bool = True
    shape: tuple = ()

@dataclass(frozen=True)
class ObservationPlan:
    specs: tuple
    image_specs: tuple = ()
    order: tuple = ()

    @property
    def offsets(self):
        result = {}; offset = 0
        for spec in self.specs:
            result[spec.name] = offset
            offset += spec.width
        return result

    @property
    def width(self):
        return sum(s.width for s in self.specs)

    def validate(self):
        known = set()
        for spec in self.specs:
            if spec.name in known or spec.width < 1 or spec.period <= 0:
                raise ValueError(f"invalid observable descriptor: {spec.name}")
            for name in spec.dependencies:
                if name not in known:
                    raise ValueError(f"dependency {name} must precede {spec.name}")
            known.add(spec.name)
        return self


def plan_from_env(env):
    """Construct descriptors from an already-built CPU environment, without stepping.

    Default identity corruption/filter/delay are supported. Custom processors and
    unknown sensors are rejected explicitly, never silently sampled on the host.
    Camera descriptors are kept separately for the renderer scheduling bridge.
    """
    from robosuite.utils.observables import NO_CORRUPTION, NO_FILTER, NO_DELAY
    robot = env.robots[0]
    prefix = robot.robot_model.naming_prefix
    mapping = {
        prefix + "joint_pos": ("qpos", tuple(robot._ref_joint_pos_indexes), ()),
        prefix + "joint_vel": ("qvel", tuple(robot._ref_joint_vel_indexes), ()),
        prefix + "joint_pos_cos": ("cos", (), (prefix + "joint_pos",)),
        prefix + "joint_pos_sin": ("sin", (), (prefix + "joint_pos",)),
        prefix + "eef_pos": ("site_pos", (int(robot.eef_site_id),), ()),
        prefix + "eef_quat": ("body_quat", (int(env.sim.model.body_name2id(robot.robot_model.eef_name)),), ()),
        prefix + "gripper_qpos": ("qpos", tuple(robot._ref_gripper_joint_pos_indexes), ()),
        prefix + "gripper_qvel": ("qvel", tuple(robot._ref_gripper_joint_vel_indexes), ()),
        "world_pose_in_gripper": ("inverse_pose", (), (prefix + "eef_pos", prefix + "eef_quat")),
    }
    for obj in env.objects:
        name = obj.name
        mapping[name + "_pos"] = ("body_pos", (int(env.obj_body_id[name]),), ())
        mapping[name + "_quat"] = ("body_quat", (int(env.obj_body_id[name]),), ())
        mapping[name + "_to_" + prefix + "eef_pos"] = ("relative_pos", (), (name + "_pos", name + "_quat", "world_pose_in_gripper"))
        mapping[name + "_to_" + prefix + "eef_quat"] = ("relative_quat", (), (name + "_to_" + prefix + "eef_pos",))
    specs = []; images = []
    for name, obs in env._observables.items():
        if obs._corrupter is not NO_CORRUPTION or obs._filter is not NO_FILTER or obs._delayer is not NO_DELAY:
            raise NotImplementedError(f"custom observable processor: {name}")
        shape = tuple(obs._data_shape)
        if obs.modality == "image":
            images.append({"name": name, "shape": shape, "period": obs._sampling_timestep,
                           "enabled": obs.is_enabled(), "active": obs.is_active()})
            continue
        if name not in mapping:
            raise NotImplementedError(f"unmapped low-dimensional sensor: {name}")
        op, indexes, dependencies = mapping[name]
        specs.append(ObservableSpec(name, obs.modality, op, int(np.prod(shape)) if shape else 1,
                                    indexes, dependencies, obs._sampling_timestep,
                                    obs.is_enabled(), obs.is_active(), shape))
    return ObservationPlan(tuple(specs), tuple(images), tuple(env._observables)).validate()


def pack_observations(plan, values, images=None, concatenate_images=False):
    """Package one world at an action boundary, preserving insertion order.

    values is the copied device observed-value buffer, not newly evaluated sensors.
    """
    values = np.asarray(values)
    if values.shape != (plan.width,):
        raise ValueError(f"observed buffer shape {values.shape}, expected {(plan.width,)}")
    offsets = plan.offsets; result = OrderedDict(); modalities = OrderedDict()
    def add(name, modality, value):
        result[name] = value
        arr = np.array([value]) if np.ndim(value) == 0 else np.asarray(value)
        modalities.setdefault(modality + "-state", []).append(arr)
    for spec in plan.specs:
        if spec.enabled and spec.active:
            data = values[offsets[spec.name]:offsets[spec.name]+spec.width].copy()
            value = data[0] if spec.shape == (1,) else data.reshape(spec.shape or ())
            add(spec.name, spec.modality, value)
    images = {} if images is None else images
    for spec in plan.image_specs:
        if spec["enabled"] and spec["active"]:
            value = np.asarray(images[spec["name"]])
            if value.shape != tuple(spec["shape"]):
                raise ValueError(f"image shape mismatch: {spec['name']}")
            add(spec["name"], "image", value)
    if plan.order:
        result = OrderedDict((name, result[name]) for name in plan.order if name in result)
        modalities = OrderedDict()
        by_name = {s.name:s.modality for s in plan.specs}
        by_name.update({s["name"]:"image" for s in plan.image_specs})
        for name,value in result.items():
            arr=np.array([value]) if np.ndim(value)==0 else np.asarray(value)
            modalities.setdefault(by_name[name]+"-state",[]).append(arr)
    for modality, group in modalities.items():
        if modality == "image-state" and not concatenate_images:
            continue
        result[modality] = np.concatenate(group, axis=-1)
    return result
def plan_from_metadata(*,prefix,joint_qpos_indices,joint_qvel_indices,gripper_qpos_indices,
                       gripper_qvel_indices,eef_site_id,eef_body_id,object_body_ids,
                       control_freq=20.0,cameras=(),use_object_obs=True):
    """Source-qualified default LIBERO plan without constructing CPU sensors.

    IDs/indexes come from an actual compiled model-only recipe. Non-default
    processors/custom sensors are outside this constructor and require an explicit
    plan. Actual first GPU forward supplies the values; no dummy dynamic readings
    are introduced merely to make Observable constructors succeed.
    cameras is an ordered sequence of dict(name,height,width[,depth,segmentation]).
    """
    period=1.0/float(control_freq);specs=[];images=[];order=[]
    def add(name,modality,operation,width,indices=(),dependencies=(),active=True,shape=None):
        shape=(width,) if shape is None else shape
        specs.append(ObservableSpec(name,modality,operation,width,tuple(indices),tuple(dependencies),period,True,active,shape))
        order.append(name)
    modality=prefix+"proprio"
    add(prefix+"joint_pos",modality,"qpos",len(joint_qpos_indices),joint_qpos_indices)
    add(prefix+"joint_pos_cos",modality,"cos",len(joint_qpos_indices),dependencies=(prefix+"joint_pos",))
    add(prefix+"joint_pos_sin",modality,"sin",len(joint_qpos_indices),dependencies=(prefix+"joint_pos",))
    add(prefix+"joint_vel",modality,"qvel",len(joint_qvel_indices),joint_qvel_indices)
    add(prefix+"eef_pos",modality,"site_pos",3,(eef_site_id,))
    add(prefix+"eef_quat",modality,"body_quat",4,(eef_body_id,))
    if gripper_qpos_indices:add(prefix+"gripper_qpos",modality,"qpos",len(gripper_qpos_indices),gripper_qpos_indices)
    if gripper_qvel_indices:add(prefix+"gripper_qvel",modality,"qvel",len(gripper_qvel_indices),gripper_qvel_indices)
    for camera in cameras:
        camera_name=camera["name"]
        h,w=int(camera["height"]),int(camera["width"])
        if h<=0 or w<=0:raise ValueError("camera dimensions must be positive")
        def image(kind,channels):
            name=camera_name+"_"+kind
            images.append({"name":name,"shape":(h,w,channels),"camera":camera_name,
                           "kind":kind,"period":period,"enabled":True,"active":True})
            order.append(name)
        image("image",3)
        if camera.get("depth"):image("depth",1)
        segmentations=camera.get("segmentation")
        if isinstance(segmentations,str):segmentations=[segmentations]
        for kind in segmentations or ():
            if kind not in ("instance","class","element"):raise ValueError("unknown camera segmentation: "+str(kind))
            image("segmentation_"+kind,1)
    if not use_object_obs:
        return ObservationPlan(tuple(specs),tuple(images),tuple(order)).validate()
    add("world_pose_in_gripper","object","inverse_pose",16,dependencies=(prefix+"eef_pos",prefix+"eef_quat"),active=False,shape=(4,4))
    for name,bodyid in object_body_ids:
        add(name+"_pos","object","body_pos",3,(bodyid,))
        add(name+"_quat","object","body_quat",4,(bodyid,))
        rel=name+"_to_"+prefix+"eef_pos"
        add(rel,"object","relative_pos",3,dependencies=(name+"_pos",name+"_quat","world_pose_in_gripper"))
        add(name+"_to_"+prefix+"eef_quat","object","relative_quat",4,dependencies=(rel,))
    return ObservationPlan(tuple(specs),tuple(images),tuple(order)).validate()