"""Original robosuite name/address APIs backed by cold copied model metadata."""
from robosuite.utils.binding_utils import MjModel

NAME_DIMENSIONS = {"body": "nbody", "joint": "njnt", "geom": "ngeom", "site": "nsite",
                   "light": "nlight", "camera": "ncam", "actuator": "nu", "sensor": "nsensor",
                   "tendon": "ntendon", "mesh": "nmesh"}


def bind_original_name_maps(view):
    for kind, dimension in NAME_DIMENSIONS.items():
        key = kind + "_name_to_id"
        if key not in view._metadata:
            continue
        mapping = view._metadata[key]
        count = int(view._metadata.get(dimension, max(mapping.values(), default=-1) + 1))
        id2name = {index: None for index in range(count)}
        for name, index in mapping.items():
            if index not in id2name:
                raise ValueError("cold name index exceeds model dimension")
            id2name[index] = name
        # Match MjModel._extract_mj_names exactly, including the final None ID.
        name2id = {name: index for index, name in id2name.items()}
        names = tuple(id2name[index] for index in sorted(name2id.values()))
        object.__setattr__(view, kind + "_names", names)
        object.__setattr__(view, "_" + kind + "_name2id", name2id)
        object.__setattr__(view, "_" + kind + "_id2name", id2name)


def install_original_name_methods(cls):
    for kind in NAME_DIMENSIONS:
        for suffix in ("name2id", "id2name"):
            name = kind + "_" + suffix
            setattr(cls, name, getattr(MjModel, name))
    for name in ("get_joint_qpos_addr", "get_joint_qvel_addr"):
        setattr(cls, name, getattr(MjModel, name))
