"""Cold exact native model clone owned exclusively by graphics.

No CPU physics or disk file is required. Graphics may mutate visual parameters
and framebuffer dimensions only on this model, never the immutable intake owner.
"""
import numpy as np


def model_bytes(model):
    import mujoco
    buffer=np.empty(int(mujoco.mj_sizeModel(model)),dtype=np.uint8)
    mujoco.mj_saveModel(model,None,buffer)
    return buffer.tobytes()


def clone_render_model(model):
    import mujoco
    source=model_bytes(model)
    name="libero_gpu_render.mjb"
    clone=mujoco.MjModel.from_binary_path(name,assets={name:source})
    if clone is model or model_bytes(clone)!=source:
        raise RuntimeError("native render model clone changed its source MJB bytes")
    return clone
