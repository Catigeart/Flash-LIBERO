"""Native graphics owners may never mutate the physical intake model."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
BASE=Path(__file__).resolve().parents[2]/"libero/libero/gpu/host/adapters/observations"
pkg=types.ModuleType("render_owner_tests");pkg.__path__=[str(BASE)];sys.modules[pkg.__name__]=pkg
def load(name):
    spec=importlib.util.spec_from_file_location(pkg.__name__+"."+name,BASE/(name+".py"))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
helper=load("render_model");boundary=load("boundary")
class InjectedRenderer:
    def __init__(self,model,height,width,max_geom):
        self._scene=mujoco.MjvScene(model,maxgeom=max_geom)
        self._mjr_context=object();self._gl_context=object()
        self.height=height;self.width=width
class ModelOwner(unittest.TestCase):
    def model(self):
        return mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="cam"/><geom type="sphere" size=".1"/><site name="site"/></worldbody></mujoco>')
    def test_exact_memory_clone_has_distinct_visual_storage(self):
        model=self.model();source=helper.model_bytes(model);clone=helper.clone_render_model(model)
        self.assertIsNot(clone,model)
        self.assertEqual(helper.model_bytes(clone),source)
        self.assertFalse(np.shares_memory(clone.geom_rgba,model.geom_rgba))
        clone.geom_rgba[:]=.25;clone.site_rgba[:]=.5;clone.vis.global_.offwidth=1024
        self.assertEqual(helper.model_bytes(model),source)
        self.assertNotEqual(helper.model_bytes(clone),source)
    def test_bridge_publication_keeps_native_intake_byte_exact(self):
        model=self.model();source=helper.model_bytes(model)
        with ExitStack() as stack:
            stack.enter_context(patch.object(mujoco,"Renderer",InjectedRenderer))
            for name in ("mj_forward","mj_step","mj_kinematics","mj_fwdPosition","mj_fwdVelocity"):
                stack.enter_context(patch.object(mujoco,name,side_effect=AssertionError("CPU physics forbidden")))
            bridge=boundary.DerivedRenderBridge(model,width=1024,height=768)
            fields={name:getattr(bridge.data,name).copy() for name in boundary.RENDER_FIELDS}
            fields["geom_rgba"]=np.full(model.geom_rgba.shape,.5,np.float64)
            fields["site_rgba"]=np.full(model.site_rgba.shape,.25,np.float64)
            bridge.publish(fields)
            self.assertTrue(bridge.scene_valid)
            self.assertEqual(helper.model_bytes(model),source)
            self.assertIsNot(bridge.model,model)
            np.testing.assert_array_equal(bridge.model.geom_rgba,np.full(model.geom_rgba.shape,.5,np.float32))
if __name__=="__main__":unittest.main()
