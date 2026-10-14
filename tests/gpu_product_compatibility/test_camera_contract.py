"""Camera compatibility tests without CUDA or OpenGL initialization."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
BASE=Path(__file__).resolve().parents[2]/"libero/libero/gpu/host/adapters/observations"
def load(name):
    spec=importlib.util.spec_from_file_location("camera_contract_"+name,BASE/(name+".py"))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
schema=load("schema");sampled=load("sampled_images")
class FakeBridge:
    def __init__(self,h,w):self.h=h;self.w=w;self.calls=[]
    def publish(self,fields):self.calls.append("publish")
    def render_rgb(self,camera,convention,options,depth=False):
        self.calls.append("rgb")
        rgb=np.ones((self.h,self.w,3),np.uint8)
        return (rgb,np.ones((self.h,self.w),np.float32)*.25) if depth else rgb
    def render_segmentation(self,*args):
        self.calls.append("seg")
        out=np.full((self.h,self.w,2),-1,np.int32)
        out[:,:,1]=np.resize([-1,7,8],(self.h,self.w));return out
class CameraContract(unittest.TestCase):
    def test_original_camera_keys_sizes_and_task_labels(self):
        plan=schema.plan_from_metadata(prefix="robot0_",joint_qpos_indices=[0],joint_qvel_indices=[0],gripper_qpos_indices=[],gripper_qvel_indices=[],eef_site_id=0,eef_body_id=0,object_body_ids=[],use_object_obs=False,cameras=[dict(name="cam",height=2,width=3,depth=True,segmentation=["instance","class","element"]),dict(name="other",height=1,width=2)])
        self.assertFalse(any(s.name=="world_pose_in_gripper" for s in plan.specs))
        descriptors=[sampled.image_descriptor(s) for s in plan.image_specs]
        mappings=sampled.segmentation_mapping(descriptors,{"instance":{7:0,8:3},"class":{7:2,8:0}})
        bridges={(2,3):FakeBridge(2,3),(1,2):FakeBridge(1,2)}
        images=[np.zeros(s["shape"]) for s in plan.image_specs];gens=np.ones(6,np.int32);rendered=np.zeros(6,np.int32)
        args=(plan.image_specs,descriptors,mappings,{"xpos":(0,(1,3))},np.arange(18).reshape(6,3),gens,rendered,images,bridges,"opengl",None)
        sampled.render_sampled_images(*args)
        self.assertEqual(bridges[(2,3)].calls.count("rgb"),1)
        self.assertEqual(images[1].dtype,np.float32)
        np.testing.assert_array_equal(images[2].ravel(),[0,1,4,0,1,4])
        np.testing.assert_array_equal(images[3].ravel(),[0,3,1,0,3,1])
        np.testing.assert_array_equal(images[4].ravel(),[-1,7,8,-1,7,8])
        self.assertEqual(images[5].shape,(1,2,3))
        counts=[len(b.calls) for b in bridges.values()]
        sampled.render_sampled_images(*args)
        self.assertEqual([len(b.calls) for b in bridges.values()],counts)
    def test_task_mapping_required(self):
        with self.assertRaises(ValueError):sampled.segmentation_mapping([("cam","segmentation_instance")],{})
    def test_invalid_segmentation_rejected(self):
        with self.assertRaises(NotImplementedError):sampled.image_descriptor({"name":"cam_wrong"})
class DepthClockContract(unittest.TestCase):
    def test_depth_holds_rgb_event_with_different_sampling_rate(self):
        specs=[dict(name="cam_image",shape=(1,1,3),enabled=True),dict(name="cam_depth",shape=(1,1,1),enabled=True)]
        class EventBridge(FakeBridge):
            def publish(self,fields):self.event=float(fields["xpos"][0]);self.calls.append("publish")
            def render_rgb(self,*args,depth=False):
                self.calls.append("rgb")
                return (np.ones((1,1,3),np.uint8),np.full((1,1),self.event,np.float32))
        bridge=EventBridge(1,1);images=[np.zeros(s["shape"]) for s in specs]
        sampled.render_sampled_images(specs,[("cam","image"),("cam","depth")],{}, {"xpos":(0,(1,))},
                                     np.array([[40.],[10.]]),np.array([2,3]),np.zeros(2,np.int32),images,
                                     {(1,1):bridge},"opengl",None,np.array([0,1]))
        self.assertEqual(images[1][0,0,0],10)
        self.assertEqual(bridge.calls.count("rgb"),2)
    def test_depth_without_rgb_cache_is_original_zero(self):
        specs=[dict(name="cam_image",shape=(1,1,3),enabled=False),dict(name="cam_depth",shape=(1,1,1),enabled=True)]
        images=[np.zeros(s["shape"]) for s in specs];bridge=FakeBridge(1,1)
        sampled.render_sampled_images(specs,[("cam","image"),("cam","depth")],{}, {"xpos":(0,(1,))},
                                     np.zeros((2,1)),np.array([5,1]),np.array([5,0]),images,
                                     {(1,1):bridge},"opengl",None,np.array([0,0]))
        self.assertFalse(bridge.calls)
        np.testing.assert_array_equal(images[1],np.zeros((1,1,1)))
    def test_device_capture_reads_parent_scene_and_reset_validity(self):
        import ast,types
        source=ast.parse((BASE/"device.py").read_text())
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="capture_scene_kernel")
        node.decorator_list=[];node.returns=None
        for arg in node.args.args:arg.annotation=None
        wp=types.SimpleNamespace(tid=lambda:0,float64=float)
        namespace={"wp":wp};exec(compile(ast.Module(body=[node],type_ignores=[]),"capture_scene_kernel","exec"),namespace)
        source=np.array([10.]);scene=np.zeros(2);periods=np.array([.004,.002]);enabled=np.ones(2,np.int32)
        elapsed=np.zeros(2);sampled_flags=np.zeros(2,np.int32);generation=np.zeros(2,np.int32)
        parents=np.array([-1,0],np.int32);parent_generation=np.zeros(2,np.int32);captured=np.zeros(2,np.int32)
        args=(2,1,source,scene,periods,enabled,elapsed,sampled_flags,generation,parents,parent_generation,captured,np.ones(1,np.int32),.002,0)
        namespace["capture_scene_kernel"](*args)
        np.testing.assert_array_equal(scene,[10,10])
        source[0]=20;namespace["capture_scene_kernel"](*args)
        self.assertEqual(scene[1],10);self.assertEqual(parent_generation[1],1)
        captured[:]=0;enabled[0]=0;sampled_flags[:]=0
        namespace["capture_scene_kernel"](*args)
        self.assertEqual(parent_generation[1],0)
if __name__=="__main__":unittest.main()
