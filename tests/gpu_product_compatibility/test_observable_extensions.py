"""User Observable callbacks keep original clocks/order over a GPU facade."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy as np
BASE=Path(__file__).resolve().parents[2]/"libero/libero/gpu/host/adapters/observations"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
compat=load("observable_compatibility_unit",BASE/"observable_compatibility.py")
robo_spec=importlib.util.find_spec("robosuite")
original=load("original_observables_unit",Path(next(iter(robo_spec.submodule_search_locations)))/"utils/observables.py")
class FakeGPUFacade:
    value=4
    def read(self):return np.array([self.value],float)
    def forward(self):raise AssertionError("CPU physics forbidden")
    def step(self):raise AssertionError("CPU physics forbidden")
class FakeEnvironment:
    def __init__(self):self.sim=FakeGPUFacade();self.trace=[];self._obs_cache={}
    def _setup_observables(self):
        @original.sensor(modality="custom")
        def first(cache):self.trace.append("first");return self.sim.read()
        @original.sensor(modality="custom")
        def second(cache):self.trace.append("second");return np.array([float(cache.get("first",0))*3])
        return {"first":original.Observable("first",first,sampling_rate=20),"second":original.Observable("second",second,sampling_rate=20)}
class Extensions(unittest.TestCase):
    def setUp(self):
        self.old=sys.modules.get("robosuite")
        sys.modules["robosuite"]=types.SimpleNamespace(macros=types.SimpleNamespace(CONCATENATE_IMAGES=False))
    def tearDown(self):
        if self.old is None:sys.modules.pop("robosuite",None)
        else:sys.modules["robosuite"]=self.old
    def test_original_dependency_order_and_filter(self):
        env=FakeEnvironment();owner=compat.HostObservableCompatibility(env)
        owner.modify("first","filter",lambda value:value+2)
        env.trace.clear();owner.update(.002,force=True)
        self.assertEqual(env.trace,["first","second"])
        self.assertEqual(owner.pack()["first"],6)
        self.assertEqual(owner.pack()["second"],18)
        np.testing.assert_array_equal(owner.pack()["custom-state"],[6,18])
    def test_original_delay_and_clock(self):
        env=FakeEnvironment();owner=compat.HostObservableCompatibility(env)
        owner.modify("first","sampling_rate",10)
        owner.modify("first","delayer",lambda:.09)
        owner.observables["first"].reset()
        owner.update(.02)
        self.assertEqual(owner.observables["first"].obs,0)
        self.assertEqual(owner.observables["first"]._time_since_last_sample,.02)
        owner.update(.002,force=True)
        self.assertEqual(owner.observables["first"].obs,4)
    def test_added_sensor_and_live_owner_transfer(self):
        env=FakeEnvironment();owner=compat.HostObservableCompatibility(env)
        @original.sensor(modality="user")
        def added(cache):return np.array([5.])
        observable=original.Observable("added",added)
        owner.add(observable);owner.update(.002,force=True)
        packet=owner.snapshot_owner()
        restored=compat.HostObservableCompatibility.restore_owner(packet,env)
        self.assertIs(restored,owner)
        self.assertIs(restored.observables["added"],observable)
        with self.assertRaises(ValueError):compat.HostObservableCompatibility.restore_owner(packet,FakeEnvironment())
        owner.modify("added","enabled",False)
        self.assertNotIn("added",owner.pack())
if __name__=="__main__":unittest.main()
