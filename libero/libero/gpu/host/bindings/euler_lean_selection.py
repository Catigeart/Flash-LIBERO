"""Cold E01 selection for the executed g09 Euler readset/SM120/W1-W2.

Existing installers/defaults select the original plan without an explicit
selection. Only a verified actual model snapshot and exact live Euler inputs
can create this context. All comparisons run at cold construction/install;
normal NativePlan admission, source invalidation and owner checks stay intact.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
import numpy as np

QUALIFIED = {'dimensions': {'nbody': 38, 'nq': 48, 'nv': 43, 'nm': 153, 'njnt': 18, 'nquat': 5}, 'joint_counts': [5, 0, 5, 8], 'dt_bits': 'fca9f1d24d62603f', 'static_readset': {'dof_damping': {'dtype': '<f8', 'shape': [43], 'raw_sha256': 'f32f440b12ed7c94ae5efb779d07f2f68a88717b60d4c0959b17ad5e1999cb31'}, 'dof_Madr': {'dtype': '<i4', 'shape': [43], 'raw_sha256': '3e27466135207e6124bbf87bad323aa0795dd99f58cb135885cce724e1f7b27d'}, 'dof_parentid': {'dtype': '<i4', 'shape': [43], 'raw_sha256': 'a57334b5f8e08a96647a0de0fc27eaac9e1fbfa1c76f007f5308360f2f4fd76a'}, 'dof_simplenum': {'dtype': '<i4', 'shape': [43], 'raw_sha256': '58f218aeb259309eeb545345a8f5c715987967b0ec724806d448394da578d0b0'}, 'jnt_type': {'dtype': '<i4', 'shape': [18], 'raw_sha256': '4c0e0d380d48f2a6a7f66baa70f16e9bef2974dcffe93545a1da175cd5c9f2ce'}, 'jnt_qposadr': {'dtype': '<i4', 'shape': [18], 'raw_sha256': '38ae4abac73ad2b845e1908b24e91cd00b1d3df40b5c6f885075fe46adb9ab35'}, 'jnt_dofadr': {'dtype': '<i4', 'shape': [18], 'raw_sha256': '13c9448fa632713f92cc44c1c525c473e12ce7105bc7b23c30d8fcdf8b90a02b'}}, 'native_source_hashes': {'cuda/integration/euler_lean.cuh': '2b520134dd42ff96a80cd65af85186d18bf6a4bd0b9628ce9e8a8210278cf5ac', 'cuda/integration/euler.cuh': 'd0d37b2bdf2d47cdf94b5ce97f24e2fef04e011384e87caafdb0881c3c732574', 'cuda/common/math.cuh': '933c7a0833e20362eab6cd9407d2e2f04c07876430dbebc4e6319b7c7a902029', 'cuda/common/vector.cuh': '61c655690e0bb5997994668123f9a95ac9ab02f71d6835c3d4e97ef95df53a22', 'cuda/common/warp.cuh': 'c271cb665af358642072fdb87f790f6dfb66a85af213fc1be2ecf83585a548a7', 'cuda/common/packed_mass.cuh': '8bf08e93da3c1dd7870a6fed42047d74f7d6530572490e519ac091e121ce60f3', 'cuda/common/packed_mass_gather.cuh': '554d82a89a389d642585cad4db57c2303100c485d2e1d20b1f0ba29d2091f36e'}, 'tested_binding_hashes': {'host/bindings/euler_lean.py': '4000e09be0532fe4a0b3c2f20cadac35d5c50e576102c855b0928aaac6bac32f'}, 'qualified_defines': {'CF_EULER_DIAGNOSTICS': 0, 'CF_EULER_LEAN_JOINT_PARALLEL': 1, 'CF_EULER_NQ': 48, 'CF_EULER_NV': 43, 'CF_EULER_NM': 153, 'CF_EULER_QUAT_STORAGE': 5}, 'architecture': 120}
READSET = tuple(QUALIFIED["static_readset"])


def _host(value):
    return np.asarray(value.numpy() if hasattr(value,"numpy") else value)


def _static(e):
    return {"dof_damping":e.dof_damping,"dof_Madr":e.mass.dof_Madr,
        "dof_parentid":e.mass.dof_parentid,"dof_simplenum":e.mass.dof_simplenum,
        "jnt_type":e.jnt_type,"jnt_qposadr":e.jnt_qposadr,"jnt_dofadr":e.jnt_dofadr}


def _same_static(value,name):
    a=_host(value);pin=QUALIFIED["static_readset"][name]
    return (a.dtype.str==pin["dtype"] and list(a.shape)==pin["shape"]
        and sha256(a.tobytes(order="C")).hexdigest()==pin["raw_sha256"])


def _matches(e,r,profile,snapshot,native,gpu_root):
    if profile!="fast" or r is None or getattr(r,"euler",None) is not e:
        return False
    if getattr(r,"integration_tape",None) is not None:
        return False  # the tested local sequence includes the original gate
    if not hasattr(r,"integration_count") or getattr(r,"expected_state_epoch",None) is None:
        return False
    if e.capacity not in (1,2) or e.device.arch!=120 or native.nbody!=38:
        return False
    if (e.nq,e.nv,e.nM,e.njnt,e.nquat)!=(48,43,153,18,5):
        return False
    for name,width in (("qpos",48),("qvel",43),("qacc",43),("qM",153),
                       ("qfrc_smooth",43),("qfrc_constraint",43)):
        value=getattr(e,name)
        dtype_name=getattr(value.dtype,"__name__",getattr(value.dtype,"name",None))
        if (tuple(value.shape)!=(e.capacity,width) or dtype_name!="float64"
                or tuple(value.strides)!=(width*8,8)):
            return False
    time_dtype=getattr(e.time.dtype,"__name__",getattr(e.time.dtype,"name",None))
    if tuple(e.time.shape)!=(e.capacity,) or tuple(e.time.strides)!=(8,) or time_dtype!="float64":
        return False
    if (native.nq,native.nv,native.nM,native.njnt)!=(48,43,153,18):
        return False
    if struct.pack("<d",float(e.dt)).hex()!=QUALIFIED["dt_bits"]:
        return False
    if not snapshot.validate_model(native,READSET):
        return False  # retained byte snapshot alone is not a live-owner proof
    cpu=snapshot.arrays(READSET)
    if not all(_same_static(cpu[name],name) for name in READSET):
        return False
    if not all(_same_static(value,name) for name,value in _static(e).items()):
        return False
    pins=dict(QUALIFIED["native_source_hashes"],**QUALIFIED["tested_binding_hashes"])
    root=Path(gpu_root)
    return all(sha256((root/name).read_bytes()).hexdigest()==pin for name,pin in pins.items())


@dataclass(frozen=True,eq=False)
class E01ColdSelection:
    euler: object
    runtime: object
    snapshot: object
    native: object
    gpu_root: object

    def matches(self,e,r,profile):
        if e is not self.euler or r is not self.runtime:
            return False
        try:
            return _matches(e,r,profile,self.snapshot,self.native,self.gpu_root)
        except (AttributeError,KeyError,OSError,TypeError,ValueError):
            return False


def prepare_e01_selection(e,r,*,profile,snapshot,native_model,gpu_root,enabled=False):
    """Explicit cold opt-in. Root passes the actual validated model owner.

    Compatible, tape, other topology/size/device/dt/source or changed owners
    retain the original route. The context is not a reusable execution token.
    NativePlan keeps its normal qualification/admission checks after selection.
    """
    if enabled is not True:
        return None
    context=E01ColdSelection(e,r,snapshot,native_model,Path(gpu_root))
    return context if context.matches(e,r,profile) else None


def use_e01(selection,e,r,profile):
    return isinstance(selection,E01ColdSelection) and selection.matches(e,r,profile)
