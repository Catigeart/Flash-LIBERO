"""Cold retained Euler selection for the exact g09 readset/SM120/W1-W2.

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

QUALIFIED = {'dimensions': {'nbody': 38, 'nq': 48, 'nv': 43, 'nm': 153, 'njnt': 18, 'nquat': 5}, 'joint_counts': [5, 0, 5, 8], 'dt_bits': 'fca9f1d24d62603f', 'static_readset': {'dof_damping': {'dtype': '<f8', 'shape': [43], 'raw_sha256': 'f32f440b12ed7c94ae5efb779d07f2f68a88717b60d4c0959b17ad5e1999cb31'}, 'dof_Madr': {'dtype': '<i4', 'shape': [43], 'raw_sha256': '3e27466135207e6124bbf87bad323aa0795dd99f58cb135885cce724e1f7b27d'}, 'dof_parentid': {'dtype': '<i4', 'shape': [43], 'raw_sha256': 'a57334b5f8e08a96647a0de0fc27eaac9e1fbfa1c76f007f5308360f2f4fd76a'}, 'dof_simplenum': {'dtype': '<i4', 'shape': [43], 'raw_sha256': '58f218aeb259309eeb545345a8f5c715987967b0ec724806d448394da578d0b0'}, 'jnt_type': {'dtype': '<i4', 'shape': [18], 'raw_sha256': '4c0e0d380d48f2a6a7f66baa70f16e9bef2974dcffe93545a1da175cd5c9f2ce'}, 'jnt_qposadr': {'dtype': '<i4', 'shape': [18], 'raw_sha256': '38ae4abac73ad2b845e1908b24e91cd00b1d3df40b5c6f885075fe46adb9ab35'}, 'jnt_dofadr': {'dtype': '<i4', 'shape': [18], 'raw_sha256': '13c9448fa632713f92cc44c1c525c473e12ce7105bc7b23c30d8fcdf8b90a02b'}}, 'native_source_hashes': {'cuda/integration/euler_lean.cuh': '2b520134dd42ff96a80cd65af85186d18bf6a4bd0b9628ce9e8a8210278cf5ac', 'cuda/integration/euler.cuh': 'd0d37b2bdf2d47cdf94b5ce97f24e2fef04e011384e87caafdb0881c3c732574', 'cuda/common/math.cuh': '933c7a0833e20362eab6cd9407d2e2f04c07876430dbebc4e6319b7c7a902029', 'cuda/common/vector.cuh': '61c655690e0bb5997994668123f9a95ac9ab02f71d6835c3d4e97ef95df53a22', 'cuda/common/warp.cuh': 'c271cb665af358642072fdb87f790f6dfb66a85af213fc1be2ecf83585a548a7', 'cuda/common/packed_mass.cuh': '6cbc9a5f253920413de5020732f09e90e02182514ff8b7c43096484eb8878364', 'cuda/common/packed_mass_gather.cuh': '554d82a89a389d642585cad4db57c2303100c485d2e1d20b1f0ba29d2091f36e', 'cuda/integration/euler_candidates.cuh': '261f0a26530ebb723ac58b085d52fb4e9315f6ff58d43825a5d25f8aa5e61b1d'}, 'tested_binding_hashes': {'host/bindings/euler_lean.py': '4000e09be0532fe4a0b3c2f20cadac35d5c50e576102c855b0928aaac6bac32f', 'host/bindings/euler_candidates.py': '552e8c8253b118863a177ca268ca5e38dd1602a09395e223e52998513034a3f2', 'host/bindings/euler.py': '5a85c703a8bb1041c03842b7de92324eb4367488e5b5493faed0aa5f9a2751ea'}, 'qualified_defines': {'CF_EULER_DIAGNOSTICS': 0, 'CF_EULER_LEAN_JOINT_PARALLEL': 1, 'CF_EULER_NQ': 48, 'CF_EULER_NV': 43, 'CF_EULER_NM': 153, 'CF_EULER_QUAT_STORAGE': 5}, 'architecture': 120}
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
    if Path(gpu_root).resolve()!=Path(__file__).resolve().parents[2]:
        return False  # exact selected project namespace, not another byte copy
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


# Completed E-selected W1/W2 independent receipts bind this cold-only opt-in.
# Historical E01-E09 receipts grant no additional recipe/model/helper domains.
# Default stays disabled; normal native source/build/owner admission still applies.
QUALIFICATION_RECEIPT = {'qualified_world_recipes': [(1, 'scalar'), (1, 'exactcache'), (2, 'short'), (2, 'scalar'), (2, 'exactcache')], 'source_pins': {'cuda/integration/euler_candidates.cuh': '261f0a26530ebb723ac58b085d52fb4e9315f6ff58d43825a5d25f8aa5e61b1d', 'host/bindings/euler_candidates.py': '552e8c8253b118863a177ca268ca5e38dd1602a09395e223e52998513034a3f2', 'host/bindings/euler.py': '5a85c703a8bb1041c03842b7de92324eb4367488e5b5493faed0aa5f9a2751ea'}, 'shared_helper_sha256': '6cbc9a5f253920413de5020732f09e90e02182514ff8b7c43096484eb8878364', 'receipts': [{'path': 'D:\\GPT\\LIBERO-GPU\\kernel-fusion\\optimization-development\\20261012-r01\\owners\\integration\\E-selected-W1_RECEIPT.json', 'sha256': '6d4ec97e974d891c903109b3ff8a0ca29f3e0398b4aa77b2d316af520525aca3', 'result_sha256': '7138916d6e202b8b4d3e11a8129fb2cd288169be236bddc2433f0c8f62f60889', 'source_freeze_sha256': '9d978431dd31b5be9a430b8ce43f98b2ec0b5862f996a090e3a0604fcb6a1dea'}, {'path': 'D:\\GPT\\LIBERO-GPU\\kernel-fusion\\optimization-development\\20261012-r01\\owners\\integration\\E-selected-W2_RECEIPT.json', 'sha256': 'd7fb739fd2ed4166a35d01635811076b1221bd03f3a1c92bd1c84416d64f51ec', 'result_sha256': '5a8af33b0f2a328f348239e422ab3529251aa17f4091b23aba4ec7d98bc28dde', 'source_freeze_sha256': '739a6dc404f1e7c10fa5d5b1e85cd0cee178b887c8aa655bbbd1c1c16f04765c'}], 'scope': 'exact g09 SM120 NV43 dt.002 W1/W2 fast no-tape original Runtime gate/core/publication, tightly packed FP64 and exact live static owner/readset/namespace; no arbitrary positive-flag combinations'}
RECIPES = {'scalar': {'scalar_io': True}, 'short': {'small_angle': True}, 'exactcache': {'single_rhs': True, 'factor_cache': True}}
ALLOWED = {1: ('scalar', 'exactcache'), 2: ('short', 'scalar', 'exactcache')}


def recipe_matches(recipe, options):
    if recipe not in RECIPES or options is None:
        return False
    names=("single_rhs","small_angle","scalar_io","reuse_velocity","factor_cache")
    try:
        return all(type(getattr(options,name)) is bool and
            getattr(options,name)==RECIPES[recipe].get(name,False) for name in names)
    except AttributeError:
        return False


def receipt_allows(worlds, recipe):
    return (QUALIFICATION_RECEIPT is not None and
        recipe in ALLOWED.get(worlds, ()) and
        (worlds,recipe) in QUALIFICATION_RECEIPT.get("qualified_world_recipes", ()))


@dataclass(frozen=True,eq=False)
class EulerColdSelection:
    euler: object
    runtime: object
    snapshot: object
    native: object
    gpu_root: object
    recipe: str

    def matches(self,e,r,profile,options):
        if e is not self.euler or r is not self.runtime:
            return False
        if not receipt_allows(e.capacity,self.recipe) or not recipe_matches(self.recipe,options):
            return False
        try:
            return _matches(e,r,profile,self.snapshot,self.native,self.gpu_root)
        except (AttributeError,KeyError,OSError,TypeError,ValueError):
            return False


def prepare_selected_selection(e,r,*,profile,snapshot,native_model,gpu_root,
        recipe="exactcache",enabled=False):
    """Exact cold owner/source/domain selection, default disabled until receipt.

    The retained recipes remain independent. No arbitrary mix of positive
    flags or negative W1 single/short/reuse route can construct a selection.
    """
    if enabled is not True or not receipt_allows(e.capacity,recipe):
        return None
    context=EulerColdSelection(e,r,snapshot,native_model,Path(gpu_root),recipe)
    options=type("ColdOptions",(),{name: RECIPES[recipe].get(name,False) for name in
        ("single_rhs","small_angle","scalar_io","reuse_velocity","factor_cache")})()
    return context if context.matches(e,r,profile,options) else None


def use_selected(selection,e,r,profile,options):
    return isinstance(selection,EulerColdSelection) and selection.matches(e,r,profile,options)
