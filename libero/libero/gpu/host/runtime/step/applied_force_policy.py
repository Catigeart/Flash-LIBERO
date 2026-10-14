"""Explicit captured fresh-target qualification; default/unknown/mixed use reference.

The cold caller must already qualify its platform/compiler and private owner/input
contract. This table grants neither an old downstream dirty transaction nor a
warm ordinary enqueue gain. sm_120 alone is insufficient.
"""
from dataclasses import dataclass
from types import MappingProxyType
import hashlib

RECEIPT_SHA256 = '8f09c5959eaea1b41ca8216fb384c124da18985b92f53ff206b307df8d46d9d9'
PLATFORM = 'NVIDIA GeForce RTX 5070 Laptop GPU / sm_120; Warp1.17.0 Toolkit12.9 Driver13.1'
RETAINED = MappingProxyType({(21, 26, 1): 'f65cdb48bc6d19c7d4b5eab70562ab030220e99ce06157f0ab9125e8d6b8e206', (43, 38, 1): 'f2127326e3dea7b4eab17952159f163fd4e884c0d666221a671496a4d8a42622', (43, 38, 2): 'f2127326e3dea7b4eab17952159f163fd4e884c0d666221a671496a4d8a42622', (57, 29, 1): '90614ba675e3f3a12620419a24d0e1b0738519b7cdf93340d640e27f5a0fe8ce', (57, 29, 2): '90614ba675e3f3a12620419a24d0e1b0738519b7cdf93340d640e27f5a0fe8ce'})
DEPENDENCIES = MappingProxyType({'host/build/native.py': '2ce0255fa7ca4dc172ac5f28e7670dff570d97fd4eb15741fa892aff69efce5f', 'cuda/common/math.cuh': '933c7a0833e20362eab6cd9407d2e2f04c07876430dbebc4e6319b7c7a902029', 'cuda/common/warp.cuh': 'c271cb665af358642072fdb87f790f6dfb66a85af213fc1be2ecf83585a548a7', 'cuda/common/vector.cuh': '61c655690e0bb5997994668123f9a95ac9ab02f71d6835c3d4e97ef95df53a22', 'cuda/runtime/copies.cuh': '2902375c9a86eebd8782de4e3ce71ce234b2e3296e458843d805b3b5441b0aa7', 'cuda/runtime/applied_force_write.cuh': '5f31162d937a832a131e65fc18489218d99ba1af9f9e5b180566b35294ada98b'})

@dataclass(frozen=True)
class AppliedForceQualification:
    receipt_sha256: str
    metadata_sha256: str
    platform: str
    same_platform_compiler: bool = False
    same_private_input_owner_contract: bool = False
    fresh_current_root_build_scope: bool = False
    execution_mode: str = 'unknown'
    block_threads: int = 64
    batch: int = 128
    rounds: int = 3

def select(worlds,nv,nbody,source_root,q=None):
    if not isinstance(q,AppliedForceQualification):return False
    if (q.receipt_sha256!=RECEIPT_SHA256 or q.platform!=PLATFORM
            or q.same_platform_compiler is not True or q.same_private_input_owner_contract is not True
            or q.fresh_current_root_build_scope is not True
            or q.execution_mode!='captured' or q.block_threads!=64 or q.batch!=128 or q.rounds!=3):return False
    if any(type(x) is not int for x in (worlds,nv,nbody)):return False
    key=(nv,nbody,worlds)
    if key not in RETAINED or RETAINED[key]!=q.metadata_sha256:return False
    try:
        return all(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest for name,digest in DEPENDENCIES.items())
    except (OSError,AttributeError):return False
