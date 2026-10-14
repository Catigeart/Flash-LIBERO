"""Detached cold S05/S07 consumers; no mutable-array/epoch cache credentials.

This module performs serialization and exact metadata selection only. S05
normalized bytes must come from the accepted CUDA producer, never Python math.
S07 accepts only nonnumeric topology payloads; numeric preflight remains an
explicit original consumer operation before using a topology hit.
"""
from dataclasses import dataclass
import hashlib
import struct
import numpy as np


def freeze_array(value,dtype,shape=None):
    a=np.asarray(value)
    if a.dtype!=np.dtype(dtype) or not a.flags.c_contiguous or (shape is not None and a.shape!=shape):
        raise ValueError('Exact complete cold array dtype/shape/contiguity required')
    return tuple(a.shape),a.tobytes()


def thaw(record,dtype):
    shape,raw=record
    return np.frombuffer(raw,dtype).reshape(shape).copy()




@dataclass(frozen=True)
class ExactTopologyOwner:
    keys: tuple
    payload: tuple
    domain: tuple
    byte_budget: int

    @staticmethod
    def freeze_domain(value):
        if type(value) in (str,bytes,int,bool,type(None)):return (type(value).__name__,value)
        if type(value) is float:return ('float',struct.pack('!d',value))
        if type(value) in (tuple,list):return (type(value).__name__,tuple(ExactTopologyOwner.freeze_domain(x) for x in value))
        raise ValueError('Exact detached primitive domain required')

    @classmethod
    def publish(cls,keys,payload,*,domain,byte_budget):
        required={'body1','body2','dim','exclude','limit_sides','counts_capacity'}
        allowed={'rowstart','contact_queue','dim_offsets','compact_heads','head_type','head_id','head_dim','mu_owner'}
        if set(keys)!=required or not payload or not set(payload)<=allowed:
            raise ValueError('Complete structural keys and nonnumeric payload only')
        if type(byte_budget) is not int or byte_budget<0:raise ValueError('Nonnegative exact budget required')
        k=tuple((name,freeze_array(value,np.int32)) for name,value in sorted(keys.items()))
        p=tuple((name,freeze_array(value,np.int32)) for name,value in sorted(payload.items()))
        if sum(len(x[1][1]) for x in (*k,*p))>byte_budget:return None
        return cls(k,p,cls.freeze_domain(domain),byte_budget)

    def consume(self,keys,*,domain,valid,numeric_preflight,original_rebuild):
        # Run the ORIGINAL current numeric preflight on hit and miss. A cache
        # hit cannot replace it, and no caller-supplied epoch licenses a skip.
        preflight=numeric_preflight()
        if preflight is not True:return None
        current=tuple((name,freeze_array(value,np.int32)) for name,value in sorted(keys.items()))
        if valid is not True or current!=self.keys or self.freeze_domain(domain)!=self.domain:
            return original_rebuild()
        return {name:thaw(value,np.int32) for name,value in self.payload}
