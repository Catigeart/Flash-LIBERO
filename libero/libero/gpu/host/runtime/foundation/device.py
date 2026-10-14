"""Device-only shared admission, epochs and sticky first-error.

One thread owns a world. Source producer statuses keep their original codes.
This module routes metadata; it never computes physics or writes producer data.
"""
from enum import IntEnum
import warp as wp
from .schema import checked_product

wp.set_module_options({"enable_backward":False,"fast_math":False,"fuse_fp":True})

class SharedError(IntEnum):
    OK=0
    BAD_ACTIVE=1
    BAD_COUNT=2
    CAPACITY=3
    UNSUPPORTED_ROUTE=4
    NONFINITE=5
    NUMERIC_DOMAIN=6
    STALE_EPOCH=7
    BINDING=8
    WATCHDOG=9

from .protocols import LOCAL_PROTOCOLS, PROTOCOL_IDS, MAX_LOCAL_CODE, LOCAL_CLASSES

@wp.func
def _first(w:int,kind:int,stage:int,local:int,item:int,
           event:wp.int64,errors:wp.array(dtype=wp.int32),
           first_stage:wp.array(dtype=wp.int32),
           first_local:wp.array(dtype=wp.int32),
           first_item:wp.array(dtype=wp.int32),
           first_event:wp.array(dtype=wp.int64)):
    if errors[w]==0:
        errors[w]=kind
        first_stage[w]=stage
        first_local[w]=local
        first_item[w]=item
        first_event[w]=event

@wp.kernel
def _gate(admission:wp.array(dtype=wp.int32),
          count:wp.array(dtype=wp.int32),
          producer_epoch:wp.array(dtype=wp.int64),
          expected_epoch:wp.array(dtype=wp.int64),
          producer_valid:wp.array(dtype=wp.int32),
          event:wp.array(dtype=wp.int64),
          errors:wp.array(dtype=wp.int32),
          first_stage:wp.array(dtype=wp.int32),
          first_local:wp.array(dtype=wp.int32),
          first_item:wp.array(dtype=wp.int32),
          first_event:wp.array(dtype=wp.int64),
          active:wp.array(dtype=wp.int32),
          valid:wp.array(dtype=wp.int32),
          capacity:int,stage:int):
    w=wp.tid()
    active[w]=0
    valid[w]=0
    if errors[w]!=0 or admission[w]==0:
        return
    if admission[w]!=1:
        _first(w,1,stage,admission[w],-1,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    if count[w]<0:
        _first(w,2,stage,count[w],-1,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    if count[w]>capacity:
        _first(w,3,stage,count[w],capacity,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    if producer_valid[w]!=1 or producer_epoch[w]!=expected_epoch[w] or producer_epoch[w]<wp.int64(0) or expected_epoch[w]<wp.int64(0):
        _first(w,7,stage,producer_valid[w],-1,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    active[w]=1

@wp.kernel
def _route(active:wp.array(dtype=wp.int32),
           local_status:wp.array(dtype=wp.int32),
           error_item:wp.array(dtype=wp.int32),
           event:wp.array(dtype=wp.int64),
           expected_epoch:wp.array(dtype=wp.int64),
           success:wp.array(dtype=wp.int32),
           error_class:wp.array(dtype=wp.int32),
           errors:wp.array(dtype=wp.int32),
           first_stage:wp.array(dtype=wp.int32),
           first_local:wp.array(dtype=wp.int32),
           first_item:wp.array(dtype=wp.int32),
           first_event:wp.array(dtype=wp.int64),
           valid:wp.array(dtype=wp.int32),
           valid_epoch:wp.array(dtype=wp.int64),
           reason_status:wp.array(dtype=wp.int32),reason_classes:wp.array(dtype=wp.int32),
           reason_event:wp.array(dtype=wp.int64),reason_epoch:wp.array(dtype=wp.int64),
           reason_armed:wp.array(dtype=wp.int32),use_reason:int,protocol:int,stage:int):
    w=wp.tid()
    valid[w]=0
    if errors[w]!=0 or active[w]!=1:
        return
    if expected_epoch[w]<wp.int64(0):
        _first(w,7,stage,1,-1,event[w],errors,first_stage,first_local,first_item,first_event)
        active[w]=0
        return
    code=local_status[w]
    if code<0 or code>=32:
        _first(w,4,stage,code,error_item[w],event[w],errors,first_stage,
               first_local,first_item,first_event)
        active[w]=0
        return
    index=protocol*32+code
    kind=error_class[index]
    if use_reason==1:
        if reason_armed[w]!=2 or reason_status[w]!=code or reason_event[w]!=event[w] or reason_epoch[w]!=expected_epoch[w]:
            kind=8
        elif success[index]==1:
            if reason_classes[w]!=0:kind=8
            else:kind=0
        elif kind<0:
            kind=reason_classes[w]
            if kind<1 or kind>9:kind=8
        elif success[index]!=1 and reason_classes[w]!=kind:
            kind=8
        reason_armed[w]=0
    elif kind<0:
        kind=8
    if success[index]==1 and (use_reason==0 or kind==0):
        valid[w]=1
        valid_epoch[w]=expected_epoch[w]
    else:
        _first(w,kind,stage,code,error_item[w],event[w],
               errors,first_stage,first_local,first_item,first_event)
        active[w]=0

@wp.kernel
def _reset(mask:wp.array(dtype=wp.int32),event:wp.array(dtype=wp.int64),
           errors:wp.array(dtype=wp.int32),first_stage:wp.array(dtype=wp.int32),
           first_local:wp.array(dtype=wp.int32),first_item:wp.array(dtype=wp.int32),
           first_event:wp.array(dtype=wp.int64),active:wp.array(dtype=wp.int32),
           valid:wp.array(dtype=wp.int32),valid_epoch:wp.array(dtype=wp.int64),stage:int):
    w=wp.tid()
    if mask[w]==1:
        errors[w]=0
        first_stage[w]=0
        first_local[w]=0
        first_item[w]=-1
        first_event[w]=wp.int64(-1)
        active[w]=0
        valid[w]=0
        valid_epoch[w]=wp.int64(0)
    elif mask[w]!=0:
        _first(w,1,stage,mask[w],-1,event[w],errors,first_stage,first_local,first_item,first_event)
        active[w]=0
        valid[w]=0

@wp.kernel
def _next_epoch(admission:wp.array(dtype=wp.int32),event:wp.array(dtype=wp.int64),
                errors:wp.array(dtype=wp.int32),first_stage:wp.array(dtype=wp.int32),
                first_local:wp.array(dtype=wp.int32),first_item:wp.array(dtype=wp.int32),
                first_event:wp.array(dtype=wp.int64),epoch:wp.array(dtype=wp.int64),
                active:wp.array(dtype=wp.int32),valid:wp.array(dtype=wp.int32),stage:int):
    w=wp.tid()
    active[w]=0
    valid[w]=0
    if errors[w]!=0 or admission[w]==0:return
    if admission[w]!=1:
        _first(w,1,stage,admission[w],-1,event[w],errors,first_stage,first_local,first_item,first_event)
        return
    if epoch[w]<wp.int64(0):
        _first(w,7,stage,1,-1,event[w],errors,first_stage,first_local,first_item,first_event)
        return
    if epoch[w]==wp.int64(9223372036854775807):
        _first(w,7,stage,2,-1,event[w],errors,first_stage,first_local,first_item,first_event)
        return
    epoch[w]=epoch[w]+wp.int64(1)

class SharedProtocol:
    def __init__(self,nworld,capacity,*,device="cuda:0",stream=None):
        if type(nworld)!=int or nworld<1:
            raise ValueError("positive integer world capacity required")
        if type(capacity)!=int or capacity<0:
            raise ValueError("nonnegative static capacity")
        checked_product([nworld,max(1,capacity)])
        self.nworld=nworld
        self.capacity=capacity
        self.device=wp.get_device(device)
        if not self.device.is_cuda: raise ValueError("CUDA required")
        self.stream=stream or wp.get_stream(self.device)
        if self.stream.device!=self.device: raise ValueError("cross-device stream")
        self.active=wp.zeros(nworld,dtype=wp.int32,device=self.device)
        self.valid=wp.zeros(nworld,dtype=wp.int32,device=self.device)
        self.valid_epoch=wp.zeros(nworld,dtype=wp.int64,device=self.device)
        self.forward_epoch=wp.zeros(nworld,dtype=wp.int64,device=self.device)
        for name in ("errors","first_stage","first_local","first_item"):
            setattr(self,name,wp.zeros(nworld,dtype=wp.int32,device=self.device))
        self.first_event=wp.zeros(nworld,dtype=wp.int64,device=self.device)
        self.reasons={}
        self.no_reason_status=wp.full(nworld,-2,dtype=wp.int32,device=self.device)
        self.no_reason_classes=wp.full(nworld,8,dtype=wp.int32,device=self.device)
        self.no_reason_event=wp.zeros(nworld,dtype=wp.int64,device=self.device)
        self.no_reason_epoch=wp.zeros(nworld,dtype=wp.int64,device=self.device)
        self.no_reason_armed=wp.zeros(nworld,dtype=wp.int32,device=self.device)
        import numpy as np
        success=np.zeros((len(PROTOCOL_IDS),32),np.int32)
        error_class=np.full_like(success,int(SharedError.NUMERIC_DOMAIN))
        for name,p in LOCAL_PROTOCOLS.items():
            success[PROTOCOL_IDS[name],p["success"]]=1
        # Explicit authentic namespaces. Ambiguous original codes cannot be
        # relabelled from a bare integer; until an actual reason is bound they
        # fail closed BINDING. Unknown original codes are UNSUPPORTED_ROUTE.
        error_class[:]=int(SharedError.UNSUPPORTED_ROUTE)
        for name,m in LOCAL_CLASSES.items():
            for code,kind in m.items():
                error_class[PROTOCOL_IDS[name],code]=kind
        self.success=wp.array(success.ravel(),dtype=wp.int32,device=self.device)
        self.error_class=wp.array(error_class.ravel(),dtype=wp.int32,device=self.device)

    def _require(self,arrays):
        from .binding import array_token
        for name,array,dtype in arrays:
            token=array_token(array)
            if array.shape!=(self.nworld,) or token[4]!=dtype or not array.is_contiguous or array.device!=self.device:
                raise ValueError("shared device field dtype/shape/stride/device mismatch: "+name)

    def _state(self):
        self._require([(n,getattr(self,n),"int64" if n in ("first_event","valid_epoch","forward_epoch") else "int32") for n in ("errors","first_stage","first_local","first_item","first_event","active","valid","valid_epoch","forward_epoch")])

    @staticmethod
    def _stage(stage):
        if type(stage)!=int or not 0<stage<=2**31-1:raise ValueError("positive int32 frozen stage ID")

    def _errors(self):
        return [self.errors,self.first_stage,self.first_local,self.first_item,self.first_event]

    def gate(self,admission,count,producer_epoch,expected_epoch,producer_valid,event,*,stage):
        self._stage(stage);self._state()
        self._require([("admission",admission,"int32"),("count",count,"int32"),
            ("producer_epoch",producer_epoch,"int64"),("expected_epoch",expected_epoch,"int64"),
            ("producer_valid",producer_valid,"int32"),("event",event,"int64")])
        wp.launch(_gate,dim=self.nworld,inputs=[
            admission,count,producer_epoch,expected_epoch,producer_valid,event,
            *self._errors(),self.active,self.valid,self.capacity,stage],
            device=self.device,stream=self.stream)

    def prepare_reason(self,registry,protocol,producer_command,event,expected_epoch,*,producer_forest=None):
        if protocol=="S07.linesearch_status":
            if producer_forest is not None:raise ValueError("S07 requires the canonical original native operation, not a caller-supplied replacement forest")
            from .native_reason import NativeSourceReasonWitness
            return NativeSourceReasonWitness(self,registry,protocol,producer_command,event,expected_epoch)
        from .reason import SourceReasonWitness
        return SourceReasonWitness(self,registry,protocol,producer_command,event,expected_epoch,producer_forest=producer_forest)

    def route(self,local_status,error_item,event,expected_epoch,*,protocol,stage,witness=None):
        if protocol not in PROTOCOL_IDS: raise ValueError("unknown local protocol")
        self._stage(stage);self._state()
        self._require([("local_status",local_status,"int32"),("error_item",error_item,"int32"),
                       ("event",event,"int64"),("expected_epoch",expected_epoch,"int64")])
        reasons=(self.no_reason_status,self.no_reason_classes,self.no_reason_event,self.no_reason_epoch,self.no_reason_armed)
        if witness is not None:
            from .reason import SourceReasonWitness
            if not isinstance(witness,SourceReasonWitness) or witness.proto is not self:
                raise ValueError("actual prepared source reason from this F owner required")
            reasons=witness.refresh(local_status,event,expected_epoch,protocol)
        wp.launch(_route,dim=self.nworld,inputs=[
            self.active,local_status,error_item,event,expected_epoch,self.success,
            self.error_class,*self._errors(),self.valid,self.valid_epoch,
            *reasons,int(witness is not None),PROTOCOL_IDS[protocol],stage],device=self.device,stream=self.stream)

    def reset_at_boundary(self,mask,*,event=None,stage=1):
        self._stage(stage);self._state()
        event=self.forward_epoch if event is None else event
        self._require([("reset_mask",mask,"int32"),("event",event,"int64")])
        wp.launch(_reset,dim=self.nworld,inputs=[mask,event,*self._errors(),self.active,self.valid,self.valid_epoch,stage],
                  device=self.device,stream=self.stream)

    def begin_forward(self,admission,*,event=None,stage=1):
        self._stage(stage);self._state()
        event=self.forward_epoch if event is None else event
        self._require([("admission",admission,"int32"),("event",event,"int64")])
        wp.launch(_next_epoch,dim=self.nworld,
                  inputs=[admission,event,*self._errors(),self.forward_epoch,self.active,self.valid,stage],
                  device=self.device,stream=self.stream)
