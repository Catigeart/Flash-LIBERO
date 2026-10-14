"""F-owned cold load capsule for the authentic frozen S07 device PTX.

The genuine original bridge/module is retained. The adapter's module is loaded
from the exact retained PTX bytes through real CUDA in the actual source context.
Preparation is separate from the repeated hot loop; no kernel source/options
or numerical computation is changed. Actual device proof remains necessary.
"""
import ctypes as ct,hashlib
from .binding import BindingError

PTX_SHA256='d471d3f281563ba4bb673a3834de1b1bb7b9be8e2b6007922face074bd332fac'
PTX_BYTES=240522
SYMBOL=b'_Z13search_kernel7S07Argsiiiii'

def _handle(value):
    return int(value if isinstance(value,int) else ct.cast(value,ct.c_void_p).value or 0)

class NativeModuleCapsule:
    def __init__(self,bridge,device):
        ptx=getattr(bridge,'_ptx',None)
        if type(ptx) is not bytes or len(ptx)!=PTX_BYTES or hashlib.sha256(ptx).hexdigest()!=PTX_SHA256:raise BindingError('native capsule requires authentic original PTX bytes')
        if not device.is_cuda:raise BindingError('native module actual CUDA owner required')
        self.bridge=bridge;self.device=device;self.original_module=bridge._module;self.original_function=bridge._function
        if not self.original_module or not self.original_function:raise BindingError('prepare genuine source bridge load before F cold capsule')
        self.ptx=ptx;self.buffer=ct.create_string_buffer(ptx)
        self.cuda=ct.CDLL('libcuda.so.1')
        signatures={'cuCtxGetCurrent':([ct.POINTER(ct.c_void_p)],ct.c_int),'cuModuleLoadData':([ct.POINTER(ct.c_void_p),ct.c_void_p],ct.c_int),'cuModuleGetFunction':([ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.c_int),'cuFuncGetParamInfo':([ct.c_void_p,ct.c_size_t,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)],ct.c_int)}
        self.api={}
        for name,(args,result) in signatures.items():
            fn=getattr(self.cuda,name);fn.argtypes=args;fn.restype=result;self.api[name]=fn
        self.api_tokens={name:(id(fn),int(ct.cast(fn,ct.c_void_p).value)) for name,fn in self.api.items()}
        current=ct.c_void_p();self._call('cuCtxGetCurrent',ct.byref(current));self.context=int(current.value or 0)
        if not self.context or self.context!=_handle(device.context):raise BindingError('native cold module load has different actual source device context')
        self.module=ct.c_void_p();self.function=ct.c_void_p();self._call('cuModuleLoadData',ct.byref(self.module),ct.cast(self.buffer,ct.c_void_p));self._call('cuModuleGetFunction',ct.byref(self.function),self.module,SYMBOL)
        self.loaded_token=(int(self.module.value or 0),int(self.function.value or 0),self.context,id(ptx),id(self.buffer),id(bridge),id(device))
        if not self.loaded_token[0] or not self.loaded_token[1]:raise BindingError('native source load returned null module/function')
        self.abi=[]
        for index,size in enumerate((272,4,4,4,4,4)):
            offset=ct.c_size_t();actual=ct.c_size_t();self._call('cuFuncGetParamInfo',self.function,index,ct.byref(offset),ct.byref(actual))
            if actual.value!=size:raise BindingError('native original compiled six-parameter ABI mismatch')
            self.abi.append((offset.value,actual.value))
        offset=ct.c_size_t();actual=ct.c_size_t();rc=self.api['cuFuncGetParamInfo'](self.function,6,ct.byref(offset),ct.byref(actual))
        if rc!=1:raise BindingError('native original ABI has unrepresented seventh parameter')
        self.load_record={'api':'cuModuleLoadData','jit_options':'original default (original bridge same API)','ptx_sha256':PTX_SHA256,'ptx_bytes':PTX_BYTES,'ptx_buffer_prefix_sha256':hashlib.sha256(ct.string_at(self.buffer,PTX_BYTES)).hexdigest(),'buffer_terminator':ct.string_at(ct.addressof(self.buffer)+PTX_BYTES,1).hex(),'context':self.context,'module':self.loaded_token[0],'function':self.loaded_token[1],'symbol_hex':SYMBOL.hex(),'actual_compiled_parameter_offsets_sizes':self.abi,'original_module_retained':int(getattr(self.original_module,'value',self.original_module) or 0),'original_function_retained':int(getattr(self.original_function,'value',self.original_function) or 0)}
        bridge._module=self.module;bridge._function=self.function
        self.validate(bridge,device)
    def _call(self,name,*args):
        rc=self.api[name](*args)
        if rc:raise BindingError('actual native module '+name+' failed: '+str(rc))
    def validate(self,bridge,device):
        if bridge is not self.bridge or device is not self.device or bridge._ptx is not self.ptx or bridge._module is not self.module or bridge._function is not self.function:raise BindingError('native capsule source/owner/function/module replaced')
        if len(self.ptx)!=PTX_BYTES or hashlib.sha256(self.ptx).hexdigest()!=PTX_SHA256 or ct.string_at(self.buffer,PTX_BYTES)!=self.ptx or ct.sizeof(self.buffer)!=PTX_BYTES+1 or ct.string_at(ct.addressof(self.buffer)+PTX_BYTES,1)!=b'\0':raise BindingError('actual retained native module source buffer changed')
        if self.loaded_token!=(int(self.module.value or 0),int(self.function.value or 0),self.context,id(self.ptx),id(self.buffer),id(bridge),id(device)):raise BindingError('actual native capsule handles changed')
        if self.api_tokens!={name:(id(fn),int(ct.cast(fn,ct.c_void_p).value)) for name,fn in self.api.items()}:raise BindingError('native source load/query API replaced')
        if _handle(device.context)!=self.context:raise BindingError('native source device context changed')
        function=ct.c_void_p();self._call('cuModuleGetFunction',ct.byref(function),self.module,SYMBOL)
        if int(function.value or 0)!=self.loaded_token[1]:raise BindingError('actual native source module symbol changed')
        return True
    def to_json(self):
        self.validate(self.bridge,self.device)
        return {'schema':'actual-original-S07-cold-module-source-capsule-v1','load':self.load_record,'scope':'exact original retained PTX loaded in source CUDA context; native actual Args/pointers/caller/error/device numerical evidence separately required'}
