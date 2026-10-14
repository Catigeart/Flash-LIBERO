"""Actual CUDA Graph parameter witness for retained typed source commands.

Immutable single-stream kernel/D2D/memset plans retain exact field owners and
logical spans. Unregistered graphs, allocation nodes and other node types fail closed. No device
array contents or dynamic counts are read by this preparation/launch check.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
import warp as wp
from .binding import BindingError, lookup, array_token, path_objects
from .graph_static_audit import single_stream_order

class _KernelParams(ct.Structure):
    # CUDA_KERNEL_NODE_PARAMS_v2: installed original CUDA header ABI.
    _fields_=[("func",ct.c_void_p),("gridX",ct.c_uint),("gridY",ct.c_uint),("gridZ",ct.c_uint),
              ("blockX",ct.c_uint),("blockY",ct.c_uint),("blockZ",ct.c_uint),
              ("shared",ct.c_uint),("params",ct.POINTER(ct.c_void_p)),
              ("extra",ct.POINTER(ct.c_void_p)),("kern",ct.c_void_p),("ctx",ct.c_void_p)]


class _MemcpyParams(ct.Structure):
    _fields_=[('srcX',ct.c_size_t),('srcY',ct.c_size_t),('srcZ',ct.c_size_t),('srcLOD',ct.c_size_t),
        ('srcType',ct.c_int),('srcHost',ct.c_void_p),('srcDevice',ct.c_uint64),('srcArray',ct.c_void_p),('reserved0',ct.c_void_p),('srcPitch',ct.c_size_t),('srcHeight',ct.c_size_t),
        ('dstX',ct.c_size_t),('dstY',ct.c_size_t),('dstZ',ct.c_size_t),('dstLOD',ct.c_size_t),('dstType',ct.c_int),('dstHost',ct.c_void_p),('dstDevice',ct.c_uint64),('dstArray',ct.c_void_p),('reserved1',ct.c_void_p),('dstPitch',ct.c_size_t),('dstHeight',ct.c_size_t),
        ('width',ct.c_size_t),('height',ct.c_size_t),('depth',ct.c_size_t)]

class _MemsetParams(ct.Structure):
    _fields_=[('dst',ct.c_uint64),('pitch',ct.c_size_t),('value',ct.c_uint),('elementSize',ct.c_uint),('width',ct.c_size_t),('height',ct.c_size_t)]

def _struct_dict(value):
    return {name:int(getattr(value,name) or 0) for name,_ in value._fields_}

def _copy_binding(reg,name,access):
    if name not in reg.bindings:raise BindingError('copy/memset actual field is not registered')
    b=reg.bindings[name];a=lookup(reg.roots[b.root],b.path)
    if b.descriptor.access not in (access,'read_write'):
        raise BindingError('copy/memset actual source access mismatch')
    if b.descriptor.logical_elements==0:
        raise BindingError('unread zero-length sentinel cannot be copied or written')
    return b,a

@dataclass
class MemcpyCommand:
    source:str
    destination:str
    count:int
    source_offset:int=0
    destination_offset:int=0
    writer:str=''
    def validate(self,reg):
        from .schema import DTYPE_BYTES
        src,a=_copy_binding(reg,self.source,'read');dst,b=_copy_binding(reg,self.destination,'write')
        if src.descriptor.dtype!=dst.descriptor.dtype:
            raise BindingError('D2D source/destination native dtype differs')
        if not self.writer or self.writer not in dst.descriptor.writers:
            raise BindingError('D2D writer is not owned by the destination descriptor')
        if any(type(x) is not int or x<0 for x in (self.count,self.source_offset,self.destination_offset)) or self.count==0:
            raise BindingError('positive static D2D span required; zero logical calls need explicit no-op provenance')
        if self.source_offset+self.count>src.descriptor.logical_elements or self.destination_offset+self.count>dst.descriptor.logical_elements:
            raise BindingError('actual D2D copy exceeds retained logical source/destination span')
        item=DTYPE_BYTES[src.descriptor.dtype];self._src=int(a.ptr)+self.source_offset*item;self._dst=int(b.ptr)+self.destination_offset*item;self._bytes=self.count*item
        if src.allocation is dst.allocation and max(self._src,self._dst)<min(self._src+self._bytes,self._dst+self._bytes):
            raise BindingError('overlapping actual D2D spans require separate original no-op/overlap proof')
        self._driver=_Driver()
        return {'node_kind':1,'source_binding':self.source,'destination_binding':self.destination,'src':self._src,'dst':self._dst,'bytes':self._bytes,'writer':self.writer,'source_token':array_token(a),'destination_token':array_token(b)}
    def launch(self,stream):self._driver.call('cuMemcpyDtoDAsync_v2',self._dst,self._src,self._bytes,int(stream.cuda_stream))

@dataclass
class MemsetCommand:
    destination:str
    count_bytes:int
    offset_bytes:int=0
    value:int=0
    writer:str=''
    def validate(self,reg):
        from .schema import DTYPE_BYTES
        dst,b=_copy_binding(reg,self.destination,'write')
        if not self.writer or self.writer not in dst.descriptor.writers:
            raise BindingError('memset writer is not owned by the destination descriptor')
        if any(type(x) is not int or x<0 for x in (self.count_bytes,self.offset_bytes,self.value)) or not self.count_bytes or self.value>255:
            raise BindingError('positive static memset span and literal byte value required')
        if self.offset_bytes+self.count_bytes>dst.descriptor.logical_elements*DTYPE_BYTES[dst.descriptor.dtype]:raise BindingError('memset span exceeds actual retained logical field')
        self._dst=int(b.ptr)+self.offset_bytes;self._driver=_Driver()
        return {'node_kind':2,'destination_binding':self.destination,'dst':self._dst,'bytes':self.count_bytes,'value':self.value,'writer':self.writer,'destination_token':array_token(b)}
    def launch(self,stream):self._driver.call('cuMemsetD8Async',self._dst,self.value,self.count_bytes,int(stream.cuda_stream))


def _raw(value):return ct.string_at(ct.addressof(value),ct.sizeof(value))
def _pointer(value):
    return int(value if isinstance(value,int) else ct.cast(value,ct.c_void_p).value or 0)

class _Driver:
    def __init__(self):
        self.lib=ct.CDLL("libcuda.so.1")
        signatures={
         "cuGraphNodeSetEnabled":([ct.c_void_p,ct.c_void_p,ct.c_uint],ct.c_int),
         "cuGraphNodeGetEnabled":([ct.c_void_p,ct.c_void_p,ct.POINTER(ct.c_uint)],ct.c_int),
         "cuGraphGetNodes":([ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)],ct.c_int),
         "cuGraphNodeGetType":([ct.c_void_p,ct.POINTER(ct.c_int)],ct.c_int),
         "cuGraphNodeGetDependencies":([ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)],ct.c_int),
         "cuFuncGetParamInfo":([ct.c_void_p,ct.c_size_t,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)],ct.c_int),
         "cuGraphMemcpyNodeGetParams":([ct.c_void_p,ct.POINTER(_MemcpyParams)],ct.c_int),
         "cuGraphMemsetNodeGetParams":([ct.c_void_p,ct.POINTER(_MemsetParams)],ct.c_int),
         "cuGraphExecMemcpyNodeSetParams":([ct.c_void_p,ct.c_void_p,ct.POINTER(_MemcpyParams),ct.c_void_p],ct.c_int),
         "cuGraphExecMemsetNodeSetParams":([ct.c_void_p,ct.c_void_p,ct.POINTER(_MemsetParams),ct.c_void_p],ct.c_int),
         "cuMemcpyDtoDAsync_v2":([ct.c_uint64,ct.c_uint64,ct.c_size_t,ct.c_void_p],ct.c_int),
         "cuMemsetD8Async":([ct.c_uint64,ct.c_ubyte,ct.c_size_t,ct.c_void_p],ct.c_int),
         "cuGraphClone":([ct.POINTER(ct.c_void_p),ct.c_void_p],ct.c_int),
         "cuLaunchKernel":([ct.c_void_p,*([ct.c_uint]*7),ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p)],ct.c_int),
        }
        self.exec_params_name="cuGraphExecKernelNodeSetParams_v2" if hasattr(self.lib,"cuGraphExecKernelNodeSetParams_v2") else "cuGraphExecKernelNodeSetParams"
        signatures[self.exec_params_name]=([ct.c_void_p,ct.c_void_p,ct.POINTER(_KernelParams)],ct.c_int)
        self.kernel_params_name="cuGraphKernelNodeGetParams_v2" if hasattr(self.lib,"cuGraphKernelNodeGetParams_v2") else "cuGraphKernelNodeGetParams"
        signatures[self.kernel_params_name]=([ct.c_void_p,ct.POINTER(_KernelParams)],ct.c_int)
        for name,(args,res) in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=res
    def call(self,name,*args):
        rc=getattr(self.lib,name)(*args)
        if rc:raise BindingError("CUDA Graph metadata "+name+" failed:"+str(rc))
    def parameter_sizes(self,func,values):
        offsets=[]
        for index,value in enumerate(values):
            offset=ct.c_size_t();size=ct.c_size_t()
            self.call("cuFuncGetParamInfo",func,index,ct.byref(offset),ct.byref(size))
            if size.value!=ct.sizeof(value):raise BindingError("actual compiled kernel parameter ABI/size mismatch")
            offsets.append((offset.value,size.value))
        # Check the actual compiler signature ends where the typed plan ends.
        offset=ct.c_size_t();size=ct.c_size_t()
        rc=self.lib.cuFuncGetParamInfo(func,len(values),ct.byref(offset),ct.byref(size))
        if rc!=1:raise BindingError("actual compiled kernel has unrepresented parameters or unknown signature")
        return offsets

def _bound_array(reg,array,label):
    token=array_token(array)
    matches=[name for name,b in reg.bindings.items()
             if lookup(reg.roots[b.root],b.path) is array and b.token==token]
    if not matches:raise BindingError("Graph actual array argument not bound to retained source: "+label)
    return matches[0]

def _walk_warp(reg,value,label):
    if isinstance(value,wp.array):return [{"argument":label,"binding":_bound_array(reg,value,label)}]
    # Warp StructInstance exposes its authentic source type through _cls.
    cls=getattr(value,"_cls",None)
    if cls is not None and hasattr(cls,"vars"):
        out=[]
        for name in cls.vars:
            if hasattr(value,name):out+=_walk_warp(reg,getattr(value,name),label+"."+name)
        return out
    if isinstance(value,(tuple,list,dict)):
        raise BindingError("untyped Graph argument container: "+label)
    return []

@dataclass
class WarpKernelCommand:
    command:object
    def validate(self,reg):
        from warp._src.context import Launch,pack_arg
        cmd=self.command
        if not isinstance(cmd,Launch) or cmd.device!=reg.stream.device or cmd.adjoint:
            raise BindingError("actual forward CUDA Warp launch command required")
        if cmd.fwd_args is None or len(cmd.fwd_args)!=len(cmd.kernel.adj.args):
            raise BindingError("actual retained Graph command arguments missing")
        if cmd.cluster_dim!=1:raise BindingError("clustered Graph command ABI not proven")
        fields=[]
        for index,(arg,value) in enumerate(zip(cmd.kernel.adj.args,cmd.fwd_args),1):
            fields+=_walk_warp(reg,value,arg.label)
            # Do not trust mutable cmd.params independently of original objects.
            actual=pack_arg(cmd.kernel,arg.type,arg.label,value,cmd.device,False)
            if _raw(actual)!=_raw(cmd.params[index]):
                raise BindingError("packed Graph argument differs from original typed source: "+arg.label)
        if _raw(cmd.params[0])!=_raw(cmd.bounds):raise BindingError("Graph launch bounds substituted")
        if len(cmd.params_addr)!=len(cmd.params) or any(_pointer(cmd.params_addr[i])!=ct.addressof(v) for i,v in enumerate(cmd.params)):
            raise BindingError("Graph native parameter address table differs from retained typed values")
        return {"function":_pointer(cmd.hooks.forward),"kernel":cmd.kernel.key,
                "parameter_bytes":[_raw(v).hex() for v in cmd.params],
                "parameter_sizes":[ct.sizeof(v) for v in cmd.params],"bound_fields":fields,
                "bounds_size":int(cmd.bounds.size),"bounds_shape":tuple(int(x) for x in cmd.bounds.shape),
                "bounds_coord_mult":int(getattr(cmd.bounds,"coord_mult",1)),"block_dim":int(cmd.block_dim),
                "max_blocks":int(cmd.max_blocks),"grid_stride":bool(cmd.grid_stride),
                "parameter_object_ids":[id(v) for v in cmd.params],
                "module_hash":str(cmd.module_exec.module_hash)}
    def launch(self,stream):self.command.launch(stream=stream)
    def values(self):return self.command.params

@dataclass
class NativeKernelCommand:
    """Exact native source fn/Args: every void-pointer field must be registered.

    Function and argument objects are actual retained root/path values. Scalars
    are typed static launch parameters; dynamic inputs must be device pointers.
    """
    function_root:str
    function_path:str
    arguments_root:str
    arguments_paths:tuple
    grid:tuple
    block:tuple
    shared:int=0
    def validate(self,reg):
        if self.function_root not in reg.roots or self.arguments_root not in reg.roots:
            raise BindingError("native Graph source root missing")
        parents=path_objects(reg.roots[self.function_root],self.function_path)
        func=_pointer(parents[-1])
        if not func:raise BindingError("null native Graph function")
        params=[lookup(reg.roots[self.arguments_root],path) for path in self.arguments_paths]
        fields=[]
        def walk(v,label):
            if isinstance(v,ct.Structure):
                for name,ty in v._fields_:
                    child=getattr(v,name)
                    if ty is ct.c_void_p or issubclass(ty,ct._Pointer):
                        matches=[]
                        for n,b in reg.pointers.items():
                            ps=path_objects(reg.roots[b["root"]],b["path"])
                            if len(ps)>=2 and ps[-2] is v and b["path"].split(".")[-1]==name and int(child or 0)==b["ptr"]:matches.append(n)
                        if not matches:raise BindingError("native Graph pointer has no actual retained field: "+label+"."+name)
                        fields.append({"argument":label+"."+name,"pointer_binding":matches[0]})
                    elif isinstance(child,ct.Structure):walk(child,label+"."+name)
            elif isinstance(v,ct.c_void_p) or isinstance(v,ct._Pointer):
                raise BindingError("standalone opaque Graph pointer requires an authentic Args struct field")
            elif not isinstance(v,ct._SimpleCData):
                raise BindingError("native Graph parameters require original ctypes ABI values")
        for path,value in zip(self.arguments_paths,params):walk(value,path)
        if len(self.grid)!=3 or len(self.block)!=3 or any(type(x)!=int or x<=0 for x in self.grid+self.block) or type(self.shared)!=int or self.shared<0:
            raise BindingError("positive static native CUDA dimensions required")
        self._params=params;self._func=func
        return {"function":func,"kernel":"native:"+self.function_root+"."+self.function_path,
                "parameter_bytes":[_raw(v).hex() for v in params],"parameter_sizes":[ct.sizeof(v) for v in params],
                "bound_fields":fields,"grid":self.grid,"block":self.block,"shared":self.shared,
                "source_parent_ids":[id(p) for p in parents[:-1]],
                "argument_object_ids":[id(v) for v in params]}
    def values(self):return self._params
    def launch(self,stream):
        addresses=(ct.c_void_p*len(self._params))(*(ct.addressof(x) for x in self._params))
        _Driver().call("cuLaunchKernel",self._func,*self.grid,*self.block,self.shared,int(stream.cuda_stream),addresses,None)

def _nodes(driver,graph):
    count=ct.c_size_t()
    driver.call("cuGraphGetNodes",graph.graph,None,ct.byref(count))
    if not count.value:raise BindingError("empty Graph has no typed source plan")
    nodes=(ct.c_void_p*count.value)();driver.call("cuGraphGetNodes",graph.graph,nodes,ct.byref(count))
    ids=[int(p) for p in nodes];deps={}
    for node in ids:
        kind=ct.c_int();driver.call("cuGraphNodeGetType",node,ct.byref(kind))
        if kind.value not in (0,1,2):raise BindingError("Graph node type has no typed source witness:"+str(kind.value))
        n=ct.c_size_t();driver.call("cuGraphNodeGetDependencies",node,None,ct.byref(n))
        if n.value:
            src=(ct.c_void_p*n.value)();driver.call("cuGraphNodeGetDependencies",node,src,ct.byref(n))
            deps[node]={int(p) for p in src}
        else:
            # A root has no dependency buffer. CUDA rejects a non-NULL
            # zero-capacity buffer even though the preceding count query passed.
            deps[node]=set()
    try:return list(single_stream_order(tuple(ids),tuple(tuple(sorted(deps[n])) for n in ids)))
    except ValueError as error:raise BindingError("Graph is not the frozen single-stream source kernel order") from error

def audit_graph(reg,graph,commands,expected,driver=None):
    driver=driver or _Driver();nodes=_nodes(driver,graph)
    if len(nodes)!=len(commands):raise BindingError("Graph node/typed source command count differs")
    out=[]
    for index,(node,cmd,want) in enumerate(zip(nodes,commands,expected)):
        actual=cmd.validate(reg)
        if actual!=want:raise BindingError("retained Graph command source was substituted")
        node_kind=ct.c_int();driver.call("cuGraphNodeGetType",node,ct.byref(node_kind))
        if isinstance(cmd,MemcpyCommand):
            if node_kind.value!=1:raise BindingError("actual D2D source command node type differs")
            p=_MemcpyParams();driver.call("cuGraphMemcpyNodeGetParams",node,ct.byref(p))
            if p.srcType!=2 or p.dstType!=2 or any((p.srcHost,p.dstHost,p.srcArray,p.dstArray,p.reserved0,p.reserved1,p.srcY,p.dstY,p.srcZ,p.dstZ,p.srcLOD,p.dstLOD)) or p.height!=1 or p.depth!=1 or p.width!=want["bytes"] or p.srcDevice+p.srcX!=want["src"] or p.dstDevice+p.dstX!=want["dst"]:
                raise BindingError("actual Graph D2D owner/span/order differs from typed source")
            out.append(dict(want,node_id=node,actual_copy=_struct_dict(p)))
            continue
        if isinstance(cmd,MemsetCommand):
            if node_kind.value!=2:raise BindingError("actual memset source command node type differs")
            p=_MemsetParams();driver.call("cuGraphMemsetNodeGetParams",node,ct.byref(p))
            if p.dst!=want["dst"] or p.height!=1 or p.elementSize!=1 or p.width!=want["bytes"] or p.value!=want["value"]:
                raise BindingError("actual Graph memset owner/span/value differs from typed source")
            out.append(dict(want,node_id=node,actual_memset=_struct_dict(p)))
            continue
        if node_kind.value!=0:raise BindingError("actual typed kernel node type differs")
        abi=driver.parameter_sizes(want["function"],cmd.values())
        p=_KernelParams();driver.call(driver.kernel_params_name,node,ct.byref(p))
        if int(p.func or 0)!=want["function"] or not p.params or bool(p.extra):
            raise BindingError("actual Graph function/parameter mechanism differs from typed source")
        raw=[ct.string_at(p.params[i],size).hex() for i,(_,size) in enumerate(abi)]
        if raw!=want["parameter_bytes"]:raise BindingError("actual Graph parameters differ from bound typed source")
        grid=(p.gridX,p.gridY,p.gridZ);block=(p.blockX,p.blockY,p.blockZ)
        if isinstance(cmd,NativeKernelCommand) and (grid!=tuple(want["grid"]) or block!=tuple(want["block"]) or p.shared!=want["shared"]):
            raise BindingError("actual native Graph launch dimensions differ")
        if isinstance(cmd,WarpKernelCommand) and (block!=(want["block_dim"],1,1) or p.shared!=0):
            raise BindingError("actual Warp Graph block/shared-memory differs")
        out.append(dict(want,node_id=node,actual_grid=grid,actual_block=block,actual_shared=p.shared,
                        actual_parameter_offsets_sizes=abi))
    return out

class GraphRecord:
    def __init__(self,reg,graph,commands,expected,nodes,private,source_sequences=()):
        self.graph=graph;self.commands=commands;self.expected=expected;self.nodes=nodes
        self.source_sequences=tuple(source_sequences)
        self.private=private;self.source_handle=_pointer(graph.graph)
        self.source_capture_id=graph.capture_id;self.private_handle=_pointer(private.graph)
        self.private_exec=private.graph_exec;self.stream_token=reg._stream_token
        self.driver=_Driver()
        self.private_nodes=tuple(audit_graph(reg,private,commands,expected,self.driver))
        self.exec_rebind_parameters=[]
        self.exec_rebind_addresses=[]
        # All executable node storage is prepared once. A later native executable
        # SetParams is not observable through source-graph metadata; explicit
        # SetParams of every node restores the sealed actual arguments before
        # each launch. This is prevention, not a claim to detect opaque exec edits.
        self.context=_pointer(private.device.context)
        for node,cmd in zip(self.private_nodes,commands):
            if isinstance(cmd,(MemcpyCommand,MemsetCommand)):
                ty=_MemcpyParams if isinstance(cmd,MemcpyCommand) else _MemsetParams
                p=ty()
                for key,value in node["actual_copy" if isinstance(cmd,MemcpyCommand) else "actual_memset"].items():setattr(p,key,value)
                self.exec_rebind_addresses.append(None)
                self.exec_rebind_parameters.append(p)
                continue
            values=tuple(cmd.values())
            addresses=(ct.c_void_p*len(values))(*(ct.addressof(v) for v in values))
            p=_KernelParams()
            p.func=node["function"]
            p.gridX,p.gridY,p.gridZ=node["actual_grid"]
            p.blockX,p.blockY,p.blockZ=node["actual_block"]
            p.shared=node["actual_shared"]
            p.params=addresses
            self.exec_rebind_addresses.append((values,addresses))
            self.exec_rebind_parameters.append(p)
    def validate(self,reg):
        for command in self.commands:
            if isinstance(command,WarpKernelCommand):reg.require_warp(command.command)
        for start,sequence in self.source_sequences:
            sequence.validate(reg)
            if any(self.commands[start+i] is not command for i,command in enumerate(sequence.commands)):raise BindingError("actual source sequence order no longer matches complete Graph plan")
        if self.graph.device!=reg.stream.device or _pointer(self.graph.graph)!=self.source_handle or self.graph.capture_id!=self.source_capture_id or self.graph.graph_exec is not None:
            raise BindingError("Graph owner/native source graph/exec substituted")
        if self.stream_token!=reg._stream_token or _pointer(self.private.graph)!=self.private_handle or _pointer(self.private.graph_exec)!=_pointer(self.private_exec):
            raise BindingError("Graph private execution/stream substituted")
        observed=audit_graph(reg,self.graph,self.commands,self.expected,self.driver)
        if observed!=self.nodes:raise BindingError("actual source Graph nodes/ABI changed after capture")
        private_observed=tuple(audit_graph(reg,self.private,self.commands,self.expected,self.driver))
        if private_observed!=self.private_nodes:
            raise BindingError("actual private Graph nodes/grid/ABI changed after capture")
    def rebind_exec(self):
        for node,p in zip(self.private_nodes,self.exec_rebind_parameters):
            if isinstance(p,_MemcpyParams):self.driver.call("cuGraphExecMemcpyNodeSetParams",self.private_exec,node["node_id"],ct.byref(p),self.context)
            elif isinstance(p,_MemsetParams):self.driver.call("cuGraphExecMemsetNodeSetParams",self.private_exec,node["node_id"],ct.byref(p),self.context)
            else:self.driver.call(self.driver.exec_params_name,self.private_exec,node["node_id"],ct.byref(p))
            # Executable node enablement is separate from source-graph params.
            # Every admitted original typed node must actually reach enqueue.
            self.driver.call('cuGraphNodeSetEnabled',self.private_exec,node['node_id'],1)
            enabled=ct.c_uint();self.driver.call('cuGraphNodeGetEnabled',self.private_exec,node['node_id'],ct.byref(enabled))
            if enabled.value!=1:raise BindingError('actual typed producer/metadata Graph node enablement failed')
    def to_json(self):
        return {"graph_owner":id(self.graph),"capture_id":self.source_capture_id,
                "source_handle":self.source_handle,"private_handle":self.private_handle,
                "stream":self.stream_token,"actual_nodes":self.nodes,
                "private_actual_nodes":self.private_nodes,
                "exec_integrity":"prepared private executable; every node explicitly rebound and enabled1 before launch",
                "exec_mutation_detection":"opaque native executable parameter edits cannot be inspected by source Graph APIs",
                "concurrency":"registry mutex; all admitted host Graph operations share this owner",
                "source_sequences":[dict(sequence.to_json(),first_node=start) for start,sequence in self.source_sequences],
                "scope":"actual original CUDA kernel metadata, no GPU contents downloaded"}

def capture(reg,commands):
    if not reg._sealed:raise BindingError("seal complete source arguments before Graph capture")
    reg.validate()
    from .reason import SourceReasonSequence
    flattened=[];source_sequences=[]
    for command in commands:
        if isinstance(command,SourceReasonSequence):
            command.validate(reg);source_sequences.append((len(flattened),command));flattened.extend(command.commands)
        else:
            typed=command if isinstance(command,(WarpKernelCommand,NativeKernelCommand,MemcpyCommand,MemsetCommand)) else WarpKernelCommand(command)
            metadata=typed.validate(reg)
            if isinstance(typed,NativeKernelCommand):
                native_witness=reg._native_source_abi.get(id(typed))
                if native_witness is None or native_witness.producer is not typed:raise BindingError('native Graph source ABI is not admitted; size-only scalar/pointer guesses are forbidden')
                native_witness.validate(native_witness.status,native_witness.event,native_witness.epoch,native_witness.protocol)
            names=[field['binding'] if 'binding' in field else reg.pointers[field['pointer_binding']]['source'] for field in metadata.get('bound_fields',[])]
            names += [metadata[k] for k in ('source_binding','destination_binding') if k in metadata]
            if any(id(reg.bindings[name].allocation) in reg._protected_source_arrays for name in names):raise BindingError('bare metadata/copy/native command touches protected source witness allocation; complete source sequence required')
            flattened.append(typed)
    commands=tuple(flattened)
    if not commands:raise BindingError("typed source Graph command plan required")
    partial=[sequence.witness for _,sequence in source_sequences if not sequence.complete_route]
    if len({id(w) for w in partial})!=len(partial):raise BindingError('repeated source stages require complete device-route sequences')
    for command in commands:
        if isinstance(command,WarpKernelCommand):reg.require_warp(command.command)
    expected=tuple(c.validate(reg) for c in commands)
    wp.capture_begin(stream=reg.stream,force_module_load=False)
    try:
        for cmd in commands:cmd.launch(reg.stream)
    except Exception:
        # End the failed capture so no model wait retains a capturing stream.
        try:wp.capture_end(stream=reg.stream)
        except Exception:pass
        raise
    graph=wp.capture_end(stream=reg.stream)
    actual=audit_graph(reg,graph,commands,expected)
    clone=ct.c_void_p();_Driver().call("cuGraphClone",ct.byref(clone),graph.graph)
    from warp._src.context import Graph
    private=Graph(reg.stream.device);private.graph=clone
    private.module_execs=set(graph.module_execs)
    from warp._src.context import runtime
    g=ct.c_void_p()
    if not runtime.core.wp_cuda_graph_create_exec(private.device.context,reg.stream.cuda_stream,private.graph,ct.byref(g)):
        raise BindingError("private Graph executable preparation failed: "+runtime.get_error_string())
    private.graph_exec=g
    record=GraphRecord(reg,graph,commands,expected,actual,private,source_sequences)
    reg.graphs[id(graph)]=record;reg._retained.extend([graph,private,*commands])
    return graph

def launch(reg,graph):
    if not reg._sealed or id(graph) not in reg.graphs:
        raise BindingError("Graph has no actual typed-source capture witness")
    reg.validate();record=reg.graphs[id(graph)];record.validate(reg)
    for _,sequence in record.source_sequences:
        if not sequence.complete_route and sequence.witness._source_pending:raise BindingError('previous actual source result is still unconsumed')
    record.rebind_exec()
    wp.capture_launch(record.private,stream=reg.stream)
    record.private_exec=record.private.graph_exec
    for _,sequence in record.source_sequences:sequence.note_enqueued(reg)
