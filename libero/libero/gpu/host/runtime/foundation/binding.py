from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
import warp as wp
from .schema import DTYPE_BYTES, FieldDescriptor

class BindingError(ValueError):pass

def dtype_name(dtype):
    return getattr(dtype,"__name__",str(dtype)).removeprefix("wp.").removeprefix("warp.types.")

def array_token(array):
    if not isinstance(array,wp.array) or not array.device.is_cuda:
        raise BindingError("actual CUDA Warp array required")
    return (id(array),int(array.ptr),tuple(array.shape),tuple(array.strides),
            dtype_name(array.dtype),str(array.device),int(array.capacity),
            bool(array.is_contiguous))

def lookup(root,path):
    value=root
    for part in path.split("."):
        value=value[part] if isinstance(value,dict) else getattr(value,part)
    return value

def path_objects(root,path):
    values=[root];value=root
    for part in path.split("."):
        value=value[part] if isinstance(value,dict) else getattr(value,part)
        values.append(value)
    return values

def allocation_origin(array):
    """Follow Warp's actual retained view parents to its owning allocation.

    A raw ptr wrapper has no intrinsic owner and is deliberately not inferred
    from address equality. It can only bind as an explicit canonical alias.
    """
    array_token(array);seen=set();chain=[]
    while isinstance(array,wp.array):
        if id(array) in seen:raise BindingError("cyclic view owner chain")
        seen.add(id(array));chain.append(array)
        parent=getattr(array,"_ref",None)
        if parent is None:break
        if not isinstance(parent,wp.array):raise BindingError("external allocation owner unsupported without explicit native allocation proof")
        array=parent;array_token(array)
    allocator=getattr(array,"_allocator",None);deleter=getattr(array,"deleter",None)
    if allocator is None or deleter is None or getattr(deleter,"__self__",None) is not allocator:
        raise BindingError("array has no authentic Warp allocator/deleter owner")
    if int(array.capacity)<=0 or int(array.ptr)<=0:raise BindingError("empty physical allocation; use one unread sentinel")
    return array,chain

def origin_token(allocation):
    root,chain=allocation_origin(allocation)
    if root is not allocation:raise BindingError("allocation argument is a view, not the actual owner")
    return (array_token(root),id(root._allocator),id(root.deleter),
            id(root.deleter.__self__),tuple(id(x) for x in chain))

def extent(array):
    # Physical access span even if caller supplies a fabricated raw view.
    if any(d<0 for d in array.shape):raise BindingError("negative array dimension")
    if not array.size:return 0
    itemsize=DTYPE_BYTES[dtype_name(array.dtype)]
    return itemsize+sum((n-1)*s for n,s in zip(array.shape,array.strides))

@dataclass
class _Binding:
    descriptor: FieldDescriptor
    root: str
    path: str
    allocation: object
    token: tuple
    allocation_token: tuple
    parent_ids: tuple
    view_ids: tuple
    alias_of: str|None

class BindingRegistry:
    """Retain actual allocation owners, fields, canonical aliases and streams.

    Only host metadata is inspected. Device counts/epochs/values are never
    downloaded. Scientific source/epoch masks are a separate device protocol.
    """
    def __init__(self,roots,stream,*,source_records=None,compile_session=None):
        if not isinstance(roots,dict) or not roots:raise BindingError("explicit retained root owners required")
        if not stream.device.is_cuda:raise BindingError("CUDA stream required")
        self.roots=roots;self.stream=stream
        self._root_ids={k:id(v) for k,v in roots.items()}
        self._stream_token=(id(stream),str(stream.device),int(stream.cuda_stream))
        self.bindings={};self.pointers={};self.stream_bindings={};self.graphs={};self._allocation_owners={}
        self._retained=list(roots.values())+[stream];self._sealed=False
        self._graph_mutex=RLock()
        self._protected_source_arrays={}
        self._native_source_abi={}
        self._warp_source_effects={};self._native_module_capsules={};self.compile_session=compile_session;self._warp_compiled={}
        from .authenticated_source import FrozenSources
        self.frozen_sources=FrozenSources(source_records) if source_records else None

    def source_writer(self,kernel):
        from .source_effects import source_writer
        if self.frozen_sources is None:raise BindingError('explicit frozen authentic source set required for kernel writers')
        return source_writer(self.frozen_sources,kernel)

    def admit_warp(self,command):
        from .source_effects import SourceEffectWitness
        if self._sealed:raise BindingError('admit authentic source effects before seal')
        if id(command) in self._warp_source_effects:
            witness=self._warp_source_effects[id(command)]
            if witness.command is not command:raise BindingError('actual source command identity reused')
            witness.validate(self);return witness
        witness=SourceEffectWitness(self,command,self.frozen_sources)
        if self.compile_session is None:raise BindingError('explicit original-source compiler session required before Warp command admission')
        compiled=self.compile_session.bind(self,command)
        self._warp_compiled[id(command)]=compiled
        self._warp_source_effects[id(command)]=witness;self._retained.extend([command,witness,compiled]);return witness

    def require_warp(self,command):
        witness=self._warp_source_effects.get(id(command))
        if witness is None or witness.command is not command:raise BindingError('actual Warp command lacks mandatory authentic source-effect writer authorization')
        witness.validate(self)
        compiled=self._warp_compiled.get(id(command))
        if compiled is None or compiled.command is not command:raise BindingError('actual Warp command lacks sealed original compiled function')
        compiled.validate();return witness

    def prepare_native_module(self,bridge):
        from .native_module import NativeModuleCapsule
        if self._sealed:raise BindingError('prepare native source module before seal')
        if id(bridge) in self._native_module_capsules:
            capsule=self._native_module_capsules[id(bridge)];capsule.validate(bridge,self.stream.device);return capsule
        capsule=NativeModuleCapsule(bridge,self.stream.device)
        self._native_module_capsules[id(bridge)]=capsule;self._retained.append(capsule);return capsule

    def _owners(self):
        if {k:id(v) for k,v in self.roots.items()}!=self._root_ids:
            raise BindingError("same-pointer different owner or replaced root")
        if (id(self.stream),str(self.stream.device),int(self.stream.cuda_stream))!=self._stream_token:
            raise BindingError("stream replaced after bind")

    def add(self,descriptor,root,path,allocation,*,alias_of=None):
        self._owners()
        if self._sealed or descriptor.name in self.bindings or descriptor.name in self.pointers:
            raise BindingError("sealed/duplicate field")
        if root not in self.roots or descriptor.owner not in self.roots:
            raise BindingError("descriptor allocation owner mismatch")
        if alias_of is None and descriptor.owner!=root:raise BindingError("canonical allocation owner must anchor the source root")
        parents=path_objects(self.roots[root],path);array=parents[-1]
        token=array_token(array);alloc_token=origin_token(allocation)
        if token[4]!=descriptor.dtype or tuple(array.shape)!=descriptor.physical_shape:
            raise BindingError("field dtype/shape mismatch: "+descriptor.name)
        if not token[7] or tuple(array.strides)!=descriptor.strides:
            raise BindingError("noncontiguous/stride mismatch: "+descriptor.name)
        if token[5]!=str(self.stream.device) or str(allocation.device)!=token[5]:
            raise BindingError("device/stream mismatch")
        if int(array.ptr)!=int(allocation.ptr)+descriptor.offset:
            raise BindingError("field view offset mismatch")
        needed=descriptor.physical_elements*DTYPE_BYTES[descriptor.dtype]
        if descriptor.offset+needed>int(allocation.capacity) or extent(array)>needed:
            raise BindingError("actual allocation undercapacity")
        if id(allocation) in self._allocation_owners and self._allocation_owners[id(allocation)]!=descriptor.owner:raise BindingError("physical allocation already has a distinct owner")
        canonical=None
        if alias_of is not None:
            if alias_of not in self.bindings:raise BindingError("canonical source must bind before alias")
            canonical=self.bindings[alias_of]
            source=lookup(self.roots[canonical.root],canonical.path)
            if canonical.allocation is not allocation or canonical.descriptor.owner!=descriptor.owner:
                raise BindingError("alias allocation/owner differs from canonical field")
            start=canonical.descriptor.offset
            length=canonical.descriptor.physical_elements*DTYPE_BYTES[canonical.descriptor.dtype]
            if descriptor.offset<start or descriptor.offset+needed>start+length:
                raise BindingError("alias outside canonical field")
            if descriptor.dtype!=canonical.descriptor.dtype or descriptor.epoch!=canonical.descriptor.epoch:
                raise BindingError("alias dtype/epoch mismatch")
            if descriptor.writers!=canonical.descriptor.writers:
                raise BindingError("alias producer write-set mismatch")
            self._retained.append(source)
        try:
            actual_owner,view_chain=allocation_origin(array)
            if actual_owner is not allocation:raise BindingError("actual view originates from a different allocation")
        except BindingError:
            if canonical is None:raise
            # N's raw ptr wrappers deliberately have no _ref. Exact canonical
            # retained source is the external owner proof; never infer by ptr.
            if getattr(array,"_ref",None) is not None or getattr(array,"deleter",None) is not None:
                raise
            view_chain=[array]
        for old in self.bindings.values():
            if old.allocation is allocation:
                d=old.descriptor
                hi=min(d.offset+d.physical_elements*DTYPE_BYTES[d.dtype],descriptor.offset+needed)
                lo=max(d.offset,descriptor.offset)
                same=(canonical is old or canonical is not None and old.alias_of==alias_of)
                if lo<hi and not same and d.access!="read" and descriptor.access!="read" and set(d.writers)&set(descriptor.writers):
                    raise BindingError("overlapping simultaneous writers")
        self._allocation_owners[id(allocation)]=descriptor.owner
        self.bindings[descriptor.name]=_Binding(descriptor,root,path,allocation,token,alloc_token,
                tuple(id(x) for x in parents[:-1]),tuple(id(x) for x in view_chain),alias_of)
        self._retained.extend(parents+view_chain+[allocation])
        return array

    def bind_pointer(self,name,root,path,source,*,offset=0):
        self._owners()
        if self._sealed or name in self.pointers or name in self.bindings or source not in self.bindings:
            raise BindingError("sealed/duplicate/unbound native pointer")
        if type(offset)!=int or offset<0:raise BindingError("native pointer offset")
        b=self.bindings[source];a=lookup(self.roots[b.root],b.path)
        parents=path_objects(self.roots[root],path);actual=parents[-1]
        if type(actual) is not int or actual!=int(a.ptr)+offset:
            raise BindingError("native pointer does not reference retained source")
        length=b.descriptor.physical_elements*DTYPE_BYTES[b.descriptor.dtype]
        if offset>=length or offset%DTYPE_BYTES[b.descriptor.dtype]:raise BindingError("native pointer outside/alignment")
        self.pointers[name]={"root":root,"path":path,"source":source,"offset":offset,
             "ptr":actual,"parent_ids":tuple(id(x) for x in parents[:-1])}
        self._retained.extend(parents[:-1])

    def bind_stream(self,root,path,*,native_handle_path=None):
        self._owners()
        if self._sealed or root not in self.roots:raise BindingError("sealed/missing stream owner")
        parents=path_objects(self.roots[root],path)
        if parents[-1] is not self.stream:raise BindingError("consumer stream object differs")
        if native_handle_path is not None:
            h=lookup(self.roots[root],native_handle_path)
            if type(h)!=int or h!=int(self.stream.cuda_stream):raise BindingError("native consumer stream handle differs")
        self.stream_bindings[(root,path)]={"native_handle_path":native_handle_path,
                "parent_ids":tuple(id(x) for x in parents[:-1])}
        self._retained.extend(parents)

    def seal(self):
        self.validate();self._sealed=True;return self

    def validate(self):
        self._owners()
        for name,b in self.bindings.items():
            try:
                parents=path_objects(self.roots[b.root],b.path);actual=parents[-1]
                if tuple(id(x) for x in parents[:-1])!=b.parent_ids:
                    raise BindingError("replaced parent owner: "+name)
                if array_token(actual)!=b.token or origin_token(b.allocation)!=b.allocation_token:
                    raise BindingError("replaced allocation/array/view/Graph argument: "+name)
                if b.alias_of is not None and getattr(actual,"_ref",None) is None and getattr(actual,"deleter",None) is None:
                    if b.view_ids!=(id(actual),):raise BindingError("lost canonical alias view lineage: "+name)
                else:
                    owner,chain=allocation_origin(actual)
                    if owner is not b.allocation or tuple(id(x) for x in chain)!=b.view_ids:
                        raise BindingError("changed view allocation lineage: "+name)
            except (KeyError,AttributeError) as ex:raise BindingError("missing actual kernel argument: "+name) from ex
        for name,b in self.pointers.items():
            parents=path_objects(self.roots[b["root"]],b["path"])
            source=self.bindings[b["source"]];a=lookup(self.roots[source.root],source.path)
            if tuple(id(x) for x in parents[:-1])!=b["parent_ids"] or parents[-1]!=b["ptr"] or parents[-1]!=int(a.ptr)+b["offset"]:
                raise BindingError("replaced native pointer/owner: "+name)
        for (root,path),b in self.stream_bindings.items():
            parents=path_objects(self.roots[root],path)
            if parents[-1] is not self.stream or tuple(id(x) for x in parents[:-1])!=b["parent_ids"]:
                raise BindingError("consumer stream owner changed")
            if b["native_handle_path"] is not None and lookup(self.roots[root],b["native_handle_path"])!=self._stream_token[2]:
                raise BindingError("native stream handle changed")
        return True

    def manifest(self):
        self.validate()
        return {"roots":self._root_ids,"stream":self._stream_token,
                "fields":{k:{"descriptor":b.descriptor.to_json(),"actual":b.token,
                    "allocation":b.allocation_token,"parent_ids":b.parent_ids,"view_ids":b.view_ids,
                    "alias_of":b.alias_of} for k,b in self.bindings.items()},
                "native_pointers":self.pointers,
                "consumer_streams":[dict(v,root=k[0],path=k[1]) for k,v in self.stream_bindings.items()],
                "source_effects":[w.metadata() for w in self._warp_source_effects.values()],
                "compiled_warp_commands":[c.to_json() for c in self._warp_compiled.values()],
                "native_module_capsules":[c.to_json() for c in self._native_module_capsules.values()],
                "graph_witnesses":[g.to_json() for g in self.graphs.values()]}

    def capture_graph(self,commands):
        from .graph import capture
        with self._graph_mutex:
            return capture(self,commands)

    def launch_graph(self,graph):
        from .graph import launch
        with self._graph_mutex:
            return launch(self,graph)
