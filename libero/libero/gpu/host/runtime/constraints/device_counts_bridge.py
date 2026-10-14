"""Source draft: typed resident S counts extraction, not full F/N acceptance."""
import warp as wp

@wp.kernel(enable_backward=False)
def extract_solver_counts(counts:wp.array(dtype=wp.int32),active:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32),committed:wp.array(dtype=wp.int32),ref_epoch:wp.array(dtype=wp.int64),expected_epoch:wp.array(dtype=wp.int64),nefc:wp.array(dtype=wp.int32),ne:wp.array(dtype=wp.int32),nf:wp.array(dtype=wp.int32),ready:wp.array(dtype=wp.int32),rows:wp.int32):
    w=wp.tid()
    ready[w]=0
    if active[w]!=1 or status[w]!=0 or committed[w]!=1:
        return
    e=counts[w*5];f=counts[w*5+1];n=counts[w*5+3]
    if n<0 or n>rows or e<0 or e>n or f<0 or f>n:
        return
    if f>n-e:
        return
    if n>0 and ref_epoch[w]!=expected_epoch[w]:
        return
    nefc[w]=n;ne[w]=e;nf[w]=f;ready[w]=1

class SolverCountsBridge:
    """Retain actual producer owner; copy using actual device counts only.

    ready is a local bridge admissibility witness. It does not manufacture
    F error namespaces, replace active/status, or claim whole N eligibility.
    Consumers must bind authentic source reason/first-error and all missing
    D/C/model inputs. No host count download or CPU truth is used here.
    """
    def __init__(self,source,stream):
        if not isinstance(stream,wp.Stream) or stream.device!=wp.get_device(source.device):
            raise ValueError('explicit producer CUDA stream/device required')
        self.source=source;self._source_root=source;self.stream=stream;self._stream_root=stream;self._stream_token=(id(stream),int(stream.cuda_stream),str(stream.device));self.nworld=source.nworld;self.rows=source.rows
        self.device=wp.get_device(source.device)
        if not self.device.is_cuda or self.nworld<=0 or self.rows<=0 or self.nworld>429496729 or self.rows>2147483647:
            raise ValueError('valid CUDA world/row shape required')
        self.inputs={k:getattr(source,k) for k in ['counts','active','status','committed','ref_epoch','expected_qvel_epoch']}
        for k,a in self.inputs.items():
            dtype=wp.int64 if k in ['ref_epoch','expected_qvel_epoch'] else wp.int32
            size=self.nworld*5 if k=='counts' else self.nworld
            if not isinstance(a,wp.array) or a.device!=self.device or a.dtype!=dtype or a.size!=size or tuple(a.shape)!=(size,) or not a.is_contiguous:
                raise ValueError('authentic typed contiguous S source required: '+k)
        self.tokens={k:(id(a),int(a.ptr),tuple(a.shape),tuple(a.strides),a.dtype,str(a.device)) for k,a in self.inputs.items()}
        with wp.ScopedStream(self.stream):
            self.nefc=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
            self.ne=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
            self.nf=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
            self.ready=wp.zeros(self.nworld,dtype=wp.int32,device=self.device)
        self.output_owner={k:getattr(self,k) for k in ['nefc','ne','nf','ready']}
        self.output_tokens={k:(id(a),int(a.ptr),tuple(a.shape),tuple(a.strides),a.dtype,str(a.device)) for k,a in self.output_owner.items()}
        self.producer_methods={k:getattr(source,k).__func__ for k in ['launch_position','launch_reference']}
        self.producer_roots={k:getattr(source,k) for k in ("s01","s02","s03")}
        self.protected_source=self._source_arrays()
        self.protected_tokens={k:self._token(a) for k,a in self.protected_source.items()}
        self.full_chain_ready=False
    @staticmethod
    def _token(a):return (id(a),int(a.ptr),tuple(a.shape),tuple(a.strides),a.dtype,str(a.device))
    def _source_arrays(self):
        arrays={}
        for prefix,root in [('source',self.source)]+[(k,getattr(self.source,k)) for k in ('s01','s02','s03')]:
            for name,value in vars(root).items():
                if isinstance(value,wp.array):arrays[prefix+'.'+name]=value
                elif isinstance(value,dict):
                    for key,a in value.items():
                        if isinstance(a,wp.array):arrays[prefix+'.'+name+'.'+str(key)]=a
        return arrays
    @staticmethod
    def _span(a):
        itemsize=wp.types.type_size_in_bytes(a.dtype)
        if not a.size:return (int(a.ptr),int(a.ptr))
        if any(stride<0 for stride in a.strides):raise ValueError('negative source stride unsupported')
        return (int(a.ptr),int(a.ptr)+sum((dim-1)*stride for dim,stride in zip(a.shape,a.strides))+itemsize)
    def validate_source(self):
        if self.source is not self._source_root or self.stream is not self._stream_root or (id(self.stream),int(self.stream.cuda_stream),str(self.stream.device))!=self._stream_token:
            raise ValueError('retained producer root or stream replaced')
        if set(self.inputs)!={"counts","active","status","committed","ref_epoch","expected_qvel_epoch"} or set(self.tokens)!=set(self.inputs):
            raise ValueError('source input path set replaced')
        if self.source.nworld!=self.nworld or self.source.rows!=self.rows or wp.get_device(self.source.device)!=self.device:
            raise ValueError('producer capacity/device replaced')
        for k,a in self.inputs.items():
            current=getattr(self.source,k)
            token=(id(current),int(current.ptr),tuple(current.shape),tuple(current.strides),current.dtype,str(current.device))
            if current is not a or token!=self.tokens[k]:
                raise ValueError('producer array identity/layout replaced: '+k)
        if set(self.output_owner)!={'nefc','ne','nf','ready'} or set(self.output_tokens)!={'nefc','ne','nf','ready'}:
            raise ValueError('output path set replaced')
        if set(self.producer_methods)!={'launch_position','launch_reference'}:
            raise ValueError('producer method path set replaced')
        for k in ('nefc','ne','nf','ready'):
            a=self.output_owner[k]
            current=getattr(self,k)
            if not isinstance(current,wp.array) or current.dtype!=wp.int32 or current.device!=self.device or tuple(current.shape)!=(self.nworld,) or not current.is_contiguous:
                raise ValueError("required int32 output layout replaced: "+k)
            token=(id(current),int(current.ptr),tuple(current.shape),tuple(current.strides),current.dtype,str(current.device))
            if current is not a or token!=self.output_tokens[k]:raise ValueError('output allocation/layout replaced: '+k)
        if set(self.producer_roots)!={"s01","s02","s03"} or any(getattr(self.source,k) is not self.producer_roots[k] for k in ("s01","s02","s03")):
            raise ValueError("retained producer stage root replaced")
        current_protected=self._source_arrays()
        if set(current_protected)!=set(self.protected_source) or set(self.protected_tokens)!=set(self.protected_source):raise ValueError('protected source array path set replaced')
        for k,a in current_protected.items():
            if a is not self.protected_source[k] or self._token(a)!=self.protected_tokens[k]:raise ValueError('producer allocation/layout replaced: '+k)
        spans=[(k,)+self._span(a) for k,a in current_protected.items()]
        written=[]
        for k,a in self.output_owner.items():
            lo=int(a.ptr);hi=lo+self.nworld*4
            if any(lo<end and start<hi for _,start,end in spans+written):raise ValueError('output overlaps source/other output: '+k)
            written.append((k,lo,hi))
        for k,method in self.producer_methods.items():
            if getattr(self.source,k).__func__ is not method:raise ValueError('producer method replaced: '+k)
    def launch(self):
        self.validate_source()
        # This fixed caller records/executes both authentic S entries before
        # extraction on one retained stream; no cross-stream assumption.
        self.source.launch_position(stream=self.stream)
        self.source.launch_reference(stream=self.stream)
        wp.launch(extract_solver_counts,dim=self.nworld,inputs=[self.inputs[k] for k in ['counts','active','status','committed','ref_epoch','expected_qvel_epoch']]+[self.nefc,self.ne,self.nf,self.ready,wp.int32(self.rows)],device=self.device,stream=self.stream)
