"""Prepared S07 reason bound to actual retained native Args/module/entry.

This is source/metadata implementation awaiting actual native/device validation.
It must not be used as a bare local-code oracle or full shared ABI declaration.
"""
import ctypes as ct,hashlib,inspect
from pathlib import Path
from types import SimpleNamespace
import warp as wp
from .binding import BindingError, lookup, path_objects, array_token
from .graph import NativeKernelCommand, WarpKernelCommand, _pointer
from .protocols import SOURCE_BINDINGS, PROTOCOL_IDS, validate_native_symbol_binding
from .schema import FieldDescriptor
from .authenticated_source import FrozenSources
from .reason import SourceReasonWitness, _prepare_reason_arrays, _mark, _post
from .native_reason_kernels import _reason_linesearch

BRIDGE_SOURCE_SHA256='e1b079b03bb649eebc46171f415310894b8d52179a3d38a80e110b0e61daa816'
BRIDGE_SOURCE_BYTES=20564
PTX_SHA256='1592dfd1c12faad8db7d83ee391954d43da4118c401b0bb8bd9cf006f364efb6'
PTX_BYTES=223960
WORKSPACE_SOURCE_SHA256='eff1dc8e4419fc1bb29f56fa459cfa6ab9e713309e9674715c5d870bc946f5ec'
WORKSPACE_SOURCE_BYTES=11199
FMA_POLICY_PROVENANCE='20261010 global FMA; actual source and newly compiled gated NVRTC PTX'
NATIVE_TO_WORKSPACE={'Jaref':'jar','smooth':'qfrc_smooth','parent':'parentid','status':'s07_status'}
S07_FIELDS=('search','Ma','Jaref','quad0','smooth','qM','J','D','R','loss','cone_mu','cone_friction','efc_type','efc_id','cone_dim','madr','parent','simplenum','nefc','ne','nf','active','model_scalars','Mv','Jv','quadGauss','quad','alpha','slope','eval_trace','LSiter','LSresult','eval_tag','status')

class NativeSourceReasonWitness(SourceReasonWitness):
    def __init__(self,proto,registry,protocol,producer_command,event,epoch):
        if registry._sealed:raise BindingError('prepare native source reason before registry seal')
        if protocol!='S07.linesearch_status' or not isinstance(producer_command,NativeKernelCommand):raise BindingError('actual typed S07 native command required')
        self.proto=proto;self.registry=registry;self.protocol=protocol;self.producer=producer_command
        self.source_metadata=producer_command.validate(registry)
        parents=path_objects(registry.roots[producer_command.function_root],producer_command.function_path)
        if len(parents)<2:raise BindingError('native S07 module owner absent')
        self.bridge=parents[-2]
        self.module_capsule=registry._native_module_capsules.get(id(self.bridge))
        if self.module_capsule is None:raise BindingError('actual original native loaded-source module capsule required before native source admission')
        self.module_capsule.validate(self.bridge,registry.stream.device)
        if producer_command.function_path.split('.')[-1]!='_function' or len(parents)<3:raise BindingError('actual S07 function/containing workspace path differs')
        self.workspace=parents[-3]
        workspace_source=Path(inspect.getsourcefile(type(self.workspace)));workspace_raw=workspace_source.read_bytes()
        if type(self.workspace).__name__!='NewtonFwdWorkspace' or len(workspace_raw)!=WORKSPACE_SOURCE_BYTES or hashlib.sha256(workspace_raw).hexdigest()!=WORKSPACE_SOURCE_SHA256 or self.workspace.s07 is not self.bridge:raise BindingError('native S07 actual containing source workspace differs')

        class_source=Path(inspect.getsourcefile(type(self.bridge)));raw=class_source.read_bytes()
        if type(self.bridge).__name__!='S07DeviceBridge' or len(raw)!=BRIDGE_SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=BRIDGE_SOURCE_SHA256:raise BindingError('native S07 bridge is not the frozen actual source entry')
        kernel_source=Path(self.bridge.kernel_source);source=kernel_source.read_bytes();pin=SOURCE_BINDINGS['reference/newton/newton_linesearch/kernel.cu']
        if len(source)!=pin['bytes'] or hashlib.sha256(source).hexdigest()!=pin['sha256']:raise BindingError('native S07 original source differs')
        ptx=self.bridge._ptx
        if not isinstance(ptx,bytes) or len(ptx)!=PTX_BYTES or hashlib.sha256(ptx).hexdigest()!=PTX_SHA256:raise BindingError('actual retained native PTX differs')
        if not self.bridge._module or not self.bridge._function:raise BindingError('native S07 load is preparation, not lazy hot work')
        self.native_python_sources=FrozenSources([{'path':str(class_source),'bytes':BRIDGE_SOURCE_BYTES,'sha256':BRIDGE_SOURCE_SHA256},{'path':str(workspace_source),'bytes':WORKSPACE_SOURCE_BYTES,'sha256':WORKSPACE_SOURCE_SHA256}])
        self.native_loaded_methods={name:self.native_python_sources.function(getattr(type(self.bridge),name)) for name in ('__init__','_cache_key','_compile','_nvrtc_compile','_gate_ptx','_driver','load','launch')}
        self.workspace_loaded_methods={name:self.native_python_sources.function(getattr(type(self.workspace),name)) for name in ('__init__','_build_s07_args','launch_s07')}
        self.bridge_token=(id(self.bridge),id(ptx),_pointer(self.bridge._module),_pointer(self.bridge._function))
        self.source_binding={'path':str(kernel_source),'sha256':pin['sha256'],'bytes':pin['bytes'],'function':'search_kernel','bridge_sha256':BRIDGE_SOURCE_SHA256,'ptx_sha256':PTX_SHA256}
        params=producer_command.values()
        if len(params)!=6 or not isinstance(params[0],ct.Structure) or tuple((name,ty) for name,ty in params[0]._fields_)!=tuple((name,ct.c_void_p) for name in S07_FIELDS):raise BindingError('actual native Args does not match original34-pointer ABI')
        if any(type(v) is not ct.c_int for v in params[1:]):raise BindingError('native S07 scalar Args must be retained original int32 objects')
        nworld,nv,nM,rows,tracecap=(v.value for v in params[1:])
        if nworld!=proto.nworld or nv!=43 or nM!=153 or rows<0 or tracecap<0:raise BindingError('native S07 fixed model/static scalar binding differs')
        if tuple(producer_command.grid)!=((nworld+127)//128,1,1) or tuple(producer_command.block)!=(128,1,1) or producer_command.shared!=0:raise BindingError('native S07 launch differs from source wrapper')
        args=params[0];self.native_inputs=SimpleNamespace();self.native_fields={}
        if args is not self.workspace._s07_args or nworld!=self.workspace.nworld or rows!=self.workspace.rows or self.workspace.stream is not registry.stream or self.workspace.s07_stream_ptr!=int(registry.stream.cuda_stream):raise BindingError('native S07 actual workspace Args/scalars/stream differs')
        self.native_shape=(nworld,rows)
        self.native_scalar_objects=tuple(params[1:])
        for name in S07_FIELDS:
            matches=[(key,v) for key,v in registry.pointers.items() if path_objects(registry.roots[v['root']],v['path'])[-2] is args and v['path'].split('.')[-1]==name]
            if len(matches)!=1:raise BindingError('one actual native S07 field pointer required: '+name)
            key,pointer=matches[0];b=registry.bindings[pointer['source']];a=lookup(registry.roots[b.root],b.path)
            dtype='int32' if name in ('efc_type','efc_id','cone_dim','madr','parent','simplenum','nefc','ne','nf','active','LSiter','LSresult','eval_tag','status') else 'float64'
            if b.descriptor.dtype!=dtype or pointer['offset']!=0:raise BindingError('native S07 reason actual dtype/span differs: '+name)
            field=getattr(self.workspace,NATIVE_TO_WORKSPACE.get(name,name))
            if a is not field or getattr(args,name)!=int(field.ptr):raise BindingError('native S07 pointer source differs from actual workspace: '+name)
            needed={'search':nworld*43,'Ma':nworld*43,'Jaref':nworld*rows,'quad0':nworld,'smooth':nworld*43,'qM':nworld*153,'J':nworld*rows*43,'D':nworld*rows,'R':nworld*rows,'loss':nworld*rows,'cone_mu':nworld*rows,'cone_friction':nworld*rows*5,'efc_type':nworld*rows,'efc_id':nworld*rows,'cone_dim':nworld*rows,'madr':43,'parent':43,'simplenum':43,'nefc':nworld,'ne':nworld,'nf':nworld,'active':nworld,'model_scalars':2,'Mv':nworld*43,'Jv':nworld*rows,'quadGauss':nworld*3,'quad':nworld*rows*3,'alpha':nworld,'slope':nworld,'eval_trace':nworld*52*4,'LSiter':nworld,'LSresult':nworld,'eval_tag':nworld*52,'status':nworld}[name]
            if b.descriptor.logical_elements<needed:raise BindingError('native S07 reason source logical capacity insufficient: '+name)
            self.native_fields[name]={'source':pointer['source'],'pointer_binding':key,'descriptor':b.descriptor,'array':a}
        self.status=self.native_fields['status']['array']
        self.source_active=self.native_fields['active']['array']
        if self.status.shape!=(nworld,) or self.status.dtype!=wp.int32:raise BindingError('native S07 actual status schema differs')
        _prepare_reason_arrays(self,proto,registry,protocol,event,epoch)
        reason_key='r'+str(PROTOCOL_IDS[protocol])
        for name in ('active','nefc','ne','nf','efc_type','cone_dim','cone_mu','status'):
            field=self.native_fields[name];a=field['array'];flat=a if len(a.shape)==1 else a.flatten();setattr(self.native_inputs,name,flat)
            if flat is not a:
                src=field['descriptor'];d=FieldDescriptor('F.reason.'+reason_key+'.native.'+name,src.dtype,(src.logical_elements,),src.owner,src.lifetime,src.epoch,src.writers,src.valid_write,src.read_domain,src.retain,access='read',offset=src.offset)
                registry.add(d,self.owner,'reasons.'+reason_key+'.native_inputs.'+name,registry.bindings[field['source']].allocation,alias_of=field['source'])
        ins=[getattr(self.native_inputs,n) for n in ('active','nefc','ne','nf','efc_type','cone_dim','cone_mu')]
        self.entry_commands=[wp.launch(_reason_linesearch,dim=nworld,inputs=[*ins,rows,tracecap,self.pre_code,self.pre_kind],device=proto.device,stream=proto.stream,record_cmd=True),wp.launch(_mark,dim=nworld,inputs=[event,epoch,self.saved_event,self.saved_epoch,self.armed,self.source_active,proto.active,self.saved_source_active,self.saved_shared_active],device=proto.device,stream=proto.stream,record_cmd=True)]
        self.post_commands=[wp.launch(_post,dim=nworld,inputs=[self.status,self.pre_code,self.pre_kind,event,epoch,self.saved_event,self.saved_epoch,self.armed,self.saved_source_active,self.saved_shared_active,proto.active,self.status_codes,self.classes,proto.error_class,proto.success,PROTOCOL_IDS[protocol],5],device=proto.device,stream=proto.stream,record_cmd=True)]
        self._source_pending=False
        self.clock_tokens=(array_token(event),array_token(epoch));self.command_metadata=[WarpKernelCommand(c).validate(registry) for c in self.entry_commands+self.post_commands]
        self._check_native_symbol()
        if id(producer_command) in registry._native_source_abi:raise BindingError('native source command already has a different retained witness')
        registry._native_source_abi[id(producer_command)]=self

    def _check_native_symbol(self):
        cuda=self.bridge._cuda
        if cuda is None:raise BindingError('native S07 driver owner is unprepared')
        function=ct.c_void_p();rc=cuda.cuModuleGetFunction(ct.byref(function),self.bridge._module,b'_Z13search_kernel7S07Argsiiiii')
        if rc!=0:raise BindingError('native S07 original module symbol query failed')
        actual=self.producer.validate(self.registry)
        try:validate_native_symbol_binding(actual,self.producer._func,_pointer(function),self.producer.function_path)
        except ValueError as e:raise BindingError(str(e)) from e
        if _pointer(self.bridge._function)!=_pointer(function):raise BindingError('native S07 bridge function differs from module symbol')

    def validate(self,local,event,epoch,protocol):
        self.native_python_sources.validate()
        if {name:self.native_python_sources.function(getattr(type(self.bridge),name)) for name in self.native_loaded_methods}!=self.native_loaded_methods or {name:self.native_python_sources.function(getattr(type(self.workspace),name)) for name in self.workspace_loaded_methods}!=self.workspace_loaded_methods:raise BindingError('actual loaded native bridge/workspace source code changed')
        if self.bridge_token!=(id(self.bridge),id(self.bridge._ptx),_pointer(self.bridge._module),_pointer(self.bridge._function)):raise BindingError('native S07 source/PTX/module/function owner replaced')
        self.module_capsule.validate(self.bridge,self.registry.stream.device)
        self._check_native_symbol()
        if self.native_shape!=(self.workspace.nworld,self.workspace.rows) or self.workspace._s07_args is not self.producer.values()[0] or self.workspace.s07 is not self.bridge or self.workspace.stream is not self.registry.stream or self.workspace.s07_stream_ptr!=int(self.registry.stream.cuda_stream):raise BindingError('native S07 actual source workspace/static scope/Args/stream changed')
        for name,field in self.native_fields.items():
            actual=getattr(self.workspace,NATIVE_TO_WORKSPACE.get(name,name))
            if actual is not field['array'] or getattr(self.workspace._s07_args,name)!=int(actual.ptr):raise BindingError('native S07 actual retained pointer source replaced: '+name)
        super().validate(local,event,epoch,protocol)
