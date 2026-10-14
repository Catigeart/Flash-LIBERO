"""Private original-source Warp compilation and CUDA function provenance.

A recorded Warp Launch is an untrusted preparation input. Its source, semantic
types and static options are checked against a fresh execution of frozen module
bytes; actual execution is then prepared from that independently regenerated
source. The old executable is retained, never used as artifact provenance.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import ModuleType
import ast,builtins,copy,ctypes as ct,hashlib,importlib.util,json,re,sys,types,uuid,subprocess,os
import warp as wp
from .authenticated_source import FrozenSources, fingerprint
from .binding import BindingError
from .runtime_baseline import SemanticDeclarations, original_immutable

def digest(raw):return hashlib.sha256(raw).hexdigest()
def handle(v):return int(v if isinstance(v,int) else ct.cast(v,ct.c_void_p).value or 0)
def stable(v):
    if isinstance(v,dict):return {str(k):stable(x) for k,x in sorted(v.items())}
    if isinstance(v,(tuple,list)):return [stable(x) for x in v]
    if v is None or type(v) in (str,int,float,bool):return v
    if hasattr(v,"value") and hasattr(v,"name"):return [type(v).__module__,type(v).__qualname__,v.name,stable(v.value)]
    raise BindingError("unrepresented original compile option: "+repr(v))
def typed(ty):
    """Native semantic layout, not sizeof-only or mutable repr identity."""
    from warp._src.codegen import Struct
    from warp._src.types import is_array
    if is_array(ty):
        if type(ty) is not wp.array:raise BindingError("only original ordinary Warp array ABI is supported")
        return ("array",int(ty.ndim),typed(ty.dtype))
    if isinstance(ty,Struct):
        fields=tuple((name,typed(var.type)) for name,var in ty.vars.items())
        layout=tuple((name,getattr(ty.ctype,name).offset,ct.sizeof(native)) for name,native in ty.ctype._fields_)
        if tuple(name for name,_ in fields)!=tuple(name for name,_,_ in layout):raise BindingError("struct type field/ctypes order differs")
        return ("struct",ty.key,fields,ct.sizeof(ty.ctype),ct.alignment(ty.ctype),layout)
    canonical=[name for name in ("bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64","float16","bfloat16","float32","float64","vec2","vec3","vec4","quat","mat22","mat33","mat44","spatial_vector","spatial_matrix") if ty is getattr(wp,name,None)]
    if ty in (int,float,bool):return ("python scalar",ty.__name__)
    if canonical:return ("warp native",canonical[0])
    if issubclass(ty,ct.Array) and hasattr(ty,"_wp_scalar_type_"):
        return ("warp composite",tuple(getattr(ty,"_shape_",())),typed(ty._wp_scalar_type_))
    raise BindingError("unresolved original scalar/composite type: "+repr(ty))

class WarpCompileSession:
    def __init__(self,source_records,runtime_records,output,*,compiler_configuration,baseline_configuration):
        self.sources=FrozenSources(source_records);self.records=tuple(dict(x) for x in runtime_records)
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.compiler_configuration=copy.deepcopy(compiler_configuration)
        self.baseline_configuration=copy.deepcopy(baseline_configuration)
        self.source_records=tuple(dict(x) for x in source_records);self.baselines={}
        if not self.compiler_configuration:raise BindingError("frozen original global compiler configuration required")
        self.modules={};self.capsules={};self.prefix="_f_original_"+uuid.uuid4().hex
        self._import=builtins.__import__;self._module_names={}
        self._validate_runtime_files()
        from warp._src import context,build,codegen
        self.context=context;self.build=build;self.codegen=codegen
        self.apis={"build_cuda":build.build_cuda,"ModuleBuilder":context.ModuleBuilder,
                   "ModuleHasher":context.ModuleHasher,"ModuleExec":context.ModuleExec,
                   "cuda_kernel_forward_name":codegen.cuda_kernel_forward_name,
                   "make_full_qualified_name":codegen.make_full_qualified_name,
                   "pack_arg":context.pack_arg}
        self.api_codes={}
        runtime_source=FrozenSources(self.records)
        for name,value in self.apis.items():
            fn=value.__init__ if isinstance(value,type) else value
            self.api_codes[name]=runtime_source.code(fn.__code__)
        self.api_tokens={name:id(value) for name,value in self.apis.items()}
        self.runtime_source=runtime_source
        self.compiler_native=self.context.runtime.core.wp_cuda_compile_program
        self.compiler_native_token=(id(self.context.runtime),id(self.context.runtime.core),id(self.compiler_native),handle(self.compiler_native),str(self.context.runtime.core._name))
        settings=self.baseline_configuration
        private_root=Path(settings["private_root"])
        if not private_root.is_absolute() or str(private_root.resolve())!=str(private_root) or not self.output.is_relative_to(private_root):
            raise BindingError("original declaration session output outside exact frozen private root")
        try:self.semantic_world=SemanticDeclarations(wp,runtime_source,compiler_configuration=self.compiler_configuration,cache_base=settings["parent_kernel_cache_dir"],private_root=str(private_root))
        except (ValueError,TypeError,AttributeError) as error:raise BindingError(str(error)) from error
        self.original_global_declarations=original_immutable(runtime_source)
        self._validate_global_declarations()
        self._verify_original_semantics()
        self.loaded_runtime_tokens=self._validate_loaded_runtime()
        self.builtin_tokens=self._builtin_semantics()
        self.wp_tokens={name:id(value) for name,value in vars(wp).items() if not name.startswith("__")}
        self.cuda=ct.CDLL("libcuda.so.1");self.native={}
        declarations={
            "cuCtxGetCurrent":([ct.POINTER(ct.c_void_p)],ct.c_int),
            "cuModuleLoadData":([ct.POINTER(ct.c_void_p),ct.c_void_p],ct.c_int),
            "cuModuleGetFunction":([ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.c_int)}
        for name,(args,result) in declarations.items():
            fn=getattr(self.cuda,name);fn.argtypes=args;fn.restype=result;self.native[name]=fn
        self.native_tokens={name:(id(fn),handle(fn)) for name,fn in self.native.items()}

    def _runtime_module_names(self):
        return sorted(name for name,module in sys.modules.items() if name.startswith("warp") and getattr(module,"__file__",None) and str(module.__file__).endswith(".py"))
    def _validate_global_declarations(self):
        for name in self._runtime_module_names():
            module=sys.modules[name];filename=str(Path(module.__file__).resolve())
            for key,want in self.original_global_declarations.get(filename,{}).items():
                if not hasattr(module,key):raise BindingError("loaded original global declaration missing: "+name+"."+key)
                try:self.semantic_world.configuration.global_value(self.semantic_world,name,key,getattr(module,key),want)
                except (ValueError,TypeError,AttributeError) as error:raise BindingError(str(error)) from error
    def _original_baseline(self,names):
        key=tuple(names)
        if key not in self.baselines:
            settings=self.baseline_configuration
            folder=self.output/("baseline-"+str(len(self.baselines)));folder.mkdir(exist_ok=False)
            worker=(GPU_ROOT / 'host/runtime/foundation/runtime_baseline.py')
            if str(worker) not in self.sources.inventory:raise BindingError("original declaration worker source pin missing")
            env=dict(settings["environment"]);env["PWD"]=str(folder)
            request={"source_records":self.source_records,"runtime_records":self.records,
                     "native_libraries":settings["native_libraries"],"python_sha256":settings["python_sha256"],
                     "cwd":str(folder),"environment":env,"output":str(folder/"ORIGINAL_DECLARATIONS.json"),
                     "module_names":names,"compiler_configuration":self.compiler_configuration,
                     "kernel_cache_dir":str(folder/"warp-cache"),"private_root":settings["private_root"]}
            path=folder/"REQUEST.json"
            with path.open("x") as out:json.dump(request,out,sort_keys=True,indent=2)
            argv=list(settings["native_prefix"])+[str(worker),str(path)]
            with (folder/"COMMAND.json").open("x") as out:json.dump({"argv":argv,"cwd":str(folder),"environment":env,"request_sha256":digest(path.read_bytes())},out,sort_keys=True,indent=2)
            process_env=dict(os.environ);process_env.update(env)
            with (folder/"STDOUT.log").open("xb") as out,(folder/"STDERR.log").open("xb") as err:
                result=subprocess.run(argv,cwd=str(folder),env=process_env,stdout=out,stderr=err)
            with (folder/"EXIT.json").open("x") as out:json.dump({"exit_code":result.returncode},out)
            if result.returncode:raise BindingError("independent original declaration child failed; complete raw retained: "+str(folder))
            data=json.loads(Path(request["output"]).read_bytes())
            if data["request_sha256"]!=digest(path.read_bytes()) or data["physics_evaluated"] is not False:raise BindingError("independent declaration baseline request/protocol differs")
            data["baseline_artifacts"]={name:{"path":str(artifact),"sha256":digest(artifact.read_bytes()),"bytes":artifact.stat().st_size} for name,artifact in (("request",path),("result",Path(request["output"])),("command",folder/"COMMAND.json"))}
            self.baselines[key]=data
        return self.baselines[key]
    def _verify_original_semantics(self):
        names=self._runtime_module_names();baseline=self._original_baseline(names)
        try:actual=self.semantic_world.snapshot(names)
        except (ValueError,TypeError,AttributeError) as error:raise BindingError(str(error)) from error
        if actual!=baseline["declarations"]:
            categories=[name for name in actual if actual[name]!=baseline["declarations"].get(name)]
            raise BindingError("actual initial compiler/helper/native/type/overload declarations differ from independent original bytes: "+",".join(categories))
        self.current_original_baseline=baseline

    def _validate_runtime_files(self):
        if not self.records:raise BindingError("independently frozen compiler/runtime/native include records required")
        expected={str(Path(r["path"]).resolve()):r for r in self.records}
        root=Path(wp.__file__).resolve().parent
        actual={str(p.resolve()) for p in root.rglob("*") if p.is_file() and p.suffix in (".py",".h",".hpp",".c",".cpp",".cu",".so",".bc",".a")}
        if not actual<=expected.keys():raise BindingError("complete original Warp Python/compiler/native include inventory required")
        for key,row in expected.items():
            raw=Path(key).read_bytes()
            if len(raw)!=row["bytes"] or digest(raw)!=row["sha256"]:raise BindingError("original compile dependency bytes differ: "+key)
    def _generated_dataclass_helper(self,value,label):
        import dataclasses
        allowed={"__init__","__repr__","__eq__","__lt__","__le__","__gt__","__ge__","__hash__","__setattr__","__delattr__"}
        module=sys.modules.get(value.__module__)
        if module is None or not label.startswith(module.__name__+"."):return False
        suffix=label[len(module.__name__)+1:].split(".")
        if len(suffix)!=2 or suffix[1] not in allowed:return False
        cls=getattr(module,suffix[0],None)
        if not isinstance(cls,type) or not dataclasses.is_dataclass(cls):return False
        fn=vars(cls).get(suffix[1]);fn=getattr(fn,"__func__",fn)
        if fn is not value:return False
        path=str(Path(module.__file__).resolve())
        if path not in self.runtime_source.syntax:return False
        tree=self.runtime_source.syntax[path]
        declarations=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==suffix[0]]
        if len(declarations)!=1:return False
        declaration=declarations[0]
        if any(isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==suffix[1] for x in declaration.body):return False
        names={a.asname or a.name for x in tree.body if isinstance(x,ast.ImportFrom) and x.module=="dataclasses" and not x.level for a in x.names if a.name=="dataclass"}
        decorators=[x.func if isinstance(x,ast.Call) else x for x in declaration.decorator_list]
        if not any(isinstance(x,ast.Name) and x.id in names and getattr(module,x.id,None) is dataclasses.dataclass for x in decorators):return False
        # Typed factory source and every generated field/default/annotation
        # are compared with a separate original-byte worker, never self-truth.
        actual_state=self.semantic_world.dataclass_state(cls)
        expected=self.current_original_baseline["declarations"]["helpers"]
        state_label=module.__name__+"."+cls.__qualname__+".__dataclass_state__"
        if expected.get(state_label)!=actual_state:return False
        return expected.get(label)==self.semantic_world.value(value)

    def _generated_namedtuple_helper(self,value,label):
        import collections,typing
        parts=label.rsplit(".",2)
        if len(parts)!=3:return False
        module=sys.modules.get(parts[0])
        if module is None or not module.__name__.startswith("warp"):return False
        suffix=parts[1:]
        if len(suffix)!=2 or suffix[1]!="__new__":return False
        cls=getattr(module,suffix[0],None)
        if not isinstance(cls,type) or not issubclass(cls,tuple) or "_fields" not in vars(cls):return False
        if getattr(vars(cls).get("__new__"),"__func__",vars(cls).get("__new__")) is not value:return False
        path=str(Path(module.__file__).resolve())
        if path not in self.runtime_source.syntax:return False
        tree=self.runtime_source.syntax[path];declared=False
        aliases={a.asname or a.name:x.module for x in tree.body if isinstance(x,ast.ImportFrom) and not x.level for a in x.names if (x.module,a.name) in (("typing","NamedTuple"),("collections","namedtuple"))}
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name==suffix[0]:
                declared=any(isinstance(base,ast.Name) and aliases.get(base.id)=="typing" and getattr(module,base.id,None) is typing.NamedTuple for base in node.bases)
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==suffix[0] and isinstance(node.value,ast.Call):
                call=node.value
                declared=isinstance(call.func,ast.Name) and aliases.get(call.func.id)=="collections" and getattr(module,call.func.id,None) is collections.namedtuple
        if not declared:return False
        if value.__globals__.get("_tuple_new") is not tuple.__new__:return False
        expected=self.current_original_baseline["declarations"]["helpers"]
        state_label=module.__name__+"."+cls.__qualname__+".__namedtuple_state__"
        return expected.get(state_label)==self.semantic_world.namedtuple_state(cls) and expected.get(label)==self.semantic_world.value(value)

    def _validate_loaded_runtime(self):
        tokens={}
        def function(value,label,external_allowed=False,generated_parent=False):
            if not isinstance(value,types.FunctionType):return
            filename=str(Path(value.__code__.co_filename).resolve())
            dataclass_generated=self._generated_dataclass_helper(value,label)
            namedtuple_generated=self._generated_namedtuple_helper(value,label)
            generated=dataclass_generated or namedtuple_generated
            if dataclass_generated:
                import dataclasses
                cls=getattr(sys.modules[value.__module__],label.split(".")[-2])
                for name in self.semantic_world.dataclass_factory_names(cls):
                    factory=getattr(dataclasses,name)
                    tokens["dataclasses.actual-factory."+name]=(id(factory),id(factory.__code__),fingerprint(factory.__code__))
            if namedtuple_generated:
                import collections,typing
                for name,fn in (("collections.namedtuple",collections.namedtuple),("typing._make_nmtuple",typing._make_nmtuple),("typing.NamedTupleMeta.__new__",typing.NamedTupleMeta.__new__)):
                    tokens[name]=(id(fn),id(fn.__code__),fingerprint(fn.__code__))
                tokens[label+"._tuple_new"]=(id(value.__globals__["_tuple_new"]),)
                parts=label.rsplit(".",2);cls=getattr(sys.modules[parts[0]],parts[1])
                for field in cls._fields:tokens[label+".getter."+field]=(id(vars(cls)[field]),)
            covered_generated=generated or generated_parent
            if filename not in self.runtime_source.files:
                if not external_allowed and not (value.__code__.co_filename=="<string>" and covered_generated):
                    raise BindingError("foreign loaded compiler/helper code: "+label+":"+filename)
            else:self.runtime_source.code(value.__code__)
            closures=[]
            for cell in value.__closure__ or ():
                child=cell.cell_contents
                if isinstance(child,types.FunctionType):function(child,label+".<closure>",generated_parent=covered_generated)
                closures.append((type(child).__module__,type(child).__qualname__,id(child),repr(child) if type(child) in (str,int,float,bool,type(None)) else None))
            tokens[label]=(id(value),id(value.__code__),fingerprint(value.__code__),tuple(closures))
            wrapped=getattr(value,"__wrapped__",None)
            if isinstance(wrapped,types.FunctionType):function(wrapped,label+".__wrapped__",generated_parent=covered_generated)
        for name,module in tuple(sys.modules.items()):
            if not name.startswith("warp"):continue
            filename=getattr(module,"__file__",None)
            if not filename or not filename.endswith(".py"):continue
            filename=str(Path(filename).resolve())
            if filename not in self.runtime_source.files:raise BindingError("loaded Warp module lacks original source pin: "+name)
            syntax=self.runtime_source.syntax[filename]
            external={}
            for node in syntax.body:
                if isinstance(node,ast.ImportFrom) and node.module and not node.module.startswith("warp") and not node.level:
                    for alias in node.names:external[alias.asname or alias.name]=(node.module,alias.name)
            for field,value in tuple(vars(module).items()):
                if isinstance(value,types.FunctionType):
                    allowed=False
                    if field in external:
                        target,original=external[field]
                        allowed=getattr(self._import(target,fromlist=[original]),original,None) is value
                    function(value,name+"."+field,allowed)
                elif isinstance(value,type) and value.__module__==name:
                    for field,method in tuple(vars(value).items()):
                        if isinstance(method,(staticmethod,classmethod)):method=method.__func__
                        if isinstance(method,types.FunctionType):function(method,name+"."+value.__qualname__+"."+field)
        return tokens
    def _builtin_semantics(self):
        from warp._src.context import Function
        active=set()
        def token(value):
            if value is None or type(value) in (str,int,float,bool):return (type(value).__name__,repr(value))
            if id(value) in active:return ("retained-reference",id(value))
            active.add(id(value))
            try:
                if isinstance(value,types.FunctionType):
                    self.runtime_source.code(value.__code__)
                    return ("original-callable",id(value),id(value.__code__),fingerprint(value.__code__),tuple(token(c.cell_contents) for c in value.__closure__ or ()))
                if isinstance(value,Function):
                    fields=("key","namespace","native_func","native_snippet","adj_native_snippet","replay_snippet","input_types","defaults","variadic","value_func","value_type","initializer_list_func","overloads")
                    return ("builtin",id(value),tuple((key,token(getattr(value,key,None))) for key in fields))
                if isinstance(value,dict):return ("dict",tuple((token(k),token(v)) for k,v in value.items()))
                if isinstance(value,(tuple,list)):return (type(value).__name__,tuple(token(v) for v in value))
                if isinstance(value,type):
                    fields=("_type_","_wp_scalar_type_","_wp_c_type_","_wp_type_","_length_","_shape_")
                    return ("type",id(value),tuple((key,token(getattr(value,key,None))) for key in fields if hasattr(value,key)))
                return (type(value).__module__,type(value).__qualname__,id(value))
            finally:active.remove(id(value))
        return token(self.context.builtin_functions)
    def validate_runtime(self):
        if any(stable(getattr(wp.config,key))!=value for key,value in self.compiler_configuration.items()):raise BindingError("actual global compiler option differs from frozen original batch pin")
        self._validate_global_declarations()
        self._verify_original_semantics()
        now_loaded=self._validate_loaded_runtime()
        if any(now_loaded.get(key)!=token for key,token in self.loaded_runtime_tokens.items()):raise BindingError("complete loaded compiler/helper callable chain changed")
        if self._builtin_semantics()!=self.builtin_tokens:raise BindingError("original Warp builtin overload/type/codegen semantics changed")
        now={"build_cuda":self.build.build_cuda,"ModuleBuilder":self.context.ModuleBuilder,
             "ModuleHasher":self.context.ModuleHasher,"ModuleExec":self.context.ModuleExec,
             "cuda_kernel_forward_name":self.codegen.cuda_kernel_forward_name,
             "make_full_qualified_name":self.codegen.make_full_qualified_name,"pack_arg":self.context.pack_arg}
        if {name:id(value) for name,value in now.items()}!=self.api_tokens:raise BindingError("original codegen/compiler/packing API substituted")
        for name,value in now.items():
            fn=value.__init__ if isinstance(value,type) else value
            if id(fn.__code__)!=self.api_codes[name]["loaded_code_identity"] or fingerprint(fn.__code__)!=self.api_codes[name]["loaded_code_sha256"]:raise BindingError("loaded original codegen/compiler code object altered")
        if any(id(getattr(wp,name,None))!=token for name,token in self.wp_tokens.items()):raise BindingError("original Warp intrinsic/type export substituted")
        if {name:(id(fn),handle(fn)) for name,fn in self.native.items()}!=self.native_tokens:raise BindingError("actual CUDA artifact load/symbol API substituted")
        native=self.context.runtime.core.wp_cuda_compile_program
        if (id(self.context.runtime),id(self.context.runtime.core),id(native),handle(native),str(self.context.runtime.core._name))!=self.compiler_native_token:raise BindingError("actual native compiler/runtime library API substituted")
    def _find(self,name):
        suffix="/"+name.replace(".","/")
        matches=[key for key in self.sources.files if key.endswith(suffix+".py") or key.endswith(suffix+"/__init__.py")]
        if len(matches)>1:raise BindingError("ambiguous frozen import: "+name)
        return matches[0] if matches else None
    def _import_original(self,name,globals_=None,locals_=None,fromlist=(),level=0):
        if level:
            package=(globals_ or {}).get("__original_package__","")
            name=importlib.util.resolve_name("."*level+name,package)
        path=self._find(name)
        if path:
            module=self.clone(path,name)
            if fromlist:
                for item in fromlist:
                    if item!="*" and not hasattr(module,item):
                        child=self._find(name+"."+item)
                        if child:setattr(module,item,self.clone(child,name+"."+item))
                return module
            # Dotted imports need the exact package chain, not a foreign original.
            top=name.split(".")[0];base=self._find(top)
            if not base:raise BindingError("frozen dotted import lacks package root")
            root=self.clone(base,top);cursor=root;parts=name.split(".")
            for i,part in enumerate(parts[1:],1):
                parent_name=".".join(parts[:i+1]);parent_path=self._find(parent_name)
                if not parent_path:raise BindingError("frozen dotted import lacks package: "+parent_name)
                child=self.clone(parent_path,parent_name);setattr(cursor,part,child);cursor=child
            return root
        if name.startswith("libero"):raise BindingError("original custom import not in frozen source set: "+name)
        return self._import(name,globals_,locals_,fromlist,0)
    def clone(self,path,original_name):
        path=str(Path(path).resolve())
        if path not in self.sources.files:raise BindingError("original module has no byte pin")
        if path in self.modules:return self.modules[path]
        row=self.sources.files[path];raw=Path(path).read_bytes()
        if digest(raw)!=row["sha256"]:raise BindingError("original clone bytes differ")
        private_name=self.prefix+"_"+str(len(self.modules))
        module=ModuleType(private_name);module.__file__=path
        package=original_name if path.endswith("/__init__.py") else original_name.rpartition(".")[0]
        module.__package__=package;module.__original_package__=package
        if path.endswith("/__init__.py"):module.__path__=[str(Path(path).parent)]
        module.__builtins__=dict(vars(builtins),__import__=self._import_original)
        sys.modules[private_name]=module;self.modules[path]=module;self._module_names[path]=original_name
        exec(compile(raw,path,"exec",dont_inherit=True),module.__dict__)
        return module
    def original(self,kernel):
        fn,row,node=self.sources.syntax_function(kernel)
        if "<locals>" in row["qualname"]:raise BindingError("captured original kernel factories need explicit value binding")
        module=self.clone(row["path"],fn.__module__);fresh=module
        for part in row["qualname"].split("."):fresh=getattr(fresh,part)
        if getattr(fresh,"func",None) is None:raise BindingError("original declaration did not recreate a Warp entry")
        return fresh,row
    def bind(self,registry,command):
        self.sources.validate();self.validate_runtime()
        fresh,row=self.original(command.kernel)
        expected_key=re.sub("[^0-9a-zA-Z_]+","",row["qualname"].replace(".","__"))
        if command.kernel.key!=expected_key:raise BindingError("original canonical Warp kernel key differs")
        if stable(command.kernel.options)!=stable(fresh.options):raise BindingError("original kernel compile options differ")
        actual_types=tuple((a.label,typed(a.type)) for a in command.kernel.adj.args)
        expected_types=tuple((a.label,typed(a.type)) for a in fresh.adj.args)
        if actual_types!=expected_types:raise BindingError("actual formal dtype/ndim/struct layout differs from original source annotation")
        original_options=command.kernel.module.resolve_options(wp.config,block_dim=command.block_dim)
        fresh_options=fresh.module.resolve_options(wp.config,block_dim=command.block_dim)
        if stable(original_options)!=stable(fresh_options):raise BindingError("original module compiler/codegen options differ")
        if fresh_options["extra_cuda_include_dirs"] or fresh_options["extra_cuda_preamble"] or fresh_options["extra_build_dependencies"]:raise BindingError("extra compiler inputs require separate source pins")
        # Query the untrusted input module directly: cached hooks are never proof.
        symbol=self.codegen.cuda_kernel_forward_name(command.kernel)
        function=ct.c_void_p()
        with command.device.context_guard:
            rc=self.native["cuModuleGetFunction"](ct.byref(function),command.module_exec.handle,symbol.encode())
        if rc or handle(command.hooks.forward)!=handle(function):raise BindingError("input command function differs from independently queried original symbol")
        key=(id(fresh.module),handle(command.device.context),command.block_dim)
        if key not in self.capsules:self.capsules[key]=WarpModuleCapsule(self,fresh.module,command.device,fresh_options)
        capsule=self.capsules[key]
        return CompiledWarpCommand(self,capsule,command,fresh,row,actual_types)

class WarpModuleCapsule:
    def __init__(self,session,module,device,options):
        self.session=session;self.module=module;self.device=device;self.options=copy.deepcopy(options)
        self.context=handle(device.context);self.folder=session.output/("module-"+str(len(session.capsules)))
        self.folder.mkdir(exist_ok=False)
        # Fresh original objects avoid cached mutable Adjoint lowering/hashers.
        session._validate_runtime_files();session.validate_runtime()
        hasher=session.context.ModuleHasher(module._get_live_kernels(),options)
        builder=session.context.ModuleBuilder(module,options,hasher=hasher)
        self.source=builder.codegen("cuda");self.meta=builder.build_meta();ltoirs,fatbins=builder.get_link_inputs()
        self.roots=tuple(module._get_live_kernels())
        self.original_roots=[session.sources.warp_function(k) for k in self.roots]
        self.original_helpers=[session.sources.warp_function(f) for f in builder.functions]
        self.struct_types=[typed(s) for s in builder.structs]
        self.arch=module._get_compile_arch(device);suffix=device._get_cuda_arch_suffix(self.arch)
        self.source_path=self.folder/"original.cu";self.source_path.write_text(self.source)
        extension=".ptx" if module._use_ptx(device) else ".cubin";self.binary_path=self.folder/("original"+extension)
        optimization=options["optimization_level"] if options["optimization_level"] is not None else 3
        self.flags=dict(pch_dir=session.context.runtime.get_nvrtc_pch_dir(),config=options["mode"],optimization_level=optimization,
                        verify_fp=options["verify_fp"],fast_math=options["fast_math"],fuse_fp=options["fuse_fp"],
                        lineinfo=options["lineinfo"],compile_time_trace=options["compile_time_trace"],
                        arch_suffix=suffix,llvm_cuda=options["llvm_cuda"],use_precompiled_headers=options["use_precompiled_headers"],
                        extra_include_dirs=options["extra_cuda_include_dirs"])
        # Fresh link inputs come from source reconstruction and are retained.
        self.links={"ltoirs":[{"bytes":len(v),"sha256":digest(v)} for v in ltoirs],"fatbins":[{"bytes":len(v),"sha256":digest(v)} for v in fatbins]}
        session.build.build_cuda(str(self.source_path),self.arch,str(self.binary_path),ltoirs=ltoirs,fatbins=fatbins,**self.flags)
        self.binary=self.binary_path.read_bytes();self.buffer=ct.create_string_buffer(self.binary)
        self.source_sha256=digest(self.source_path.read_bytes());self.binary_sha256=digest(self.binary)
        current=ct.c_void_p()
        with device.context_guard:
            if session.native["cuCtxGetCurrent"](ct.byref(current)) or handle(current)!=self.context:raise BindingError("actual artifact load context differs")
            self.handle=ct.c_void_p()
            if session.native["cuModuleLoadData"](ct.byref(self.handle),ct.cast(self.buffer,ct.c_void_p)):raise BindingError("actual original artifact byte load failed")
        self.module_hash=module.get_module_hash(options["block_dim"])
        self.module_exec=session.context.ModuleExec(handle(self.handle),self.module_hash,device,self.meta,options["block_dim"],self.arch)
        self.load_record={"api":"cuModuleLoadData","jit_options":"original default; PTX driver JIT when selected production output uses PTX",
                          "context":self.context,"module":handle(self.handle),"artifact_path":str(self.binary_path),
                          "artifact_sha256":self.binary_sha256,"artifact_bytes":len(self.binary),
                          "actual_loaded_buffer_sha256":digest(ct.string_at(self.buffer,len(self.binary))),
                          "generated_source_path":str(self.source_path),"generated_source_sha256":self.source_sha256,
                          "compile_arch":self.arch,"compile_flags":stable(self.flags),"resolved_options":stable(options),
                          "link_inputs":self.links,"original_roots":self.original_roots,"original_helpers":self.original_helpers,
                          "original_struct_layouts":self.struct_types,
                           "independent_original_declaration_baseline":{
                               "request_sha256":session.current_original_baseline["request_sha256"],
                               "declarations_sha256":digest(json.dumps(session.current_original_baseline["declarations"],sort_keys=True).encode()),
                               "identity":session.current_original_baseline["identity"],"artifacts":session.current_original_baseline["baseline_artifacts"],"physics_evaluated":False}}
    def symbol(self,kernel):
        name=self.session.codegen.cuda_kernel_forward_name(kernel);value=ct.c_void_p()
        with self.device.context_guard:
            rc=self.session.native["cuModuleGetFunction"](ct.byref(value),self.handle,name.encode())
        if rc or not value.value:raise BindingError("actual sealed source module symbol query failed")
        return name,handle(value)
    def validate(self):
        self.session.validate_runtime()
        if handle(self.device.context)!=self.context or handle(self.module_exec.handle)!=handle(self.handle):raise BindingError("retained original compile module/context changed")
        if digest(self.source_path.read_bytes())!=self.source_sha256 or self.binary_path.read_bytes()!=self.binary or ct.string_at(self.buffer,len(self.binary))!=self.binary:raise BindingError("retained generated source/artifact/load buffer changed")
        if stable(self.module.resolve_options(wp.config,block_dim=self.options["block_dim"]))!=stable(self.options):raise BindingError("original compiler configuration changed")
        return True

class CompiledWarpCommand:
    def __init__(self,session,capsule,command,fresh,source,types_):
        self.session=session;self.capsule=capsule;self.command=command;self.fresh=fresh;self.source=source;self.types=types_
        self.original_module=command.module_exec;self.original_hooks=command.hooks
        self.original_function=handle(command.hooks.forward)
        self.symbol,self.function=capsule.symbol(fresh)
        # Prepare a private execution hook. Do not change the original kernel,
        # arguments, numeric options or retained initial module.
        self.hooks=copy.copy(command.hooks);self.hooks.forward=self.function;self.hooks.backward=None
        if capsule.meta[self.symbol+"_smem_bytes"]!=self.hooks.forward_smem_bytes:raise BindingError("actual original compiled shared-memory metadata differs")
        command.hooks=self.hooks;command.module_exec=capsule.module_exec
        self.kernel=command.kernel;self.hook_meta=(self.hooks.forward_smem_bytes,self.hooks.cluster_dim)
        self.parameter_source=self._parameter_source()
        from .graph import _Driver
        self.actual_compiled_abi=_Driver().parameter_sizes(self.function,command.params)
        self.validate()
    def _parameter_source(self):
        from warp._src.codegen import Struct
        from .graph import _raw
        def convert(formal,original,value):
            if isinstance(formal,Struct):
                if getattr(value,"_cls",None) is not original:raise BindingError("actual struct instance differs from original formal owner type")
                rebuilt=formal()
                for name,var in formal.vars.items():setattr(rebuilt,name,convert(var.type,original.vars[name].type,getattr(value,name)))
                return rebuilt
            return value
        if len(self.command.fwd_args)!=len(self.fresh.adj.args):raise BindingError("original actual/formal argument count differs")
        original=[]
        for index,(actualarg,sourcearg,value) in enumerate(zip(self.command.kernel.adj.args,self.fresh.adj.args,self.command.fwd_args),1):
            regenerated=self.session.context.pack_arg(self.fresh,sourcearg.type,sourcearg.label,convert(sourcearg.type,actualarg.type,value),self.command.device,False)
            if _raw(regenerated)!=_raw(self.command.params[index]):raise BindingError("actual packed parameter differs from authentic original semantic type: "+sourcearg.label)
            original.append({"argument":sourcearg.label,"semantic_type":typed(sourcearg.type),"bytes":_raw(regenerated).hex(),"native_size":ct.sizeof(regenerated)})
        return original
    def validate(self):
        c=self.command;self.capsule.validate()
        if c.kernel is not self.kernel or c.hooks is not self.hooks or c.module_exec is not self.capsule.module_exec:raise BindingError("actual sealed original command kernel/hook/module substituted")
        if handle(c.hooks.forward)!=self.function or (c.hooks.forward_smem_bytes,c.hooks.cluster_dim)!=self.hook_meta:raise BindingError("actual sealed original CUDA function/options substituted")
        if self.capsule.symbol(self.fresh)!=(self.symbol,self.function):raise BindingError("actual artifact/module/symbol/function chain changed")
        if tuple((a.label,typed(a.type)) for a in c.kernel.adj.args)!=self.types:raise BindingError("actual original semantic argument type changed")
        if self._parameter_source()!=self.parameter_source:raise BindingError("actual original packed source/parameter ABI changed")
        return True
    def to_json(self):
        self.validate()
        return {"schema":"actual-original-source-compiled-warp-command-v1","source":self.source,
                "semantic_types":self.types,"actual_symbol":self.symbol,"actual_function":self.function,
                "actual_compiled_parameter_offsets_sizes":self.actual_compiled_abi,"actual_source_parameters":self.parameter_source,
                "input_module_retained":handle(self.original_module.handle),"input_function_retained":self.original_function,
                "actual_load_compile_chain":self.capsule.load_record,
                "scope":"fresh frozen original Python module/types/helpers/options to generated source to real compiler artifact to actual byte-loaded CUDA module/symbol/function; initial cached executable not asserted as original-image provenance"}
