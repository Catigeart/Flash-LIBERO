"""Independent original-byte Warp declaration baseline; no physics evaluation.

The command is a child of the admitted scientific job. It imports pinned source
bytes in a new interpreter, never a parent module/cache, and emits semantic
declarations rather than object identities or host physical state.
"""
from pathlib import Path
import ast,ctypes as ct,hashlib,json,sys,types,typing,enum,inspect,struct

def sha(raw):return hashlib.sha256(raw).hexdigest()

def original_immutable(source):
    result={}
    for filename,tree in source.syntax.items():
        # Original functions explicitly declaring a global identify live
        # framework state, not a source constant. Dict/list registries are also
        # live state. Neither is promoted to an immutable mathematical datum.
        live={name for node in ast.walk(tree) if isinstance(node,ast.Global) for name in node.names}
        declarations={};unrepresented=object()
        for node in tree.body:
            if isinstance(node,ast.Assign):targets=node.targets;value=node.value
            elif isinstance(node,ast.AnnAssign):targets=[node.target];value=node.value
            else:continue
            if len(targets)!=1 or not isinstance(targets[0],ast.Name) or value is None:continue
            name=targets[0].id
            if name in declarations:declarations[name]=unrepresented;continue
            try:v=ast.literal_eval(value)
            except (ValueError,TypeError):continue
            def immutable(x):
                return type(x) in (str,bytes,int,float,bool,type(None)) or type(x) is tuple and all(immutable(z) for z in x)
            declarations[name]=v if immutable(v) else unrepresented
        result[filename]={name:value for name,value in declarations.items() if name not in live and value is not unrepresented}
    return result

def original_data_graph(source):
    """Module-declared compiler data and source-described runtime mutation edges.

    Tables built with classes/comprehensions/imports are data, too. Only real
    writes in original function bodies classify a container as runtime state;
    module import-time population does not exempt its final semantic contents.
    """
    result={}
    mutators={"append","extend","insert","remove","pop","popitem","clear","update",
              "setdefault","add","discard","difference_update","intersection_update",
              "symmetric_difference_update","sort","reverse"}
    def root_name(node):
        while isinstance(node,(ast.Attribute,ast.Subscript)):node=node.value
        return node.id if isinstance(node,ast.Name) else None
    for filename,tree in source.syntax.items():
        declared={};reads={};writes={}
        def top_level(nodes):
            for node in nodes:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
                if isinstance(node,(ast.Assign,ast.AnnAssign)):
                    targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                    for target in targets:
                        for child in ast.walk(target):
                            if isinstance(child,ast.Name) and isinstance(child.ctx,ast.Store):
                                declared.setdefault(child.id,[]).append(node.lineno)
                elif isinstance(node,ast.ImportFrom):
                    for alias in node.names:
                        if alias.name!="*":declared.setdefault(alias.asname or alias.name,[]).append(node.lineno)
                elif isinstance(node,ast.Import):
                    for alias in node.names:declared.setdefault(alias.asname or alias.name.split(".")[0],[]).append(node.lineno)
                for field in ("body","orelse","finalbody"):
                    children=getattr(node,field,None)
                    if isinstance(children,list):top_level(children)
                if isinstance(node,ast.Try):
                    for handler in node.handlers:top_level(handler.body)
        top_level(tree.body)
        for fn in (x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))):
            globals_={name for node in ast.walk(fn) if isinstance(node,ast.Global) for name in node.names}
            local={arg.arg for arg in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs)}
            if fn.args.vararg:local.add(fn.args.vararg.arg)
            if fn.args.kwarg:local.add(fn.args.kwarg.arg)
            local.update(node.id for node in ast.walk(fn) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id not in globals_)
            for node in ast.walk(fn):
                if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id in declared and node.id not in local:
                    reads.setdefault(node.id,set()).add((fn.name,node.lineno))
                target=None
                if isinstance(node,(ast.Subscript,ast.Attribute)) and isinstance(node.ctx,(ast.Store,ast.Del)):target=root_name(node)
                elif isinstance(node,ast.Name) and isinstance(node.ctx,(ast.Store,ast.Del)) and node.id in globals_:target=node.id
                elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in mutators:target=root_name(node.func.value)
                if target in declared and target not in local:writes.setdefault(target,set()).add((fn.name,node.lineno))
        result[filename]={name:{"declaration_lines":sorted(set(lines)),
                               "read_edges":[list(x) for x in sorted(reads.get(name,set()))],
                               "runtime_write_edges":[list(x) for x in sorted(writes.get(name,set()))]}
                          for name,lines in sorted(declared.items()) if not name.startswith("__")}
    return result

class ConfigurationPurposes:
    """Exact source-declared options plus one validated private cache purpose."""
    def __init__(self,wp,source,pins,cache_base,private_root):
        self.wp=wp;self.source=source;self.pins=dict(pins)
        self.filename=str(Path(wp.config.__file__).resolve())
        if self.filename not in source.syntax:raise ValueError("original config source pin missing")
        declarations=original_immutable(source).get(self.filename,{})
        if "kernel_cache_dir" in pins:raise ValueError("cache location must have its separate frozen private purpose")
        self.fields={}
        for key in (*sorted(pins),"kernel_cache_dir"):
            nodes=[node for node in source.syntax[self.filename].body if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id==key]
            if len(nodes)!=1 or key not in declarations:raise ValueError("frozen option lacks a unique original annotated literal declaration: "+key)
            node=nodes[0];read_edges=[]
            for filename,tree in source.syntax.items():
                for child in ast.walk(tree):
                    if isinstance(child,ast.Attribute) and child.attr==key and (isinstance(child.value,ast.Name) and child.value.id=="config" or isinstance(child.value,ast.Attribute) and child.value.attr=="config"):
                        read_edges.append({"path":filename,"line":child.lineno,"source_sha256":source.inventory[filename]["sha256"]})
            self.fields[key]={"source_path":self.filename,"source_sha256":source.inventory[self.filename]["sha256"],
                              "declaration_line":node.lineno,"annotation":ast.unparse(node.annotation),
                              "original_default":declarations[key],"read_edges":sorted(read_edges,key=lambda x:(x["path"],x["line"])),
                              "purpose":"private-kernel-cache" if key=="kernel_cache_dir" else "explicit-frozen-compiler-option"}
        root=Path(private_root);base=Path(cache_base)
        if not root.is_absolute() or not base.is_absolute() or str(root.resolve())!=str(root) or str(base.resolve())!=str(base) or not base.is_relative_to(root):
            raise ValueError("kernel cache base is outside its exact frozen private root")
        version=declarations.get("version")
        if type(version) is not str or getattr(wp.config,"version",None)!=version or "/" in version or version in (".",".."):
            raise ValueError("private cache version lacks an unchanged original literal source")
        self.base=str(base);self.expected=str(base/version);self.private_root=str(root)
        build=sys.modules.get("warp._src.build")
        build_path=str(Path(build.__file__).resolve()) if build is not None else ""
        if build_path not in source.syntax:raise ValueError("original cache-resolution source missing")
        functions=[node for node in source.syntax[build_path].body if isinstance(node,ast.FunctionDef) and node.name=="init_kernel_cache"]
        if len(functions)!=1:raise ValueError("original cache-resolution declaration ambiguous")
        self.cache_resolution={"source_path":build_path,"source_sha256":source.inventory[build_path]["sha256"],
                               "declaration_line":functions[0].lineno,"original_version":version,
                               "rule":"original init_kernel_cache explicit fresh base joined with original config.version"}
    def global_value(self,semantic,module_name,key,value,original_default):
        if module_name!="warp.config" or key not in self.fields:
            if semantic.value(value)!=semantic.value(original_default):raise ValueError("loaded original compiler template/immutable global differs: "+module_name+"."+key)
            return semantic.value(value)
        if key=="kernel_cache_dir":
            build=sys.modules["warp._src.build"]
            if type(value) is not str or value!=self.expected or getattr(build,"_resolved_kernel_cache_dir",None)!=self.expected or str(Path(value).resolve())!=value:
                raise ValueError("actual original versioned cache differs from frozen private base/resolution")
            return ["source-validated-private-cache",self.fields[key],self.cache_resolution]
        if semantic.value(value)!=semantic.value(self.pins[key]):raise ValueError("actual original compiler option differs from its explicit frozen pin: "+key)
        return semantic.value(value)
    def manifest(self,semantic):
        return {key:dict(binding,authorized_value=["private-versioned-cache-purpose"] if key=="kernel_cache_dir" else semantic.value(self.pins[key]))
                for key,binding in self.fields.items()}

class SemanticDeclarations:
    """Identity-free source, callable, native, type and overload description."""
    def __init__(self,wp,source,*,compiler_configuration,cache_base,private_root):
        self.wp=wp;self.source=source;self.active=set()
        self.configuration=ConfigurationPurposes(wp,source,compiler_configuration,cache_base,private_root)
        self.original_globals=original_immutable(source);self.data_graph=original_data_graph(source)
        from warp._src.context import Function
        self.Function=Function
        self.core=wp._src.context.runtime.core
        self.core_probe=ct.CDLL(self.core._name)
    def original_sync_object(self,value):
        import _thread,threading
        if type(value) is not _thread.RLock:return None
        module=sys.modules.get("warp._src.codegen")
        if module is None:return None
        if getattr(module,"_codegen_lock",None) is not value:
            # Original synchronized() also creates private locks. Their
            # provenance is the pinned factory and actual wrapped method,
            # never an address or held state shared with another process.
            self.source.code(threading.RLock.__code__)
            self.source.code(module.synchronized.__code__)
            for name,owner in tuple(sys.modules.items()):
                if not name.startswith("warp") or owner is None:continue
                for member in tuple(vars(owner).values()):
                    methods=vars(member).items() if isinstance(member,type) and member.__module__==name else ((getattr(member,"__name__",""),member),)
                    for field,fn in methods:
                        fn=getattr(fn,"__func__",fn)
                        if not isinstance(fn,types.FunctionType):continue
                        if fn.__code__.co_name!="locked_call" or not any(c.cell_contents is value for c in fn.__closure__ or ()):continue
                        wrapped=getattr(fn,"__wrapped__",None)
                        if not isinstance(wrapped,types.FunctionType):continue
                        if fn.__code__.co_filename!=module.synchronized.__code__.co_filename:continue
                        self.source.code(fn.__code__);self.source.code(wrapped.__code__)
                        path=str(Path(wrapped.__code__.co_filename).resolve())
                        # Decorators start co_firstlineno; match by name and
                        # pinned exact wrapper code when there are many lines.
                        nodes=[n for n in ast.walk(self.source.syntax[path]) if isinstance(n,ast.FunctionDef) and n.name==wrapped.__name__ and any(isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=="synchronized" and not d.args and not d.keywords for d in n.decorator_list)]
                        if len(nodes)!=1 or getattr(owner,"synchronized",None) is not module.synchronized:continue
                        return ["original-builtin-private-sync","_thread.RLock",self.source.files[path],wrapped.__qualname__]
            return None
        filename=str(Path(module.__file__).resolve())
        if filename not in self.source.syntax:return None
        declarations=[x for x in self.source.syntax[filename].body if isinstance(x,ast.Assign)
            and len(x.targets)==1 and isinstance(x.targets[0],ast.Name) and x.targets[0].id=="_codegen_lock"]
        if len(declarations)!=1:return None
        call=declarations[0].value
        if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and isinstance(call.func.value,ast.Name) and call.func.value.id=="threading"
                and call.func.attr=="RLock" and not call.args and not call.keywords):return None
        if getattr(module,"threading",None) is not threading:return None
        self.source.code(threading.RLock.__code__)
        self.source.code(module.synchronized.__code__)
        return ["original-builtin-sync","_thread.RLock",self.source.files[filename],"_codegen_lock"]

    def value(self,v):
        if isinstance(v,types.BuiltinFunctionType):
            import builtins,_thread
            if v is tuple.__new__:return ["trusted-builtin-callable","builtins.tuple.__new__"]
            for module in (builtins,_thread):
                name=getattr(v,"__name__",None)
                if name and getattr(module,name,None) is v:
                    return ["trusted-builtin-callable",module.__name__,name]
            raise ValueError("unbound builtin helper callable")
        sync=self.original_sync_object(v)
        if sync is not None:return sync
        dataclass_module=sys.modules.get("dataclasses")
        if dataclass_module is not None:
            for name in ("MISSING","_HAS_DEFAULT_FACTORY","_FIELD","_FIELD_CLASSVAR","_FIELD_INITVAR"):
                if v is getattr(dataclass_module,name,None):return ["dataclass-factory-sentinel",name]
        if v is None or type(v) in (str,int,bool):return [type(v).__name__,v]
        if type(v) is float:return ["float-bits",struct.pack("!d",v).hex()]
        if type(v) is bytes:return ["bytes",v.hex()]
        if id(v) in self.active:
            if isinstance(v,self.Function):return ["recursive-function",v.key,v.namespace]
            if isinstance(v,types.FunctionType):return ["recursive-callable",sha(v.__code__.co_code),v.__code__.co_name]
            if isinstance(v,type):return ["recursive-type",v.__module__,v.__qualname__]
            raise ValueError("unrepresented cyclic declaration: "+type(v).__name__)
        self.active.add(id(v))
        try:
            if isinstance(v,types.FunctionType):
                from .authenticated_source import fingerprint
                filename=str(Path(v.__code__.co_filename).resolve())
                if filename in self.source.files:self.source.code(v.__code__)
                return ["callable",("<string>" if v.__code__.co_filename=="<string>" else filename),fingerprint(v.__code__),self.value(v.__defaults__),self.value(v.__kwdefaults__),[self.value(c.cell_contents) for c in v.__closure__ or ()]]
            if isinstance(v,self.Function):
                # Retain all initial registration data, not a hand selected
                # key/native_func whitelist or a mutable self-baseline.
                return ["builtin",[[key,self.value(value)] for key,value in sorted(vars(v).items())]]
            if isinstance(v,dict):return ["dict",[[self.value(k),self.value(x)] for k,x in v.items()]]
            if isinstance(v,(tuple,list)):return [type(v).__name__,[self.value(x) for x in v]]
            if isinstance(v,(set,frozenset)):return [type(v).__name__,sorted([self.value(x) for x in v],key=lambda x:json.dumps(x,sort_keys=True))]
            if isinstance(v,enum.Enum):return ["enum",type(v).__module__,type(v).__qualname__,v.name,self.value(v.value)]
            if isinstance(v,type):
                fields=set(key for key in vars(v) if key.startswith("_wp_"))
                fields.update(key for key in ("_type_","_length_","_shape_","_fields_") if hasattr(v,key))
                values=[[key,self.value(getattr(v,key))] for key in sorted(fields)]
                native=None
                if issubclass(v,(ct.Array,ct.Structure,ct.Union,ct._SimpleCData)):
                    native=[ct.sizeof(v),ct.alignment(v)]
                    if issubclass(v,(ct.Structure,ct.Union)):
                        native.append([[name,getattr(v,name).offset,ct.sizeof(native_ty)] for name,native_ty,*_ in v._fields_])
                return ["type",v.__module__,v.__qualname__,values,native]
            if isinstance(v,ct._CFuncPtr):
                name=getattr(v,"__name__",None)
                if not name:raise ValueError("native declaration lacks original symbol")
                # A name alone is not provenance: query original library.
                queried=getattr(self.core_probe,name)
                if ct.cast(v,ct.c_void_p).value!=ct.cast(queried,ct.c_void_p).value:
                    raise ValueError("actual builtin native symbol differs from original library: "+name)
                return ["native-symbol",str(Path(self.core._name).resolve()),name]
            if isinstance(v,self.wp.array) or type(v).__module__.startswith("warp") and hasattr(v,"dtype") and hasattr(v,"ndim"):
                return ["array-type",type(v).__module__,type(v).__qualname__,self.value(v.dtype),int(v.ndim)]
            context_module=sys.modules["warp._src.context"]
            if type(v) is context_module.Module:
                self.source.code(context_module.Module.__init__.__code__)
                # Module.name is a compiler namespace, including original
                # synthetic namespaces; it need not name a Python module.
                # Exact namespace is compared with original-byte registration.
                path=str(Path(context_module.__file__).resolve())
                if type(v.name) is not str:raise ValueError("invalid registered compiler namespace")
                return ["original-registration-module",v.name,self.source.files[path],"mutable-compile-cache-state-not-certified"]
            codegen_module=sys.modules["warp._src.codegen"]
            if type(v) is codegen_module.Adjoint:
                # Registration-source identity is distinct from a compiler's
                # mutable AST/IR. Actual selected source/formal/struct/loaded
                # artifact/effect are authenticated by the compile session.
                self.source.code(codegen_module.Adjoint.__init__.__code__)
                if not isinstance(v.func,types.FunctionType):raise ValueError("unbound registered adjoint source callable")
                self.source.code(v.func.__code__)
                return ["original-registration-source-adjoint",self.value(v.func),self.value(v.source),"mutable-AST-IR-not-certified"]
            scalar_module=sys.modules["warp._src.types"]
            if isinstance(v,scalar_module.scalar_base) and getattr(scalar_module,type(v).__name__,None) is type(v):
                self.source.code(scalar_module.scalar_base.__init__.__code__)
                return ["original-scalar-registration",self.value(type(v)),self.value(v.value)]
            if type(v) is scalar_module.Reference:
                self.source.code(scalar_module.Reference.__init__.__code__)
                return ["original-reference-annotation",self.value(v.value_type)]
            if type(v) in (self.wp.tile,self.wp.tile_stack):
                # Source-owned tile annotations occur in original builtin
                # registration. Compare their complete small ABI descriptor;
                # this does not evaluate a tile or admit arbitrary objects.
                self.source.code(type(v).__init__.__code__)
                return ["original-tile-annotation",self.value(type(v)),[[k,self.value(x)] for k,x in sorted(vars(v).items())]]
            if isinstance(v,typing.TypeVar):
                return ["TypeVar",v.__name__,self.value(v.__constraints__),self.value(v.__bound__),v.__covariant__,v.__contravariant__]
            origin=typing.get_origin(v)
            if origin is not None:return ["typing",self.value(origin),self.value(typing.get_args(v))]
            if type(v).__module__=="typing":
                return ["typing-declaration",type(v).__qualname__,str(v)]
            if type(v).__module__.startswith("numpy") and hasattr(v,"str") and hasattr(v,"descr") and hasattr(v,"itemsize"):
                return ["numpy-dtype",v.str,self.value(v.descr),int(v.itemsize),int(v.alignment),bool(v.isalignedstruct),self.value(v.metadata)]
            if isinstance(v,type(__import__("re").compile(""))):return ["regex",v.pattern,int(v.flags)]
            if isinstance(v,inspect.Signature):
                return ["signature",[[name,self.value(p.annotation),self.value(p.default),p.kind.name] for name,p in v.parameters.items()],self.value(v.return_annotation)]
            if isinstance(v,ct._SimpleCData):return ["ctypes-value",self.value(type(v)),bytes(v).hex()]
            sentinel_module=sys.modules.get("warp._src.marching_cubes")
            if sentinel_module is not None and type(v) is sentinel_module._DeprecatedArgumentDefault:
                path=str(Path(sentinel_module.__file__).resolve())
                self.source.code(type(v).__init__.__code__)
                for node in self.source.syntax[path].body:
                    if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
                    call=node.value
                    if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=="_DeprecatedArgumentDefault" and len(call.args)==1 and not call.keywords):continue
                    label=node.targets[0].id
                    if getattr(sentinel_module,label,None) is not v:continue
                    literal=ast.literal_eval(call.args[0])
                    if vars(v)!={"value":literal}:raise ValueError("source default sentinel contents differ")
                    return ["original-source-default-sentinel",self.source.files[path],label,self.value(literal)]
                raise ValueError("unbound source default sentinel")
            import functools
            if type(v) is functools.partial:
                return ["original-partial-binding",self.value(v.func),self.value(v.args),self.value(v.keywords)]
            wrapped=getattr(v,"__wrapped__",None)
            if callable(v) and isinstance(wrapped,types.FunctionType):return ["original-wrapper",type(v).__module__,type(v).__qualname__,self.value(wrapped)]
            raise ValueError("unrepresented original declaration data: "+type(v).__module__+"."+type(v).__qualname__)
        finally:self.active.remove(id(v))
    @staticmethod
    def dataclass_factory_names(cls):
        params=cls.__dataclass_params__
        names=["dataclass","_process_class","_get_field","_fields_in_init_order","_set_new_attribute","_create_fn"]
        if params.init:names.extend(("_init_fn","_init_param","_field_init","_field_assign"))
        if params.repr:names.extend(("_repr_fn","_recursive_repr"))
        if params.eq or params.order:names.append("_cmp_fn")
        if params.frozen:names.append("_frozen_get_del_attr")
        if params.unsafe_hash or params.eq and params.frozen:names.append("_hash_fn")
        return tuple(dict.fromkeys(names))

    def dataclass_state(self,cls):
        import dataclasses
        filename=str(Path(dataclasses.__file__).resolve())
        if filename not in self.source.files:raise ValueError("dataclass factory source pin missing")
        for name in self.dataclass_factory_names(cls):
            self.source.code(getattr(dataclasses,name).__code__)
        def item(value):
            for name in ("MISSING","_HAS_DEFAULT_FACTORY","_FIELD","_FIELD_CLASSVAR","_FIELD_INITVAR"):
                if value is getattr(dataclasses,name):return ["dataclass-factory-sentinel",name]
            return self.value(value)
        fields=[]
        for name,field in cls.__dataclass_fields__.items():
            fields.append([name,[[key,item(getattr(field,key))] for key in
                ("name","type","default","default_factory","init","repr","hash","compare","kw_only","_field_type")]])
        params=cls.__dataclass_params__
        return {"fields":fields,"annotations":self.value(vars(cls).get("__annotations__",{})),
                "params":[[key,self.value(getattr(params,key))] for key in
                    ("init","repr","eq","order","unsafe_hash","frozen")],
                "factory_source":self.source.files[filename],
                "participating_factory_helpers":[[name,self.value(getattr(dataclasses,name))] for name in self.dataclass_factory_names(cls)]}

    def namedtuple_state(self,cls):
        import collections
        self.source.code(collections.namedtuple.__code__)
        factories=[collections.namedtuple]
        if vars(cls).get("__annotations__"):
            factories.extend((typing._make_nmtuple,typing.NamedTupleMeta.__new__))
        for fn in factories:self.source.code(fn.__code__)
        import _collections
        getters=[]
        for name in cls._fields:
            descriptor=vars(cls)[name]
            if type(descriptor) is not _collections._tuplegetter:raise ValueError("foreign namedtuple field getter")
            reduced=descriptor.__reduce__()
            if reduced[0] is not _collections._tuplegetter or type(reduced[1][0]) is not int:raise ValueError("unbound namedtuple field getter")
            getters.append([name,reduced[1][0]])
        return {"getters":getters,"fields":self.value(cls._fields),"defaults":self.value(cls._field_defaults),
                "annotations":self.value(vars(cls).get("__annotations__",{})),
                "factories":[self.value(fn) for fn in factories]}

    def snapshot(self,module_names):
        helpers={};exports={};globals_={};compiler_data={};runtime_data={}
        original=self.original_globals
        for name in sorted(module_names):
            module=sys.modules[name]
            filename=str(Path(module.__file__).resolve())
            if filename not in self.source.files:raise ValueError("unfrozen original module: "+name)
            globals_[name]={key:self.configuration.global_value(self,name,key,getattr(module,key),want) for key,want in original.get(filename,{}).items()}
            compiler_data[name]={};runtime_data[name]={}
            for key,binding in self.data_graph.get(filename,{}).items():
                if key in original.get(filename,{}) or not hasattr(module,key):continue
                value=getattr(module,key)
                if not isinstance(value,(dict,list,tuple,set,frozenset,type(__import__("re").compile("")))):continue
                description=dict(binding,source_path=filename,source_sha256=self.source.inventory[filename]["sha256"])
                if binding["runtime_write_edges"]:
                    # This is an explicit source mutation forest, not a claim
                    # that a mutable cache's contents are a constant table.
                    runtime_data[name][key]=description
                else:compiler_data[name][key]={"binding":description,"value":self.value(value)}
            for field,value in sorted(vars(module).items()):
                if isinstance(value,types.FunctionType):
                    helpers[name+"."+field]=self.value(value)
                elif isinstance(value,type) and value.__module__==name:
                    if issubclass(value,tuple) and "_fields" in vars(value) and "_field_defaults" in vars(value):
                        helpers[name+"."+value.__qualname__+".__namedtuple_state__"]=self.namedtuple_state(value)
                    if "__dataclass_fields__" in vars(value):
                        helpers[name+"."+value.__qualname__+".__dataclass_state__"]=self.dataclass_state(value)
                    for attr,method in sorted(vars(value).items()):
                        if isinstance(method,(staticmethod,classmethod)):method=method.__func__
                        if isinstance(method,types.FunctionType):helpers[name+"."+value.__qualname__+"."+attr]=self.value(method)
                        elif isinstance(method,property):
                            for kind,fn in (("fget",method.fget),("fset",method.fset),("fdel",method.fdel)):
                                if fn is not None:helpers[name+"."+value.__qualname__+"."+attr+"."+kind]=self.value(fn)
        for field,value in sorted(vars(self.wp).items()):
            if field.startswith("__"):continue
            if isinstance(value,(type,types.FunctionType,self.Function)):exports[field]=self.value(value)
        native_api={name:[self.value(value),self.value(value.argtypes),self.value(value.restype)] for name,value in sorted(vars(self.core).items()) if isinstance(value,ct._CFuncPtr)}
        return {"helpers":helpers,"exports":exports,"global_constants":globals_,"compiler_data":compiler_data,"source_runtime_mutation_forests":runtime_data,"configuration_purposes":self.configuration.manifest(self),"native_api":native_api,"builtin_registry":self.value(self.wp._src.context.builtin_functions)}

def identity():
    import os
    lib=ct.CDLL("libm.so.6");lib.fegetround.restype=ct.c_int
    buf=ct.create_string_buffer(128);rc=lib.fegetenv(buf)
    maps=Path("/proc/self/maps").read_text();paths=sorted({line.split()[-1] for line in maps.splitlines() if line.split()[-1].startswith("/") and ".so" in line and Path(line.split()[-1]).is_file()})
    return {"pid":os.getpid(),"birth":Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],
            "boot":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"argv":sys.argv,
            "cwd":str(Path.cwd()),"affinity":sorted(os.sched_getaffinity(0)),"nice":os.getpriority(os.PRIO_PROCESS,0),
            "fenv":{"round":lib.fegetround(),"getenv_rc":rc,"storage128hex":buf.raw.hex()},
            "raw_maps":maps,"loaded_images":[{"path":p,"bytes":Path(p).stat().st_size,"sha256":sha(Path(p).read_bytes())} for p in paths],
            "environment":{k:os.environ.get(k) for k in ("HOME","PWD","LIBERO_CONFIG_PATH","XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME","XDG_STATE_HOME","XDG_RUNTIME_DIR","TMPDIR","TMP","TEMP","PYTHONDONTWRITEBYTECODE","CUDA_CACHE_PATH")}}

def main():
    import os,importlib.abc,importlib.machinery,importlib,traceback
    request=Path(sys.argv[1]);r=json.loads(request.read_bytes());out=Path(r["output"])
    assert sys.flags.isolated and sys.flags.dont_write_bytecode
    assert Path.cwd()==Path(r["cwd"])
    assert all(os.environ.get(k)==v for k,v in r["environment"].items())
    assert sha(Path(sys.executable).read_bytes())==r["python_sha256"]
    records=r["source_records"]+r["runtime_records"]+r["native_libraries"]
    modules={}
    for row in records:
        raw=Path(row["path"]).read_bytes()
        if len(raw)!=row["bytes"] or sha(raw)!=row["sha256"]:raise ValueError("baseline source pin changed: "+row["path"])
        if "snapshot_path" in row:
            snapshot=Path(row["snapshot_path"]).read_bytes()
            if len(snapshot)!=row["bytes"] or sha(snapshot)!=row["sha256"]:raise ValueError("baseline snapshot pin changed")
        if not row["path"].endswith(".py"):continue
        name=None
        if "stdlib_module" in row:
            if row["stdlib_module"]!="dataclasses":raise ValueError("unsupported original stdlib factory module")
            name=row["stdlib_module"]
        elif "runtime_relative_path" in row:name="warp."+row["runtime_relative_path"][:-3].replace("/",".")
        elif row.get("relative_path","").startswith("libero/libero/"):name="libero."+row["relative_path"][len("libero/libero/"):-3].replace("/",".")
        if name:
            package=name.endswith(".__init__")
            if package:name=name[:-9]
            modules[name]=(row,package)
    class OriginalByteLoader(importlib.abc.MetaPathFinder,importlib.abc.Loader):
        def find_spec(self,fullname,path=None,target=None):
            if fullname not in modules:return None
            row,package=modules[fullname];spec=importlib.machinery.ModuleSpec(fullname,self,origin=row["path"],is_package=package)
            if package:spec.submodule_search_locations=[str(Path(row["path"]).parent)]
            return spec
        def create_module(self,spec):return None
        def exec_module(self,module):
            row,package=modules[module.__name__];raw=Path(row.get("snapshot_path",row["path"])).read_bytes()
            assert len(raw)==row["bytes"] and sha(raw)==row["sha256"]
            module.__file__=row["path"];module.__source_sha256__=row["sha256"];module.__frozen_source_without_pyc__=True
            if package:module.__path__=[str(Path(row["path"]).parent)]
            exec(compile(raw,row["path"],"exec",dont_inherit=True),module.__dict__)
    assert not any(name.startswith("warp") for name in sys.modules)
    sys.meta_path.insert(0,OriginalByteLoader())
    # Reconstruct the factory from pinned bytes even if stdlib imports loaded
    # it earlier. No Warp/user modules exist yet in this isolated worker.
    for name in sorted(row["stdlib_module"] for row in records if "stdlib_module" in row):
        sys.modules.pop(name,None)
        original=importlib.import_module(name)
        if getattr(original,"__frozen_source_without_pyc__",False) is not True:
            raise ValueError("original stdlib factory was not executed from bytes")
    import warp as wp
    for key,value in r["compiler_configuration"].items():setattr(wp.config,key,value)
    wp.config.kernel_cache_dir=r["kernel_cache_dir"]
    wp.init()
    for name in r["module_names"]:importlib.import_module(name)
    from libero.gpu.host.runtime.foundation.authenticated_source import FrozenSources
    from libero.gpu.host.runtime.foundation.runtime_baseline import SemanticDeclarations
    source=FrozenSources(r["runtime_records"])
    result={"schema":"independent-original-warp-declaration-baseline-v1",
            "request_sha256":sha(request.read_bytes()),"source_identity":[dict(path=x["path"],sha256=x["sha256"],bytes=x["bytes"]) for x in records],
            "module_names":r["module_names"],"declarations":SemanticDeclarations(wp,source,compiler_configuration=r["compiler_configuration"],cache_base=r["kernel_cache_dir"],private_root=r["private_root"]).snapshot(r["module_names"]),
            "identity":identity(),"physics_evaluated":False,"full_F":False}
    with out.open("x") as f:json.dump(result,f,sort_keys=True,indent=2)
    print(json.dumps({"baseline":str(out),"sha256":sha(out.read_bytes()),"physics_evaluated":False}))
if __name__=="__main__":
    main()
