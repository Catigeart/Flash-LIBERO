"""Authenticated conservative source effects bound to concrete typed arguments.

This proves permitted source fields, not an exact dynamic predicate, epoch,
compiled binary identity, capacity certificate or a complete host call forest.
Every admitted Graph Warp command needs this binding before capture.
"""
from .authenticated_source import FrozenSources
from .binding import BindingError, lookup, array_token

def source_writer(sources,kernel):
    row=sources.warp_function(kernel)['source']
    return 'source:'+row['sha256']+':'+row['qualname']+':'+str(row['original_first_line'])

class SourceEffectWitness:
    def __init__(self,registry,command,sources):
        from .graph import _walk_warp
        if registry._sealed:raise BindingError('bind authentic source effects before sealing registry')
        if not isinstance(sources,FrozenSources):raise BindingError('independently frozen complete Python source set required')
        self.registry=registry;self.command=command;self.sources=sources
        self.writer=source_writer(sources,command.kernel)
        self.lowering=sources.warp_function(command.kernel)
        try:self.census=sources.transitive_access(command.kernel,require_warp_lowering=True)
        except ValueError as error:raise BindingError('source field effects incomplete: '+str(error)) from error
        self.formal_types=tuple((arg.label,id(arg.type),repr(arg.type)) for arg in command.kernel.adj.args)
        fields=[]
        for arg,value in zip(command.kernel.adj.args,command.fwd_args):fields+=_walk_warp(registry,value,arg.label)
        paths={}
        for field in fields:
            key=field['argument']
            if key in paths:raise BindingError('ambiguous actual formal field binding')
            paths[key]=field['binding']
        effects={key:set() for key in paths}
        self.source_effects=[]
        for effect in self.census['effects']:
            key=effect['root_argument']
            if key not in paths:
                # Known scalar/vector local indexing is not a device allocation.
                # A root/struct reference without an actual array is unsupported.
                head=key.split('.')[0]
                actual=next((value for arg,value in zip(command.kernel.adj.args,command.fwd_args) if arg.label==head),None)
                if any(f['argument']==key or f['argument'].startswith(key+'.') for f in fields):raise BindingError('source effect is not an exact actual array field: '+key)
                if getattr(actual,'_cls',None) is not None or hasattr(actual,'ptr'):raise BindingError('source indexing has unresolved actual device field: '+key)
                continue
            mode=effect['context']
            if mode not in ('Load','Store','ReadWrite'):raise BindingError('unsupported original source effect context')
            if mode in ('Load','ReadWrite'):effects[key].add('read')
            if mode in ('Store','ReadWrite'):effects[key].add('write')
            self.source_effects.append(dict(effect,actual_binding=paths[key]))
        self.fields=tuple((key,paths[key],tuple(sorted(effects[key]))) for key in sorted(paths))
        self.tokens={name:array_token(lookup(registry.roots[registry.bindings[name].root],registry.bindings[name].path)) for _,name,_ in self.fields}
        self.validate(registry)
    def validate(self,registry):
        from .graph import _walk_warp
        if registry is not self.registry:raise BindingError('source effects belong to another registry')
        self.sources.validate()
        try:
            if source_writer(self.sources,self.command.kernel)!=self.writer or self.sources.warp_function(self.command.kernel)!=self.lowering or self.sources.transitive_access(self.command.kernel,require_warp_lowering=True)!=self.census:raise BindingError('authentic source effects/lowering/helper graph changed')
        except ValueError as error:raise BindingError(str(error)) from error
        if self.formal_types!=tuple((arg.label,id(arg.type),repr(arg.type)) for arg in self.command.kernel.adj.args):raise BindingError('actual authenticated typed formal ABI changed')
        fields=[]
        for arg,value in zip(self.command.kernel.adj.args,self.command.fwd_args):fields+=_walk_warp(registry,value,arg.label)
        paths={row['argument']:row['binding'] for row in fields}
        if len(paths)!=len(fields) or paths!={key:name for key,name,_ in self.fields}:raise BindingError('actual source effect argument path changed')
        for path,name,modes in self.fields:
            bound=registry.bindings[name];actual=lookup(registry.roots[bound.root],bound.path)
            if array_token(actual)!=self.tokens[name]:raise BindingError('source effect actual buffer/owner changed')
            for mode in modes:
                if bound.descriptor.access not in (mode,'read_write'):raise BindingError('authentic kernel source '+mode+' forbidden by actual descriptor: '+path+' -> '+name)
            if 'write' in modes and self.writer not in bound.descriptor.writers:raise BindingError('authentic kernel source writer is not owned by actual destination: '+path+' -> '+name)
            if 'write' in modes and bound.alias_of is not None:
                canonical=registry.bindings[bound.alias_of]
                if canonical.descriptor.access=='read' or self.writer not in canonical.descriptor.writers:raise BindingError('actual source alias cannot elevate canonical owner write permission: '+path)

            if modes and bound.descriptor.logical_elements==0:raise BindingError('unread zero logical sentinel has reachable source effect: '+path)
        return True
    def metadata(self):
        import hashlib,json
        return {'writer':self.writer,'source':self.lowering,'fields':[{'argument':key,'binding':name,'modes':modes} for key,name,modes in self.fields],'census_sha256':hashlib.sha256(json.dumps(self.census,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'scope':'symbolic field authorization; original compiled ABI/dynamic-domain/fullcaller evidence separately required'}
    def to_json(self):
        return {'schema':'authentic-source-effect-typed-binding-v1','writer':self.writer,'original_source':self.lowering,'fields':[{'argument':key,'binding':name,'modes':modes} for key,name,modes in self.fields],'source_effects':self.source_effects,'source_census':self.census,'scope':'conservative transitive source field authorization only; guards not dynamic-domain or epoch proof; compiled binary/fullcaller evidence separately required'}
