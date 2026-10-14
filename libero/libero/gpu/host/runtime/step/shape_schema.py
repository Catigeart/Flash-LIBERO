"""Cold native dimensions and retained typed views; no physics or ready grant."""
from dataclasses import dataclass
from math import prod
from types import MappingProxyType


@dataclass(frozen=True)
class RuntimeShape:
    model_sha256: str
    nq: int
    nv: int
    nM: int
    njnt: int
    nu: int
    nbody: int
    ngeom: int
    nsite: int
    nsensordata: int

    @classmethod
    def from_native_metadata(cls, metadata):
        digest=metadata['model_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('actual native model SHA required')
        dimensions=metadata['dimensions']
        for name in ('na','nmocap','neq','ntendon','nplugin'):
            if type(dimensions[name]) is not int or dimensions[name]!=0:
                raise ValueError('separate state/component support required: '+name)
        names=tuple(cls.__dataclass_fields__)[1:]
        values={name:dimensions[name] for name in names}
        for name,value in values.items():
            minimum=0 if name in ('nu','nsite','nsensordata') else 1
            if type(value) is not int or not minimum<=value<=2147483647:
                raise ValueError('positive native int32 dimension required: '+name)
        if values['nM'] < values['nv']:
            raise ValueError('native mass storage must cover every velocity coordinate')
        return cls(digest,**values)

    def state_shapes(self, nworld):
        if isinstance(nworld,bool) or not isinstance(nworld,int) or nworld<=0:
            raise ValueError('positive world capacity required')
        return dict(qpos=(nworld,self.nq),qvel=(nworld,self.nv),time=(nworld,),
            ctrl=(nworld,self.nu),qacc_warmstart=(nworld,self.nv),
            qfrc_applied=(nworld,self.nv),xfrc_applied=(nworld,self.nbody,6))

    def solver_shapes(self,nworld,rows,contacts):
        self.state_shapes(nworld)
        if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (rows,contacts)):
            raise ValueError('positive unsliced row/contact capacities required')
        return dict(J=(nworld,rows,self.nv),qM=(nworld,self.nM),
            qacc=(nworld,self.nv),contact_dim=(nworld,contacts),
            contact_friction=(nworld,contacts,5),efc_force=(nworld,rows),
            cfrc_int=(nworld,self.nbody,6),sensor=(nworld,self.nsensordata))


def array_token(array):
    return (id(array),int(array.ptr),array.dtype,tuple(array.shape),
            tuple(array.strides),str(array.device),int(array.capacity))


class RetainedShapeViews:
    """Prepare contiguous exact-span aliases once; validate actual parents later.

    Provider functions are cold fixed references. The enclosing producer must
    additionally enforce its original source/clock/stream/allocation guards.
    This helper grants only layout compatibility.
    """
    def __init__(self,providers,schema,*,device):
        if set(providers)!=set(schema):raise ValueError('typed provider keyset differs')
        self.providers=MappingProxyType(dict(providers))
        self.schema=MappingProxyType({k:(tuple(shape),dtype) for k,(shape,dtype) in schema.items()})
        self.device=device
        self.parents={};self.flats={};self.views={};self.tokens={}
        for name,provider in self.providers.items():
            if not callable(provider):raise TypeError('actual parent provider required')
            source=provider();shape,dtype=self.schema[name]
            if (source.dtype!=dtype or source.device!=device or
                    not source.is_contiguous or source.size!=prod(shape)):
                raise ValueError('actual typed shape/span differs: '+name)
            flat=source.flatten();view=flat.reshape(shape)
            if int(view.ptr)!=int(source.ptr):raise ValueError('reshape must be true alias')
            self.parents[name]=source;self.flats[name]=flat;self.views[name]=view
            self.tokens[name]=tuple(array_token(a) for a in (source,flat,view))
        self._parents=dict(self.parents);self._flats=dict(self.flats);self._views=dict(self.views)
        self._providers=self.providers;self._schema=self.schema

    def validate(self):
        if self.providers is not self._providers or self.schema is not self._schema:
            raise ValueError('cold provider/schema identity changed')
        for name,provider in self.providers.items():
            current=(provider(),self.flats[name],self.views[name])
            if (self.parents[name] is not self._parents[name] or
                self.flats[name] is not self._flats[name] or self.views[name] is not self._views[name] or
                tuple(array_token(a) for a in current)!=self.tokens[name]):
                raise ValueError('actual retained parent/view changed: '+name)
        return self.views
