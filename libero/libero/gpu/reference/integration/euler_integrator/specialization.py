"""CPU structural validation for newly scoped Euler specializations.

This never admits a model under either old fixed-layout licence. Structural
validation precedes allocation; independent per-model numerical acceptance
is required separately by the runtime capability registry.
"""
import hashlib
import json
import numpy as np

INT_ARRAYS=('dof_Madr','dof_parentid','dof_simplenum','jnt_type','jnt_qposadr','jnt_dofadr')

def _integers(model,name,length):
    raw=np.asarray(getattr(model,name))
    if raw.shape!=(length,) or raw.dtype.kind not in 'iu':
        raise TypeError('Euler model integer schema: '+name)
    if np.any(raw<np.iinfo(np.int32).min) or np.any(raw>np.iinfo(np.int32).max):
        raise ValueError('Euler model integer overflow: '+name)
    return np.ascontiguousarray(raw,dtype=np.int32)

def validate_specialization(model):
    dimensions={n:int(getattr(model,n)) for n in ('nq','nv','nM','njnt','nu','na','nplugin','nmocap','nsensordata')}
    if any(v<0 for v in dimensions.values()) or any(dimensions[n]==0 for n in ('nq','nv','nM','njnt')):
        raise ValueError('Euler dimensions require nonempty joints and valid counts')
    if dimensions['na'] or dimensions['nplugin'] or int(model.opt.integrator)!=0:
        raise ValueError('generic Euler currently requires Euler/na=0/nplugin=0')
    if int(model.opt.disableflags)!=0 or int(model.opt.enableflags)!=0:
        raise ValueError('Euler option flags need separate qualification')
    dt=float(model.opt.timestep)
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('Euler timestep must be finite and positive')
    nv,nm,nq,nj=(dimensions[n] for n in ('nv','nM','nq','njnt'))
    arrays={n:_integers(model,n,nj if n.startswith('jnt_') else nv) for n in INT_ARRAYS}
    damping=np.asarray(model.dof_damping)
    if damping.shape!=(nv,) or damping.dtype!=np.float64 or not np.isfinite(damping).all():
        raise TypeError('Euler damping requires finite float64[nv]')
    arrays['dof_damping']=np.ascontiguousarray(damping)
    parent,madr,simple=(arrays[n] for n in ('dof_parentid','dof_Madr','dof_simplenum'))
    address=0
    for i in range(nv):
        if parent[i]<-1 or parent[i]>=i:
            raise ValueError('Euler mass parent must precede child: '+str(i))
        if simple[i]<0 or simple[i]>nv-i:
            raise ValueError('Euler simple block out of bounds: '+str(i))
        # Packed MuJoCo qM rows store diagonal then every ordered ancestor.
        length=1;j=int(parent[i])
        while j>=0:
            length+=1;j=int(parent[j])
        if madr[i]!=address:
            raise ValueError('Euler packed mass row mismatch: '+str(i))
        address+=length
    if address!=nm:
        raise ValueError('Euler packed mass final size mismatch')
    pa=va=0
    for i in range(nj):
        kind=int(arrays['jnt_type'][i])
        if kind not in (0,1,2,3):raise ValueError('Euler unknown joint type')
        if arrays['jnt_qposadr'][i]!=pa or arrays['jnt_dofadr'][i]!=va:
            raise ValueError('Euler joint spans must uniquely cover ordered state')
        pa+=7 if kind==0 else 4 if kind==1 else 1
        va+=6 if kind==0 else 3 if kind==1 else 1
    if pa!=nq or va!=nv:raise ValueError('Euler joint spans do not cover state')
    digests={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in arrays.items()}
    metadata={'schema':'generic-euler-layout-v1','dimensions':dimensions,
              'static_digests':digests,'dt_hex':dt.hex(),'integrator':0,
              'disableflags':0,'enableflags':0}
    metadata['specialization_key']=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return metadata,arrays
