"""Hash every static field consumed by N01, N03, and N06."""
import hashlib,json
import numpy as np

FIELDS=('nbody','njnt','nq','nv','nM','nmocap','body_mass','body_subtreemass',
        'body_inertia','body_parentid','body_rootid','jnt_type','jnt_bodyid',
        'jnt_dofadr','dof_bodyid','dof_parentid','dof_Madr','dof_simplenum',
        'dof_M0','dof_armature')

def digest(model):
    h=hashlib.sha256()
    for name in FIELDS:
        a=np.asarray(getattr(model,name))
        if name in ('nbody','njnt','nq','nv','nM','nmocap'):
            a=np.asarray(int(a),dtype=np.int32)
        if a.dtype.kind not in 'iuf':raise TypeError('static model dtype: '+name)
        if not a.flags.c_contiguous:a=np.ascontiguousarray(a)
        h.update(json.dumps([name,list(a.shape),a.dtype.str],separators=(',',':')).encode())
        h.update(b'\0');h.update(a.tobytes())
    return h.hexdigest()
