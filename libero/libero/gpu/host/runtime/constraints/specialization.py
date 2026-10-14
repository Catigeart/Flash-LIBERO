"""Source-derived, model-qualified dense elliptic S specialization generator.

Original accepted modules are read-only templates. A generated candidate has
new model dimensions, exact static digests, and isolated module identities.
It is pending until the shared GPU runner qualifies its actual outputs.
"""
from __future__ import annotations
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import hashlib
import json
from pathlib import Path
import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def _expr(value):
    return ast.parse(repr(value),mode='eval').body


def _target_names(node):
    if isinstance(node,ast.Name): return [node.id]
    if isinstance(node,(ast.Tuple,ast.List)):
        return sum((_target_names(x) for x in node.elts),[])
    return []


def _kernel_bodies(tree):
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
            if isinstance(n,ast.FunctionDef) and any('kernel' in ast.unparse(d) for d in n.decorator_list)}


def validate_manifest(manifest, static):
    from .identity import STATIC_SCHEMA
    facts=manifest['model_facts']; dims=manifest['dimensions']
    if facts['neq'] or facts['ntendon'] or facts['nplugin'] or facts['na'] or facts['nmocap']:
        raise ValueError('unsupported equality/tendon/plugin/state family')
    if facts['solver']!=2 or facts['cone']!=1 or facts['noslip_iterations']!=0:
        raise ValueError('only primal Newton elliptic supported')
    if not (facts['jacobian']==0 or facts['jacobian']==2 and facts['nv']<60):
        raise ValueError('sparse Jacobian family requires separate implementation')
    if facts['disableflags']!=0 or facts['enableflags']!=0:
        raise ValueError('option flags require separate source qualification')
    if not np.isfinite(facts['timestep']) or facts['timestep']<=0 or not np.isfinite(facts['impratio']) or facts['impratio']<=0:
        raise ValueError('actual native float options required')
    if set(static)!=set(STATIC_SCHEMA) or set(manifest['static'])!=set(static):
        raise ValueError('complete original S static readset required')
    for key,a in static.items():
        pin=manifest['static'][key]
        if list(a.shape)!=pin['shape'] or str(a.dtype)!=pin['dtype'] or hashlib.sha256(a.tobytes()).hexdigest()!=pin['sha256']:
            raise ValueError('model static pin mismatch: '+key)
        if a.dtype.kind=='f' and not np.isfinite(a).all(): raise ValueError('nonfinite static '+key)
    nq,nv,nj,nb,ng=(facts[k] for k in ('nq','nv','njnt','nbody','ngeom'))
    expected={'dof_frictionloss':(nv,), 'jnt_type':(nj,), 'jnt_limited':(nj,),
              'jnt_qposadr':(nj,), 'jnt_dofadr':(nj,), 'jnt_range':(nj,2), 'jnt_margin':(nj,),
              'dof_solref':(nv,2),'dof_solimp':(nv,5),'jnt_solref':(nj,2),'jnt_solimp':(nj,5),
              'geom_bodyid':(ng,), 'body_invweight0':(nb,2),'dof_invweight0':(nv,),
              'body_rootid':(nb,),'body_parentid':(nb,),'body_dofnum':(nb,),
              'body_dofadr':(nb,),'dof_parentid':(nv,)}
    for key,shape in expected.items():
        if static[key].shape!=shape or static[key].dtype!=np.dtype(STATIC_SCHEMA[key][1]):
            raise ValueError('actual dimension/type mismatch: '+key)
    if any(dims[k]!=facts[k] for k in ('nq','nv','njnt','nbody','ngeom','neq','ntendon','nplugin','na','nu','nM')):
        raise ValueError('model dimension/fact mismatch')
    if np.any(~np.isin(static['jnt_type'][static['jnt_limited']!=0],[2,3])):
        raise ValueError('limited ball/free joints require separate source')
    if np.any(static['jnt_qposadr']<0) or np.any(static['jnt_qposadr']>=nq) or np.any(static['jnt_dofadr']<0) or np.any(static['jnt_dofadr']>=nv):
        raise ValueError('joint address range')
    for key in ('dof_solimp','jnt_solimp'):
        power=np.maximum(1.,static[key][:,4])
        if np.any((power!=1.)&(power!=2.)): raise ValueError('static solimp power family')
    from libero.gpu.reference.constraints.constraint_contact.operator import ContactStatic
    topo={k:static[k] for k in ContactStatic.fields}
    ContactStatic(topo,manifest['model_sha256'])
    return facts


class _Transform(ast.NodeTransformer):
    def __init__(self, assignments=None, replacements=None):
        self.assignments=assignments or {}; self.replacements=replacements or {}
    def visit_Assign(self,node):
        text=ast.unparse(node.targets[0])
        names=_target_names(node.targets[0])
        if text in self.assignments:
            node.value=_expr(self.assignments[text]); return node
        if names and all(k in self.assignments for k in names):
            v=[self.assignments[k] for k in names]
            node.value=_expr(v[0] if len(names)==1 else tuple(v)); return node
        return self.generic_visit(node)
    def visit_Compare(self,node):
        node=self.generic_visit(node)
        key=ast.unparse(node.left)
        if key in self.replacements and len(node.comparators)==1 and isinstance(node.ops[0],ast.NotEq):
            node.comparators[0]=_expr(self.replacements[key])
        return node


def generate_specialization(manifest, static, output_dir):
    """Generate isolated importable package; refuses existing output directory."""
    facts=validate_manifest(manifest,static)
    root=(GPU_ROOT / '.')
    templates={'identity':(GPU_ROOT / 'host/runtime/constraints/identity.py'),
               'scalar':(GPU_ROOT / 'reference/constraints/constraint_scalar/kernel.py'),
               'impedance':(GPU_ROOT / 'reference/constraints/constraint_impedance/kernel.py'),
               'workspace':(GPU_ROOT / 'host/runtime/constraints/workspace.py'),
               'schema':(GPU_ROOT / 'host/runtime/constraints/schema.py'),
               'kernels':(GPU_ROOT / 'host/runtime/constraints/kernels.py'),
               'runtime_adapter':(GPU_ROOT / 'host/runtime/constraints/runtime_adapter.py')}
    nv,nq,nj=(facts[k] for k in ('nv','nq','njnt'))
    nf=int(np.count_nonzero(static['dof_frictionloss']>0))
    nl=int(np.count_nonzero(static['jnt_limited']))
    sha={k:v['sha256'] for k,v in manifest['static'].items()}
    dimensions=dict(zip(('NQ','NV','NU','NM','NBODY','NGEOM','NSITE','NJNT'),
                        (facts[k] for k in ('nq','nv','nu','nM','nbody','ngeom','nsite','njnt'))))
    dimensions.update(dict(NEQ=0,NTENDON=0,NPLUGIN=0,NMOCAP=facts['nmocap'],NA=0,NPAIR=manifest['dimensions']['npair'],
                      NCONMAX=manifest['dimensions']['nconmax'],TIMESTEP=facts['timestep'],IMPRATIO=facts['impratio'],
                      SOLVER=2,CONE=1,JACOBIAN=facts['jacobian'],DISABLEFLAGS=0,ENABLEFLAGS=0,NOSLIP=0,
                      NF_FRICTION_DOF=nf,NLIMIT_JOINTS=nl,NL_LIMIT_MAX=2*nl,NF_CAP=max(nf,1),NL_CAP=max(2*nl,1),
                      MJB_SHA256=manifest['model_sha256'],FIXED_STATIC_SHA=sha,MODEL_FACTS=facts))
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    (out/'__init__.py').write_text('"""Generated S candidate; pending actual GPU qualification."""\n')
    report={'status':'candidate_pending_gpu','model_sha256':manifest['model_sha256'],
            'model_facts':facts,'dimensions':dimensions,'template_pins':{},'generated_pins':{},
            'kernel_arithmetic_bodies_unchanged':{},'floating_options':{'fuse_fp':True,'fast_math':False}}
    for name,path in templates.items():
        text=path.read_text(); tree=ast.parse(text); original_kernels=_kernel_bodies(tree)
        if name=='identity':
            tree=_Transform(dimensions).visit(tree)
            shape_expr='{'+','.join(repr(k)+':('+repr(tuple(a.shape))+', np.'+str(a.dtype)+')' for k,a in static.items())+'}'
            for n in tree.body:
                if isinstance(n,ast.Assign) and _target_names(n.targets[0])==['STATIC_SCHEMA']:
                    n.value=ast.parse(shape_expr,mode='eval').body
        elif name=='scalar':
            shape_expr='{'+','.join(repr(k)+':('+repr(tuple(static[k].shape))+', np.'+str(static[k].dtype)+')' for k in ('dof_frictionloss','jnt_type','jnt_limited','jnt_qposadr','jnt_dofadr','jnt_range','jnt_margin'))+'}'
            tree=_Transform(dict(NV=nv,NQ=nq,NJ=nj,NF_CAP=max(nf,1),NL_CAP=max(2*nl,1),_FIXED_STATIC_SHA={k:sha[k] for k in ('dof_frictionloss','jnt_type','jnt_limited','jnt_qposadr','jnt_dofadr','jnt_range','jnt_margin')}),
                 {"np.count_nonzero(checked['dof_frictionloss'] > 0)":nf,"np.count_nonzero(checked['jnt_limited'])":nl}).visit(tree)
            for n in ast.walk(tree):
                if isinstance(n,ast.Assign) and _target_names(n.targets[0])==['_STATIC']:
                    n.value=ast.parse(shape_expr,mode='eval').body
        elif name=='impedance':
            tree=_Transform({'FIXED_STATIC_SHA256':{k:sha[k] for k in ('dof_solref','dof_solimp','jnt_solref','jnt_solimp')},
                             'expect':dict(dof_solref=(nv,2),dof_solimp=(nv,5),jnt_solref=(nj,2),jnt_solimp=(nj,5)),
                             '(self.nv, self.njnt)':(nv,nj)},
                            {'timestep':facts['timestep'],'impratio':facts['impratio']}).visit(tree)
        elif name=='workspace':
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom) and n.module=='libero.gpu.reference.constraints.constraint_scalar.kernel': n.module='scalar';n.level=1
                if isinstance(n,ast.ImportFrom) and n.module=='libero.gpu.reference.constraints.constraint_impedance.kernel': n.module='impedance';n.level=1
        ast.fix_missing_locations(tree)
        generated=ast.unparse(tree)+'\n'
        if _kernel_bodies(ast.parse(generated))!=original_kernels:
            raise ValueError('unexpected kernel arithmetic transformation: '+name)
        target=out/(name+'.py'); target.write_text(generated)
        report['template_pins'][name]={'path':str(path),'sha256':_sha(path)}
        report['generated_pins'][name]={'path':str(target),'sha256':_sha(target)}
        report['kernel_arithmetic_bodies_unchanged'][name]=True
    (out/'MODEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (out/'SOURCE_GENERATION.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report