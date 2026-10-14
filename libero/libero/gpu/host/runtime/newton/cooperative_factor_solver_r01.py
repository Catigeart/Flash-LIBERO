"""Cold complete S05 candidate solver selection; all qualification pending."""
from pathlib import Path
import ast,hashlib,inspect,textwrap
from .cooperative_factor_bridge_r01 import CooperativeFactorModule

def factor_arrays(solver,kind):
    ws=solver.ws
    return dict(count=ws.count,ncon=ws.ncon,active=ws.active,kind=kind,
        J=solver._J_flat,D=solver._D_flat,state=solver._state_flat,oldstate=solver._oldstate_flat,
        qM=solver._qM_flat,madr=ws.madr,parentid=ws.parentid,simplenum=ws.simplenum,
        efc_id=solver._id_flat,contact_dim=solver._dim_flat,contact_H=solver._H_contact_flat,
        ncone=ws.ncone,H=solver._H_flat,Hcone=solver._Hcone_flat,nnz=ws.nnz,nupdate=ws.nupdate,
        local=solver._H_local_flat,ltj=solver._H_ltj_flat,vec=solver._H_vec_flat,status=ws.status,route=ws.route)

def candidate_method(original):
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source);fn=tree.body[0]
    launches=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='wp.launch']
    if len(launches)!=3 or [ast.unparse(fn.body[i].value.args[0]) for i in launches]!=['s05.validate','s05.assemble','s05.factor_update']:raise ValueError('original S05 source layout drift')
    i=launches[2];original_node=fn.body[i]
    fn.body[i]=ast.parse('self._cooperative_factor(kind_arr)').body[0]
    product=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    recovered=ast.parse(product);recovered.body[0].body[i]=original_node
    if ast.dump(recovered,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('retained validate/assemble source changed')
    env=original.__globals__.copy();exec(compile(ast.fix_missing_locations(tree),'<complete-S05-candidate>','exec'),env)
    return env[original.__name__],dict(original_method_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_method_sha256=hashlib.sha256(product.encode()).hexdigest(),only_factor_launch_replaced=True,source=product)

def cooperative_factor_solver_type(original_type,*,build_root):
    owner=next(c for c in original_type.__mro__ if '_s05_launch' in c.__dict__)
    original=owner.__dict__['_s05_launch'];method,proof=candidate_method(original)
    nv=int(original.__globals__['NV']);nm=int(original.__globals__['NM']) if 'NM' in original.__globals__ else int(original.__globals__['s05'].NM)
    class CooperativeFactorSolver(original_type):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            ws=self.ws
            self.factor_module=CooperativeFactorModule(Path(build_root)/'native_factor',device=ws.device)
            self.factor_bindings={}
            for kind in (self._kind_direct,self._kind_incr):
                self.factor_bindings[id(kind)]=self.factor_module.bind(factor_arrays(self,kind),nv=nv,nm=nm,nworld=ws.nworld,rows=ws.rows,contacts=ws.contacts,stream=ws.stream)
        def _cooperative_factor(self,kind):
            # Cold bound owner identity only, never a host device-value branch.
            bound=self.factor_bindings.get(id(kind))
            if bound is None:raise ValueError('unbound factor kind owner')
            bound.launch()
    CooperativeFactorSolver._s05_launch=method
    CooperativeFactorSolver.factor_source_proof=proof
    return CooperativeFactorSolver
