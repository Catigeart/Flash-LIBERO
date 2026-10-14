"""Cold bind handwritten forward guards and S-to-N projection."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import MethodType
from libero.gpu.host.build.native import NativePlan


def install(bundle, build_root):
    from libero.gpu.host.runtime.step import physics, shape_physics
    from libero.gpu.host.assembly.pipeline import forward as previous
    f=bundle.runtime.forward; d=f.d; sw=f.s.workspace; n=f.n.ws
    source=(GPU_ROOT / 'cuda/runtime/forward.cuh'); root=Path(build_root)
    plans={}; originals={}
    def plan(name, stages, worlds=f.capacity):
        result=NativePlan(name,[GPU_ROOT / 'cuda/constraints/constraints_projection.cuh',source],stages,int(worlds),root)
        plans[name]=result
        return result
    routes={}
    def bind(kernel,fn,values):
        p=plan(fn,[('forward_cuda::'+fn,values,'lane0')])
        routes[kernel]=(p,tuple(values))
    bind(physics._c_mask,'collision_mask',[d.active,d.valid['position'],d.written['position'],f.expected_forward,d.first_error,f.c_mask])
    bind(physics._reference_mask,'reference_mask',[d.valid['velocity'],d.written['velocity'],f.expected_forward,d.first_error,f.transport.active])
    bind(physics._publish_forward,'publish',[d.active,f.final_valid,n.err_code,d.first_error,f.projection_status,d.source_state_epoch,d.state_epoch,f.valid,f.written,f.error])
    bind(shape_physics._invalidate_full,'invalidate',[f.valid])
    # step1 no longer consumes the standalone increment, but retain its API.
    bind(physics._increment_plan,'increment',[f.mask,d.first_error,f.expected_forward])
    originals['forward_launch']=f._launch
    def launch(owner,kernel,inputs):
        route=routes.get(kernel)
        if route is None: return originals['forward_launch'](kernel,inputs)
        p,expected=route
        if len(inputs)!=len(expected) or any(a is not b for a,b in zip(inputs,expected)):
            raise ValueError('prepared CUDA forward binding changed')
        p.launch(owner.stream)
    f._launch=MethodType(launch,f)
    project=getattr(f,'_cuda_s_projection',None)
    if project is None:project=plan('project_newton',[
        ('forward_cuda::newton_mask',[d.active,d.first_error,d.valid['smooth'],sw.reference_ready,sw.status,f.n_admitted,f.n_upstream],'lane0'),
        ('forward_cuda::counts',[f.n_admitted,f._counts,f.transport.ncon,n.nefc,n.ne,n.nf,n.ncon,n.count,n.empty,n.newton_world,n.active,f.projection_status,f.projection_item,n.rows,n.contacts,f.runtime_shape.nv],'lane0'),
        ('cf_s::project_cone_heads',[f.n_admitted,n.nefc,n.ne,n.nf,n.ncon,f._s_views['type'],f._s_views['id'],f._s_views['dim'],f._s_views['mu'],f._s_views['friction'],n.cone_dim,n.cone_mu,n.cone_friction,f.projection_status,f.projection_item,n.rows,n.contacts],'block'),
        ('forward_cuda::increment',[f.mask,d.first_error,f.expected_newton],'lane0')])
    else:plans['project_newton']=project
    originals['project']=f._fusion_project_newton
    f._fusion_project_newton=MethodType(lambda self:project.launch(self.stream),f)
    begin=plan('begin_forward', [('forward_cuda::begin',[f.f.admitted,d.first_error,d.state_epoch,d.source_state_epoch,d.forward_epoch,d.active,*[d.valid[k] for k in ('position','velocity','bias','smooth')],f.expected_forward],'lane0')])
    originals['dynamics_launch']=d._launch
    def dynamics_launch(owner,kernel,inputs):
        if kernel is previous.begin_and_plan: return begin.launch(owner.stream)
        return originals['dynamics_launch'](kernel,inputs)
    d._launch=MethodType(dynamics_launch,d)
    for context in getattr(d,'_cuda_dynamics_contexts',{}).values():
        context.forward_launch_router=dynamics_launch
    foundation=f.f
    originals['gate_copy']=foundation._fusion_gate_copy
    gates=[]
    for world,(gate,row) in enumerate(zip(foundation.gates,foundation._views)):
        count,actual,expected,valid=row
        values=[foundation._mask_views[world],count,actual,expected,valid,actual,
                gate.errors,gate.first_stage,gate.first_local,gate.first_item,gate.first_event,
                gate.active,gate.valid,gate.capacity,101,foundation.admitted,
                foundation.errors,foundation.first_stage,foundation.first_local,
                foundation.first_item,foundation.first_event,world]
        gates.append(plan('foundation_gate%d'%world,[('forward_cuda::gate_copy',values,'lane0')],worlds=foundation.nworld))
    def gate_copy(owner,w,gate,count,actual,expected,valid):
        gate._stage(101);gate._state()
        gate._require([('admission',owner._mask_views[w],'int32'),('count',count,'int32'),
            ('producer_epoch',actual,'int64'),('expected_epoch',expected,'int64'),
            ('producer_valid',valid,'int32'),('event',actual,'int64')])
        gates[w].launch(owner.stream)
    foundation._fusion_gate_copy=MethodType(gate_copy,foundation)
    bundle.cuda_forward_plans=plans
    return dict(installed=True,plans={k:v.manifest for k,v in plans.items()},
                scope='forward admission, begin, masks, projection and publication',unresolved=[])
