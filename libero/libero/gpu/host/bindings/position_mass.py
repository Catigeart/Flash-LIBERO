"""ABI bindings for handwritten compos, CRB and tree-sparse mass stages."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
HEADER = (GPU_ROOT / 'cuda/position/position_mass.cuh')

def factor_stages(m):
    return [('cf_pm::factor_solve', [m.qm,m.rhs,m.count,m.active,m.dof_Madr,
        m.dof_parentid,m.ld,m.inv,m.sqrt_inv,m.solution,m.pivot,m.status,
        m.warning_count,m.warning_last,m.nv,m.nM], 'lane0')]

def stages(p):
    """Position workspace stages, beginning after its frame/mass mask gate."""
    c=p.compos; r=p.crb_model; m=p.factor; b=p.body
    from .position import _topology,_body_schedule
    topology=_topology(p);typed=topology['qualified']
    body=topology['body'].arrays if typed else None
    compos_schedule=[];crb_schedule=[]
    if typed:
        joint=topology['joint'].arrays;packed=topology['packed'].arrays
        tree=[*_body_schedule(topology),body['children_offsets'],body['children_indices']]
        compos_schedule=[joint['joint_type_offsets'],joint['joint_type_indices'],*tree]
        crb_schedule=[packed['dof_chain_offsets'],packed['dof_chain_indices'],packed['dof_chain_packed'],
            packed['dof_simple_offsets'],packed['dof_simple_indices'],*tree]
    return [
        ('cf_pm::compos',[p.mass_active,p.xipos,p.ximat,b.xanchor,b.xmat,b.xaxis,
            c.mass,c.mass_subtree,c.inertia,c.parent,c.root,c.jtype,c.jbody,c.jdof,
            c.nbody,c.njnt,c.nv,p.subtree_com,p.cinert,p.cdof,p.compos_status,*compos_schedule],'block'),
        ('cf_pm::gate1',[p.mass_active,p.compos_status,p.crb_active,
            p.mass_first_stage,p.mass_first_code],'lane0'),
        ('cf_pm::crb',[p.crb_active,p.cinert,p.cdof,r.body_parent,r.dof_body,r.dof_parent,
            r.dof_madr,r.dof_simple,r.dof_m0,r.dof_armature,r.nbody,r.nv,r.nM,
            p.crb,m.qm,p.crb_status,*crb_schedule],'block'),
        ('cf_pm::gate2',[p.crb_active,p.crb_status,m.active,
            p.mass_first_stage,p.mass_first_code],'lane0'),
        *factor_stages(m),
        ('cf_pm::finish',[p.mass_active,m.active,m.status,p.mass_first_stage,
            p.mass_first_code,p.mass_result_kind],'lane0')]
