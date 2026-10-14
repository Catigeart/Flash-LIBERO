"""Finite-corpus GPU comparator: schema/identity/labels, then interval stages."""
from __future__ import annotations
import hashlib,json,math,struct
from pathlib import Path
import numpy as np
from .budget import I, pt, trig_envelope, compute, validate_scalar_ops
STAGE_INT=('body_route','joint_route','body_mat_identity','joint_rot_identity','hinge_zero','offcenter_rot_identity')
FULL_FIELDS=('qpos_after','mocap_after','xpos','xquat','xmat','xanchor','xaxis','world_xipos','world_ximat',
             'norm_before','mocap_before','qpos_after_stage','mocap_after_stage','pre_pos','pre_quat','joint_pos','joint_quat','body_norm_before',*STAGE_INT)

def canonical(fields):
    h=hashlib.sha256()
    for name in sorted(fields):
        a=np.asarray(fields[name]);a=np.ascontiguousarray(a)
        h.update(json.dumps([name,a.shape,a.dtype.str],separators=(',',':')).encode());h.update(b'\0');h.update(a.tobytes())
    return h.hexdigest()

CONT=tuple(x for x in FULL_FIELDS if x not in STAGE_INT)

def _flat(x):
    if isinstance(x,I):yield x
    elif isinstance(x,(list,tuple)):
        for y in x:yield from _flat(y)
    else:raise TypeError(type(x))

def schema(reference,candidate,model):
    dims=model.dims;nb=dims['nbody'];nj=dims['njnt'];nm=dims['nmocap'];nq=dims['nq']
    sizes={'qpos_after':(nq,),'mocap_after':(nm,4),'xpos':(nb,3),'xquat':(nb,4),'xmat':(nb,9),
           'xanchor':(nj,3),'xaxis':(nj,3),'world_xipos':(3,),'world_ximat':(9,),
           'norm_before':(nj,),'mocap_before':(nm,),'qpos_after_stage':(nq,),'mocap_after_stage':(nm,4),
           'pre_pos':(nb,3),'pre_quat':(nb,4),'joint_pos':(nj,3),'joint_quat':(nj,4),'body_norm_before':(nb,),
           'body_route':(nb,),'joint_route':(nj,),'body_mat_identity':(nb,),'joint_rot_identity':(nj,),
           'hinge_zero':(nj,),'offcenter_rot_identity':(nj,)}
    N=len(reference['in_qpos'])
    for field in FULL_FIELDS:
        for side,values in (('reference',reference),('candidate',candidate)):
            if field not in values:return {'stage':'schema','field':field,'side':side,'reason':'missing'}
            a=values[field];shape=(N,)+sizes[field];dtype=np.int32 if field in STAGE_INT else np.float64
            if not isinstance(a,np.ndarray) or a.shape!=shape or a.dtype!=dtype:return {'stage':'schema','field':field,'side':side,'actual_shape':getattr(a,'shape',None),'expected_shape':shape,'actual_dtype':str(getattr(a,'dtype',None))}
            if dtype==np.float64 and not np.isfinite(a).all():return {'stage':'schema','field':field,'side':side,'reason':'nonfinite'}
    if 'status' not in candidate or candidate['status'].shape!=(N,) or candidate['status'].dtype!=np.int32:return {'stage':'schema','field':'status'}
    for field,size,dtype in (('norm_branch',nj,np.int32),('mocap_branch',nm,np.int32),('body_norm_branch',nb,np.int32),
                             ('trig_arg',nj,np.float64),('trig_sin',nj,np.float64),('trig_cos',nj,np.float64)):
        a=candidate.get(field)
        if not isinstance(a,np.ndarray) or a.shape!=(N,size) or a.dtype!=dtype:return {'stage':'schema','field':field}
        if dtype==np.float64 and not np.isfinite(a).all():return {'stage':'schema','field':field,'reason':'nonfinite'}
    return None

def first_diff(reference,candidate,model,seal,identity,critical_out=None):
    issue=schema(reference,candidate,model)
    if issue:return issue
    if identity.get('model_sha256')!=model.field_sha256 or identity.get('input_sha256')!=canonical({k:reference['in_'+k][identity['row']] for k in ('qpos','mocap_pos','mocap_quat')}):
        return {'stage':'identity','field':'source'}
    row=identity['row'];m=model.arrays
    if candidate['status'][row]!=0:return {'stage':'status','row':row,'actual':int(candidate['status'][row])}
    for field in STAGE_INT:
        a=reference[field][row];b=candidate[field][row]
        bad=np.argwhere(a!=b)
        if len(bad):return {'stage':'route','field':field,'row':row,'index':bad[0].tolist(),'expected':int(a[tuple(bad[0])]),'actual':int(b[tuple(bad[0])])}
    for field,expected in (('xpos',np.array([0.,0.,0.])),('xquat',np.array([1.,0.,0.,0.])),('world_xipos',np.zeros(3)),('xmat',np.eye(3).reshape(-1)),('world_ximat',np.eye(3).reshape(-1))):
        a=reference[field][row] if field.startswith('world') else reference[field][row,0]
        b=candidate[field][row] if field.startswith('world') else candidate[field][row,0]
        if a.tobytes()!=expected.tobytes():return {'stage':'world','field':field,'reason':'bad_reference'}
        if a.tobytes()!=b.tobytes():return {'stage':'world','field':field,'row':row}
    # These stages precede every hinge trig operation and are exact RN copies.
    for field in ('norm_before','mocap_before','qpos_after_stage','mocap_after_stage','qpos_after','mocap_after'):
        if reference[field][row].tobytes()!=candidate[field][row].tobytes():
            return {'stage':'normalization_exact','field':field,'row':row}
    for b in range(1,model.dims['nbody']):
        if reference['body_route'][row,b]==1:
            j=int(m['body_jntadr'][b])
            for field,x,y in (('xanchor',candidate['xanchor'][row,j],candidate['pre_pos'][row,b]),
                              ('xaxis',candidate['xaxis'][row,j],np.array([0.,0.,1.]))):
                if x.tobytes()!=y.tobytes():return {'stage':'free_exact','field':field,'row':row,'body':b}
        if reference['body_mat_identity'][row,b]==1 and candidate['xmat'][row,b].tobytes()!=np.eye(3).reshape(-1).tobytes():
            return {'stage':'matrix_identity_exact','field':'xmat','row':row,'body':b}
    for j in range(model.dims['njnt']):
        if reference['joint_rot_identity'][row,j]==1 and candidate['xaxis'][row,j].tobytes()!=m['jnt_axis'][j].tobytes():
            return {'stage':'rotation_identity_exact','field':'xaxis','row':row,'joint':j}
        t=int(m['jnt_type'][j]);qa=int(m['jnt_qposadr'][j])
        expected_norm=int(reference['norm_before'][row,j]<1.e-15) if t in (0,1) else 0
        if int(candidate['norm_branch'][row,j])!=expected_norm:return {'stage':'norm_branch','field':'norm_branch','row':row,'joint':j}
        if t==3:
            angle=float(reference['in_qpos'][row,qa]-m['qpos0'][qa])
            if angle==0.:
                if any(candidate[x][row,j].tobytes()!=np.float64(0.).tobytes() for x in ('trig_arg','trig_sin','trig_cos')):
                    return {'stage':'hinge_zero_exact','row':row,'joint':j}
            else:
                z=float(angle*.5)
                if candidate['trig_arg'][row,j].tobytes()!=np.float64(z).tobytes():return {'stage':'trig_argument','row':row,'joint':j}
                sn,cs=trig_envelope(pt(z),z,seal,[])
                if not sn.contains(candidate['trig_sin'][row,j]) or not cs.contains(candidate['trig_cos'][row,j]):
                    return {'stage':'trig_value','row':row,'joint':j}
    for k in range(model.dims['nmocap']):
        if int(candidate['mocap_branch'][row,k])!=int(reference['mocap_before'][row,k]<1.e-15):
            return {'stage':'mocap_norm_branch','row':row,'mocap':k}
    for b in range(1,model.dims['nbody']):
        if int(candidate['body_norm_branch'][row,b])!=int(reference['body_norm_before'][row,b]<1.e-15):
            return {'stage':'body_norm_branch','row':row,'body':b}
    source={k:reference['in_'+k][row] for k in ('qpos','mocap_pos','mocap_quat')}
    critical=[];bounds=compute(model,source,seal,critical)
    if critical_out is not None:critical_out.extend(critical)
    sequence=[('norm_before',None),('qpos_after_stage',None),('mocap_before',None),('mocap_after_stage',None),('qpos_after',None),('mocap_after',None)]
    for b in range(1,model.dims['nbody']):
        sequence.extend((('pre_pos',b),('pre_quat',b)))
        adr=int(m['body_jntadr'][b]);cnt=int(m['body_jntnum'][b])
        for j in range(adr,adr+cnt):sequence.extend((('xanchor',j),('xaxis',j),('joint_pos',j),('joint_quat',j)))
        sequence.extend((('body_norm_before',b),('xpos',b),('xquat',b),('xmat',b)))
    sequence.extend((('pre_pos',0),('pre_quat',0),('body_norm_before',0)))
    for field,index in sequence:
        aa=reference[field][row] if index is None else reference[field][row,index]
        bb=candidate[field][row] if index is None else candidate[field][row,index]
        ix=bounds[field] if index is None else bounds[field][index]
        iv=list(_flat(ix));flat_a=aa.reshape(-1);flat_b=bb.reshape(-1)
        if len(iv)!=len(flat_a):return {'stage':'budget','field':field,'reason':'shape_graph'}
        for k,(bound,x,y) in enumerate(zip(iv,flat_a,flat_b)):
            if not bound.contains(x):return {'stage':'budget','field':field,'row':row,'index':index,'element':k,'reason':'cpu_outside','lo':bound.lo,'hi':bound.hi,'cpu':float(x)}
            if not bound.contains(y):return {'stage':'value','field':field,'row':row,'index':index,'element':k,'lo':bound.lo,'hi':bound.hi,'gpu':float(y)}
    # Original C never executes these world diagnostic slots; the zero
    # sentinels are checked last, separately from the five executed writes.
    for field in ('pre_pos','pre_quat','body_norm_before'):
        a=reference[field][row,0];b=candidate[field][row,0]
        if a.tobytes()!=np.zeros_like(a).tobytes():return {'stage':'sentinel','field':field,'reason':'bad_reference','row':row}
        if a.tobytes()!=b.tobytes():return {'stage':'sentinel','field':field,'row':row}
    return None
