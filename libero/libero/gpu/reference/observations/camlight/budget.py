"""Independent source-node outward interval certificate for D01 FP64 fields.

This host checker does not import the Warp kernel or use observed GPU errors.
Every +,-,*,/,sqrt endpoint is widened by one binary64 successor.  Input
points are the exact original binary64 arrays after dtype/shape validation.
"""
from __future__ import annotations

import numpy as np

MINVAL=np.float64(1.0e-15)
NEG=np.float64(-np.inf);POS=np.float64(np.inf)

class NumericRange(ArithmeticError):pass

class I:
    def __init__(self,lo,hi=None):
        self.lo=np.asarray(lo,dtype=np.float64)
        self.hi=self.lo if hi is None else np.asarray(hi,dtype=np.float64)
    @staticmethod
    def wide(lo,hi):
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():raise NumericRange('nonfinite node')
        return I(np.nextafter(lo,NEG),np.nextafter(hi,POS))
    def __add__(self,other):
        b=ii(other)
        return I.wide(self.lo+b.lo,self.hi+b.hi)
    def __sub__(self,other):
        b=ii(other)
        return I.wide(self.lo-b.hi,self.hi-b.lo)
    def __mul__(self,other):
        b=ii(other)
        p=(self.lo*b.lo,self.lo*b.hi,self.hi*b.lo,self.hi*b.hi)
        return I.wide(np.minimum.reduce(p),np.maximum.reduce(p))
    def sqrt(self):
        if np.any(self.lo<0):raise NumericRange('sqrt interval crosses negative')
        return I.wide(np.sqrt(self.lo),np.sqrt(self.hi))
    def reciprocal(self):
        if np.any(self.lo<=0):raise NumericRange('division interval crosses zero')
        return I.wide(np.float64(1)/self.hi,np.float64(1)/self.lo)

def ii(x):return x if isinstance(x,I) else I(x)
def stack(items):return I(np.stack([v.lo for v in items],axis=-1),np.stack([v.hi for v in items],axis=-1))
def scalar(a,i):return I(a.lo[...,i],a.hi[...,i])

def position(xpos,xmat,local):
    pos=ii(xpos);mat=ii(xmat);v=ii(local)
    out=[]
    for row in range(3):
        z=scalar(mat,3*row)*scalar(v,0)
        z=z+scalar(mat,3*row+1)*scalar(v,1)
        z=z+scalar(mat,3*row+2)*scalar(v,2)
        out.append(z+scalar(pos,row))
    return stack(out)

def orientation(body,local):
    a=[scalar(ii(body),i) for i in range(4)]
    b=[scalar(ii(local),i) for i in range(4)]
    q0=((a[0]*b[0]-a[1]*b[1])-a[2]*b[2])-a[3]*b[3]
    q1=((a[0]*b[1]+a[1]*b[0])+a[2]*b[3])-a[3]*b[2]
    q2=((a[0]*b[2]-a[1]*b[3])+a[2]*b[0])+a[3]*b[1]
    q3=((a[0]*b[3]+a[1]*b[2])-a[2]*b[1])+a[3]*b[0]
    q=(q0,q1,q2,q3)
    p00=q0*q0;p01=q0*q1;p02=q0*q2;p03=q0*q3
    p11=q1*q1;p12=q1*q2;p13=q1*q3
    p22=q2*q2;p23=q2*q3;p33=q3*q3
    two=I(np.float64(2))
    mat=stack((((p00+p11)-p22)-p33,two*(p12-p03),two*(p13+p02),
               two*(p12+p03),((p00-p11)+p22)-p33,two*(p23-p01),
               two*(p13-p02),two*(p23+p01),((p00-p11)-p22)+p33))
    return stack(q),mat

def direction(body,local):
    a=[scalar(ii(body),i) for i in range(4)]
    v=[scalar(ii(local),i) for i in range(3)]
    # mju_rotVecQuat's exact identity route is selected from original input.
    raw=np.asarray(body)
    identity=np.all(raw==np.array([1.,0.,0.,0.]),axis=-1)
    t0=(a[0]*v[0]+a[2]*v[2])-a[3]*v[1]
    t1=(a[0]*v[1]+a[3]*v[0])-a[1]*v[2]
    t2=(a[0]*v[2]+a[1]*v[1])-a[2]*v[0]
    two=I(np.float64(2))
    regular=[v[0]+two*(a[2]*t2-a[3]*t1),
             v[1]+two*(a[3]*t0-a[1]*t2),
             v[2]+two*(a[1]*t1-a[2]*t0)]
    rot=stack([I(np.where(identity,v[j].lo,regular[j].lo),
                 np.where(identity,v[j].hi,regular[j].hi)) for j in range(3)])
    s=(scalar(rot,0)*scalar(rot,0)+scalar(rot,1)*scalar(rot,1))+scalar(rot,2)*scalar(rot,2)
    # Outward interval dependencies can put a tiny negative lower endpoint at
    # a true zero.  The exact nonnegative sum starts at zero in that case.
    if np.any(s.hi<0):raise NumericRange('negative norm square')
    norm=I(np.maximum(s.lo,0),s.hi).sqrt()
    critical=(norm.lo<MINVAL)&(norm.hi>=MINVAL)
    if np.any(critical):raise NumericRange('normalize threshold interval critical')
    below=norm.hi<MINVAL
    if np.any((~below)&(norm.lo<=0)):raise NumericRange('positive norm unproved')
    safe=I(np.where(below,1.,norm.lo),np.where(below,1.,norm.hi))
    inv=safe.reciprocal()
    result=stack([I(np.where(below,1. if j==0 else 0.,(scalar(rot,j)*inv).lo),
                    np.where(below,1. if j==0 else 0.,(scalar(rot,j)*inv).hi)) for j in range(3)])
    return result,rot,norm,identity.astype(np.int32),below.astype(np.int32)

def bounds(batch,static):
    N=len(batch['in_xpos']);nc=len(static['cam_bodyid']);nl=len(static['light_bodyid'])
    out={'cam_xpos':[],'cam_xmat':[],'cam_tmpquat':[],'light_xpos':[],'light_raw_dir':[],'light_xdir':[],
         'light_norm':[],'light_identity':[],'light_norm_branch':[],'cam_identity_critical':[]}
    for i,b in enumerate(static['cam_bodyid']):
        out['cam_xpos'].append(position(batch['in_xpos'][:,b],batch['in_xmat'][:,b],static['cam_pos'][i]))
        q,mat=orientation(batch['in_xquat'][:,b],static['cam_quat'][i])
        out['cam_tmpquat'].append(q)
        out['cam_xmat'].append(mat)
        # Exact branch input may be stable, or the interval may straddle 1/0.
        possible=np.all((q.lo<=np.array([1.,0.,0.,0.]))&(q.hi>=np.array([1.,0.,0.,0.])),axis=-1)
        out['cam_identity_critical'].append(possible)
    for i,b in enumerate(static['light_bodyid']):
        out['light_xpos'].append(position(batch['in_xpos'][:,b],batch['in_xmat'][:,b],static['light_pos'][i]))
        v,raw,norm,identity,branch=direction(batch['in_xquat'][:,b],static['light_dir'][i])
        out['light_raw_dir'].append(raw)
        out['light_xdir'].append(v);out['light_norm'].append(norm)
        out['light_identity'].append(identity);out['light_norm_branch'].append(branch)
    for name,width in (('cam_xpos',3),('cam_xmat',9),('cam_tmpquat',4),('light_xpos',3),('light_raw_dir',3),('light_xdir',3),('light_norm',1)):
        seq=out[name]
        if width==1:
            out[name]=I(np.stack([x.lo for x in seq],axis=1),np.stack([x.hi for x in seq],axis=1)) if seq else I(np.empty((N,0)),np.empty((N,0)))
        else:
            out[name]=I(np.stack([x.lo for x in seq],axis=1),np.stack([x.hi for x in seq],axis=1)) if seq else I(np.empty((N,0,width)),np.empty((N,0,width)))
    for name,dim in (('light_identity',nl),('light_norm_branch',nl),('cam_identity_critical',nc)):
        out[name]=np.stack(out[name],axis=1) if out[name] else np.empty((N,0),np.int32)
    return out

def first_outside(interval,value):
    bad=(value<interval.lo)|(value>interval.hi)|(~np.isfinite(value))
    if not np.any(bad):return None
    idx=tuple(int(v) for v in np.argwhere(bad)[0])
    return {'index':idx,'value':float(value[idx]),'lo':float(interval.lo[idx]),'hi':float(interval.hi[idx])}
