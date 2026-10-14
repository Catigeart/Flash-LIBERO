"""Independent source-ordered outward interval graph for the fixed N16 corpus.

Intervals begin at exact input bits. Each binary64 arithmetic node encloses the
exact real operation followed by RN using an outward adjacent representable.
This deliberately does not consult candidate GPU values.
"""
from __future__ import annotations
import math,struct
from fractions import Fraction as F
from dataclasses import dataclass
import numpy as np

MINVAL=1.e-15

def down(x):return math.nextafter(x,-math.inf)
def up(x):return math.nextafter(x,math.inf)
def outward_lo(x:F):
    rounded=float(x)
    return down(rounded) if F.from_float(rounded)>x else rounded
def outward_hi(x:F):
    rounded=float(x)
    return up(rounded) if F.from_float(rounded)<x else rounded

@dataclass(frozen=True)
class I:
    lo:float
    hi:float
    def __post_init__(self):
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo>self.hi:raise ArithmeticError(('bad interval',self.lo,self.hi))
    @staticmethod
    def point(x):return I(float(x),float(x))
    def is_point(a):return a.lo==a.hi
    def __add__(a,b):
        b=as_i(b)
        if a.is_point() and b.is_point():return I.point(a.lo+b.lo)
        return I(down(a.lo+b.lo),up(a.hi+b.hi))
    __radd__=__add__
    def __neg__(a):return I(-a.hi,-a.lo)
    def __sub__(a,b):return a+-as_i(b)
    def __rsub__(a,b):return as_i(b)+-a
    def __mul__(a,b):
        b=as_i(b)
        if a.is_point() and b.is_point():return I.point(a.lo*b.lo)
        v=(a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi)
        return I(down(min(v)),up(max(v)))
    __rmul__=__mul__
    def __truediv__(a,b):
        b=as_i(b)
        if b.lo<=0<=b.hi:raise ArithmeticError(('division crosses zero',b))
        if a.is_point() and b.is_point():return I.point(a.lo/b.lo)
        v=(a.lo/b.lo,a.lo/b.hi,a.hi/b.lo,a.hi/b.hi)
        return I(down(min(v)),up(max(v)))
    def __rtruediv__(a,b):return as_i(b)/a
    def sqrt(a,proved_square_sum=False):
        lo=a.lo
        if lo<0:
            if not proved_square_sum:raise ArithmeticError('negative sqrt without square-sum proof')
            lo=0.
        if a.is_point():return I.point(math.sqrt(lo))
        return I(down(math.sqrt(lo)) if lo else 0.,up(math.sqrt(a.hi)))
    def contains(a,x):return math.isfinite(float(x)) and a.lo<=float(x)<=a.hi

def as_i(x):return x if isinstance(x,I) else I.point(x)
def pt(x):return I.point(x)
def zeros(n):return [pt(0.) for _ in range(n)]
def add3(a,b):return [a[i]+b[i] for i in range(3)]
def sub3(a,b):return [a[i]-b[i] for i in range(3)]
def ident(q):return all(v.lo==v.hi==t for v,t in zip(q,(1.,0.,0.,0.)))

def norm4(q,critical):
    n=(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]).sqrt(True)
    if n.lo<MINVAL<=n.hi:critical.append(('norm_minval',n.lo,n.hi))
    if n.hi<MINVAL:return [pt(1.),pt(0.),pt(0.),pt(0.)],n,1
    if n.lo>=MINVAL:return [x*(pt(1.)/n) for x in q],n,0
    # Conservative union when both fixed-input branches are possible.
    if n.hi<=0:raise ArithmeticError('zero norm denominator')
    bounded=I(MINVAL,n.hi)
    div=[x*(pt(1.)/bounded) for x in q]
    return [I(min(div[i].lo,v),max(div[i].hi,v)) for i,v in enumerate((1.,0.,0.,0.))],n,2

def mulquat(a,b):
    return [((a[0]*b[0]-a[1]*b[1])-a[2]*b[2])-a[3]*b[3],
      ((a[0]*b[1]+a[1]*b[0])+a[2]*b[3])-a[3]*b[2],
      ((a[0]*b[2]-a[1]*b[3])+a[2]*b[0])+a[3]*b[1],
      ((a[0]*b[3]+a[1]*b[2])-a[2]*b[1])+a[3]*b[0]]

def rotquat(v,q,critical):
    if ident(q):return list(v)
    maybe_identity=all(q[k].contains(x) for k,x in enumerate((1.,0.,0.,0.)))
    if maybe_identity:critical.append(('rot_identity',))
    t=[(q[0]*v[0]+q[2]*v[2])-q[3]*v[1],
       (q[0]*v[1]+q[3]*v[0])-q[1]*v[2],
       (q[0]*v[2]+q[1]*v[1])-q[2]*v[0]]
    result=[v[0]+pt(2.)*(q[2]*t[2]-q[3]*t[1]),
            v[1]+pt(2.)*(q[3]*t[0]-q[1]*t[2]),
            v[2]+pt(2.)*(q[1]*t[1]-q[2]*t[0])]
    if maybe_identity:return [I(min(result[k].lo,v[k].lo),max(result[k].hi,v[k].hi)) for k in range(3)]
    return result

def rotmat(v,m):return [((m[3*i]*v[0]+m[3*i+1]*v[1])+m[3*i+2]*v[2]) for i in range(3)]

def quatmat(q,critical):
    if ident(q):return [pt(x) for x in (1.,0.,0.,0.,1.,0.,0.,0.,1.)]
    maybe_identity=all(q[k].contains(x) for k,x in enumerate((1.,0.,0.,0.)))
    if maybe_identity:critical.append(('mat_identity',))
    q00=q[0]*q[0];q01=q[0]*q[1];q02=q[0]*q[2];q03=q[0]*q[3]
    q11=q[1]*q[1];q12=q[1]*q[2];q13=q[1]*q[3]
    q22=q[2]*q[2];q23=q[2]*q[3];q33=q[3]*q[3]
    result=[((q00+q11)-q22)-q33,pt(2.)*(q12-q03),pt(2.)*(q13+q02),
            pt(2.)*(q12+q03),((q00-q11)+q22)-q33,pt(2.)*(q23-q01),
            pt(2.)*(q13-q02),pt(2.)*(q23+q01),((q00-q11)-q22)+q33]
    if maybe_identity:
        original=(1.,0.,0.,0.,1.,0.,0.,0.,1.)
        return [I(min(result[k].lo,original[k]),max(result[k].hi,original[k])) for k in range(9)]
    return result

def trig_gpu_interval(z,kind):
    # Actual binary64 constants in certified_trig.py; rounding on each node.
    n=23;t=z*z
    p=pt(float(F((-1)**22,math.factorial(44+(kind=='sin')))))
    for k in range(21,-1,-1):p=p*t+pt(float(F((-1)**k,math.factorial(2*k+(kind=='sin')))))
    return z*p if kind=='sin' else p

def trig_envelope(z:I,cpu_z:float,seal:dict,critical):
    """CPU fixed-input certificate + Lipschitz argument + RN Horner graph.

    The bound is a radius around the sealed CPU helper bit pattern; none of
    its terms is derived from candidate GPU output.
    """
    key=np.float64(cpu_z).tobytes().hex();item=seal[key]
    if not z.contains(cpu_z):raise ArithmeticError(('CPU trig argument outside propagated interval',z,cpu_z))
    cpu_z_exact=F.from_float(cpu_z)
    delta=max(abs(F.from_float(z.lo)-cpu_z_exact),abs(F.from_float(z.hi)-cpu_z_exact))
    result=[]
    for kind in ('sin','cos'):
        gpu=trig_gpu_interval(z,kind)
        x=F.from_float(cpu_z);poly=sum(((-1)**k*x**(2*k+(kind=='sin'))/math.factorial(2*k+(kind=='sin')) for k in range(23)),F(0))
        tail=F(4**(47 if kind=='sin' else 46),math.factorial(47 if kind=='sin' else 46))
        cpu=np.frombuffer(bytes.fromhex(item[kind+'_bits']),dtype='<f8')[0].item()
        cert=F(*item[kind+'_cpu_error_bound'])
        if abs(F.from_float(cpu)-poly)+tail!=cert:raise ArithmeticError(('CPU trig certificate drift',kind,key))
        # Mathematical function at cpu_z lies in poly +/- tail; at any z in
        # the propagated interval it moves by at most delta (derivative <=1).
        mlo=poly-tail-delta;mhi=poly+tail+delta
        polynomial_math_error=max(abs(F.from_float(gpu.lo)-mhi),abs(F.from_float(gpu.hi)-mlo))
        radius=polynomial_math_error+cert+delta
        cpu_exact=F.from_float(cpu)
        result.append(I(outward_lo(cpu_exact-radius),outward_hi(cpu_exact+radius)))
    return result[0],result[1]

def compute(model,source,trig,critical):
    """Return all continuous fields from an independent interval execution."""
    m=model.arrays;n=model.dims;nb=n['nbody'];nj=n['njnt'];nm=n['nmocap']
    a={k:np.asarray(v) for k,v in source.items()}
    qpos=[pt(x) for x in a['qpos']]
    mq=[[pt(x) for x in row] for row in a['mocap_quat']]
    r={'xpos':[zeros(3) for _ in range(nb)],'xquat':[zeros(4) for _ in range(nb)],
       'xmat':[zeros(9) for _ in range(nb)],'xanchor':[zeros(3) for _ in range(nj)],'xaxis':[zeros(3) for _ in range(nj)],
       'world_xipos':zeros(3),'world_ximat':[pt(x) for x in (1.,0.,0.,0.,1.,0.,0.,0.,1.)],
       'norm_before':zeros(nj),'mocap_before':zeros(nm),'pre_pos':[zeros(3) for _ in range(nb)],
       'pre_quat':[zeros(4) for _ in range(nb)],'joint_pos':[zeros(3) for _ in range(nj)],
       'joint_quat':[zeros(4) for _ in range(nj)],'body_norm_before':zeros(nb)}
    r['xquat'][0]=[pt(x) for x in (1.,0.,0.,0.)];r['xmat'][0]=[pt(x) for x in (1.,0.,0.,0.,1.,0.,0.,0.,1.)]
    for j,t in enumerate(m['jnt_type']):
        if t in (0,1):
            adr=int(m['jnt_qposadr'][j])+(3 if t==0 else 0)
            qpos[adr:adr+4],r['norm_before'][j],_=norm4(qpos[adr:adr+4],critical)
    for k in range(nm):mq[k],r['mocap_before'][k],_=norm4(mq[k],critical)
    r['qpos_after_stage']=qpos.copy();r['qpos_after']=qpos.copy();r['mocap_after_stage']=[x.copy() for x in mq];r['mocap_after']=[x.copy() for x in mq]
    for b in range(1,nb):
        adr=int(m['body_jntadr'][b]);jnum=int(m['body_jntnum'][b])
        if jnum==1 and m['jnt_type'][adr]==0:
            qa=int(m['jnt_qposadr'][adr]);pos=qpos[qa:qa+3].copy();quat=qpos[qa+3:qa+7].copy()
            r['pre_pos'][b]=pos.copy();r['pre_quat'][b]=quat.copy()
            r['xanchor'][adr]=pos.copy();r['xaxis'][adr]=[pt(0.),pt(0.),pt(1.)]
            r['joint_pos'][adr]=pos.copy();r['joint_quat'][adr]=quat.copy()
        else:
            pid=int(m['body_parentid'][b]);mid=int(m['body_mocapid'][b])
            bp=[pt(x) for x in (a['mocap_pos'][mid] if mid>=0 else m['body_pos'][b])]
            bq=mq[mid].copy() if mid>=0 else [pt(x) for x in m['body_quat'][b]]
            vec=rotmat(bp,r['xmat'][pid]);pos=add3(r['xpos'][pid],vec);quat=mulquat(r['xquat'][pid],bq)
            r['pre_pos'][b]=pos.copy();r['pre_quat'][b]=quat.copy()
            for j in range(adr,adr+jnum):
                qa=int(m['jnt_qposadr'][j]);t=int(m['jnt_type'][j]);axis=[pt(x) for x in m['jnt_axis'][j]];jp=[pt(x) for x in m['jnt_pos'][j]]
                xaxis=rotquat(axis,quat,critical);xanchor=add3(rotquat(jp,quat,critical),pos)
                if t==2:
                    delta=qpos[qa]-pt(m['qpos0'][qa]);pos=[pos[k]+xaxis[k]*delta for k in range(3)]
                elif t in (1,3):
                    if t==1:qloc=qpos[qa:qa+4].copy()
                    else:
                        angle=qpos[qa]-pt(m['qpos0'][qa]);cpu_angle=float(a['qpos'][qa]-m['qpos0'][qa])
                        if cpu_angle==0.:qloc=[pt(1.),pt(0.),pt(0.),pt(0.)]
                        else:
                            z=angle*pt(.5);cpu_z=float(cpu_angle*.5)
                            if z.lo<-4 or z.hi>4:raise ArithmeticError(('trig domain',z))
                            sn,cs=trig_envelope(z,cpu_z,trig,critical)
                            qloc=[cs,axis[0]*sn,axis[1]*sn,axis[2]*sn]
                    quat=mulquat(quat,qloc)
                    vec=sub3(xanchor,pos);vec1=rotquat(jp,quat,critical)
                    pos=[pos[k]+(vec[k]-vec1[k]) for k in range(3)]
                else:raise ArithmeticError(('type',t))
                r['xanchor'][j]=xanchor;r['xaxis'][j]=xaxis;r['joint_pos'][j]=pos.copy();r['joint_quat'][j]=quat.copy()
        quat,r['body_norm_before'][b],_=norm4(quat,critical)
        r['xquat'][b]=quat;r['xpos'][b]=pos;r['xmat'][b]=quatmat(quat,critical)
    return r

def validate_scalar_ops():
    # Exact rational midpoint test proves each sampled point node is RN;
    # interval tests below separately prove containment of uncertain nodes.
    def even(x):return (struct.unpack('<Q',struct.pack('<d',x))[0]&1)==0
    def rn(exact:F,rounded:float):
        if exact==F.from_float(rounded):return True
        below=math.nextafter(rounded,-math.inf);above=math.nextafter(rounded,math.inf)
        lowmid=(F.from_float(below)+F.from_float(rounded))/2
        highmid=(F.from_float(rounded)+F.from_float(above))/2
        return (exact>lowmid or (exact==lowmid and even(rounded))) and (exact<highmid or (exact==highmid and even(rounded)))
    def rn_sqrt(exact:F,rounded:float):
        if F.from_float(rounded)**2==exact:return True
        below=math.nextafter(rounded,-math.inf);above=math.nextafter(rounded,math.inf)
        lowmid=(F.from_float(below)+F.from_float(rounded))/2
        highmid=(F.from_float(rounded)+F.from_float(above))/2
        return (exact>lowmid**2 or (exact==lowmid**2 and even(rounded))) and (exact<highmid**2 or (exact==highmid**2 and even(rounded)))
    examples=[(0.,0.),(1.,-1.),(math.ldexp(1.,-1074),1.),(-1.e100,1.e-100),(1.,3.)]
    for x,y in examples:
        a=pt(x);b=pt(y);fx=F.from_float(x);fy=F.from_float(y)
        for op,exact in ((a+b,fx+fy),(a-b,fx-fy),(a*b,fx*fy)):
            assert op.is_point() and rn(exact,op.lo)
        if y:
            op=a/b
            assert op.is_point() and rn(fx/fy,op.lo)
        if x>=0:
            op=a.sqrt(True)
            assert op.is_point() and rn_sqrt(fx,op.lo)
        wide=I(min(x,y),max(x,y))
        for op,exact in ((wide+a,F.from_float(wide.lo)+F.from_float(x)),
                         (wide-a,F.from_float(wide.lo)-F.from_float(x)),
                         (wide*a,F.from_float(wide.lo)*F.from_float(x))):
            assert F.from_float(op.lo)<=exact<=F.from_float(op.hi)
