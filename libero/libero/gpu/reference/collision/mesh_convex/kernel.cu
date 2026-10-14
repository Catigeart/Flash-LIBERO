// C07 Mesh-Mesh MPR, independently executing on CUDA. Algorithmic order from
// MuJoCo 2.3.7 engine_collision_convex.c:32-239,278-364 and pinned libccd
// 7931e764 src/mpr.c, support.c, vec3.c. No source code is loaded at runtime.
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

struct V { double x,y,z; };
struct S { V v,v1,v2; };
struct Portal { S p[4]; int size; };
struct Model {
  const float *vert;
  const int *graph,*vertadr,*vertnum,*graphadr,*dataid,*geomtype;
  int nvert,ngraph,nmesh,ngeom,maxiter;
  double tolerance;
};
struct Query {
  Model m;
  const double *pose;
  int g1,g2,world,watchdog,reason,discover,refine,penetrate,n_support,stage;
  int *support_ids;
  V last_dir;
  S last_support;
  Portal portal;
};
static __device__ __forceinline__ V v(double x,double y,double z){return {x,y,z};}
static __device__ __forceinline__ V add(V a,V b){return v(a.x+b.x,a.y+b.y,a.z+b.z);}
static __device__ __forceinline__ V sub(V a,V b){return v(a.x-b.x,a.y-b.y,a.z-b.z);}
static __device__ __forceinline__ V scale(V a,double b){return v(a.x*b,a.y*b,a.z*b);}
static __device__ __forceinline__ double dot(V a,V b){double d=a.x*b.x;d+=a.y*b.y;d+=a.z*b.z;return d;}
static __device__ __forceinline__ V cross(V a,V b){return v(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
static __device__ __forceinline__ V norm(V a){double k=1.0/sqrt(dot(a,a));return scale(a,k);}
static __device__ __forceinline__ bool zero(double a){return fabs(a)<DBL_EPSILON;}
static __device__ __forceinline__ bool eq(double a,double b){
  double d=fabs(a-b);if(fabs(d)<DBL_EPSILON)return true;
  a=fabs(a);b=fabs(b);return d<DBL_EPSILON*(b>a?b:a);
}
static __device__ __forceinline__ bool veq(V a,V b){return eq(a.x,b.x)&&eq(a.y,b.y)&&eq(a.z,b.z);}
static __device__ __forceinline__ V load3(const double *a){return v(a[0],a[1],a[2]);}
static __device__ __forceinline__ V rot(V a,const double *m){
  return v(m[0]*a.x+m[1]*a.y+m[2]*a.z,
           m[3]*a.x+m[4]*a.y+m[5]*a.z,
           m[6]*a.x+m[7]*a.y+m[8]*a.z);
}
static __device__ __forceinline__ V rotT(V a,const double *m){
  return v(m[0]*a.x+m[3]*a.y+m[6]*a.z,
           m[1]*a.x+m[4]*a.y+m[7]*a.z,
           m[2]*a.x+m[5]*a.y+m[8]*a.z);
}
static __device__ __forceinline__ void write3(double *out,V a){out[0]=a.x;out[1]=a.y;out[2]=a.z;}

// Caller owns the two ordered geom poses, [margin,pos1,mat1,pos2,mat2].
static __device__ V mesh_support(Query &q,int which,V direction,int &local_index){
  int geom=which==0?q.g1:q.g2,id=q.m.dataid[geom];
  const double *p=q.pose+(which==0?1:13),*mat=p+3;
  V dir=rotT(direction,mat);
  int adr=q.m.vertadr[id],ga=q.m.graphadr[id],best=-1;
  double tmp=-1E+10,vdot;
  if(ga<0){
    for(int i=0;i<q.m.vertnum[id];++i){
      int vi=3*(adr+i);
      vdot=dir.x*(double)q.m.vert[vi]+dir.y*(double)q.m.vert[vi+1]+dir.z*(double)q.m.vert[vi+2];
      if(vdot>tmp){tmp=vdot;best=i;}
    }
    local_index=best;
  }else{
    int nv=q.m.graph[ga];
    const int *edgeadr=q.m.graph+ga+2,*global=q.m.graph+ga+2+nv;
    const int *edge=q.m.graph+ga+2+2*nv;
    best=0;int vi=3*(adr+global[0]);
    tmp=dir.x*(double)q.m.vert[vi]+dir.y*(double)q.m.vert[vi+1]+dir.z*(double)q.m.vert[vi+2];
    int change=1,steps=0;
    while(change){
      if(++steps>q.watchdog){q.reason=6;return v(0,0,0);}
      change=0;int i=edgeadr[best],loc;
      while((loc=edge[i])>=0){
        vi=3*(adr+global[loc]);
        vdot=dir.x*(double)q.m.vert[vi]+dir.y*(double)q.m.vert[vi+1]+dir.z*(double)q.m.vert[vi+2];
        if(vdot>tmp){tmp=vdot;best=loc;change=1;}
        i++;
      }
    }
    local_index=best;best=global[best];
  }
  if(best<0){q.reason=8;return v(0,0,0);}
  int vi=3*(adr+best);
  V res=v((double)q.m.vert[vi],(double)q.m.vert[vi+1],(double)q.m.vert[vi+2]);
  // Preserve MuJoCo's (dir * margin) / 2 association separately per axis.
  res.x+=dir.x*q.pose[0]/2;res.y+=dir.y*q.pose[0]/2;res.z+=dir.z*q.pose[0]/2;
  return add(rot(res,mat),load3(p));
}
static __device__ S support(Query &q,V dir){
  int a=-1,b=-1;S r;
  r.v1=mesh_support(q,0,dir,a);
  r.v2=mesh_support(q,1,scale(dir,-1.0),b);
  r.v=sub(r.v1,r.v2);
  if(q.n_support<256){q.support_ids[3*q.n_support]=a;q.support_ids[3*q.n_support+1]=b;
    q.support_ids[3*q.n_support+2]=q.stage;}
  else q.reason=7;
  q.n_support++;
  q.last_dir=dir;q.last_support=r;
  return r;
}
static __device__ __forceinline__ V portal_dir(const Portal &p){
  return norm(cross(sub(p.p[2].v,p.p[1].v),sub(p.p[3].v,p.p[1].v)));
}
static __device__ __forceinline__ bool encapsulates(const Portal &p,V dir){
  double x=dot(dir,p.p[1].v);return zero(x)||x>0.0;
}
static __device__ __forceinline__ bool can_encapsulate(S s,V dir){
  double x=dot(s.v,dir);return zero(x)||x>0.0;
}
static __device__ __forceinline__ bool reach_tolerance(const Portal &p,S s,V dir,double tol){
  double d1=dot(p.p[1].v,dir),d2=dot(p.p[2].v,dir),d3=dot(p.p[3].v,dir),d4=dot(s.v,dir);
  double x=d4-d1,y=d4-d2,z=d4-d3;
  x=fmin(x,y);x=fmin(x,z);
  return eq(x,tol)||x<tol;
}
static __device__ __forceinline__ void expand(Portal &p,S s){
  V v4v0=cross(s.v,p.p[0].v);
  double d=dot(p.p[1].v,v4v0);
  if(d>0.0){d=dot(p.p[2].v,v4v0);if(d>0.0)p.p[1]=s;else p.p[3]=s;}
  else{d=dot(p.p[3].v,v4v0);if(d>0.0)p.p[2]=s;else p.p[1]=s;}
}

// Return -1 for separation, 1/2 for touch/segment, 0 for full portal.
static __device__ int discover(Query &q,Portal &p){
  q.stage=1;
  V va,vb,dir;double d;int cont;
  p.p[0].v1=load3(q.pose+1);p.p[0].v2=load3(q.pose+13);
  p.p[0].v=sub(p.p[0].v1,p.p[0].v2);p.size=1;
  if(veq(p.p[0].v,v(0,0,0)))p.p[0].v=add(p.p[0].v,v(DBL_EPSILON*10.0,0,0));
  dir=norm(scale(p.p[0].v,-1.0));
  p.p[1]=support(q,dir);p.size=2;
  if(q.reason)return -2;
  d=dot(p.p[1].v,dir);if(zero(d)||d<0.0){q.reason=10;return -1;}
  dir=cross(p.p[0].v,p.p[1].v);
  if(zero(dot(dir,dir))){
    if(veq(p.p[1].v,v(0,0,0))){q.reason=11;return 1;}
    q.reason=12;return 2;
  }
  dir=norm(dir);p.p[2]=support(q,dir);if(q.reason)return -2;
  d=dot(p.p[2].v,dir);if(zero(d)||d<0.0){q.reason=13;return -1;}
  p.size=3;
  va=sub(p.p[1].v,p.p[0].v);vb=sub(p.p[2].v,p.p[0].v);
  dir=norm(cross(va,vb));d=dot(dir,p.p[0].v);
  if(d>0.0){S tmp=p.p[1];p.p[1]=p.p[2];p.p[2]=tmp;dir=scale(dir,-1.0);}
  while(p.size<4){
    if(++q.discover>q.watchdog){q.reason=6;return -2;}
    p.p[3]=support(q,dir);if(q.reason)return -2;
    d=dot(p.p[3].v,dir);if(zero(d)||d<0.0){q.reason=14;return -1;}
    cont=0;
    va=cross(p.p[1].v,p.p[3].v);d=dot(va,p.p[0].v);
    if(d<0.0&&!zero(d)){p.p[2]=p.p[3];cont=1;}
    if(!cont){va=cross(p.p[3].v,p.p[2].v);d=dot(va,p.p[0].v);
      if(d<0.0&&!zero(d)){p.p[1]=p.p[3];cont=1;}}
    if(cont){va=sub(p.p[1].v,p.p[0].v);vb=sub(p.p[2].v,p.p[0].v);
      dir=norm(cross(va,vb));}
    else p.size=4;
  }
  return 0;
}
static __device__ int refine(Query &q,Portal &p){
  q.stage=2;
  for(;;){
    if(++q.refine>q.watchdog){q.reason=6;return -2;}
    V dir=portal_dir(p);
    if(encapsulates(p,dir)){q.reason=20;return 0;}
    S s=support(q,dir);if(q.reason)return -2;
    if(!can_encapsulate(s,dir)||reach_tolerance(p,s,dir,q.m.tolerance)){
      q.reason=21;return -1;
    }
    expand(p,s);
  }
}

static __device__ double segment_dist2(V P,V A,V B,V &witness){
  V d=sub(B,A),a=sub(A,P);
  double t=-1.0*dot(a,d);t/=dot(d,d);
  if(t<0.0||zero(t)){witness=A;return dot(sub(A,P),sub(A,P));}
  if(t>1.0||eq(t,1.0)){witness=B;return dot(sub(B,P),sub(B,P));}
  witness=add(scale(d,t),A);V diff=sub(witness,P);return dot(diff,diff);
}
// Pinned libccd vec3.c:137-216, retaining branch tests and strict tie order.
static __device__ double triangle_dist2(V P,V A,V B,V C,V &witness){
  V d1=sub(B,A),d2=sub(C,A),a=sub(A,P);
  double u=dot(a,a),vv=dot(d1,d1),w=dot(d2,d2),pp=dot(a,d1),q=dot(a,d2),r=dot(d1,d2);
  double det=w*vv-r*r,s,t,dist,dist2;V witness2;
  if(zero(det))s=t=-1.0;
  else{s=(q*r-w*pp)/det;t=(-s*r-q)/w;}
  if((zero(s)||s>0.0)&&(eq(s,1.0)||s<1.0)&&
     (zero(t)||t>0.0)&&(eq(t,1.0)||t<1.0)&&
     (eq(t+s,1.0)||t+s<1.0)){
    witness=add(add(A,scale(d1,s)),scale(d2,t));
    V delta=sub(witness,P);dist=dot(delta,delta);
  }else{
    dist=segment_dist2(P,A,B,witness);
    dist2=segment_dist2(P,A,C,witness2);if(dist2<dist){dist=dist2;witness=witness2;}
    dist2=segment_dist2(P,B,C,witness2);if(dist2<dist){dist=dist2;witness=witness2;}
  }
  (void)u;return dist;
}
static __device__ V find_pos(const Portal &p){
  V dir=portal_dir(p),vec;
  double b[4],sum,inv;
  vec=cross(p.p[1].v,p.p[2].v);b[0]=dot(vec,p.p[3].v);
  vec=cross(p.p[3].v,p.p[2].v);b[1]=dot(vec,p.p[0].v);
  vec=cross(p.p[0].v,p.p[1].v);b[2]=dot(vec,p.p[3].v);
  vec=cross(p.p[2].v,p.p[1].v);b[3]=dot(vec,p.p[0].v);
  sum=b[0]+b[1]+b[2]+b[3];
  if(zero(sum)||sum<0.0){
    b[0]=0.0;
    vec=cross(p.p[2].v,p.p[3].v);b[1]=dot(vec,dir);
    vec=cross(p.p[3].v,p.p[1].v);b[2]=dot(vec,dir);
    vec=cross(p.p[1].v,p.p[2].v);b[3]=dot(vec,dir);
    sum=b[1]+b[2]+b[3];
  }
  inv=1.0/sum;V p1=v(0,0,0),p2=v(0,0,0);
  for(int i=0;i<4;i++){p1=add(p1,scale(p.p[i].v1,b[i]));p2=add(p2,scale(p.p[i].v2,b[i]));}
  p1=scale(p1,inv);p2=scale(p2,inv);
  return scale(add(p1,p2),0.5);
}
static __device__ void find_penetration(Query &q,Portal &p,double &depth,V &dir,V &pos){
  q.stage=3;
  int iter=0;
  for(;;){
    if(++q.penetrate>q.watchdog){q.reason=6;return;}
    dir=portal_dir(p);S s=support(q,dir);if(q.reason)return;
    bool tolerance_hit=reach_tolerance(p,s,dir,q.m.tolerance);
    if(tolerance_hit||iter>q.m.maxiter){
      V witness;depth=triangle_dist2(v(0,0,0),p.p[1].v,p.p[2].v,p.p[3].v,witness);
      depth=sqrt(depth);
      dir=zero(depth)?v(0,0,0):norm(witness);
      pos=find_pos(p);
      q.reason=tolerance_hit?30:31;
      return;
    }
    expand(p,s);iter++;
  }
}
static __device__ int mpr(Query &q,double &depth,V &dir,V &pos){
  Portal p;int rc=discover(q,p);
  if(rc==-2)return -2;
  if(rc<0){q.portal=p;return -1;}
  if(rc==1){depth=0.0;dir=v(0,0,0);pos=scale(add(p.p[1].v1,p.p[1].v2),0.5);q.portal=p;return 0;}
  if(rc==2){pos=scale(add(p.p[1].v1,p.p[1].v2),0.5);
    dir=p.p[1].v;depth=sqrt(dot(dir,dir));dir=norm(dir);q.portal=p;return 0;}
  rc=refine(q,p);if(rc==-2)return -2;
  if(rc<0){q.portal=p;return -1;}
  q.reason=0;
  find_penetration(q,p,depth,dir,pos);q.portal=p;
  return q.reason==6||q.reason==7||q.reason==8?-2:0;
}

__global__ void mesh_mpr_kernel(Model model,const double *pose,const int *pair,
    const int *active,const int *entered,int n,double *raw,int *out_count,
    int *status,int *diag,int *supports,double *portal_out){
  int w=blockIdx.x*blockDim.x+threadIdx.x;if(w>=n)return;
  if(active[w]==0)return;
  if(active[w]!=1){status[w]=1;return;}
  if(entered[w]<0||entered[w]>1){status[w]=2;return;}
  if(entered[w]==0){status[w]=0;return;}
  int a=pair[2*w],b=pair[2*w+1];
  if(a<0||b<0||a>=model.ngeom||b>=model.ngeom||
     model.geomtype[a]!=7||model.geomtype[b]!=7||
     model.dataid[a]<0||model.dataid[b]<0||
     model.dataid[a]>=model.nmesh||model.dataid[b]>=model.nmesh){status[w]=3;return;}
  const double *x=pose+25*w;
  for(int i=0;i<25;i++)if(!isfinite(x[i])){status[w]=4;return;}
  if(model.maxiter<0||!isfinite(model.tolerance)||model.tolerance<0.0){status[w]=5;return;}
  Query q{};q.m=model;q.pose=x;q.g1=a;q.g2=b;q.world=w;
  q.support_ids=supports+768*w;q.watchdog=4096;
  double depth=0.0;V direction=v(0,0,0),position=v(0,0,0);
  int rc=mpr(q,depth,direction,position);
  if(rc==-2){status[w]=q.reason==7?7:q.reason==8?8:6;return;}
  int count=rc==0&&!veq(direction,v(0,0,0))?1:0;
  if(count){
    double *y=raw+10*w;y[0]=x[0]-depth;write3(y+1,position);write3(y+4,direction);
    y[7]=0.0;y[8]=0.0;y[9]=0.0;
  }
  out_count[w]=count;status[w]=0;
  int *d=diag+12*w;
  d[0]=q.n_support;d[1]=q.discover;d[2]=q.refine;d[3]=q.penetrate;
  d[4]=q.reason;d[5]=rc;d[6]=q.portal.size;
  d[7]=q.g1;d[8]=q.g2;d[9]=q.m.dataid[a];d[10]=q.m.dataid[b];d[11]=0;
  double *po=portal_out+36*w;
  for(int i=0;i<q.portal.size;i++){
    write3(po+9*i,q.portal.p[i].v);
    write3(po+9*i+3,q.portal.p[i].v1);
    write3(po+9*i+6,q.portal.p[i].v2);
  }
}

extern "C" int launch_mesh_mpr(const float *vert,const int *graph,const int *vertadr,
  const int *vertnum,const int *graphadr,const int *dataid,const int *geomtype,
  int nvert,int ngraph,int nmesh,int ngeom,int maxiter,double tolerance,
  const double *pose,const int *pair,const int *active,const int *entered,int n,
  double *raw,int *count,int *status,int *diag,int *supports,double *portal,
  cudaStream_t stream){
  Model m{vert,graph,vertadr,vertnum,graphadr,dataid,geomtype,
          nvert,ngraph,nmesh,ngeom,maxiter,tolerance};
  mesh_mpr_kernel<<<(n+63)/64,64,0,stream>>>(m,pose,pair,active,entered,n,
      raw,count,status,diag,supports,portal);
  return (int)cudaGetLastError();
}
