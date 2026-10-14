// Handwritten MuJoCo impedance passes; private stage outputs only.
#pragma once
#ifndef CF_S_BATCH_COMPACT
#define CF_S_BATCH_COMPACT 0
#endif
#ifndef CF_S_GROUP_PARAMETER_REUSE
#define CF_S_GROUP_PARAMETER_REUSE 0
#endif
namespace cf_s {
// Preserve the source's deterministic first rejection before parallel math.
// The only new writes are private group flags; public output is transactional.
// Leader parses group boundaries; a warp checks independent groups in order.
// Publish flags only through the first rejected group, matching scalar history.
__device__ void CF_S_impedance_validate {
  __shared__ int head[32],type[32],ident[32],width[32],structural[32];
  __shared__ int cursor,used,finished,boundary;
  if(threadIdx.x==0){
    status[w]=0;cursor=0;finished=0;boundary=0;
#if CF_S_BATCH_COMPACT
    compact_count[w]=0;compact_valid[w]=0;
#endif
    if(!active[w])finished=1;
    else if(active[w]!=1){status[w]=1;finished=1;}
    else if(count[w]<0||count[w]>rows||ncon[w]<0||ncon[w]>contacts||ne[w]<0||nf[w]<0||ne[w]+nf[w]>count[w]){status[w]=2;finished=1;}
  }
  __syncthreads();
  const int n=count[w],nc=ncon[w],rb=w*rows,cb=w*contacts;
  while(!finished){
    if(threadIdx.x==0){
      used=0;
      while(used<32&&cursor<n){
        const int i=cursor,k=used++,t=typ[rb+i],id=eid[rb+i];
        int d=1,code=0;
        if(i==ne[w]+nf[w])boundary=1;
        if(t==1){if(id<0||id>=nv)code=3;}
        else if(t==3){if(id<0||id>=njnt)code=3;}
        else if(t==5||t==7){
          if(id<0||id>=nc)code=3;
          else if(cdim[cb+id]<1||cdim[cb+id]>6||(t==7&&cdim[cb+id]<2))code=4;
          else if(t==7)d=cdim[cb+id];
        }else code=3;
        if(!code&&i+d>n)code=4;
        head[k]=i;type[k]=t;ident[k]=id;width[k]=d;structural[k]=code;
        if(code)break;
        cursor+=d;
      }
    }
    __syncthreads();
    if(threadIdx.x<32){
      const int k=threadIdx.x;
      int code=k<used?structural[k]:0;
      if(k<used&&!code){
        const int i=head[k],t=type[k],id=ident[k],d=width[k];
        for(int j=0;j<d;j++){
          const int q=rb+i+j;
          if(typ[q]!=t||eid[q]!=id||!isfinite(prediag[q])||!isfinite(pos[q])||!isfinite(margin[q])){code=4;break;}
        }
        if(!code){
          const double *ref=t==1?dof_ref+id*2:t==3?jnt_ref+id*2:cref+(cb+id)*2;
          const double *imp=t==1?dof_imp+id*5:t==3?jnt_imp+id*5:cimp+(cb+id)*5;
          for(int j=0;j<2;j++)if(!isfinite(ref[j]))code=5;
          if(t==5||t==7)for(int j=0;j<2;j++)if(!isfinite(creff[(cb+id)*2+j]))code=5;
          for(int j=0;j<5;j++)if(!isfinite(imp[j]))code=5;
          if(t==5||t==7)for(int j=0;j<5;j++)if(!isfinite(cfric[(cb+id)*5+j]))code=5;
          if(!code&&imp[4]>1.&&imp[4]!=2.)code=6;
          if(!code&&t==7)for(int j=1;j<d-1;j++)if(cfric[(cb+id)*5+j]==0.)code=5;
        }
      }
      const int first=libero_warp::min(code?k:libero_warp::no_index);
      const int winner=first==libero_warp::no_index?0:first;
      const int rejection=__shfl_sync(libero_warp::mask,code,winner);
      const int end=first==libero_warp::no_index?used:first+1;
      if(k<end&&!structural[k])
        for(int j=0;j<width[k];j++)group_heads[rb+head[k]+j]=j==0;
#if CF_S_BATCH_COMPACT
      // Parsed groups already have stable ordinals. Reuse the accepted batch
      // instead of re-scanning every per-row head flag after validation.
      const int offset=compact_count[w];
      if(k<end&&!structural[k]){
        const int q=rb+offset+k;
        compact_heads[q]=head[k];head_type[q]=type[k];
        head_id[q]=ident[k];head_dim[q]=width[k];
      }
      __syncwarp(libero_warp::mask);
      if(k==0)compact_count[w]=offset+(first==libero_warp::no_index?used:first);
#endif
      if(k==0){
        if(first!=libero_warp::no_index){status[w]=rejection;finished=1;}
        else if(cursor==n){
          if(ne[w]+nf[w]==n)boundary=1;
          if(!boundary)status[w]=4;
          finished=1;
        }
      }
    }
    __syncthreads();
  }
#if CF_S_BATCH_COMPACT
  if(threadIdx.x==0){
    if(status[w]||active[w]!=1)compact_count[w]=0;
    else compact_valid[w]=1;
  }
  __syncthreads();
#endif
}
__device__ void CF_S_impedance_prepare {
  CF_S_CALL_impedance_validate;
  __syncthreads();
#if !CF_S_BATCH_COMPACT
  if(threadIdx.x==0){compact_count[w]=0;compact_valid[w]=0;}
  __syncthreads();
#endif
  if(status[w]||active[w]!=1)return;
  int n=count[w],rb=w*rows,cb=w*contacts;
  // This private list is rebuilt from this invocation's validated row owners;
  // no epoch-based guess permits stale topology reuse.
#if !CF_S_BATCH_COMPACT
  if(threadIdx.x<32){
    const int lane=threadIdx.x;
    for(int base=0;base<n;base+=32){
      const int i=base+lane,keep=i<n&&group_heads[rb+i];
      const int rank=libero_warp::exclusive_sum(keep);
      const int total=libero_warp::sum(keep),offset=compact_count[w];
      if(keep){
        const int q=rb+offset+rank,t=typ[rb+i],id=eid[rb+i];
        compact_heads[q]=i;head_type[q]=t;head_id[q]=id;
        head_dim[q]=t==7?cdim[cb+id]:1;
      }
      __syncwarp(libero_warp::mask);
      if(lane==0)compact_count[w]=offset+total;
      __syncwarp(libero_warp::mask);
    }
    if(lane==0)compact_valid[w]=1;
  }
  __syncthreads();
#endif
  for(int c=threadIdx.x;c<ncon[w];c+=blockDim.x)mu_owner[cb+c]=-1;
  __syncthreads();
  for(int ordinal=threadIdx.x;ordinal<compact_count[w];ordinal+=blockDim.x){
    const int i=compact_heads[rb+ordinal];
    if(head_type[rb+ordinal]==7)atomicMax(mu_owner+cb+head_id[rb+ordinal],i);
  }
  __syncthreads();
 }
__device__ void CF_S_impedance_math {
  if(status[w]||active[w]!=1||compact_valid[w]!=1)return;
  const int n=count[w],rb=w*rows,cb=w*contacts;
#if CF_S_GROUP_TILES
  const int begin=tile*CF_S_GROUP_TILE;
  const int end=min(begin+CF_S_GROUP_TILE,compact_count[w]);
#else
  const int begin=0,end=compact_count[w];
#endif
  for(int ordinal=begin+threadIdx.x;ordinal<end;ordinal+=blockDim.x){
    const int i=compact_heads[rb+ordinal];
    int t=head_type[rb+ordinal],id=head_id[rb+ordinal],d=head_dim[rb+ordinal];
    const double *ref=t==1?dof_ref+id*2:t==3?jnt_ref+id*2:cref+(cb+id)*2;
    const double *ip=t==1?dof_imp+id*5:t==3?jnt_imp+id*5:cimp+(cb+id)*5;
    // Validated record fields only; five-double record alignment alternates.
    const double2 rv=libero_vec::load_pair(ref);
    const double2 ip01=libero_vec::load_pair(ip),ip23=libero_vec::load_pair(ip+2);
    double r0=rv.x,r1=rv.y,f0=0.,f1=0.;
    if(t==5||t==7){const double2 fv=libero_vec::load_pair(creff+(cb+id)*2);f0=fv.x;f1=fv.y;}
    if((r0>0.)!=(r1>0.)){r0=.02;r1=1.;}
    if(refsafe&&r0>0.)r0=fmax(r0,2.*timestep);
    if((f0>0.)!=(f1>0.))f0=f1=0.;
    if(refsafe&&f0>0.)f0=fmax(f0,2.*timestep);
    double p0=fmin(.9999,fmax(.0001,ip01.x)),p1=fmin(.9999,fmax(.0001,ip01.y));
    double width=fmax(0.,ip23.x),mid=fmin(.9999,fmax(.0001,ip23.y)),power=fmax(1.,ip[4]);
    double imp=0.,impP=0.;int branch=0;
    if(p0==p1||width<=1e-15)imp=.5*(p0+p1);
    else{
      double x=(pos[rb+i]-margin[rb+i])/width,sgn=1.;if(x<0.){x=-x;sgn=-1.;}
      if(x>=1.){imp=p1;branch=1;}
      else if(x<=0.){imp=p0;branch=2;}
      else{
        double y,yP;
        if(power==1.){y=x;yP=1.;branch=3;}
        else if(power==2.){
          if(x<=mid){double a=1./mid;y=a*(x*x);yP=power*a*x;branch=4;}
          else{double z=1.-x,b=1./(1.-mid);y=1.-b*(z*z);yP=power*b*z;branch=5;}
        }else{
#if CF_S_GROUP_TILES
          atomicMin(group_math_error+w,ordinal);return;
#else
          status[w]=6;return;
#endif
        }
        imp=p0+y*(p1-p0);impP=yP*sgn*(p1-p0)/width;
      }
    }
    // One lane owns the complete contact (<=6 rows). R adjustment, reciprocal
    // and postdiag use register values instead of additional global passes.
    double local_R[6];
#if CF_S_GROUP_PARAMETER_REUSE
    // Validation proves all rows in this group have the same owner. Only R0
    // survives the elliptic adjustment; tangential prediag is still validated.
    local_R[0]=fmax(1e-15,(1.-imp)*prediag[rb+i]/imp);
    const double group_k=t!=1?(r0>0.?1./fmax(1e-15,p1*p1*r0*r0*r1*r1):-r0/fmax(1e-15,p1*p1)):0.;
    const double normal_b=r1>0.?2./fmax(1e-15,p1*r0):-r1/fmax(1e-15,p1);
    double tangent_b=normal_b;
    if(t==7&&(f0!=0.||f1!=0.))
      tangent_b=f1>0.?2./fmax(1e-15,p1*f0):-f1/fmax(1e-15,p1);
#endif
    for(int j=0;j<d;j++){
#if CF_S_GROUP_PARAMETER_REUSE
      const int q=rb+i+j;const bool friction=t==7&&j>0;
      route[q]=branch;
      const double k=friction?0.:group_k,b=friction?tangent_b:normal_b;
#else
      int q=rb+i+j;bool friction=typ[q]==7&&j>0;double rr0=r0,rr1=r1;
      route[q]=branch;local_R[j]=fmax(1e-15,(1.-imp)*prediag[q]/imp);
      if(friction&&(f0!=0.||f1!=0.)){rr0=f0;rr1=f1;}
      double k=0.;
      if(typ[q]!=1&&!friction)k=rr0>0.?1./fmax(1e-15,p1*p1*rr0*rr0*rr1*rr1):-rr0/fmax(1e-15,p1*p1);
      double b=rr1>0.?2./fmax(1e-15,p1*rr0):-rr1/fmax(1e-15,p1);
#endif
      #if CF_S_KBIP_PLANES
      KBIP[(w*4+0)*rows+i+j]=k;KBIP[(w*4+1)*rows+i+j]=b;
      KBIP[(w*4+2)*rows+i+j]=imp;KBIP[(w*4+3)*rows+i+j]=impP;
#else
      libero_vec::store_pair(KBIP+q*4,k,b);
      libero_vec::store_pair(KBIP+q*4+2,imp,impP);
#endif
    }
    if(t==7){
      int fbase=(cb+id)*5;
      double friction[5];
      // Read only the prefix used by the cone (validated d<=6).
      const int used=d>2?d-1:(mu_owner[cb+id]==i?1:0);int fj=0;
      for(;fj+1<used;fj+=2){const double2 f=libero_vec::load_pair(cfric+fbase+fj);friction[fj]=f.x;friction[fj+1]=f.y;}
      if(fj<used)friction[fj]=cfric[fbase+fj];
      local_R[1]=local_R[0]/fmax(1e-15,impratio);
      if(mu_owner[cb+id]==i)mu[cb+id]=friction[0]*sqrt(local_R[1]/local_R[0]);
      for(int j=1;j<d-1;j++){
        double f=friction[j];
        local_R[j+1]=local_R[1]*friction[0]*friction[0]/(f*f);
      }
    }
    int j=0;
    for(;j+1<d;j+=2){
      const int q=rb+i+j;const double rr=local_R[j],rr1=local_R[j+1];
      const double dd=1./rr,pp=rr*imp/(1.-imp);
      const double dd1=1./rr1,pp1=rr1*imp/(1.-imp);
      libero_vec::store_pair(R+q,rr,rr1);
      libero_vec::store_pair(D+q,dd,dd1);
      libero_vec::store_pair(post+q,pp,pp1);
    }
    if(j<d){const int q=rb+i+j;const double rr=local_R[j];R[q]=rr;D[q]=1./rr;post[q]=rr*imp/(1.-imp);}
  }
}
__device__ void CF_S_impedance {
  CF_S_CALL_impedance_prepare;
  __syncthreads();
  CF_S_CALL_impedance_math;
}
__device__ void impedance_math_reset(int w,int32_t* error){if(threadIdx.x==0)error[w]=2147483647;}
__device__ void impedance_math_merge(int w,const int32_t* error,int32_t* status){
  if(threadIdx.x==0&&error[w]!=2147483647&&status[w]==0)status[w]=6;
}
}
