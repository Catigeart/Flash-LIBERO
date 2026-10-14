#pragma once
#include <math.h>
#include <stdint.h>

#ifdef D_FAST_DATAFLOW
// One block owns one world. These are private working values, never public
// aliases; their lifetime extends across the force-chain device operations.
struct DForceShared {
    double scalar_force[D_NU>0?D_NU:1];
    double actuator[D_NV],smooth[D_NV],seed[D_NV],scratch[D_NV],force[D_NV];
    double rhs[D_NV],acceleration[D_NV],ld[D_NM],inv[D_NV],sqrt_inv[D_NV];
    int pivot[D_NV];
};
__device__ __forceinline__ DForceShared& d_force_shared() {
    __shared__ DForceShared work;
    return work;
}
#endif

// Handwritten sparse rigid-body arithmetic. These routines receive whole
// allocations; only the selected world row is touched. The caller executes
// each collective tree routine on all threads. Its ordered tree arithmetic
// remains on lane zero; array checks and independent projections cooperate.
__device__ inline bool d_finite(const double* x, int n) {
    for (int k=0;k<n;++k) if (!isfinite(x[k])) return false;
    return true;
}
// Collective helpers: every thread in the world block must enter. Warp votes
// exchange predicates only; the barriers publish and protect inter-warp flags.
__device__ inline bool d1_finite_block(const double* x, int n,
    const double* y=nullptr,int ny=0,const double* z=nullptr,int nz=0,
    const double* u=nullptr,int nu=0,const double* v=nullptr,int nv=0) {
    __shared__ int warp_bad[32];
    bool bad=false;
    bad |= libero_vec::any_nonfinite_block_lane(x,n);
    bad |= libero_vec::any_nonfinite_block_lane(y,ny);
    bad |= libero_vec::any_nonfinite_block_lane(z,nz);
    bad |= libero_vec::any_nonfinite_block_lane(u,nu);
    bad |= libero_vec::any_nonfinite_block_lane(v,nv);
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    // Membership comes from block geometry, not the instantaneous active
    // lanes after unequal per-thread strides. Include every live warp lane.
    const int remaining=(int)blockDim.x-(warp<<5);
    const unsigned mask=remaining>=32?0xffffffffu:((1u<<remaining)-1u);
    const int vote=__any_sync(mask,bad);
    if(lane==0)warp_bad[warp]=vote;
    __syncthreads();
    bool any=false;
    for(int k=0;k<(blockDim.x+31)/32;++k)any |= warp_bad[k]!=0;
    __syncthreads();
    return !any;
}
__device__ inline void d1_zero_block(double* x,int n) {
    libero_vec::fill_block(x,n,0.0);
}
__device__ inline void d_zero(double* x,int n) {
    for(int k=0;k<n;++k) x[k]=0.0;
}
__device__ inline void d_gate(int w,int*active,int*error,int64_t*state,
    int64_t*source,int64_t*epoch,int*valid,int64_t*written,int*gate,int require) {
    gate[w]=0;
    if(active[w]!=1 || error[w]) return;
    if(state[w]!=source[w]) {error[w]=203;return;}
    if(require==1 && (valid[w]!=1 || written[w]!=epoch[w])) return;
    gate[w]=1;
}
__device__ inline void d_finish(int w,int*gate,int*status,int64_t*epoch,
    int*valid,int64_t*written,int*error,int repair,int code) {
    valid[w]=0;
    if(gate[w]!=1 || error[w]) return;
    if(status[w]!=0 && !(repair==1 && status[w]==25)){error[w]=code;return;}
    written[w]=epoch[w];valid[w]=1;
}
__device__ inline void d_reference_gate(int w,int*done,int*err,int64_t*written,
    int64_t*expected,int64_t*epoch,int*gate,int*first) {
    if(gate[w]==1 && (done[w]!=1 || err[w] || written[w]!=expected[w] || expected[w]!=epoch[w])) {
        gate[w]=0;if(!first[w]) first[w]=208;
    }
}
__device__ inline void d_velocity_status(int w,int*a,int*b,int*c,int*out) {
    out[w]=a[w]?100+a[w]:(b[w]?200+b[w]:(c[w]?300+c[w]:0));
}
__device__ inline void d_force_gate(int w,int*valid,int*gate) {if(valid[w]!=1)gate[w]=0;}
__device__ inline void d_actuation_finish(int w,int*gate,int*status,int*first) {
    if(gate[w]==1 && status[w]){gate[w]=0;if(!first[w])first[w]=214;}
}
__device__ inline void d_copy_admitted(int w,int*gate,double*source,double*target,int width) {
    if(gate[w]==1)for(int i=0;i<width;++i)target[w*width+i]=source[w*width+i];
}
__device__ inline void d_commit_bias(int w,int*active,int*gate,int*status,
    int64_t*epoch,int*error,double*work,double*pub,int*valid,int64_t*written,int width) {
    valid[w]=0;
    if(active[w]!=1 || gate[w]!=1 || error[w])return;
    if(status[w]){error[w]=211;return;}
    for(int i=0;i<width;++i)pub[w*width+i]=work[w*width+i];
    written[w]=epoch[w];valid[w]=1;
}
__device__ inline void d_commit_smooth(int w,int*active,int*gate,int*status,
    int64_t*epoch,int*error,double*wf,double*wa,double*pf,double*pa,
    int*valid,int64_t*written,int*repaired,int width) {
    valid[w]=0;repaired[w]=0;
    if(active[w]!=1 || gate[w]!=1 || error[w])return;
    if(status[w] && status[w]!=25){error[w]=213;return;}
    for(int i=0;i<width;++i){pf[w*width+i]=wf[w*width+i];pa[w*width+i]=wa[w*width+i];}
    written[w]=epoch[w];repaired[w]=(status[w]==25);valid[w]=1;
}
// Collective publication: callers enter with the complete world block. Gate
// decisions and scalar flags retain lane-zero order; payload stores complete
// before the epoch becomes visible. Shifted overlaps retain serial copy order.
__device__ inline bool d_copy_overlap(const double* a,const double* b,int width) {
    const uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    const uintptr_t bytes=uintptr_t(width)*sizeof(double);
    return x<y+bytes && y<x+bytes;
}
__device__ inline void d_copy_admitted_block(int w,int*gate,double*source,double*target,int width) {
    __shared__ int admitted,serial;
    double* s=source+(long long)w*width,*t=target+(long long)w*width;
    if(threadIdx.x==0){admitted=gate[w]==1;serial=d_copy_overlap(s,t,width);}
    __syncthreads();
    if(admitted){
        if(serial){if(threadIdx.x==0)for(int i=0;i<width;++i)t[i]=s[i];}
        else libero_vec::copy_block(s,t,width);
    }
    __syncthreads();
}
__device__ inline void d_commit_bias_block(int w,int*active,int*gate,int*status,
    int64_t*epoch,int*error,double*work,double*pub,int*valid,int64_t*written,int width) {
    __shared__ int admitted,serial;
    double* s=work+(long long)w*width,*t=pub+(long long)w*width;
    if(threadIdx.x==0){
        valid[w]=0;admitted=active[w]==1 && gate[w]==1 && !error[w];
        if(admitted && status[w]){error[w]=211;admitted=0;}
        serial=d_copy_overlap(s,t,width);
    }
    __syncthreads();
    if(admitted){
        if(serial){if(threadIdx.x==0)for(int i=0;i<width;++i)t[i]=s[i];}
        else libero_vec::copy_block(s,t,width);
    }
    __syncthreads();
    if(threadIdx.x==0 && admitted){written[w]=epoch[w];valid[w]=1;}
    __syncthreads();
}
__device__ inline void d_commit_smooth_block(int w,int*active,int*gate,int*status,
    int64_t*epoch,int*error,double*wf,double*wa,double*pf,double*pa,
    int*valid,int64_t*written,int*repaired,int width) {
    __shared__ int admitted,serial;
    wf+=(long long)w*width;wa+=(long long)w*width;
    pf+=(long long)w*width;pa+=(long long)w*width;
    if(threadIdx.x==0){
        valid[w]=0;repaired[w]=0;admitted=active[w]==1 && gate[w]==1 && !error[w];
        if(admitted && status[w] && status[w]!=25){error[w]=213;admitted=0;}
        serial=d_copy_overlap(wf,pf,width)||d_copy_overlap(wf,pa,width)||
               d_copy_overlap(wa,pf,width)||d_copy_overlap(wa,pa,width)||d_copy_overlap(pf,pa,width);
    }
    __syncthreads();
    if(admitted){
        if(serial){if(threadIdx.x==0)for(int i=0;i<width;++i){pf[i]=wf[i];pa[i]=wa[i];}}
        else {libero_vec::copy_block(wf,pf,width);libero_vec::copy_block(wa,pa,width);}
    }
    __syncthreads();
    if(threadIdx.x==0 && admitted){written[w]=epoch[w];repaired[w]=status[w]==25;valid[w]=1;}
    __syncthreads();
}
__device__ inline void d_scalar_transmission_velocity(int w,double*moment,
    double*qvel,int*active,int*position_status,int nv,int nu,double*velocity,int*status) {
    double*v=velocity+w*nu;const double*q=qvel+w*nv;const double*m=moment+w*nu*nv;
    status[w]=0;d_zero(v,nu);
    if(!active[w])return;if(active[w]!=1){status[w]=1;return;}
    if(position_status[w]){status[w]=3;return;}
    if(!d_finite(q,nv)){status[w]=2;return;}
    for(int a=0;a<nu;++a){
        double s0=0,s1=0,s2=0,s3=0;
        int j=0;for(;j+3<nv;j+=4){s0+=m[a*nv+j]*q[j];s1+=m[a*nv+j+1]*q[j+1];s2+=m[a*nv+j+2]*q[j+2];s3+=m[a*nv+j+3]*q[j+3];}
        double x=(s0+s2)+(s1+s3);
        if(j<nv){double tail=m[a*nv+j]*q[j];for(++j;j<nv;++j)tail+=m[a*nv+j]*q[j];x+=tail;}
        if(!isfinite(x)){status[w]=4;return;}v[a]=x;
    }
}
#ifdef D_SCALAR_SUPPORT
__device__ inline void d_scalar_transmission_velocity_support(int w,double*moment,
    double*qvel,int*active,int*position_status,int nv,int nu,double*velocity,int*status) {
    if(threadIdx.x==0){status[w]=0;d_zero(velocity+(long long)w*nu,nu);}
    __syncthreads();
    if(active[w]!=1 || position_status[w]!=0) {
        if(threadIdx.x==0)d_scalar_transmission_velocity(w,moment,qvel,active,position_status,nv,nu,velocity,status);
        __syncthreads();return;
    }
    if(nv!=D_NV || nu!=D_NU) {
        if(threadIdx.x==0)d_scalar_transmission_velocity(w,moment,qvel,active,position_status,nv,nu,velocity,status);
        __syncthreads();return;
    }
    const double* q=qvel+(long long)w*nv;const double* m=moment+(long long)w*nu*nv;
    if(!d1_finite_block(q,nv)) {if(threadIdx.x==0)status[w]=2;__syncthreads();return;}
    const int support[D_NU>0?D_NU:1]=D_SCALAR_SUPPORT;
    bool bad=false;
    for(int z=threadIdx.x;z<nu*nv;z+=blockDim.x)
        if(z%nv!=support[z/nv])bad|=__double_as_longlong(m[z])!=0;
    if(__syncthreads_or(bad)) {
        if(threadIdx.x==0)d_scalar_transmission_velocity(w,moment,qvel,active,position_status,nv,nu,velocity,status);
    } else if(threadIdx.x==0)for(int a=0;a<nu;++a) {
        // Four dot accumulators and the final tail additions normalize zero.
        const double value=(0.0+m[a*nv+support[a]]*q[support[a]])+0.0;
        if(!isfinite(value)){status[w]=4;break;}velocity[(long long)w*nu+a]=value;
    }
    __syncthreads();
}
#endif

__device__ inline void d_cross_motion(const double*v,const double*a,double*out) {
    out[0]=-v[2]*a[1]+v[1]*a[2];out[1]=v[2]*a[0]-v[0]*a[2];out[2]=-v[1]*a[0]+v[0]*a[1];
    out[3]=-v[2]*a[4]+v[1]*a[5];out[4]=v[2]*a[3]-v[0]*a[5];out[5]=-v[1]*a[3]+v[0]*a[4];
    out[3]+=-v[5]*a[1]+v[4]*a[2];out[4]+=v[5]*a[0]-v[3]*a[2];out[5]+=-v[4]*a[0]+v[3]*a[1];
}
__device__ inline void d_add_three(double*v,const double*a,const double*q) {
    double t[6]={};for(int j=0;j<3;++j)if(q[j]!=0)for(int k=0;k<6;++k)t[k]+=a[6*j+k]*q[j];
    for(int k=0;k<6;++k)v[k]+=t[k];
}
__device__ inline void d1_comvel_body(int b,const double*a,const double*q,
    const int*bp,const int*ba,const int*bn,const int*dj,const int*jt,double*v,double*d) {
    for(int k=0;k<6;++k)v[6*b+k]=v[6*bp[b]+k];
    int j=0;while(j<bn[b]){
        int i=ba[b]+j,type=jt[dj[i]];
        if(type==0){d_add_three(v+6*b,a+6*i,q+i);j+=3;i=ba[b]+j;}
        if(type<=1){for(int k=0;k<3;++k)d_cross_motion(v+6*b,a+6*(i+k),d+6*(i+k));d_add_three(v+6*b,a+6*i,q+i);j+=3;}
        else {d_cross_motion(v+6*b,a+6*i,d+6*i);for(int k=0;k<6;++k)v[6*b+k]+=a[6*i+k]*q[i];++j;}
    }
}
// Qualified topology only: levels omit world body 0 and parent IDs precede
// children. A level barrier publishes every parent before its children read.
__device__ inline void d_comvel(int w,int*active,double*cdof,double*qvel,
    int*bp,int*ba,int*bn,int*dj,int*jt,int nb,int nv,double*cvel,double*cdot,int*status,
    const int*level_offsets=nullptr,const int*level_bodies=nullptr,int nlevels=0) {
    double*v=cvel+w*nb*6,*d=cdot+w*nv*6;const double*a=cdof+w*nv*6,*q=qvel+w*nv;
    if(threadIdx.x==0)status[w]=0;d1_zero_block(v,nb*6);d1_zero_block(d,nv*6);
    __syncthreads();
    if(!active[w])return;if(active[w]!=1){if(threadIdx.x==0)status[w]=1;return;}
    if(!d1_finite_block(a,nv*6,q,nv)){if(threadIdx.x==0)status[w]=2;return;}
    if(level_offsets && level_bodies && nlevels>0) {
        for(int l=0;l<nlevels;++l) {
            for(int k=level_offsets[l]+threadIdx.x;k<level_offsets[l+1];k+=blockDim.x)
                d1_comvel_body(level_bodies[k],a,q,bp,ba,bn,dj,jt,v,d);
            __syncthreads();
        }
    } else if(threadIdx.x==0) {
        for(int b=1;b<nb;++b)d1_comvel_body(b,a,q,bp,ba,bn,dj,jt,v,d);
    }
    __syncthreads();
    if(!d1_finite_block(v,nb*6,d,nv*6)){
        if(threadIdx.x==0)status[w]=3;d1_zero_block(v,nb*6);d1_zero_block(d,nv*6);
    }
    __syncthreads();
}
// Compressed spatial inertia multiplication, not a dense BLAS operand.
__device__ inline void d_inertia(const double*i,const double*v,double*f) {
    f[0]=i[0]*v[0]+i[3]*v[1]+i[4]*v[2]-i[8]*v[4]+i[7]*v[5];
    f[1]=i[3]*v[0]+i[1]*v[1]+i[5]*v[2]+i[8]*v[3]-i[6]*v[5];
    f[2]=i[4]*v[0]+i[5]*v[1]+i[2]*v[2]-i[7]*v[3]+i[6]*v[4];
    f[3]=i[8]*v[1]-i[7]*v[2]+i[9]*v[3];f[4]=i[6]*v[2]-i[8]*v[0]+i[9]*v[4];f[5]=i[7]*v[0]-i[6]*v[1]+i[9]*v[5];
}
__device__ inline void d_cross_force(const double*v,const double*t,double*x) {
    x[0]=-v[2]*t[1]+v[1]*t[2];x[1]=v[2]*t[0]-v[0]*t[2];x[2]=-v[1]*t[0]+v[0]*t[1];
    x[3]=-v[2]*t[4]+v[1]*t[5];x[4]=v[2]*t[3]-v[0]*t[5];x[5]=-v[1]*t[3]+v[0]*t[4];
    x[0]+=-v[5]*t[4]+v[4]*t[5];x[1]+=v[5]*t[3]-v[3]*t[5];x[2]+=-v[4]*t[3]+v[3]*t[4];
}
__device__ inline void d1_rne_body(int b,const double*d,const double*q,
    const double*i,const double*v,const int*bp,const int*ba,const int*bn,double*acc,double*f,int nb=0,bool component_planes=false) {
    double t[6]={};const int adr=ba[b],n=bn[b];
    if(n==1)for(int k=0;k<6;++k)t[k]=d[6*adr+k]*q[adr];
    else for(int j=0;j<n;++j)if(q[adr+j]!=0)
        for(int k=0;k<6;++k)t[k]+=d[6*(adr+j)+k]*q[adr+j];
    double local_acc[6];
    for(int k=0;k<6;++k){
        const int out=component_planes?k*nb+b:6*b+k;
        const int parent=component_planes?k*nb+bp[b]:6*bp[b]+k;
        local_acc[k]=acc[parent]+t[k];acc[out]=local_acc[k];
    }
    double fi[6],corr[6],cross[6];
    d_inertia(i+10*b,local_acc,fi);d_inertia(i+10*b,v+6*b,corr);
    d_cross_force(v+6*b,corr,cross);
    for(int k=0;k<6;++k)f[component_planes?k*nb+b:6*b+k]=fi[k]+cross[k];
}
__device__ inline void d1_rne_forward(const double*d,const double*q,const double*i,
    const double*v,const int*bp,const int*ba,const int*bn,const double*gravity,
    int disabled,int nb,double*acc,double*f,const int*lo,const int*lb,int nl,bool component_planes=false) {
    if(threadIdx.x==0 && !disabled)for(int k=0;k<3;++k)acc[component_planes?(3+k)*nb:3+k]=-gravity[k];
    __syncthreads();
    if(lo && lb && nl>0) {
        for(int l=0;l<nl;++l) {
            for(int k=lo[l]+threadIdx.x;k<lo[l+1];k+=blockDim.x)
                d1_rne_body(lb[k],d,q,i,v,bp,ba,bn,acc,f,nb,component_planes);
            __syncthreads();
        }
    } else if(threadIdx.x==0) {
        for(int b=1;b<nb;++b)d1_rne_body(b,d,q,i,v,bp,ba,bn,acc,f,nb,component_planes);
    }
    __syncthreads();
}
__device__ inline void d1_rne_reverse(const int*bp,int nb,double*f,
    const int*lo,const int*lb,int nl,const int*co,const int*ci) {
    if(lo && lb && co && ci && nl>0) {
        // One writer per parent; descending child IDs exactly reproduce the
        // original reverse-body addition order, including cancellation.
        for(int l=nl-1;l>=0;--l) {
            for(int k=lo[l]+threadIdx.x;k<lo[l+1];k+=blockDim.x) {
                const int b=lb[k];
                for(int a=co[b];a<co[b+1];++a)
                    for(int j=0;j<6;++j)f[6*b+j]+=f[6*ci[a]+j];
            }
            __syncthreads();
        }
    } else if(threadIdx.x==0) {
        for(int b=nb-1;b>0;--b)if(bp[b]>0)
            for(int k=0;k<6;++k)f[6*bp[b]+k]+=f[6*b+k];
    }
    __syncthreads();
}
__device__ inline void d_rne_bias(int w,int*active,double*cdot,double*qvel,
    double*cinert,double*cvel,double*cdof,int*bp,int*ba,int*bn,int*db,double*gravity,
    int disabled,int nb,int nv,double*cacc,double*frc,double*result,int*status,
    const int*level_offsets=nullptr,const int*level_bodies=nullptr,int nlevels=0,
    const int*children_offsets=nullptr,const int*children_indices=nullptr) {
    const double*d=cdot+w*nv*6,*q=qvel+w*nv,*i=cinert+w*nb*10,*v=cvel+w*nb*6,*a=cdof+w*nv*6;
    double*acc=cacc+w*nb*6,*f=frc+w*nb*6,*r=result+w*nv;
    if(threadIdx.x==0)status[w]=0;d1_zero_block(acc,nb*6);d1_zero_block(f,nb*6);d1_zero_block(r,nv);
    __syncthreads();
    if(!active[w])return;if(active[w]!=1){if(threadIdx.x==0)status[w]=1;return;}
#if defined(D_FUSED_RNE_FINITE) && D_FUSED_RNE_FINITE
    // Every admitted range has the same status 2 and no gate side effects.
    if(!d1_finite_block(d,nv*6,a,nv*6,q,nv,i,nb*10,v,nb*6)){if(threadIdx.x==0)status[w]=2;return;}
#else
    if(!d1_finite_block(d,nv*6)||!d1_finite_block(a,nv*6,q,nv)||!d1_finite_block(i,nb*10)||!d1_finite_block(v,nb*6)){if(threadIdx.x==0)status[w]=2;return;}
#endif
    d1_rne_forward(d,q,i,v,bp,ba,bn,gravity,disabled,nb,acc,f,
        level_offsets,level_bodies,nlevels);
    d1_rne_reverse(bp,nb,f,level_offsets,level_bodies,nlevels,
        children_offsets,children_indices);
    for(int j=threadIdx.x;j<nv;j+=blockDim.x){const double*aa=a+6*j,*ff=f+6*db[j];r[j]=((aa[0]*ff[0]+aa[2]*ff[2])+(aa[1]*ff[1]+aa[3]*ff[3]))+(aa[4]*ff[4]+aa[5]*ff[5]);}
    __syncthreads();
    if(!d1_finite_block(acc,nb*6,f,nb*6,r,nv)){
        if(threadIdx.x==0)status[w]=3;d1_zero_block(acc,nb*6);d1_zero_block(f,nb*6);d1_zero_block(r,nv);
    }
    __syncthreads();
}

// Compute only the forward part of bias RNE while the velocity input is live.
// Its private status does not decide velocity publication or the later S gate.
// The reverse force tree and public bias commit remain after S admission.
__device__ inline void d_comvel_bias_cached(int w,int*active,double*cdof,double*qvel,
    int*bp,int*ba,int*bn,int*dj,int*jt,int nb,int nv,double*cvel,double*cdot,int*status,
    double*cinert,double*gravity,int disabled,double*cache_acc,double*cache_force,
    int*cache_status,int*cache_valid,int64_t*cache_state,int64_t*cache_forward,
    const int64_t*state_epoch,const int64_t*forward_epoch,
    const int*level_offsets=nullptr,const int*level_bodies=nullptr,int nlevels=0) {
    d_comvel(w,active,cdof,qvel,bp,ba,bn,dj,jt,nb,nv,cvel,cdot,status,level_offsets,level_bodies,nlevels);
    __syncthreads();
    if(threadIdx.x==0){cache_valid[w]=0;cache_status[w]=0;}
    double*acc=cache_acc+(long long)w*nb*6,*f=cache_force+(long long)w*nb*6;
    d1_zero_block(acc,nb*6);d1_zero_block(f,nb*6);
    __syncthreads();
    if(active[w]!=1 || status[w]!=0) return;
    const double*d=cdot+(long long)w*nv*6,*q=qvel+(long long)w*nv;
    const double*i=cinert+(long long)w*nb*10,*v=cvel+(long long)w*nb*6;
    // Successful comvel immediately above proved qvel/cdof and its cvel/dot
    // outputs finite. No intervening producer writes them inside this call.
    if(!d1_finite_block(i,nb*10)) {
        if(threadIdx.x==0)cache_status[w]=2;return;
    }
    d1_rne_forward(d,q,i,v,bp,ba,bn,gravity,disabled,nb,acc,f,
        level_offsets,level_bodies,nlevels,true);
    if(!d1_finite_block(acc,nb*6,f,nb*6)) {
        if(threadIdx.x==0)cache_status[w]=3;d1_zero_block(acc,nb*6);d1_zero_block(f,nb*6);
        __syncthreads();return;
    }
    if(threadIdx.x==0){
        cache_state[w]=state_epoch[w];cache_forward[w]=forward_epoch[w];
        cache_valid[w]=1;
    }
    __syncthreads();
}

__device__ inline void d_rne_bias_cached(int w,int*active,double*cdot,double*qvel,
    double*cinert,double*cvel,double*cdof,int*bp,int*ba,int*bn,int*db,double*gravity,
    int disabled,int nb,int nv,double*cacc,double*frc,double*result,int*status,
    const double*cache_acc,const double*cache_force,const int*cache_status,
    const int*cache_valid,const int64_t*cache_state,const int64_t*cache_forward,
    const int64_t*state_epoch,const int64_t*forward_epoch,
    const int*level_offsets=nullptr,const int*level_bodies=nullptr,int nlevels=0,
    const int*children_offsets=nullptr,const int*children_indices=nullptr) {
    if(active[w]!=1 || cache_valid[w]!=1 || cache_status[w]!=0 ||
       cache_state[w]!=state_epoch[w] || cache_forward[w]!=forward_epoch[w]) {
        d_rne_bias(w,active,cdot,qvel,cinert,cvel,cdof,bp,ba,bn,db,gravity,
                   disabled,nb,nv,cacc,frc,result,status,level_offsets,level_bodies,nlevels,
                   children_offsets,children_indices);return;
    }
    double*acc=cacc+(long long)w*nb*6,*f=frc+(long long)w*nb*6,*r=result+(long long)w*nv;
    const double*a=cdof+(long long)w*nv*6;
    const double*ca=cache_acc+(long long)w*nb*6,*cf=cache_force+(long long)w*nb*6;
    if(threadIdx.x==0)status[w]=0;
    // Cache producer writes [component][body] directly. Each linear cache
    // tile is read contiguously; public RNE diagnostics keep [body][component].
    for(int k=threadIdx.x;k<nb*6;k+=blockDim.x){
        const int body=k%nb,component=k/nb,dst=body*6+component;
        acc[dst]=ca[k];f[dst]=cf[k];
    }
    d1_zero_block(r,nv);
    __syncthreads();
    d1_rne_reverse(bp,nb,f,level_offsets,level_bodies,nlevels,
        children_offsets,children_indices);
    for(int j=threadIdx.x;j<nv;j+=blockDim.x) {
        const double*aa=a+6*j,*ff=f+6*db[j];
        r[j]=((aa[0]*ff[0]+aa[2]*ff[2])+(aa[1]*ff[1]+aa[3]*ff[3]))+
             (aa[4]*ff[4]+aa[5]*ff[5]);
    }
    __syncthreads();
    if(!d1_finite_block(acc,nb*6,f,nb*6,r,nv)) {
        if(threadIdx.x==0)status[w]=3;d1_zero_block(acc,nb*6);d1_zero_block(f,nb*6);d1_zero_block(r,nv);
    }
    __syncthreads();
}

#ifdef D_FAST_DATAFLOW
__device__ inline void d_commit_smooth_shared(int w,int*active,int*gate,int*status,
    int64_t*epoch,int*error,double*wf,double*wa,double*pf,double*pa,
    int*valid,int64_t*written,int*repaired,int width) {
    valid[w]=0;repaired[w]=0;
    if(active[w]!=1||gate[w]!=1||error[w])return;
    if(status[w]&&status[w]!=25){error[w]=213;return;}
    DForceShared& work=d_force_shared();
    pf+=(long long)w*width;pa+=(long long)w*width;
    if(d_copy_overlap(work.force,pf,width)||d_copy_overlap(work.force,pa,width)||
       d_copy_overlap(work.acceleration,pf,width)||d_copy_overlap(work.acceleration,pa,width)||
       d_copy_overlap(pf,pa,width)) {
        for(int i=0;i<width;++i){pf[i]=work.force[i];pa[i]=work.acceleration[i];}
    } else {
        libero_vec::copy_serial(work.force,pf,width);
        libero_vec::copy_serial(work.acceleration,pa,width);
    }
    written[w]=epoch[w];repaired[w]=(status[w]==25);valid[w]=1;
}
#endif
