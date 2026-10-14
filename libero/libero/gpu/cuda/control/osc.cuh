#pragma once
#include <stdint.h>
#include <math.h>
#include "../common/vector.cuh"
extern "C" __device__ void osc_inverse7(double*,double*,int*,int*);
#define OSC_DECL(M,N,K) extern "C" __device__ void osc_mm_##M##_##N##_##K(double*,double*,double*);
OSC_DECL(6,7,7) OSC_DECL(6,6,7) OSC_DECL(7,6,7) OSC_DECL(7,6,6)
OSC_DECL(7,7,6) OSC_DECL(3,3,3) OSC_DECL(6,1,6) OSC_DECL(3,1,3)
OSC_DECL(7,1,7) OSC_DECL(7,1,6)
// -1 retains runtime selection. Host plans with immutable controller config
// may set 0/1, while retaining both library routes and the alias-safe loader.
#ifndef OSC_UNCOUPLE
#define OSC_UNCOUPLE -1
#endif
namespace osc_cuda {
__device__ inline double clip(double x,double lo,double hi) {return isnan(x)?x:fmin(hi,fmax(lo,x));}

// Compatible callers may alias persisted M/J with later inputs. Preserve the
// original leader write/read sequence for overlapping ranges; otherwise each
// operand is loaded once by the first warp. Address differences avoid overflow.
__device__ inline bool input_overlap(const double* a,int na,const double* b,int nb) {
    if(!a||!b)return false;
    const uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y ? y-x<(uintptr_t)na*sizeof(double)
                : x-y<(uintptr_t)nb*sizeof(double);
}
__device__ inline bool load_alias(int w,double* mass,double* jacobian,
                                 double* bias,double* q0,double* pm,double* pj) {
    const double* m=mass+w*49;const double* j=jacobian+w*42;
    const double* b=bias+w*7;const double* q=q0+w*7;
    const double* dm=pm?pm+w*49:nullptr;
    const double* dj=pj?pj+w*42:nullptr;
    return input_overlap(dm,49,m,49)||input_overlap(dm,49,j,42)||
           input_overlap(dm,49,b,7)||input_overlap(dm,49,q,7)||
           input_overlap(dj,42,m,49)||input_overlap(dj,42,j,42)||
           input_overlap(dj,42,b,7)||input_overlap(dj,42,q,7)||
           input_overlap(dm,49,dj,42);
}


// cuSOLVERDx 25.12 GESVD exposes values only, not U/V. This specialized
// truncated inverse therefore keeps the one-sided Jacobi algorithm, scaling,
// relative 1e-15 cutoff and final retained-column orthogonality criterion.
__device__ inline void pinv(const double*a,int n,double*out,int&rank,int&ok) {
    // Column-major private workspace makes each ordered column scan contiguous.
    // Only addresses change: reduction order, rotations, scale and cutoff stay fixed.
    double b[36]={},v[36]={},norm[6]={},scale=0;
    rank=0;ok=1;
    for(int z=0;z<36;z++)out[z]=0;
    for(int i=0;i<n;i++)for(int j=0;j<n;j++)scale=fmax(scale,fabs(a[i*6+j]));
    if(!(scale>0))return;
    for(int i=0;i<n;i++){v[i*6+i]=1;for(int j=0;j<n;j++)b[j*6+i]=a[i*6+j]/scale;}
    ok=0;
    for(int sweep=0;sweep<64;sweep++){
        int rotations=0;
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
            double alpha=0,beta=0,gamma=0;
            for(int i=0;i<n;i++){double bp=b[p*6+i],bq=b[q*6+i];alpha+=bp*bp;beta+=bq*bq;gamma+=bp*bq;}
            if(fabs(gamma)>2.220446049250313e-16*sqrt(alpha)*sqrt(beta)){
                double zeta=(beta-alpha)/(2*gamma),t=(zeta<0?-1.:1.)/(fabs(zeta)+sqrt(1+zeta*zeta));
                double c=1/sqrt(1+t*t),s=c*t;
                for(int i=0;i<n;i++){double bp=b[p*6+i],bq=b[q*6+i],vp=v[p*6+i],vq=v[q*6+i];
                    b[p*6+i]=c*bp-s*bq;b[q*6+i]=s*bp+c*bq;v[p*6+i]=c*vp-s*vq;v[q*6+i]=s*vp+c*vq;}
                rotations++;
            }
        }
        if(rotations==0){ok=1;break;}
    }
    double maxnorm=0,maxcorr=0,singular[6]={};bool retained[6]={};
    for(int p=0;p<n;p++){for(int i=0;i<n;i++)norm[p]+=b[p*6+i]*b[p*6+i];singular[p]=sqrt(norm[p]);maxnorm=fmax(maxnorm,singular[p]);}
    for(int p=0;p<n;p++)retained[p]=singular[p]>1e-15*maxnorm;
    for(int p=0;p<n;p++)for(int q=p+1;q<n;q++)if(retained[p]&&retained[q]){
        double gamma=0;for(int i=0;i<n;i++)gamma+=b[p*6+i]*b[q*6+i];maxcorr=fmax(maxcorr,fabs(gamma)/(singular[p]*singular[q]));}
    if(maxcorr<2e-15)ok=1;
    for(int k=0;k<n;k++)if(retained[k]){rank++;double wt=(1/norm[k])/scale;
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)out[i*6+j]+=v[k*6+i]*b[k*6+j]*wt;}
}

__device__ inline void rotation_delta(double x,double y,double z,double*r,double*raw,float*cast) {
    double angle=sqrt(x*x+y*y+z*z),f=angle==0?0:sin(angle*.5)/angle;
    raw[0]=x*f;raw[1]=y*f;raw[2]=z*f;raw[3]=angle==0?1:cos(angle*.5);
    float q[4]={(float)raw[3],(float)raw[0],(float)raw[1],(float)raw[2]};
    double normacc=0;for(int i=0;i<4;i++){cast[i]=q[i];float product=q[i]*q[i];normacc+=(double)product;}
    float norm=(float)normacc;double factor=sqrt(2/(double)norm);
    for(int i=0;i<4;i++)q[i]=(float)((double)q[i]*factor);
    r[0]=1-(double)(q[2]*q[2])-(double)(q[3]*q[3]);
    r[4]=1-(double)(q[1]*q[1])-(double)(q[3]*q[3]);
    r[8]=1-(double)(q[1]*q[1])-(double)(q[2]*q[2]);
    r[1]=(double)(q[1]*q[2]-q[3]*q[0]);r[2]=(double)(q[1]*q[3]+q[2]*q[0]);
    r[3]=(double)(q[1]*q[2]+q[3]*q[0]);r[5]=(double)(q[2]*q[3]-q[1]*q[0]);
    r[6]=(double)(q[1]*q[3]-q[2]*q[0]);r[7]=(double)(q[2]*q[3]+q[1]*q[0]);
}

__device__ inline void refresh(int w,int*active,int*needs,int*valid,int*error,
 double*pm,double*pj,double*pq,double*pv,double*pp,double*pr,double*pe,
 double*m,double*j,double*q,double*v,double*p,double*r,double*e) {
    if(!active[w]||error[w])return;
    if(needs[w]){
        for(int i=0;i<49;i++)m[w*49+i]=pm[w*49+i];for(int i=0;i<42;i++)j[w*42+i]=pj[w*42+i];
        for(int i=0;i<7;i++){q[w*7+i]=pq[w*7+i];v[w*7+i]=pv[w*7+i];}
        for(int i=0;i<3;i++)p[w*3+i]=pp[w*3+i];for(int i=0;i<9;i++)r[w*9+i]=pr[w*9+i];
        for(int i=0;i<6;i++)e[w*6+i]=pe[w*6+i];valid[w]=1;needs[w]=0;
    }else if(!valid[w])error[w]=1;
}

__device__ inline void reset(int w,int*mask,double*p,double*r,double*q,double*gp,double*gr,double*q0,
 double*grip,int*error,int*valid,int*needs,int clear,int update){
    if(!mask[w])return;
    for(int i=0;i<3;i++)gp[w*3+i]=p[w*3+i];for(int i=0;i<9;i++)gr[w*9+i]=r[w*9+i];
    if(update)for(int i=0;i<7;i++)q0[w*7+i]=q[w*7+i];if(clear)for(int i=0;i<2;i++)grip[w*2+i]=0;
    error[w]=0;valid[w]=0;needs[w]=1;
}

// The entire block enters. Shared operands make each library operation a
// true device-side collective inside this kernel, including all dense GEMMs.
__device__ inline void control(int w,double*mass,double*jacobian,double*q,double*qvel,double*bias,double*q0,
 double*ee_pos,double*ee_rot,double*ee_velocity,double*action,double*goal_pos,double*goal_rot,double*gripper,
 int*active,double*gains,double*range,double*ctrl,double*raw_tau,int*error,int*ranks,int*policy,int*needs,
 double*mass_inv,double*A_full,double*A_pos,double*A_ori,double*lambda_full,double*lambda_pos,double*lambda_ori,
 double*nullspace_matrix,double*position_error,double*orientation_error,double*quat_input,float*quat_cast,
 double*quat_result,int*quat_valid,double*nullspace_torque,double*clipped,int diagnostics,int uncouple, int* derived_valid=nullptr, double* derived_key=nullptr, double* derived_data=nullptr, int* derived_ranks=nullptr, int* cache_hits=nullptr,double* persist_m=nullptr,double* persist_j=nullptr,double* local_state=nullptr) {
    __shared__ double m[49],lu[49],mi[49],j[42],jt[42],jw[42],a[36],ap[36],ar[36],lf[36],lp[36],lr[36];
    __shared__ double mjt[42],jbar[42],project[49],delta[9],current[9],gr[9],gp[3],desired[6],wrench[6];
    __shared__ double lp3[9],lr3[9],pd[7],pose[7],nt[7],tau[7],project_t[49];
    __shared__ int info,piv[7],admitted,reuse,serial_load;
    if(threadIdx.x==0){
        admitted=active[w]!=0&&error[w]==0;info=0;
        serial_load=admitted&&load_alias(w,mass,jacobian,bias,q0,persist_m,persist_j);
    }
    __syncthreads();
    if(!admitted)return;
    if(serial_load){
        if(threadIdx.x==0){
            for(int r=0;r<7;r++){if(!isfinite(bias[w*7+r])||!isfinite(q0[w*7+r]))admitted=0;
                for(int c=0;c<7;c++){double v=mass[w*49+r*7+c];m[r*7+c]=v;lu[r*7+c]=v;mi[r*7+c]=r==c?1:0;if(persist_m)persist_m[w*49+r*7+c]=v;if(!isfinite(v))admitted=0;}}
            for(int r=0;r<6;r++)for(int c=0;c<7;c++){double v=jacobian[w*42+r*7+c];j[r*7+c]=v;jt[c*6+r]=v;if(persist_j)persist_j[w*42+r*7+c]=v;if(!isfinite(v))admitted=0;}
            if(!admitted)error[w]=1;
        }
    }else if(threadIdx.x<32){
        bool bad=false;
        const int width=blockDim.x<32?blockDim.x:32;
        const unsigned participants=width==32?0xffffffffu:((1u<<width)-1u);
        for(int pair=threadIdx.x;pair<25;pair+=width){
            const int z=pair*2;
            if(z+1<49){const double2 v=libero_vec::load_pair(mass+w*49+z);
                libero_vec::store_pair(m+z,v.x,v.y);libero_vec::store_pair(lu+z,v.x,v.y);
                mi[z]=z/7==z%7?1:0;mi[z+1]=(z+1)/7==(z+1)%7?1:0;
                if(persist_m)libero_vec::store_pair(persist_m+w*49+z,v.x,v.y);
                bad=bad||!isfinite(v.x)||!isfinite(v.y);
            }else{const double v=mass[w*49+z];m[z]=lu[z]=v;mi[z]=z/7==z%7?1:0;
                if(persist_m)persist_m[w*49+z]=v;bad=bad||!isfinite(v);}
        }
        for(int pair=threadIdx.x;pair<21;pair+=width){
            const int z=pair*2;const double2 v=libero_vec::load_pair(jacobian+w*42+z);
            libero_vec::store_pair(j+z,v.x,v.y);
            jt[(z%7)*6+z/7]=v.x;jt[((z+1)%7)*6+(z+1)/7]=v.y;
            if(persist_j)libero_vec::store_pair(persist_j+w*42+z,v.x,v.y);
            bad=bad||!isfinite(v.x)||!isfinite(v.y);
        }
        bad=libero_vec::any_nonfinite_lane(bias+w*7,7,threadIdx.x,width)||bad;
        bad=libero_vec::any_nonfinite_lane(q0+w*7,7,threadIdx.x,width)||bad;
        const bool any_bad=__any_sync(participants,bad);
        if(threadIdx.x==0&&any_bad){admitted=0;error[w]=1;}
    }
    __syncthreads();if(!admitted)return;
    if(threadIdx.x<32){
        bool mismatch=!(derived_valid&&derived_valid[w]);
        const int width=blockDim.x<32?blockDim.x:32;
        const unsigned participants=width==32?0xffffffffu:((1u<<width)-1u);
        if(!mismatch){
            for(int pair=threadIdx.x;pair<46;pair+=width){
                const int z=pair*2;
                const double2 key=z+1<91?libero_vec::load_pair(derived_key+w*91+z):make_double2(derived_key[w*91+z],0);
                const double x=z<49?m[z]:j[z-49];
                mismatch=mismatch||(__double_as_longlong(x)!=__double_as_longlong(key.x));
                if(z+1<91){const double y=z+1<49?m[z+1]:j[z+1-49];
                    mismatch=mismatch||(__double_as_longlong(y)!=__double_as_longlong(key.y));}
            }
        }
        const bool any_mismatch=__any_sync(participants,mismatch);
        if(threadIdx.x==0)reuse=!any_mismatch;
    }
    __syncthreads();
    if(threadIdx.x==0){
        if(reuse){double* d=derived_data+w*206;
            libero_vec::copy_serial(d,mi,49);libero_vec::copy_serial(d+157,project,49);
            libero_vec::copy_serial(d+49,a,36);libero_vec::copy_serial(d+103,lf,36);
            for(int z=0;z<36;z++)ap[z]=ar[z]=lp[z]=lr[z]=0;
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;
                ap[t]=d[85+z];ar[t]=d[94+z];lp[t]=d[139+z];lr[t]=d[148+z];lp3[z]=lp[t];lr3[z]=lr[t];}
            for(int z=0;z<3;z++)ranks[w*3+z]=derived_ranks[w*3+z];if(cache_hits)cache_hits[w]++;
        }
    }__syncthreads();
    if(!reuse){
    osc_inverse7(lu,mi,piv,&info);
    if(threadIdx.x==0&&info!=0){error[w]=2;admitted=0;}
    __syncthreads();if(!admitted)return;
    for(int z=threadIdx.x;z<49;z+=blockDim.x)mass_inv[w*49+z]=mi[z];
    osc_mm_6_7_7(j,mi,jw);osc_mm_6_6_7(jw,jt,a);
    if(threadIdx.x==0){
        for(int z=0;z<36;z++){ap[z]=0;ar[z]=0;}
        for(int r=0;r<3;r++)for(int c=0;c<3;c++){ap[r*6+c]=a[r*6+c];ar[r*6+c]=a[(r+3)*6+c+3];}
        int rf,rp,rr,of,op,orr;pinv(a,6,lf,rf,of);pinv(ap,3,lp,rp,op);pinv(ar,3,lr,rr,orr);
        ranks[w*3]=rf;ranks[w*3+1]=rp;ranks[w*3+2]=rr;
        if(!of||!op||!orr){error[w]=4;admitted=0;}
        else{
            for(int z=0;z<36;z++){A_full[w*36+z]=a[z];lambda_full[w*36+z]=lf[z];}
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,s=r*6+c;
                A_pos[w*9+z]=ap[s];A_ori[w*9+z]=ar[s];lambda_pos[w*9+z]=lp[s];lambda_ori[w*9+z]=lr[s];lp3[z]=lp[s];lr3[z]=lr[s];}
        }
    }__syncthreads();if(!admitted)return;
    osc_mm_7_6_7(mi,jt,mjt);osc_mm_7_6_6(mjt,lf,jbar);osc_mm_7_7_6(jbar,j,project);
    if(threadIdx.x==0){
        for(int r=0;r<7;r++)for(int c=0;c<7;c++)project[r*7+c]=(r==c?1.:0.)-project[r*7+c];
        if(derived_valid){double* d=derived_data+w*206;
            for(int z=0;z<49;z++){derived_key[w*91+z]=m[z];d[z]=mi[z];d[157+z]=project[z];}
            for(int z=0;z<42;z++)derived_key[w*91+49+z]=j[z];
            for(int z=0;z<36;z++){d[49+z]=a[z];d[103+z]=lf[z];}
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;d[85+z]=ap[t];d[94+z]=ar[t];d[139+z]=lp[t];d[148+z]=lr[t];}
            for(int z=0;z<3;z++)derived_ranks[w*3+z]=ranks[w*3+z];derived_valid[w]=1;
        }
    }__syncthreads();
    } // exact-input derived-matrix cache miss
    // Misses already materialized these diagnostics at their original points.
    // Only hits need replay; avoid extra writes on the compatible alias path.
    if(reuse){
        for(int z=threadIdx.x;z<49;z+=blockDim.x)mass_inv[w*49+z]=mi[z];
        if(threadIdx.x==0){
            for(int z=0;z<36;z++){A_full[w*36+z]=a[z];lambda_full[w*36+z]=lf[z];}
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;A_pos[w*9+z]=ap[t];A_ori[w*9+z]=ar[t];lambda_pos[w*9+z]=lp[t];lambda_ori[w*9+z]=lr[t];}
        }
    }
    if(threadIdx.x==0){
        for(int r=0;r<7;r++)for(int c=0;c<7;c++){double v=project[r*7+c];project[r*7+c]=v;project_t[c*7+r]=v;if(diagnostics)nullspace_matrix[w*49+r*7+c]=v;}
        for(int r=0;r<3;r++){gp[r]=goal_pos[w*3+r];for(int c=0;c<3;c++){gr[r*3+c]=goal_rot[w*9+r*3+c];current[r*3+c]=(local_state?local_state[17+r*3+c]:ee_rot[w*9+r*3+c]);}}
        quat_valid[w]=0;
        if(policy[w]){
            for(int r=0;r<3;r++)gp[r]=(local_state?local_state[14+r]:ee_pos[w*3+r])+.05*clip(action[w*7+r],-1,1);
            double x=.5*clip(action[w*7+3],-1,1),y=.5*clip(action[w*7+4],-1,1),z=.5*clip(action[w*7+5],-1,1);
            if(x!=0||y!=0||z!=0){rotation_delta(x,y,z,delta,quat_input+w*4,quat_cast+w*4);quat_valid[w]=1;for(int i=0;i<9;i++)quat_result[w*9+i]=delta[i];}
        }
    }__syncthreads();if(quat_valid[w])osc_mm_3_3_3(delta,current,gr);
    if(threadIdx.x==0){
        double er[3]={};
        for(int c=0;c<3;c++){er[0]+=.5*(current[3+c]*gr[6+c]-current[6+c]*gr[3+c]);
            er[1]+=.5*(current[6+c]*gr[c]-current[c]*gr[6+c]);er[2]+=.5*(current[c]*gr[3+c]-current[3+c]*gr[c]);}
        for(int r=0;r<3;r++){position_error[w*3+r]=gp[r]-(local_state?local_state[14+r]:ee_pos[w*3+r]);orientation_error[w*3+r]=er[r];
            desired[r]=gains[w*12+r]*position_error[w*3+r]-gains[w*12+r+6]*(local_state?local_state[26+r]:ee_velocity[w*6+r]);
            desired[r+3]=gains[w*12+r+3]*er[r]-gains[w*12+r+9]*(local_state?local_state[26+r+3]:ee_velocity[w*6+r+3]);}
        for(int c=0;c<7;c++)pd[c]=10*(q0[w*7+c]-(local_state?local_state[0+c]:q[w*7+c]))-6.324555320336759*(local_state?local_state[7+c]:qvel[w*7+c]);
    }__syncthreads();
    const bool split_mode=OSC_UNCOUPLE<0 ? uncouple!=0 : OSC_UNCOUPLE!=0;
    if(split_mode){osc_mm_3_1_3(lp3,desired,wrench);osc_mm_3_1_3(lr3,desired+3,wrench+3);}else osc_mm_6_1_6(lf,desired,wrench);
    osc_mm_7_1_7(m,pd,pose);osc_mm_7_1_7(project_t,pose,nt);osc_mm_7_1_6(jt,wrench,tau);
    if(threadIdx.x==0){
        for(int r=0;r<3;r++){goal_pos[w*3+r]=gp[r];for(int c=0;c<3;c++)goal_rot[w*9+r*3+c]=gr[r*3+c];}
        for(int i=0;i<7;i++){nullspace_torque[w*7+i]=nt[i];double t=tau[i]+bias[w*7+i]+nt[i];raw_tau[w*7+i]=t;clipped[w*7+i]=clip(t,range[w*18+2*i],range[w*18+2*i+1]);ctrl[w*9+i]=clipped[w*7+i];}
        double a6=action[w*7+6],sgn=isnan(a6)?a6:(a6>0?1.:(a6<0?-1.:0.));
        for(int i=0;i<2;i++){double value=clip(gripper[w*2+i]+.01*(i==0?-1.:1.)*sgn,-1,1);gripper[w*2+i]=value;
            double lo=range[w*18+2*(i+7)],hi=range[w*18+2*(i+7)+1];ctrl[w*9+i+7]=(hi+lo)*.5+(hi-lo)*.5*value;}
        needs[w]=1;
    }__syncthreads();
}
// Leader-only paired field loader, no synchronization. Valid short fields have
// a scalar final element; alignment is checked independently at each view.
__device__ inline void state_field(const double* src,double* dst,double* local,int n){
    int z=0;
    for(;z+1<n;z+=2){const double2 v=libero_vec::load_pair(src+z);
        if(dst)libero_vec::store_pair(dst+z,v.x,v.y);
        libero_vec::store_pair(local+z,v.x,v.y);}
    if(z<n){const double v=src[z];if(dst)dst[z]=v;local[z]=v;}
}
__device__ inline void control_fast(int w,double*mass,double*jacobian,double*q,double*qvel,double*bias,double*q0,
 double*ee_pos,double*ee_rot,double*ee_velocity,double*action,double*goal_pos,double*goal_rot,double*gripper,
 int*active,double*gains,double*range,double*ctrl,double*raw_tau,int*error,int*ranks,int*policy,int*needs,
 double*mass_inv,double*A_full,double*A_pos,double*A_ori,double*lambda_full,double*lambda_pos,double*lambda_ori,
 double*nullspace_matrix,double*position_error,double*orientation_error,double*quat_input,float*quat_cast,
 double*quat_result,int*quat_valid,double*nullspace_torque,double*clipped,int diagnostics,int uncouple, int* derived_valid, double* derived_key, double* derived_data, int* derived_ranks, int* cache_hits, double* pm,double* pj,double* pq,double* pv,double* pp,double* pr,double* pe,int* cache_valid) {
    __shared__ int live;
    __shared__ double local_state[32];
    if(threadIdx.x==0){live=needs[w]!=0;
        if(active[w]&&!error[w]){
            if(live){
                // M/J materialization is paired with their shared operand load
                // inside control. Each physical matrix element is read once.
                const double* sources[5]={pq+w*7,pv+w*7,pp+w*3,pr+w*9,pe+w*6};
                double* targets[5]={q+w*7,qvel+w*7,ee_pos+w*3,ee_rot+w*9,ee_velocity+w*6};
                const int sizes[5]={7,7,3,9,6};
                bool alias=false;
                for(int i=0;i<5;i++)for(int k=0;k<5;k++)alias=alias||input_overlap(sources[i],sizes[i],targets[k],sizes[k]);
                for(int i=0;i<5;i++)for(int k=i+1;k<5;k++)alias=alias||input_overlap(targets[i],sizes[i],targets[k],sizes[k]);
                if(alias){
                for(int z=0;z<7;z++){double a=pq[w*7+z],b=pv[w*7+z];local_state[z]=q[w*7+z]=a;local_state[7+z]=qvel[w*7+z]=b;}
                for(int z=0;z<3;z++){double v=pp[w*3+z];local_state[14+z]=ee_pos[w*3+z]=v;}
                for(int z=0;z<9;z++){double v=pr[w*9+z];local_state[17+z]=ee_rot[w*9+z]=v;}
                for(int z=0;z<6;z++){double v=pe[w*6+z];local_state[26+z]=ee_velocity[w*6+z]=v;}
                }else{
                    state_field(sources[0],targets[0],local_state,7);
                    state_field(sources[1],targets[1],local_state+7,7);
                    state_field(sources[2],targets[2],local_state+14,3);
                    state_field(sources[3],targets[3],local_state+17,9);
                    state_field(sources[4],targets[4],local_state+26,6);
                }
                cache_valid[w]=1;needs[w]=0;
            }else if(!cache_valid[w])error[w]=1;
            else{
                state_field(q+w*7,nullptr,local_state,7);
                state_field(qvel+w*7,nullptr,local_state+7,7);
                state_field(ee_pos+w*3,nullptr,local_state+14,3);
                state_field(ee_rot+w*9,nullptr,local_state+17,9);
                state_field(ee_velocity+w*6,nullptr,local_state+26,6);
            }
        }
    }
    __syncthreads();
    control(w,(live?pm:mass),(live?pj:jacobian),(live?pq:q),(live?pv:qvel),bias,q0,(live?pp:ee_pos),(live?pr:ee_rot),(live?pe:ee_velocity),action,goal_pos,goal_rot,gripper,active,gains,range,ctrl,raw_tau,error,ranks,policy,needs,mass_inv,A_full,A_pos,A_ori,lambda_full,lambda_pos,lambda_ori,nullspace_matrix,position_error,orientation_error,quat_input,quat_cast,quat_result,quat_valid,nullspace_torque,clipped,diagnostics,uncouple,derived_valid,derived_key,derived_data,derived_ranks,cache_hits,live?mass:nullptr,live?jacobian:nullptr,local_state);
}
}
