// Explicit D07 leaf only. Baseline d_fluid_count remains unchanged.
namespace d_selected_passive_detail {
using namespace d_passive_detail;
__device__ inline bool fluid_body(int b, const double* cvel,const double* subtree_com,
    const double* xipos,const double* ximat,int* rootid,double* box,double* wind,
    double* opt,int diagnostic,double* stages,int stride,double* payload,
    const double* coefficients=nullptr,const double* coefficient_source=nullptr,
    const int* coefficient_valid=nullptr) {
    constexpr double pi=3.14159265358979323846;
        int root=rootid[b]; const double* m=ximat+9*b;
        Vec offset={{xipos[3*b]-subtree_com[3*root],xipos[3*b+1]-subtree_com[3*root+1],xipos[3*b+2]-subtree_com[3*root+2]}};
        const double2 cv01=libero_vec::load_pair(cvel+6*b),cv23=libero_vec::load_pair(cvel+6*b+2),cv45=libero_vec::load_pair(cvel+6*b+4);
        Vec ang={{cv01.x,cv01.y,cv23.x}};
        Vec lin={{cv23.y,cv45.x,cv45.y}};
        Vec local_ang=rot_t(m,ang), ox=cross(offset,ang);
        for(int r=0;r<3;++r) lin[r]-=ox[r];
        Vec object_lin=rot_t(m,lin), local_wind=rot_t(m,{{wind[0],wind[1],wind[2]}});
        Vec local_lin; for(int r=0;r<3;++r) local_lin[r]=object_lin[r]-local_wind[r];
        for(int r=0;r<3;++r) if(!isfinite(offset[r]) || !isfinite(local_ang[r]) || !isfinite(object_lin[r]) || !isfinite(local_wind[r]) || !isfinite(local_lin[r])) {return false;}
        double* st=diagnostic!=0 ? stages+b*stride : nullptr;
        if(diagnostic!=0) for(int r=0;r<3;++r) {
            st[r]=box[3*b+r];st[3+r]=local_ang[r];st[6+r]=object_lin[r];
            st[9+r]=0.0;st[12+r]=local_wind[r];st[15+r]=local_ang[r];st[18+r]=local_lin[r];
        }
        const double2 boxxy=libero_vec::load_pair(box+3*b);
        double x=boxxy.x,y=boxxy.y,z=box[3*b+2],visc=opt[0],density=opt[1];
        Vec torque={{0.,0.,0.}},force={{0.,0.,0.}};
        bool prepared=coefficients && coefficient_source && coefficient_valid && coefficient_valid[b];
        if(prepared) {
            const double* src=coefficient_source+5*b;
            const double actual[5]={x,y,z,visc,density};
            for(int k=0;k<5;++k)prepared&=__double_as_longlong(actual[k])==__double_as_longlong(src[k]);
            // Preserve extreme-value first-error behavior by using the exact
            // original arithmetic there. These bounds keep every cached drag
            // product/rotation well inside normal FP64 range; nothing is cut.
            for(int k=0;k<8;++k) {
                const double t=fabs(coefficients[8*b+k]);
                prepared&=t==0.0 || (t>=1e-100 && t<=1e50);
            }
            for(int k=0;k<3;++k) {
                const double a=fabs(local_ang[k]),v=fabs(local_lin[k]);
                prepared&=(a==0.0 || (a>=1e-50 && a<=1e20));
                prepared&=(v==0.0 || (v>=1e-50 && v<=1e20));
                prepared&=fabs(offset[k])<=1e20;
            }
            for(int k=0;k<9;++k)prepared&=fabs(m[k])<=1e20;
        }
        const double* cf=prepared ? coefficients+8*b : nullptr;
        if(visc>0.0) {
            double diam=(x+y+z)/3.0;
            double ang_scl=prepared?cf[0]:-pi*diam*diam*diam*visc;
            double lin_scl=prepared?cf[1]:-3.0*pi*diam*visc;
            for(int r=0;r<3;++r) {torque[r]=local_ang[r]*ang_scl;force[r]=local_lin[r]*lin_scl;}
        }
        if(density>0.0) {
            force[0]-=(prepared?cf[2]:0.5*density*y*z)*fabs(local_lin[0])*local_lin[0];
            force[1]-=(prepared?cf[3]:0.5*density*x*z)*fabs(local_lin[1])*local_lin[1];
            force[2]-=(prepared?cf[4]:0.5*density*x*y)*fabs(local_lin[2])*local_lin[2];
            torque[0]-=(prepared?cf[5]:density*x*(y*y*y*y+z*z*z*z))*fabs(local_ang[0])*local_ang[0]/64.0;
            torque[1]-=(prepared?cf[6]:density*y*(x*x*x*x+z*z*z*z))*fabs(local_ang[1])*local_ang[1]/64.0;
            torque[2]-=(prepared?cf[7]:density*z*(x*x*x*x+y*y*y*y))*fabs(local_ang[2])*local_ang[2]/64.0;
        }
        Vec wt=rot(m,torque),wf=rot(m,force);
        for(int r=0;r<3;++r) if(!isfinite(torque[r]) || !isfinite(force[r]) || !isfinite(wt[r]) || !isfinite(wf[r])) {return false;}
        if(diagnostic!=0) for(int r=0;r<3;++r) {st[21+r]=torque[r];st[24+r]=force[r];st[27+r]=wt[r];st[30+r]=wf[r];}
    for(int r=0;r<3;++r) {payload[r]=offset[r];payload[3+r]=wt[r];payload[6+r]=wf[r];}
    return true;
}
}

__device__ inline void d_selected_fluid_count(int w, double* cvel,double* subtree_com,
    double* xipos,double* ximat,double* cdof,int* active,int* rootid,double* mass,
    double* box,int* ancestor,double* wind,double* opt,int passive_disabled,
    int diagnostic,double* stages,double* qfrc,int* status,int nbody,int nv,
    const int*body_dof_offsets=nullptr,const int*body_dof_indices=nullptr,
    const double* coefficients=nullptr,const double* coefficient_source=nullptr,
    const int* coefficient_valid=nullptr,double* private_payload=nullptr,
    int* private_ok=nullptr,const int* same_call_comvel_status=nullptr) {
    using namespace d_selected_passive_detail;
    cvel+=(long long)w*nbody*6; subtree_com+=(long long)w*nbody*3;
    xipos+=(long long)w*nbody*3; ximat+=(long long)w*nbody*9;
    cdof+=(long long)w*nv*6; qfrc+=(long long)w*nv;
    int stride=33+7*nv; if(diagnostic!=0)stages+=(long long)w*nbody*stride;
    if(diagnostic!=0)libero_vec::fill_block(stages,(size_t)nbody*stride,0.0);
    __syncthreads();
    if(active[w]==0 || status[w]!=0) return;
    bool invalid=false;
    invalid|=libero_vec::any_nonfinite_block_lane(qfrc,nv);
    // Only a closed Native group can supply this current-invocation proof.
    // Standalone and producer-failure paths scan every original region.
    const bool velocity_proved=same_call_comvel_status && same_call_comvel_status[w]==0;
    if(!velocity_proved)invalid|=libero_vec::any_nonfinite_block_lane(cvel,nbody*6);
    invalid|=libero_vec::any_nonfinite_block_lane(subtree_com,nbody*3);
    invalid|=libero_vec::any_nonfinite_block_lane(xipos,nbody*3);
    invalid|=libero_vec::any_nonfinite_block_lane(ximat,nbody*9);
    if(!velocity_proved)invalid|=libero_vec::any_nonfinite_block_lane(cdof,nv*6);
    if(d_passive_detail::bad(invalid)) {
        if(threadIdx.x==0) status[w]=2;
        __syncthreads(); return;
    }
    if(passive_disabled!=0 || (opt[0]<=0.0 && opt[1]<=0.0)) return;
    if(coefficients) {
        bool extreme=nbody>=1000000;
        for(int k=threadIdx.x;k<nv*6;k+=blockDim.x)extreme|=fabs(cdof[k])>1e20;
        for(int k=threadIdx.x;k<nv;k+=blockDim.x)extreme|=fabs(qfrc[k])>1e250;
        if(__syncthreads_or(extreme))coefficients=nullptr;
    }
    const bool parallel=diagnostic==0 && private_payload && private_ok;
    if(parallel) {
        private_payload+=(long long)w*nbody*9;private_ok+=(long long)w*nbody;
        for(int b=1+threadIdx.x;b<nbody;b+=blockDim.x) {
            private_ok[b]=mass[b]<1e-15 ? 1 : fluid_body(b,cvel,subtree_com,xipos,ximat,
                rootid,box,wind,opt,0,nullptr,stride,private_payload+9*b,
                coefficients,coefficient_source,coefficient_valid);
        }
        __syncthreads();
    }
    __shared__ double payload[9];
    __shared__ int body_ok;
    for(int b=1;b<nbody;++b) {
        if(mass[b]<1e-15) continue;
        if(threadIdx.x==0) {
            if(parallel) {
                body_ok=private_ok[b];
                if(body_ok)for(int k=0;k<9;++k)payload[k]=private_payload[9*b+k];
            } else body_ok=fluid_body(b,cvel,subtree_com,xipos,ximat,
                rootid,box,wind,opt,diagnostic,stages,stride,payload,
                coefficients,coefficient_source,coefficient_valid);
        }
        __syncthreads();
        if(!body_ok) {
            if(threadIdx.x==0) status[w]=2;
            __syncthreads(); return;
        }
        Vec offset,wt,wf;
        // One shared load per warp; every lane receives the same body payload.
        for(int r=0;r<3;++r) {
            double o=0.0,t=0.0,f=0.0;
            if((threadIdx.x&31)==0) {o=payload[r];t=payload[3+r];f=payload[6+r];}
            offset[r]=__shfl_sync(0xffffffffu,o,0);
            wt[r]=__shfl_sync(0xffffffffu,t,0);
            wf[r]=__shfl_sync(0xffffffffu,f,0);
        }
        double* st=diagnostic!=0 ? stages+b*stride : nullptr;
        const bool sparse=diagnostic==0 && body_dof_offsets && body_dof_indices;
        const int begin=sparse?body_dof_offsets[b]:0;
        const int count=sparse?body_dof_offsets[b+1]-begin:nv;
        for(int base=0;base<count;base+=blockDim.x) {
            const int ordinal=base+threadIdx.x;
            const int j=ordinal<count?(sparse?body_dof_indices[begin+ordinal]:ordinal):nv;
            Vec jp={{0.,0.,0.}},jr={{0.,0.,0.}};
            bool projection_bad=false, result_bad=false;
            double result=0.0;
            if(j<nv) {
                if(sparse || ancestor[b*nv+j]!=0) {
                    jr={{cdof[6*j],cdof[6*j+1],cdof[6*j+2]}};
                    Vec cp=cross(jr,offset);
                    for(int r=0;r<3;++r) jp[r]=cdof[6*j+3+r]+cp[r];
                }
                for(int r=0;r<3;++r) projection_bad|=!isfinite(jp[r]) || !isfinite(jr[r]);
                if(!projection_bad) {
                    double qforce=0.0,qtorque=0.0;
                    for(int r=0;r<3;++r) if(wf[r]!=0.0) qforce+=jp[r]*wf[r];
                    result=qfrc[j]+qforce;
                    for(int r=0;r<3;++r) if(wt[r]!=0.0) qtorque+=jr[r]*wt[r];
                    result+=qtorque;
                    result_bad=!isfinite(result);
                }
            }
            const int local=projection_bad ? 2*j : result_bad ? 2*j+1 : libero_warp::no_index;
            const int fail=d_passive_detail::first_bad(local);
            // Preserve the scalar prefix: bad projection writes nothing; a bad
            // result writes qfrc but does not write its final diagnostic slot.
            if(j<nv && 2*j<fail) {
                if(diagnostic!=0) for(int r=0;r<3;++r) {
                    st[33+nv*r+j]=jp[r];st[33+3*nv+nv*r+j]=jr[r];
                }
                qfrc[j]=result;
                if(diagnostic!=0 && 2*j+1<fail) st[33+6*nv+j]=result;
            }
            __syncthreads();
            if(fail!=libero_warp::no_index) {
                if(sparse)for(int k=threadIdx.x;k<fail/2;k+=blockDim.x)
                    if(ancestor[b*nv+k]==0)qfrc[k]=(qfrc[k]+0.0)+0.0;
                __syncthreads();
                if(threadIdx.x==0) status[w]=2;
                __syncthreads(); return;
            }
        }
        if(sparse)for(int j=threadIdx.x;j<nv;j+=blockDim.x)
            if(ancestor[b*nv+j]==0)qfrc[j]=(qfrc[j]+0.0)+0.0;
        __syncthreads();
    }
    __syncthreads();
}

__device__ inline void d_fluid_parallel_selected(int w,double*cvel,double*com,
    double*xipos,double*ximat,double*cdof,int*active,int*rootid,double*mass,
    double*box,int*ancestor,double*wind,double*opt,int disabled,int diagnostic,
    double*stages,double*qfrc,int*status,int nb,int nv,double*payload,int*ok,
    const int*offsets=nullptr,const int*indices=nullptr) {
    // D04 coefficients and D06 producer proof are deliberately absent.
    d_selected_fluid_count(w,cvel,com,xipos,ximat,cdof,active,rootid,mass,box,ancestor,
        wind,opt,disabled,diagnostic,stages,qfrc,status,nb,nv,offsets,indices,
        nullptr,nullptr,nullptr,payload,ok,nullptr);
}
