#pragma once
#include <math.h>
#include <stdint.h>
#include <float.h>
#include "collision_bounds.cuh"

namespace collision_cuda {
// Temporal state is private. The compatibility diagnostic path never calls
// these entry points. All block threads enter, with one block per world.
__device__ inline bool cf_temporal_int_lane(const int* src,const int* shadow,
                                           int size,int offset) {
    bool bad=false;
    for(int k=threadIdx.x;k<size;k+=blockDim.x)bad|=src[k]!=shadow[offset+k];
    return bad;
}
__device__ inline bool cf_temporal_static_bad_lane(int nbody,int ngeom,
    const int* body_geomadr,const int* body_geomnum,const int* geom_bodyid,
    const int* body_weldid,const int* body_parentid,const int* can,const int* plane,
    const int* geom_type,const int* geom_contype,const int* geom_conaffinity,
    const int* gates,const int* collision_mode,const int* filterparent_disabled,
    const double* rbound,const double* margin,const int* ishadow,const double* fshadow) {
    bool bad=false;int p=0;
    bad|=cf_temporal_int_lane(body_geomadr,ishadow,nbody,p);p+=nbody;
    bad|=cf_temporal_int_lane(body_geomnum,ishadow,nbody,p);p+=nbody;
    bad|=cf_temporal_int_lane(geom_bodyid,ishadow,ngeom,p);p+=ngeom;
    bad|=cf_temporal_int_lane(body_weldid,ishadow,nbody,p);p+=nbody;
    bad|=cf_temporal_int_lane(body_parentid,ishadow,nbody,p);p+=nbody;
    bad|=cf_temporal_int_lane(can,ishadow,nbody,p);p+=nbody;
    bad|=cf_temporal_int_lane(plane,ishadow,nbody,p);p+=nbody;
    bad|=cf_temporal_int_lane(geom_type,ishadow,ngeom,p);p+=ngeom;
    bad|=cf_temporal_int_lane(geom_contype,ishadow,ngeom,p);p+=ngeom;
    bad|=cf_temporal_int_lane(geom_conaffinity,ishadow,ngeom,p);p+=ngeom;
    bad|=cf_temporal_int_lane(gates,ishadow,3,p);p+=3;
    bad|=cf_temporal_int_lane(collision_mode,ishadow,1,p);++p;
    bad|=cf_temporal_int_lane(filterparent_disabled,ishadow,1,p);
    for(int g=threadIdx.x;g<ngeom;g+=blockDim.x){
        bad|=__double_as_longlong(rbound[g])!=__double_as_longlong(fshadow[g]);
        bad|=__double_as_longlong(margin[g])!=__double_as_longlong(fshadow[ngeom+g]);
    }
    return bad;
}
__device__ inline bool cf_temporal_overlap(const double* ab,int b1,int b2) {
    for(int k=0;k<3;++k)
        if(ab[6*b1+2*k]>ab[6*b2+2*k+1] || ab[6*b1+2*k+1]<ab[6*b2+2*k])return false;
    return true;
}
// ||R||_2 <= sqrt(||R^T R||_infinity). Interval dot products protect the
// certificate from normal FP64 non-orthogonality and FMA reassociation. A
// further 32 eps encloses the native three-term point-transform roundoff.
__device__ inline double cf_temporal_norm_squared(const double* matrix) {
    double rows[3]={0,0,0};
    for(int i=0;i<3;++i)for(int j=i;j<3;++j){
        double lo=0,hi=0;
        for(int k=0;k<3;++k){
            const double a=matrix[3*k+i],b=matrix[3*k+j];
            lo=__dadd_rd(lo,__dmul_rd(a,b));hi=__dadd_ru(hi,__dmul_ru(a,b));
        }
        const double magnitude=fmax(fabs(lo),fabs(hi));
        rows[i]=__dadd_ru(rows[i],magnitude);
        if(i!=j)rows[j]=__dadd_ru(rows[j],magnitude);
    }
    return fmax(1.0,fmax(rows[0],fmax(rows[1],rows[2])));
}
__device__ inline double cf_temporal_radius(double norm_squared,double world_norm_squared,
                                          double rb,double margin) {
    const double norm=__dsqrt_ru(norm_squared);
    const double scale=__dmul_ru(norm,1.0+32.0*DBL_EPSILON);
    // Native MPR maps a local margin direction back to world space: R R^T.
    // A common maximum protects pair max-margin even when only one geom owns
    // that margin: nonnegative m1+m2 >= max(m1,m2).
    const double margin_scale=__dmul_ru(world_norm_squared,1.0+64.0*DBL_EPSILON);
    return __dadd_ru(__dmul_ru(rb,scale),__dmul_ru(margin,margin_scale));
}
// Uses a bounding sphere centered at geom_xpos. Rotation cannot escape a
// sphere; rbound and the original geom margin are guarded static inputs.
// Directed rounding encloses the FP64 addition/subtraction at each endpoint.
__device__ inline void cf_temporal_bounds_block(int w,
    const double* xpos,const int* active,const int* gates,const int* collision_mode,
    const double* geom_rbound,const double* geom_margin,const int* body_geomadr,
    const int* body_geomnum,const int* geom_bodyid,double* aabb,double* frame,
    int* entered,int* count,int* status,double* scratch,int ngeom,int nbody,
    const int* body_weldid,const int* body_parentid,const int* can,const int* plane,
    const int* geom_type,const int* geom_contype,const int* geom_conaffinity,
    const int* filterparent_disabled,const int* integer_shadow,const double* float_shadow,
    const double* geom_matrix,double* current_radius,
    const int* tree_body_num,const int* tree_num_shadow,
    const int* override_enabled,
    const int* position_valid,const int64_t* position_written,const int64_t* forward_epoch,
    const int* first_error,double* fat_bounds,int* cache_valid,int* cache_mode,int* cache_age,
    int* reason,int64_t* hit_count,int64_t* refresh_count,int64_t* fallback_count,
    double absolute_padding,double relative_padding,int max_age) {
    __shared__ int wb[32], mode;
    __shared__ double matrix_norms[32], world_norm_squared;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    bool bad=cf_temporal_static_bad_lane(nbody,ngeom,body_geomadr,body_geomnum,geom_bodyid,
        body_weldid,body_parentid,can,plane,geom_type,geom_contype,geom_conaffinity,
        gates,collision_mode,filterparent_disabled,geom_rbound,geom_margin,integer_shadow,float_shadow);
    bad|=cf_temporal_int_lane(tree_body_num,tree_num_shadow,nbody,0);
    bad|=override_enabled[0]!=0;
    if(lane==0)wb[warp]=0;
    __syncwarp(0xffffffffu);
    const int any=__any_sync(0xffffffffu,bad);
    if(lane==0)wb[warp]=any;
    __syncthreads();
    if(threadIdx.x==0){
        int why=0;for(int k=0;k<(blockDim.x>>5);++k)if(wb[k])why=1;
        if(active[w]!=1 || gates[0] || gates[1] || !gates[2] || collision_mode[0]==1 || nbody<2)why=2;
        if(position_valid[w]!=1 || first_error[w] || position_written[w]!=forward_epoch[w])why=3;
        reason[w]=why;mode=why?0:2;
        cache_mode[w]=mode;
        if(!mode){cache_valid[w]=0;++fallback_count[w];}
    }
    __syncthreads();
    if(!mode){
        cf_bounds_block(w,xpos,active,gates,collision_mode,geom_rbound,geom_margin,
            body_geomadr,body_geomnum,geom_bodyid,aabb,frame,entered,count,status,scratch,ngeom,nbody);
        return;
    }
    // Validate all computed endpoints before the first public bounds write.
    // A NaN/error fallback must retain exactly the original partial-write ABI.
    bool invalid_endpoint=false;double local_norm_squared=1.0;
    for(int g=threadIdx.x;g<ngeom;g+=blockDim.x){
        const double* matrix=geom_matrix+(w*ngeom+g)*9;
        for(int k=0;k<9;++k)invalid_endpoint|=!isfinite(matrix[k]);
        const double norm_squared=cf_temporal_norm_squared(matrix);
        current_radius[w*ngeom+g]=norm_squared;
        invalid_endpoint|=!isfinite(norm_squared);
        local_norm_squared=fmax(local_norm_squared,norm_squared);
    }
    for(int offset=16;offset;offset>>=1)
        local_norm_squared=fmax(local_norm_squared,__shfl_down_sync(0xffffffffu,local_norm_squared,offset));
    if(lane==0)matrix_norms[warp]=local_norm_squared;
    __syncthreads();
    if(threadIdx.x==0){
        world_norm_squared=1.0;
        for(int k=0;k<(blockDim.x>>5);++k)world_norm_squared=fmax(world_norm_squared,matrix_norms[k]);
    }
    __syncthreads();
    for(int g=threadIdx.x;g<ngeom;g+=blockDim.x){
        const double r=cf_temporal_radius(current_radius[w*ngeom+g],world_norm_squared,
                                          geom_rbound[g],geom_margin[g]);
        current_radius[w*ngeom+g]=r;
        invalid_endpoint|=!isfinite(r) || r<0;
        for(int k=0;k<3;++k){
            const double c=xpos[(w*ngeom+g)*3+k];
            invalid_endpoint|=!isfinite(c) || !isfinite(__dadd_rd(c,-r)) || !isfinite(__dadd_ru(c,r));
        }
    }
    const int invalid=__any_sync(0xffffffffu,invalid_endpoint);
    if(lane==0)wb[warp]=invalid;
    __syncthreads();
    if(threadIdx.x==0){
        for(int k=0;k<(blockDim.x>>5);++k)if(wb[k]){mode=0;reason[w]=4;cache_valid[w]=0;cache_mode[w]=0;++fallback_count[w];break;}
    }
    __syncthreads();
    if(!mode){
        cf_bounds_block(w,xpos,active,gates,collision_mode,geom_rbound,geom_margin,
            body_geomadr,body_geomnum,geom_bodyid,aabb,frame,entered,count,status,scratch,ngeom,nbody);
        return;
    }
    bool nonfinite=false,outside=false;
    double* ab=aabb+w*nbody*6;const double* fat=fat_bounds+w*nbody*6;
    for(int t=threadIdx.x;t<nbody*3;t+=blockDim.x){
        const int b=t/3,k=t%3;double lo=0.0,hi=0.0;
        if(b && body_geomnum[b])for(int j=0;j<body_geomnum[b];++j){
            const int g=body_geomadr[b]+j;
            const double center=xpos[(w*ngeom+g)*3+k];
            const double r=current_radius[w*ngeom+g];
            const double l=__dadd_rd(center,-r),h=__dadd_ru(center,r);
            nonfinite|=!isfinite(center) || !isfinite(r) || r<0 || !isfinite(l) || !isfinite(h);
            if(!j){lo=l;hi=h;}else{lo=fmin(lo,l);hi=fmax(hi,h);}
        }
        ab[6*b+2*k]=lo;ab[6*b+2*k+1]=hi;
        if(b && body_geomnum[b])outside|=!(lo>fat[6*b+2*k] && hi<fat[6*b+2*k+1]);
    }
    const int failed=__any_sync(0xffffffffu,nonfinite),escaped=__any_sync(0xffffffffu,outside);
    if(lane==0)wb[warp]=failed+2*escaped;
    __syncthreads();
    if(threadIdx.x==0){
        int failed_all=0,escaped_all=0;
        for(int k=0;k<(blockDim.x>>5);++k){failed_all|=wb[k]&1;escaped_all|=wb[k]&2;}
        if(failed_all){mode=0;reason[w]=4;cache_valid[w]=0;++fallback_count[w];}
        else {
            mode=cache_valid[w] && !escaped_all && cache_age[w]<max_age ? 1:2;
            reason[w]=mode==1?0:(!cache_valid[w]?5:(escaped_all?6:7));
            status[w]=0;entered[w]=1;int total=0;
            for(int g=0;g<ngeom;++g)total+=geom_bodyid[g]!=0;count[w]=total;
            for(int k=0;k<9;++k)frame[w*9+k]=(k%4==0)?1.0:0.0;
            // Fast-profile PCA scratch has no consumer. Keep its existing
            // owner rather than writing a fictitious eigendecomposition.
        }
        cache_mode[w]=mode;
    }
    __syncthreads();
    if(!mode){
        cf_bounds_block(w,xpos,active,gates,collision_mode,geom_rbound,geom_margin,
            body_geomadr,body_geomnum,geom_bodyid,aabb,frame,entered,count,status,scratch,ngeom,nbody);
        return;
    }
    if(mode==2)for(int t=threadIdx.x;t<nbody*3;t+=blockDim.x){
        const int b=t/3,k=t%3;const double lo=ab[6*b+2*k],hi=ab[6*b+2*k+1];
        const double scale=fmax(1.0,fmax(fabs(lo),fabs(hi)));
        const double pad=__dadd_ru(absolute_padding,__dmul_ru(relative_padding,scale));
        fat_bounds[w*nbody*6+6*b+2*k]=fmax(-DBL_MAX,__dadd_rd(lo,-pad));
        fat_bounds[w*nbody*6+6*b+2*k+1]=fmin(DBL_MAX,__dadd_ru(hi,pad));
    }
    __syncthreads();
}
// Static pairs are sorted by original packed body key. Both refresh and hit
// preserve this order; stagebuild still performs original explicit merges.
__device__ inline void cf_temporal_broadphase_block(int w,
    const double* aabb,const int* active,const int* gates,const int* collision_mode,
    const int* body_weldid,const int* body_parentid,const int* body_geomadr,
    const int* body_geomnum,const int* geom_bodyid,const int* geom_type,
    const int* geom_contype,const int* geom_conaffinity,const int* can,const int* plane,
    const int* filterparent_disabled,int* out_pair,int* out_npair,int* status,
    float* sortbuf,int* sorttag,int* activebuf,int nbody,int pair_capacity,
    const int* static_pair,const int* static_global,int static_count,const double* fat_bounds,
    int* cache_pair,int* cache_global,int* cache_count,int* cache_valid,const int* cache_mode,
    int* cache_age,int* flags,int* prefix,int64_t* hit_count,int64_t* refresh_count) {
    __shared__ int n,overflow;
    if(cache_mode[w]==0){
        cf_broadphase_block(w,aabb,active,gates,collision_mode,body_weldid,body_parentid,
            body_geomadr,body_geomnum,geom_bodyid,geom_type,geom_contype,geom_conaffinity,
            can,plane,filterparent_disabled,out_pair,out_npair,status,sortbuf,sorttag,
            activebuf,nbody,pair_capacity);return;
    }
    int* f=flags+w*static_count;int* p=prefix+w*(static_count+1);
    if(cache_mode[w]==2){
        const double* fat=fat_bounds+w*nbody*6;
        for(int i=threadIdx.x;i<static_count;i+=blockDim.x){
            const int key=static_pair[i],b1=(key>>16)&65535,b2=key&65535;
            f[i]=static_global[i] || cf_temporal_overlap(fat,b1,b2);
        }
        __syncthreads();
        if(threadIdx.x==0){int total=0;for(int i=0;i<static_count;++i){p[i]=total;total+=f[i];}p[static_count]=total;n=total;}
        __syncthreads();
        for(int i=threadIdx.x;i<static_count;i+=blockDim.x)if(f[i]){
            const int dst=w*static_count+p[i];cache_pair[dst]=static_pair[i];cache_global[dst]=static_global[i];
        }
        if(threadIdx.x==0){cache_count[w]=n;cache_age[w]=0;cache_valid[w]=1;++refresh_count[w];}
        __syncthreads();
    }
    if(threadIdx.x==0)n=cache_count[w];
    __syncthreads();
    const double* ab=aabb+w*nbody*6;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){
        const int key=cache_pair[w*static_count+i],b1=(key>>16)&65535,b2=key&65535;
        f[i]=cache_global[w*static_count+i] || cf_temporal_overlap(ab,b1,b2);
    }
    __syncthreads();
    if(threadIdx.x==0){
        int total=0;for(int i=0;i<n;++i){p[i]=total;total+=f[i];}p[n]=total;
        overflow=total>pair_capacity;
        if(!overflow){status[w]=0;out_npair[w]=total;++cache_age[w];if(cache_mode[w]==1)++hit_count[w];}
        else cache_valid[w]=0;
    }
    __syncthreads();
    if(overflow){
        // No optimized public pair has been committed. Retry original SAP.
        cf_broadphase_block(w,aabb,active,gates,collision_mode,body_weldid,body_parentid,
            body_geomadr,body_geomnum,geom_bodyid,geom_type,geom_contype,geom_conaffinity,
            can,plane,filterparent_disabled,out_pair,out_npair,status,sortbuf,sorttag,
            activebuf,nbody,pair_capacity);return;
    }
    for(int i=threadIdx.x;i<n;i+=blockDim.x)if(f[i])out_pair[w*pair_capacity+p[i]]=cache_pair[w*static_count+i];
    __syncthreads();
}
} // namespace collision_cuda
