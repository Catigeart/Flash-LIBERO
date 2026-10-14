#pragma once
namespace libero_scene {
// NEW private ABI. body pose publication is F's actual committed device output,
// not an enqueue sequence or qpos epoch. Static/read/output buffers are private
// cold owners. Only this scene stage writes output; reset/rebind creates new ready.
__device__ void camera_private_publication(int w,int nb,int nc,
 const double* bp,const double* bm,const int* mask,const int64_t* pose_generation,
 const int* pose_valid,const int* body,const double* lp,const double* lm,
 double* op,double* om,int64_t* retained_pose,int* ready,int* reused,int* computed){
 if(!mask[w]||!pose_valid[w])return;
 const int64_t generation=pose_generation[w];
 if(generation<=0)return;
 for(int c=threadIdx.x;c<nc;c+=blockDim.x){int o=w*nc+c,b=w*nb+body[c];
  if(ready[o]&&retained_pose[o]==generation){reused[o]+=1;continue;}
  for(int i=0;i<3;++i){double p=bp[3*b+i];
   for(int k=0;k<3;++k)p+=bm[9*b+3*i+k]*lp[3*c+k];op[3*o+i]=p;
   for(int j=0;j<3;++j){double v=0;for(int k=0;k<3;++k)v+=bm[9*b+3*i+k]*lm[9*c+3*k+j];om[9*o+3*i+j]=v;}
  }
  retained_pose[o]=generation;ready[o]=1;computed[o]+=1;
 }
}
__device__ void light_private_publication(int w,int nb,int nl,
 const double* bp,const double* bm,const int* mask,const int64_t* pose_generation,
 const int* pose_valid,const int* body,const double* lp,const double* ld,
 double* op,double* od,int64_t* retained_pose,int* ready,int* reused,int* computed){
 if(!mask[w]||!pose_valid[w])return;
 const int64_t generation=pose_generation[w];if(generation<=0)return;
 for(int c=threadIdx.x;c<nl;c+=blockDim.x){int o=w*nl+c,b=w*nb+body[c];
  if(ready[o]&&retained_pose[o]==generation){reused[o]+=1;continue;}
  for(int i=0;i<3;++i){double p=bp[3*b+i],d=0.;
   for(int k=0;k<3;++k){p+=bm[9*b+3*i+k]*lp[3*c+k];d+=bm[9*b+3*i+k]*ld[3*c+k];}op[3*o+i]=p;od[3*o+i]=d;
  }
  retained_pose[o]=generation;ready[o]=1;computed[o]+=1;
 }
}
}
