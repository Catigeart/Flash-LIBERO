// S07 standalone exact integer cache primitives. Production is unchanged.
#pragma once
namespace cf_s {
__device__ void topology_cache_probe_serial(int w,const int* current,const int* saved,
  const int* valid,const int* domain,const int* saved_domain,int* hit,int extent){
  if(threadIdx.x==0){
    bool bad=valid[w]!=1||domain[w]!=saved_domain[w];
    for(int i=0;i<extent;i++)bad|=current[w*extent+i]!=saved[w*extent+i];
    hit[w]=bad?0:1;
  }
  __syncthreads();
}
__device__ void topology_cache_probe(int w,const int* current,const int* saved,
  const int* valid,const int* domain,const int* saved_domain,int* hit,int extent){
  __shared__ int mismatch[32];
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  bool bad=valid[w]!=1||domain[w]!=saved_domain[w];
  for(int i=threadIdx.x;i<extent;i+=blockDim.x)bad|=current[w*extent+i]!=saved[w*extent+i];
  const bool vote=libero_warp::any(bad);
  if(lane==0)mismatch[warp]=vote;
  __syncthreads();
  if(warp==0){
    const bool any=libero_warp::any(lane<(blockDim.x>>5)&&mismatch[lane]);
    if(lane==0)hit[w]=any?0:1;
  }
  __syncthreads();
}
__device__ void topology_cache_restore(int w,const int* hit,const int* payload,int* output,const int* lengths,int extent){
  if(hit[w]!=1)return;
  const int n=lengths[w];if(n<0||n>extent)return;
  for(int i=threadIdx.x;i<n;i+=blockDim.x)output[w*extent+i]=payload[w*extent+i];
}
}
