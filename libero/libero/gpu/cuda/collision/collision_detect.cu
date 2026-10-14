#include "collision_detect.cuh"
extern "C" size_t collision_detect_params_size() {return sizeof(libero_collision_cuda::Params);}
__global__ void collision_detect_producer(libero_collision_cuda::Params* p) {
 const int w=blockIdx.y,bucket=blockIdx.z;
 const bool collective=p->query_subgroup_width==32&&
     (bucket==5||bucket==11||bucket==12||bucket==13);
 const int worker=collective?threadIdx.x/32:threadIdx.x;
 const int workers=collective?2:64;
 for(int q=p->bucket_offsets[16*w+bucket]+blockIdx.x*workers+worker;
     q<p->bucket_offsets[16*w+bucket+1];q+=gridDim.x*workers) {
  const int j=p->query_order[(size_t)w*p->pairs+q];
  libero_collision_cuda::detect_scheduled_query(*p,w,j,bucket,collective);
 }
}
__global__ void collision_detect_prepare(libero_collision_cuda::Params* p) {
 const int w=blockIdx.y;
 if(threadIdx.x==0)atomicAdd((unsigned long long*)(p->cuda_detector_execution_count+w),1ULL);
 libero_collision_cuda::detect_prepare_routes(*p,w);
}
__global__ void collision_detect_commit(libero_collision_cuda::Params* p) {
 libero_collision_cuda::detect_commit_world(*p,blockIdx.y);
}
__global__ void collision_detect_collective(libero_collision_cuda::Params* p) {
 libero_collision_cuda::detect_world(*p,blockIdx.y);
}
extern "C" int launch_collision_detect(void* p,int worlds,void* stream) {
 collision_detect_collective<<<dim3(1,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 return (int)cudaGetLastError();
}
extern "C" int launch_collision_detect_tiled(void* p,int worlds,int pairs,void* stream) {
 collision_detect_prepare<<<dim3(1,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 int code=(int)cudaGetLastError();if(code)return code;
 // Route is uniform across each block; warp queries use grid-stride work.
 collision_detect_producer<<<dim3((pairs+63)/64,worlds,15),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 code=(int)cudaGetLastError();if(code)return code;
 collision_detect_commit<<<dim3(1,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 return (int)cudaGetLastError();
}

__global__ void collision_detect_compact_producer(libero_collision_cuda::Params* p) {
 const int w=blockIdx.y;
 const size_t base=(size_t)w*((p->pairs+1)/2+15);
 for(int task=blockIdx.x;task<p->task_count[w];task+=gridDim.x) {
  const int bucket=p->task_bucket[base+task],start=p->task_start[base+task];
  const bool collective=p->query_subgroup_width==32&&
      (bucket==5||bucket==11||bucket==12||bucket==13);
  const int worker=collective?threadIdx.x/32:threadIdx.x;
  const int q=start+worker;
  if(q<p->bucket_offsets[16*w+bucket+1]) {
   const int j=p->query_order[(size_t)w*p->pairs+q];
   libero_collision_cuda::detect_scheduled_query(*p,w,j,bucket,collective);
  }
  // All warp collectives finish before this CTA changes its uniform route.
  __syncthreads();
 }
}
extern "C" int launch_collision_detect_compact_producer(void* p,int worlds,int blocks,void* stream) {
 if(worlds<=0||blocks<=0)return (int)cudaErrorInvalidValue;
 collision_detect_compact_producer<<<dim3(blocks,worlds),64,0,(cudaStream_t)stream>>>((libero_collision_cuda::Params*)p);
 return (int)cudaGetLastError();
}
