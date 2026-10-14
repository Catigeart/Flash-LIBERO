#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <initializer_list>
#include "cuda/common/vector.cuh"
#include "cuda/observations/position_scene.cuh"
// Exclusive small independent private pose writer and native scene consumer.
// Neither input/output GPU pointers nor static buffers leave this opaque ABI.
struct Component {
  int count,width;int *body,*oldbody,*ready,*reused,*computed;
  double *local_pos,*local_frame,*pos,*frame,*snapshot;int64_t* retained;
};
struct Context {
  int nb;double *frameA,*frameB,*bp,*bm;int *mask,*valid;int64_t* generation;
  Component camera,light;cudaStream_t stream;cudaEvent_t start,stop;
  cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
};
__global__ void commit_pose(Context c,int which) {
  if(!c.mask[0])return;
  __shared__ int bad;if(threadIdx.x==0)bad=0;__syncthreads();
  const double* source=which?c.frameB:c.frameA;
  for(int i=threadIdx.x;i<c.nb*12;i+=blockDim.x)if(!isfinite(source[i]))atomicExch(&bad,1);
  __syncthreads();
  if(bad||c.generation[0]==INT64_MAX){if(threadIdx.x==0)c.valid[0]=0;return;}
  for(int i=threadIdx.x;i<c.nb*3;i+=blockDim.x)c.bp[i]=source[i];
  for(int i=threadIdx.x;i<c.nb*9;i+=blockDim.x)c.bm[i]=source[c.nb*3+i];
  __syncthreads();
  if(threadIdx.x==0){c.generation[0]+=1;c.valid[0]=1;}
}
__global__ void scene_leaf(Context c) {
  if(!c.mask[0]||!c.valid[0]||c.generation[0]<=0)return;
  Component a=c.camera,b=c.light;
#if CF_PRIVATE
  libero_scene::camera_private_publication(0,c.nb,a.count,c.bp,c.bm,c.mask,c.generation,c.valid,
    a.body,a.local_pos,a.local_frame,a.pos,a.frame,a.retained,a.ready,a.reused,a.computed);
  libero_scene::light_private_publication(0,c.nb,b.count,c.bp,c.bm,c.mask,c.generation,c.valid,
    b.body,b.local_pos,b.local_frame,b.pos,b.frame,b.retained,b.ready,b.reused,b.computed);
#else
  libero_scene::camera_cached(0,c.nb,a.count,c.bp,c.bm,c.mask,a.body,a.local_pos,a.local_frame,
    a.pos,a.frame,a.snapshot,a.oldbody,a.ready,a.reused,a.computed);
  libero_scene::light_cached(0,c.nb,b.count,c.bp,c.bm,c.mask,b.body,b.local_pos,b.local_frame,
    b.pos,b.frame,b.snapshot,b.oldbody,b.ready,b.reused,b.computed);
#endif
}
#define OK(call) do { cudaError_t status=(call); if(status!=cudaSuccess)return int(status); } while(0)
int allocate_component(Component& a,int count,int width,const int* body,const double* position,const double* frame,cudaStream_t stream) {
  a.count=count;a.width=width;
  for(int** p : {&a.body,&a.oldbody,&a.ready,&a.reused,&a.computed}){OK(cudaMalloc(p,count*sizeof(int)));OK(cudaMemsetAsync(*p,0,count*sizeof(int),stream));}
  OK(cudaMalloc(&a.local_pos,count*3*sizeof(double)));OK(cudaMalloc(&a.local_frame,count*width*sizeof(double)));
  OK(cudaMalloc(&a.pos,count*3*sizeof(double)));OK(cudaMalloc(&a.frame,count*width*sizeof(double)));
  OK(cudaMalloc(&a.snapshot,count*(width==9?36:24)*sizeof(double)));OK(cudaMalloc(&a.retained,count*sizeof(int64_t)));
  OK(cudaMemsetAsync(a.pos,0,count*3*sizeof(double),stream));OK(cudaMemsetAsync(a.frame,0,count*width*sizeof(double),stream));
  OK(cudaMemcpyAsync(a.body,body,count*sizeof(int),cudaMemcpyHostToDevice,stream));
  OK(cudaMemcpyAsync(a.local_pos,position,count*3*sizeof(double),cudaMemcpyHostToDevice,stream));
  OK(cudaMemcpyAsync(a.local_frame,frame,count*width*sizeof(double),cudaMemcpyHostToDevice,stream));return 0;
}
extern "C" int create(int nb,int nc,int nl,const int* body,const double* local,void** result) {
  if(nb<1||nc<1||nl<1)return -1;
  for(int i=0;i<nc+nl;i++)if(body[i]<0||body[i]>=nb)return -2;
  Context* c=new Context{};c->nb=nb;*result=c;
  OK(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));OK(cudaEventCreate(&c->start));OK(cudaEventCreate(&c->stop));
  OK(cudaMalloc(&c->frameA,nb*12*sizeof(double)));OK(cudaMalloc(&c->frameB,nb*12*sizeof(double)));
  OK(cudaMalloc(&c->bp,nb*3*sizeof(double)));OK(cudaMalloc(&c->bm,nb*9*sizeof(double)));
  OK(cudaMalloc(&c->mask,sizeof(int)));OK(cudaMalloc(&c->valid,sizeof(int)));OK(cudaMalloc(&c->generation,sizeof(int64_t)));
  OK(cudaMemsetAsync(c->valid,0,sizeof(int),c->stream));OK(cudaMemsetAsync(c->generation,0,sizeof(int64_t),c->stream));
  int code=allocate_component(c->camera,nc,9,body,local,local+nc*3,c->stream);if(code)return code;
  code=allocate_component(c->light,nl,3,body+nc,local+nc*12,local+nc*12+nl*3,c->stream);if(code)return code;
  OK(cudaStreamSynchronize(c->stream));return 0;
}
extern "C" int upload(void* ptr,const double* a,const double* b,int publish) {
  Context& c=*static_cast<Context*>(ptr);int mask=1;
  OK(cudaMemcpyAsync(c.frameA,a,c.nb*12*sizeof(double),cudaMemcpyHostToDevice,c.stream));
  OK(cudaMemcpyAsync(c.frameB,b,c.nb*12*sizeof(double),cudaMemcpyHostToDevice,c.stream));
  OK(cudaMemcpyAsync(c.mask,&mask,sizeof(int),cudaMemcpyHostToDevice,c.stream));
  if(publish)commit_pose<<<1,64,0,c.stream>>>(c,1);
  else OK(cudaMemsetAsync(c.valid,0,sizeof(int),c.stream));
  OK(cudaGetLastError());OK(cudaStreamSynchronize(c.stream));return 0;
}
extern "C" int step(void* ptr,int mask,int valid,int reset) {
  Context& c=*static_cast<Context*>(ptr);
  OK(cudaMemcpyAsync(c.mask,&mask,sizeof(int),cudaMemcpyHostToDevice,c.stream));
  if(!valid)OK(cudaMemsetAsync(c.valid,0,sizeof(int),c.stream));
  if(reset){OK(cudaMemsetAsync(c.camera.ready,0,c.camera.count*sizeof(int),c.stream));OK(cudaMemsetAsync(c.light.ready,0,c.light.count*sizeof(int),c.stream));}
  scene_leaf<<<1,64,0,c.stream>>>(c);OK(cudaGetLastError());OK(cudaStreamSynchronize(c.stream));return 0;
}
extern "C" int read(void* ptr,double* output,int* reused,int* computed,int64_t* generation,int* valid) {
  Context& c=*static_cast<Context*>(ptr);Component a=c.camera,b=c.light;
  OK(cudaMemcpy(output,a.pos,a.count*3*sizeof(double),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(output+a.count*3,a.frame,a.count*9*sizeof(double),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(output+a.count*12,b.pos,b.count*3*sizeof(double),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(output+a.count*12+b.count*3,b.frame,b.count*3*sizeof(double),cudaMemcpyDeviceToHost));
  for(Component item : {a,b}) {
    OK(cudaMemcpy(reused,item.reused,item.count*sizeof(int),cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(computed,item.computed,item.count*sizeof(int),cudaMemcpyDeviceToHost));
    reused+=item.count;computed+=item.count;
  }
  OK(cudaMemcpy(generation,c.generation,sizeof(int64_t),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(valid,c.valid,sizeof(int),cudaMemcpyDeviceToHost));return 0;
}
extern "C" int capture(void* ptr,int batch,int changing) {
  Context& c=*static_cast<Context*>(ptr);if(batch<8)return -3;
  if(c.executable){OK(cudaGraphExecDestroy(c.executable));c.executable=nullptr;}
  if(c.graph){OK(cudaGraphDestroy(c.graph));c.graph=nullptr;}
  OK(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeThreadLocal));
  for(int i=0;i<batch;i++){
    if(changing)commit_pose<<<1,64,0,c.stream>>>(c,i&1);
    scene_leaf<<<1,64,0,c.stream>>>(c);
  }
  OK(cudaStreamEndCapture(c.stream,&c.graph));OK(cudaGraphInstantiate(&c.executable,c.graph,nullptr,nullptr,0));return 0;
}
extern "C" int measure(void* ptr,float* milliseconds) {
  Context& c=*static_cast<Context*>(ptr);if(!c.executable)return -4;
  OK(cudaEventRecord(c.start,c.stream));OK(cudaGraphLaunch(c.executable,c.stream));
  OK(cudaEventRecord(c.stop,c.stream));OK(cudaEventSynchronize(c.stop));
  OK(cudaEventElapsedTime(milliseconds,c.start,c.stop));return 0;
}
extern "C" int destroy(void* ptr) {
  if(!ptr)return 0;Context& c=*static_cast<Context*>(ptr);cudaError_t first=cudaStreamSynchronize(c.stream);
  auto release=[&](cudaError_t status){if(first==cudaSuccess&&status!=cudaSuccess)first=status;};
  if(c.executable)release(cudaGraphExecDestroy(c.executable));if(c.graph)release(cudaGraphDestroy(c.graph));
  for(Component a : {c.camera,c.light})for(void* p : {static_cast<void*>(a.body),static_cast<void*>(a.oldbody),static_cast<void*>(a.ready),static_cast<void*>(a.reused),static_cast<void*>(a.computed),static_cast<void*>(a.local_pos),static_cast<void*>(a.local_frame),static_cast<void*>(a.pos),static_cast<void*>(a.frame),static_cast<void*>(a.snapshot),static_cast<void*>(a.retained)})if(p)release(cudaFree(p));
  for(void* p : {static_cast<void*>(c.frameA),static_cast<void*>(c.frameB),static_cast<void*>(c.bp),static_cast<void*>(c.bm),static_cast<void*>(c.mask),static_cast<void*>(c.valid),static_cast<void*>(c.generation)})if(p)release(cudaFree(p));
  if(c.start)release(cudaEventDestroy(c.start));if(c.stop)release(cudaEventDestroy(c.stop));if(c.stream)release(cudaStreamDestroy(c.stream));delete &c;return int(first);
}
