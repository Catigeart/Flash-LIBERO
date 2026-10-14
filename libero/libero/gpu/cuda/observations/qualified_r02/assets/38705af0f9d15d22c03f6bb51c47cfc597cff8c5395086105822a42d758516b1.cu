#include <cuda_runtime.h>
#include <initializer_list>
#include "cuda/common/vector.cuh"
#include "cuda/observations/observations.cuh"
#define NS 37
#define WIDTH 153
#define STRIDE 354
#define NINT 1177
#define NSTATE 854
#define NFLAGS 112
#define META_SAMPLED 0
#define META_PRESENT 37
#define META_EVENTS 74
#define META_MASK 111
#define SAMPLER_ARGS w,NS,WIDTH,STRIDE,c.integers+0,c.integers+37,c.integers+74,c.integers+111,c.integers+148,c.integers+222,c.integers+259,c.integers+296,c.integers+333,c.integers+444,c.periods,c.integers+481,c.integers+486,c.raw+0,c.raw+58,c.raw+109,c.raw+187,c.raw+291,c.state+500,0,c.state+0,c.state+153,c.state+306,c.state+463,c.flags+0,c.flags+37,c.flags+111,dt,force,c.flags+74
#define DAG_ARGS 4,74,c.integers+491,c.integers+496,c.integers+533,c.integers+977,c.integers+982,c.integers+987,c.integers+1167
struct Context {
  int* frozen_host=nullptr;bool revoked=false;
  int* integers;double* periods;double* raw;double* state;int* flags;
  double* initial_state;int* initial_flags;unsigned long long* samples;
  cudaStream_t stream;cudaEvent_t start,stop;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
};
__global__ void sample_leaf(Context c,int force,double dt) {
  int w=0;
  if(!c.flags[META_MASK])return;
#if CF_ROUTE == 0
  if(threadIdx.x==0)libero_observe::lowdim(SAMPLER_ARGS);
#elif CF_ROUTE == 1
  libero_observe::lowdim_dag(SAMPLER_ARGS,DAG_ARGS);
#elif CF_ROUTE == 2
  libero_observe::lowdim_dag_sensor(SAMPLER_ARGS,DAG_ARGS);
#else
  libero_observe::lowdim_dag_owned_prototype(SAMPLER_ARGS,DAG_ARGS);
#endif
  __syncthreads();
  for(int sensor=threadIdx.x;sensor<NS;sensor+=blockDim.x)
    if(c.flags[META_EVENTS+sensor])atomicAdd(c.samples,1ULL);
}
#define OK(call) do { cudaError_t status=(call); if(status!=cudaSuccess)return int(status); } while(0)
extern "C" int create(void** result,const int* integers,const double* periods) {
  Context* c=new Context{};*result=c;
  c->frozen_host=new int[NINT];for(int i=0;i<NINT;i++)c->frozen_host[i]=integers[i];
  OK(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));
  OK(cudaEventCreate(&c->start));OK(cudaEventCreate(&c->stop));
  OK(cudaMalloc(&c->integers,NINT*sizeof(int)));OK(cudaMalloc(&c->periods,NS*sizeof(double)));
  OK(cudaMalloc(&c->raw,STRIDE*sizeof(double)));OK(cudaMalloc(&c->state,NSTATE*sizeof(double)));
  OK(cudaMalloc(&c->flags,NFLAGS*sizeof(int)));OK(cudaMalloc(&c->initial_state,NSTATE*sizeof(double)));
  OK(cudaMalloc(&c->initial_flags,NFLAGS*sizeof(int)));OK(cudaMalloc(&c->samples,sizeof(unsigned long long)));
  OK(cudaMemcpyAsync(c->integers,integers,NINT*sizeof(int),cudaMemcpyHostToDevice,c->stream));
  OK(cudaMemcpyAsync(c->periods,periods,NS*sizeof(double),cudaMemcpyHostToDevice,c->stream));
  OK(cudaMemsetAsync(c->samples,0,sizeof(unsigned long long),c->stream));
  OK(cudaStreamSynchronize(c->stream));return 0;
}
extern "C" int upload(void* ptr,const double* raw,const double* state,const int* flags) {
  Context& c=*static_cast<Context*>(ptr);
  if(c.revoked)return -7;
  OK(cudaMemcpyAsync(c.raw,raw,STRIDE*sizeof(double),cudaMemcpyHostToDevice,c.stream));
  OK(cudaMemcpyAsync(c.initial_state,state,NSTATE*sizeof(double),cudaMemcpyHostToDevice,c.stream));
  OK(cudaMemcpyAsync(c.initial_flags,flags,NFLAGS*sizeof(int),cudaMemcpyHostToDevice,c.stream));
  OK(cudaMemcpyAsync(c.state,c.initial_state,NSTATE*sizeof(double),cudaMemcpyDeviceToDevice,c.stream));
  OK(cudaMemcpyAsync(c.flags,c.initial_flags,NFLAGS*sizeof(int),cudaMemcpyDeviceToDevice,c.stream));
  OK(cudaMemsetAsync(c.samples,0,sizeof(unsigned long long),c.stream));
  OK(cudaStreamSynchronize(c.stream));return 0;
}
extern "C" int mutate(void* ptr,const int* integers) {
  Context& c=*static_cast<Context*>(ptr);
  if(c.revoked)return -7;
  for(int i=0;i<NINT;i++)if(integers[i]!=c.frozen_host[i]){c.revoked=true;return -6;}
  OK(cudaMemcpyAsync(c.integers,integers,NINT*sizeof(int),cudaMemcpyHostToDevice,c.stream));
  OK(cudaStreamSynchronize(c.stream));return 0;
}
extern "C" int step(void* ptr,int force,double dt) {
  Context& c=*static_cast<Context*>(ptr);
  if(c.revoked)return -7;
  sample_leaf<<<1,64,0,c.stream>>>(c,force,dt);OK(cudaGetLastError());
  OK(cudaStreamSynchronize(c.stream));return 0;
}
extern "C" int read(void* ptr,double* state,int* flags,unsigned long long* samples) {
  Context& c=*static_cast<Context*>(ptr);
  OK(cudaMemcpy(state,c.state,NSTATE*sizeof(double),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(flags,c.flags,NFLAGS*sizeof(int),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(samples,c.samples,sizeof(unsigned long long),cudaMemcpyDeviceToHost));return 0;
}
extern "C" int capture(void* ptr,int batch,int force,double dt) {
  Context& c=*static_cast<Context*>(ptr);
  if(c.revoked)return -7;if(batch<8)return -2;
  if(c.executable){OK(cudaGraphExecDestroy(c.executable));c.executable=nullptr;}
  if(c.graph){OK(cudaGraphDestroy(c.graph));c.graph=nullptr;}
  OK(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeThreadLocal));
  for(int i=0;i<batch;i++)sample_leaf<<<1,64,0,c.stream>>>(c,force,dt);
  OK(cudaStreamEndCapture(c.stream,&c.graph));
  OK(cudaGraphInstantiate(&c.executable,c.graph,nullptr,nullptr,0));return 0;
}
extern "C" int measure(void* ptr,float* milliseconds) {
  Context& c=*static_cast<Context*>(ptr);
  if(c.revoked)return -7;if(!c.executable)return -3;
  OK(cudaMemcpyAsync(c.state,c.initial_state,NSTATE*sizeof(double),cudaMemcpyDeviceToDevice,c.stream));
  OK(cudaMemcpyAsync(c.flags,c.initial_flags,NFLAGS*sizeof(int),cudaMemcpyDeviceToDevice,c.stream));
  OK(cudaMemsetAsync(c.samples,0,sizeof(unsigned long long),c.stream));
  OK(cudaEventRecord(c.start,c.stream));OK(cudaGraphLaunch(c.executable,c.stream));
  OK(cudaEventRecord(c.stop,c.stream));OK(cudaEventSynchronize(c.stop));
  OK(cudaEventElapsedTime(milliseconds,c.start,c.stop));return 0;
}
extern "C" int destroy(void* ptr) {
  if(!ptr)return 0;Context& c=*static_cast<Context*>(ptr);
  cudaError_t first=cudaStreamSynchronize(c.stream);
  auto release=[&](cudaError_t status){if(first==cudaSuccess&&status!=cudaSuccess)first=status;};
  if(c.executable)release(cudaGraphExecDestroy(c.executable));if(c.graph)release(cudaGraphDestroy(c.graph));
  for(void* allocation : {static_cast<void*>(c.integers),static_cast<void*>(c.periods),static_cast<void*>(c.raw),
      static_cast<void*>(c.state),static_cast<void*>(c.flags),static_cast<void*>(c.initial_state),
      static_cast<void*>(c.initial_flags),static_cast<void*>(c.samples)})if(allocation)release(cudaFree(allocation));
  if(c.start)release(cudaEventDestroy(c.start));if(c.stop)release(cudaEventDestroy(c.stop));
  if(c.stream)release(cudaStreamDestroy(c.stream));delete[] c.frozen_host;delete &c;return int(first);
}
