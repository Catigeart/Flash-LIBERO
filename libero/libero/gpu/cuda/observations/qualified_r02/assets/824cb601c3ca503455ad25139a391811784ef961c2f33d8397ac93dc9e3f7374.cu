#include <cuda_runtime.h>
#include <math.h>
#include <initializer_list>
namespace libero_observe {
__device__ __forceinline__ void matrix_quaternion32(const float* m,double* out){
 float k[16]={
 m[0]-m[4]-m[8],m[1]+m[3],m[2]+m[6],m[7]-m[5],
 m[1]+m[3],m[4]-m[0]-m[8],m[5]+m[7],m[2]-m[6],
 m[2]+m[6],m[5]+m[7],m[8]-m[0]-m[4],m[3]-m[1],
 m[7]-m[5],m[2]-m[6],m[3]-m[1],m[0]+m[4]+m[8]};
 for(int i=0;i<16;++i)k[i]/=3.f;
 int largest=0;for(int i=1;i<4;++i)if(k[5*i]>k[5*largest])largest=i;
 float q[4]={0,0,0,0};q[largest]=1;
 for(int it=0;it<24;++it){float v[4]={0,0,0,0};
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)v[i]+=k[4*i+j]*q[j];
  float n=0;for(int i=0;i<4;++i)n+=v[i]*v[i];n=sqrtf(n);
  for(int i=0;i<4;++i)q[i]=v[i]/n;
 }
 for(int i=0;i<4;++i)out[i]=q[3]<0?-double(q[i]):double(q[i]);
}
__device__ void matrix_quaternion32_memo(const float* m,double* out,float* key,double* value,int* ready,int* hits){
 bool hit=*ready!=0;for(int i=0;i<9;++i)hit=hit&&(__float_as_int(m[i])==__float_as_int(key[i]));
 if(hit){for(int i=0;i<4;++i)out[i]=value[i];*hits+=1;return;}
 matrix_quaternion32(m,out);for(int i=0;i<9;++i)key[i]=m[i];
 for(int i=0;i<4;++i)value[i]=out[i];*ready=1;
}
}
// Standalone independent leaf ABI; one prepared object serves W1 and W2.
struct Context {
  int worlds;
  float* matrix; float* key; double* output; double* value;
  int* ready; int* hits; int* misses; int* mask;
  cudaStream_t stream; cudaEvent_t start, stop;
  cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
};
__global__ void matrix_leaf(Context c,int reset) {
  int world=blockIdx.x;if(threadIdx.x!=0)return;
  if(reset)c.ready[world]=0;
  if(!c.mask[world])return;
  if(CF_MEMO) {
    int oldhits=c.hits[world];
    libero_observe::matrix_quaternion32_memo(c.matrix+9*world,c.output+4*world,
      c.key+9*world,c.value+4*world,c.ready+world,c.hits+world);
    if(c.hits[world]==oldhits)c.misses[world]+=1;
  } else {
    libero_observe::matrix_quaternion32(c.matrix+9*world,c.output+4*world);
    c.misses[world]+=1;
  }
}
#define OK(call) do { cudaError_t status=(call); if(status!=cudaSuccess)return int(status); } while(0)
extern "C" int create(int worlds,void** result) {
  if(worlds!=1&&worlds!=2)return -1;
  Context* c=new Context{};c->worlds=worlds;*result=c;
  OK(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));
  OK(cudaEventCreate(&c->start));OK(cudaEventCreate(&c->stop));
  OK(cudaMalloc(&c->matrix,worlds*9*sizeof(float)));OK(cudaMalloc(&c->key,worlds*9*sizeof(float)));
  OK(cudaMalloc(&c->output,worlds*4*sizeof(double)));OK(cudaMalloc(&c->value,worlds*4*sizeof(double)));
  OK(cudaMalloc(&c->ready,worlds*sizeof(int)));OK(cudaMalloc(&c->hits,worlds*sizeof(int)));
  OK(cudaMalloc(&c->misses,worlds*sizeof(int)));OK(cudaMalloc(&c->mask,worlds*sizeof(int)));
  OK(cudaMemsetAsync(c->output,0,worlds*4*sizeof(double),c->stream));
  OK(cudaMemsetAsync(c->ready,0,worlds*sizeof(int),c->stream));
  OK(cudaMemsetAsync(c->hits,0,worlds*sizeof(int),c->stream));
  OK(cudaMemsetAsync(c->misses,0,worlds*sizeof(int),c->stream));
  OK(cudaStreamSynchronize(c->stream));return 0;
}
extern "C" int upload(void* ptr,const float* matrix,const int* mask) {
  Context& c=*static_cast<Context*>(ptr);
  OK(cudaMemcpyAsync(c.matrix,matrix,c.worlds*9*sizeof(float),cudaMemcpyHostToDevice,c.stream));
  OK(cudaMemcpyAsync(c.mask,mask,c.worlds*sizeof(int),cudaMemcpyHostToDevice,c.stream));
  OK(cudaStreamSynchronize(c.stream));return 0;
}
extern "C" int step(void* ptr,int reset) {
  Context& c=*static_cast<Context*>(ptr);
  matrix_leaf<<<c.worlds,64,0,c.stream>>>(c,reset);OK(cudaGetLastError());
  OK(cudaStreamSynchronize(c.stream));return 0;
}
extern "C" int read(void* ptr,double* output,int* hits,int* misses,int* ready) {
  Context& c=*static_cast<Context*>(ptr);
  OK(cudaMemcpy(output,c.output,c.worlds*4*sizeof(double),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(hits,c.hits,c.worlds*sizeof(int),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(misses,c.misses,c.worlds*sizeof(int),cudaMemcpyDeviceToHost));
  OK(cudaMemcpy(ready,c.ready,c.worlds*sizeof(int),cudaMemcpyDeviceToHost));return 0;
}
extern "C" int capture(void* ptr,int batch,int reset) {
  Context& c=*static_cast<Context*>(ptr);
  if(batch<8)return -2;
  if(c.executable){OK(cudaGraphExecDestroy(c.executable));c.executable=nullptr;}
  if(c.graph){OK(cudaGraphDestroy(c.graph));c.graph=nullptr;}
  OK(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeThreadLocal));
  for(int i=0;i<batch;i++)matrix_leaf<<<c.worlds,64,0,c.stream>>>(c,reset);
  OK(cudaStreamEndCapture(c.stream,&c.graph));
  OK(cudaGraphInstantiate(&c.executable,c.graph,nullptr,nullptr,0));return 0;
}
extern "C" int measure(void* ptr,float* milliseconds) {
  Context& c=*static_cast<Context*>(ptr);if(!c.executable)return -3;
  OK(cudaEventRecord(c.start,c.stream));OK(cudaGraphLaunch(c.executable,c.stream));
  OK(cudaEventRecord(c.stop,c.stream));OK(cudaEventSynchronize(c.stop));
  OK(cudaEventElapsedTime(milliseconds,c.start,c.stop));return 0;
}
extern "C" int destroy(void* ptr) {
  if(!ptr)return 0;Context& c=*static_cast<Context*>(ptr);
  cudaError_t first=cudaStreamSynchronize(c.stream);
  auto release=[&](cudaError_t status){if(first==cudaSuccess&&status!=cudaSuccess)first=status;};
  if(c.executable)release(cudaGraphExecDestroy(c.executable));if(c.graph)release(cudaGraphDestroy(c.graph));
  for(void* allocation : {static_cast<void*>(c.matrix),static_cast<void*>(c.key),static_cast<void*>(c.output),
      static_cast<void*>(c.value),static_cast<void*>(c.ready),static_cast<void*>(c.hits),
      static_cast<void*>(c.misses),static_cast<void*>(c.mask)})if(allocation)release(cudaFree(allocation));
  if(c.start)release(cudaEventDestroy(c.start));if(c.stop)release(cudaEventDestroy(c.stop));
  if(c.stream)release(cudaStreamDestroy(c.stream));delete &c;return int(first);
}
