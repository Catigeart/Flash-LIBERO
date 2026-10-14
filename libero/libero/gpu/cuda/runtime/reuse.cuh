// Exact resident-input cache. Consumption publication is distinct from math execution.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
namespace forward_reuse {
__device__ __forceinline__ void start(int w,const int* active,const int* first,const int* saved,
 int* hit) { hit[w]=active[w]==1 && first[w]==0 && saved[w]==1; }
// Inputs are resident global allocations. Alignment is proved before the PTX
// loads; integer registers preserve raw bits without C++ type-punning aliases.
template<class T> __device__ __forceinline__ T raw_word(const unsigned char* p) {
 T value;memcpy(&value,p,sizeof(T));return value;
}
template<> __device__ __forceinline__ uint64_t raw_word<uint64_t>(const unsigned char* p) {
#if defined(__CUDA_ARCH__)
 uint64_t value;asm("ld.global.u64 %0, [%1];" : "=l"(value) : "l"(p));return value;
#else
 uint64_t value;memcpy(&value,p,sizeof(value));return value;
#endif
}
template<> __device__ __forceinline__ uint32_t raw_word<uint32_t>(const unsigned char* p) {
#if defined(__CUDA_ARCH__)
 uint32_t value;asm("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(p));return value;
#else
 uint32_t value;memcpy(&value,p,sizeof(value));return value;
#endif
}
template<> __device__ __forceinline__ uint4 raw_word<uint4>(const unsigned char* p) {
#if defined(__CUDA_ARCH__)
 uint4 v;asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));return v;
#else
 uint4 v;memcpy(&v,p,sizeof(v));return v;
#endif
}
template<class Word> __device__ __forceinline__ bool word_different(Word a,Word b){return a!=b;}
template<> __device__ __forceinline__ bool word_different<uint4>(uint4 a,uint4 b){return a.x!=b.x||a.y!=b.y||a.z!=b.z||a.w!=b.w;}
template<class Word> __device__ __forceinline__ bool compare_words(
 const unsigned char* current,const unsigned char* saved,int bytes,int head) {
 const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
 const int words=(bytes-head)/int(sizeof(Word));
 const int tail=head+words*int(sizeof(Word));
 bool mismatch=false;
 if(tid<head)mismatch=current[tid]!=saved[tid];
 if(tail+tid<bytes)mismatch|=current[tail+tid]!=saved[tail+tid];
 if(bytes<4096) {
  // Short/medium keys keep independent early exits without extra tile votes.
  for(int i=tid;i<words&&!mismatch;i+=blockDim.x) {
   const size_t at=size_t(head)+size_t(i)*sizeof(Word);
   mismatch=word_different(raw_word<Word>(current+at),raw_word<Word>(saved+at));
  }
 } else {
  // Four adjacent warp tiles per vote: a mismatch stops its entire warp,
  // limiting unnecessary reads without adding a block barrier per tile.
  for(int base=warp*128;base<words;base+=(int(blockDim.x)>>5)*128) {
   for(int tile=0;tile<4&&!mismatch;++tile) {
    const int i=base+tile*32+lane;
    if(i<words) {
     const size_t at=size_t(head)+size_t(i)*sizeof(Word);
     mismatch=word_different(raw_word<Word>(current+at),raw_word<Word>(saved+at));
    }
   }
   if(__any_sync(0xffffffffu,mismatch)){mismatch=true;break;}
  }
 }
 return mismatch;
}
// Collective entries require complete warps and a stage barrier after returning.
// Compare bytes directly: numeric equality would lose signed-zero / NaN payloads.
// Small fields use one warp without an internal block barrier. Long fields use
// the whole block; only the leader publishes a mismatch to the resident flag.
__device__ __forceinline__ void compare_span(const unsigned char* current,
 const unsigned char* saved,int bytes,int* hit) {
 if(!*hit)return;
 const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
 if(bytes<128) {
  if(warp)return;
  bool mismatch=false;
  for(int k=lane;k<bytes;k+=32)if(current[k]!=saved[k]) {mismatch=true;break;}
  const bool bad=__any_sync(0xffffffffu,mismatch);
  if(!lane && bad)*hit=0;
  return;
 }
 bool mismatch;
 const uintptr_t a=reinterpret_cast<uintptr_t>(current),b=reinterpret_cast<uintptr_t>(saved);
 if(((a^b)&15u)==0) {
  const int head=int((uintptr_t(0)-a)&15u);
  mismatch=compare_words<uint4>(current,saved,bytes,head);
 } else if(((a^b)&7u)==0) {
  const int head=int((uintptr_t(0)-a)&7u);
  mismatch=compare_words<uint64_t>(current,saved,bytes,head);
 } else if(((a^b)&3u)==0) {
  const int head=int((uintptr_t(0)-a)&3u);
  mismatch=compare_words<uint32_t>(current,saved,bytes,head);
 } else mismatch=compare_words<unsigned char>(current,saved,bytes,0);
 __shared__ int warp_bad[8];
 const bool bad=__any_sync(0xffffffffu,mismatch);
 if(!lane)warp_bad[warp]=bad;
 __syncthreads();
 if(!warp) {
  const bool block_bad=__any_sync(0xffffffffu,
      lane<(int(blockDim.x)>>5) && warp_bad[lane]!=0);
  if(!lane && block_bad)*hit=0;
 }
}
__device__ __forceinline__ void compare(int w,const unsigned char* current,
 const unsigned char* saved,int bytes,int* hit) {
 if(bytes<=0)return;
 const size_t offset=size_t(w)*size_t(bytes);
 compare_span(current+offset,saved+offset,bytes,hit+w);
}
__device__ __forceinline__ void compare_shared(int w,const unsigned char* current,
 const unsigned char* saved,int bytes,int* hit) {
 if(bytes<=0)return;
 compare_span(current,saved+size_t(w)*size_t(bytes),bytes,hit+w);
}
__device__ __forceinline__ void save_shared(int w,const int* active,const int* first,
 const int* bias,const int* reference,const int* status,const unsigned char* source,
 unsigned char* saved,int bytes) {
 // Each world snapshots broadcast bytes independently, including when world zero is inactive.
 if(active[w]!=1 || first[w])return;
 const size_t offset=size_t(w)*size_t(bytes);
 libero_vec::copy_bytes_lane(source,saved+offset,size_t(bytes),int(threadIdx.x),int(blockDim.x));
}
__device__ __forceinline__ void predicate(int w,const int* active,const int* hit,int worlds,int* miss) {
 if(w)return; miss[0]=0;
 for(int j=0;j<worlds;++j)if(active[j]==1 && !hit[j]) {miss[0]=1;break;}
}
__device__ __forceinline__ void save_bytes(int w,const int* active,const int* first,
 const int* bias,const int* reference,const int* status,const unsigned char* source,
 unsigned char* saved,int bytes) {
 if(active[w]!=1 || first[w])return;
 const size_t offset=size_t(w)*size_t(bytes);
 libero_vec::copy_bytes_lane(source+offset,saved+offset,size_t(bytes),int(threadIdx.x),int(blockDim.x));
}
__device__ __forceinline__ void save_valid(int w,const int* active,const int* first,
 const int* bias,const int* reference,const int* status,int* saved,uint64_t* math) {
 if(active[w]!=1)return;
 saved[w]=!first[w] && bias[w]==1 && reference[w]==1 && !status[w];
 if(saved[w])++math[w];
}
__device__ __forceinline__ void publish(int w,const int* active,const int* hit,
 const int* first,const int64_t* epoch,int* position,int* velocity,int* bias,
 int64_t* position_written,int64_t* velocity_written,int64_t* bias_written,
 int64_t* ref_epoch,uint64_t* consumed,int* collision_mask,int* transport_active) {
 collision_mask[w]=transport_active[w]=active[w]==1 && hit[w]==1 && !first[w];
 if(active[w]!=1 || hit[w]!=1 || first[w])return;
 position[w]=velocity[w]=bias[w]=1;
 position_written[w]=velocity_written[w]=bias_written[w]=ref_epoch[w]=epoch[w];
 ++consumed[w];
}
}
